"""One fixed-input GEMM replay. CPU --plan; GPU only with explicit --run.

The parent launches a worker so rocBLAS and TunableOp logs are flushed before
asserting that the requested CSV signature was actually invoked. Never imports
torch in the parent or --plan path. Operating CSVs are read-only.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import traceback

from v10_tunable_inventory import QWEN_CSV, read_csv, sha

ROOT = Path(__file__).resolve().parent


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=QWEN_CSV)
    p.add_argument("--entry", type=int, default=16, help="Zero-based non-validator row")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261108)
    p.add_argument("--repeats", type=int, default=32)
    p.add_argument("--timing-repeats", type=int, default=12)
    p.add_argument("--timing-groups", type=int, default=5)
    p.add_argument("--solution", default="original", help="original, Default or exact Gemm_Rocblas_ID")
    p.add_argument("--alpha", type=float, default=1)
    p.add_argument("--beta", type=float, default=0)
    p.add_argument("--batch-strides", help="Exact A,B,C strides in elements; otherwise canonical packed")
    p.add_argument("--max-storage-gib", type=float, default=2)
    p.add_argument("--save-outputs", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def layout(entry, batch_strides):
    e = entry
    shapes = [(e["m"], e["k"]) if e["transa"] == "n" else (e["k"], e["m"]),
              (e["k"], e["n"]) if e["transb"] == "n" else (e["n"], e["k"]),
              (e["m"], e["n"])]
    lds = [e["lda"], e["ldb"], e["ldc"]]
    if e["transa"] not in ("n", "t") or e["transb"] not in ("n", "t"):
        raise ValueError("This probe supports real N/T operands only")
    batch = e["batch"]
    strides = [ld * shape[1] for shape, ld in zip(shapes, lds)]
    if batch_strides:
        strides = [int(v) for v in batch_strides.split(",")]
        if len(strides) != 3 or any(v < 0 for v in strides):
            raise ValueError("Expected non-negative A,B,C batch strides")
    layouts = []
    for shape, ld, bs in zip(shapes, lds, strides):
        if ld < shape[0]:
            raise ValueError("Leading dimension smaller than physical row count")
        span = (shape[1] - 1) * ld + shape[0]
        if e["batched"] and bs < span and bs != 0:
            # Interleaved heads can have a smaller batch stride without aliasing,
            # but this CPU prototype deliberately requires exact captured layouts.
            raise ValueError("Overlapping/interleaved batch layout requires captured tensors; not guessed")
        if e["batched"] and len(layouts) == 2 and bs < span:
            raise ValueError("Output batches may not overlap")
        layouts.append({"physical_shape": list(shape), "ld": ld, "batch_stride": bs,
                        "storage_elements": span + (batch - 1) * bs})
    return layouts


def prepare(a):
    validators, rows = read_csv(a.csv)
    if not 0 <= a.entry < len(rows):
        raise ValueError(f"entry must be in [0,{len(rows)-1}]")
    e = rows[a.entry]
    if e["dtype"] not in ("float", "Half"):
        raise ValueError("Only float/Half CSV dtypes supported")
    selected = e["solution"] if a.solution == "original" else a.solution
    if selected != "Default" and not re.fullmatch(r"Gemm_Rocblas_-?\d+", selected):
        raise ValueError("Unsupported explicit solution")
    layouts = layout(e, a.batch_strides)
    element_bytes = 2 if e["dtype"] == "Half" else 4
    storage = sum(x["storage_elements"] for x in layouts) * element_bytes
    if storage > a.max_storage_gib * 2**30:
        raise ValueError("Requested tensor storage exceeds explicit limit")
    return {"status": "plan_gpu_not_run", "csv": str(a.csv.resolve()),
            "csv_sha256": sha(a.csv), "validators": validators,
            "entry": e, "requested_solution": selected, "layouts": layouts,
            "alpha": a.alpha, "beta": a.beta, "seed": a.seed,
            "repeats": a.repeats, "timing_repeats": a.timing_repeats,
            "timing_groups": a.timing_groups, "tensor_storage_bytes": storage,
            "scope": "seeded synthetic inputs; CSV geometry, not captured model tensor values",
            "batch_stride_source": "explicit" if a.batch_strides else "canonical packed assumption"}


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def make_private_csv(a, plan):
    target = a.out / "replay.csv"
    rows = list(csv.reader(a.csv.open(newline="")))
    for row in rows:
        if row[:2] == [plan["entry"]["op"], plan["entry"]["signature"]]:
            row[2] = plan["requested_solution"]
    with target.open("w", newline="") as fp:
        csv.writer(fp, lineterminator="\n").writerows(rows)
    return target


def metrics(torch, got, reference):
    d = got.double() - reference.double()
    finite = bool(torch.isfinite(got).all() and torch.isfinite(reference).all())
    rms = float(reference.double().square().mean().sqrt())
    if not finite:
        return {"finite": False, "max_abs": None, "rmse": None, "nrmse": None}
    err = float(d.square().mean().sqrt())
    return {"finite": True, "max_abs": float(d.abs().max()), "rmse": err,
            "nrmse": err / max(rms, 1e-30)}


def worker(a, plan):
    # All imports which may create a GPU client are below the guard.
    from gpu_guard_v06 import GpuAudit
    result = dict(plan, status="gpu_worker_started", completed=False)
    audit = GpuAudit()
    try:
        with audit:
            audit.ensure_clear()
            import torch
            import torch.cuda.tunable as tunable
            result["runtime"] = {"python": sys.executable, "torch": torch.__version__,
                                 "hip": torch.version.hip}
            dtype = torch.float16 if plan["entry"]["dtype"] == "Half" else torch.float32
            gen = torch.Generator(device="cpu").manual_seed(a.seed)
            tensors, bases, input_hashes = [], [], []
            e = plan["entry"]
            for i, spec in enumerate(plan["layouts"]):
                cpu = torch.randn(spec["storage_elements"], dtype=dtype, generator=gen)
                input_hashes.append(hashlib.sha256(cpu.numpy().tobytes()).hexdigest())
                base = cpu.to("cuda")
                bases.append(base)
                physical = tuple(spec["physical_shape"])
                dims = (e["batch"], *physical) if e["batched"] else physical
                strides = (spec["batch_stride"], 1, spec["ld"]) if e["batched"] else (1, spec["ld"])
                tensors.append(torch.as_strided(base, dims, strides))
            raw_a, raw_b, out = tensors
            aa = raw_a if e["transa"] == "n" else raw_a.transpose(-2, -1)
            bb = raw_b if e["transb"] == "n" else raw_b.transpose(-2, -1)
            initial_c = out.clone()
            result["input_storage_sha256"] = input_hashes
            result["actual_tensors"] = [{"shape": list(t.shape), "stride": list(t.stride()),
                                         "dtype": str(t.dtype), "storage_offset": t.storage_offset()}
                                        for t in (aa, bb, out)]

            def invoke():
                if e["batched"]:
                    return torch.baddbmm(out, aa, bb, beta=a.beta, alpha=a.alpha, out=out)
                return torch.addmm(out, aa, bb, beta=a.beta, alpha=a.alpha, out=out)

            def reset_output():
                out.copy_(initial_c)

            # Empty initial DB records the actual dispatcher signature. The parent
            # checks the flushed untuned CSV before marking this probe valid.
            tunable.set_filename(str(a.out / "observed.csv"), insert_device_ordinal=False)
            tunable.enable(True)
            tunable.tuning_enable(False)
            tunable.record_untuned_enable(True)
            tunable.set_rotating_buffer_size(0)
            reset_output()
            invoke()
            torch.cuda.synchronize()
            tunable.record_untuned_enable(False)
            observed_now = []
            for observed_file in a.out.glob("observed_untuned*.csv"):
                observed_now.extend(row[:2] for row in csv.reader(observed_file.open(newline="")) if row)
            result["pre_replay_signature_rows"] = observed_now
            if observed_now != [[e["op"], e["signature"]]]:
                raise RuntimeError("Actual dispatcher signature differs from requested CSV row")
            loaded = tunable.read_file(str(a.out / "replay.csv"))
            result["read_file_return"] = loaded
            known = list(tunable.get_results())
            matched = [list(row) for row in known if list(row)[:2] == [e["op"], e["signature"]]]
            result["loaded_target_rows"] = matched
            if not loaded or len(matched) != 1 or matched[0][2] != plan["requested_solution"]:
                raise RuntimeError("Target CSV solution was not loaded exactly")
            if tunable.tuning_is_enabled():
                raise RuntimeError("Tuning unexpectedly enabled")
            result["validators_loaded"] = list(tunable.get_validators())
            result["modes"] = {}
            firsts = {}
            for mode, enabled in (("on", True), ("off", False)):
                tunable.enable(enabled)
                if bool(tunable.is_enabled()) != enabled:
                    raise RuntimeError("TunableOp toggle overridden by inherited environment")
                samples, hashes = [], []
                reference = None
                for repeat in range(a.repeats):
                    reset_output()
                    invoke()
                    torch.cuda.synchronize()
                    value = out.detach().cpu().contiguous()
                    digest = hashlib.sha256(value.numpy().tobytes()).hexdigest()
                    if reference is None:
                        reference = value.clone()
                        firsts[mode] = reference
                    hashes.append(digest)
                    samples.append(metrics(torch, value, reference))
                medians = []
                for _ in range(a.timing_groups):
                    reset_output()
                    if a.beta:
                        # For beta!=0 use single calls with identical initial C.
                        starts, ends = [], []
                        for _ in range(a.timing_repeats):
                            reset_output()
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            start.record(); invoke(); end.record()
                            starts.append(start); ends.append(end)
                        torch.cuda.synchronize()
                        medians.append(statistics.median(s.elapsed_time(t) for s, t in zip(starts, ends)))
                    else:
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(a.timing_repeats):
                            invoke()
                        end.record()
                        torch.cuda.synchronize()
                        medians.append(start.elapsed_time(end) / a.timing_repeats)
                result["modes"][mode] = {"hashes": hashes, "distinct_output_hashes": len(set(hashes)),
                                         "bitwise_repeatable": len(set(hashes)) == 1,
                                         "all_finite": all(sample["finite"] for sample in samples),
                                         "repeat_vs_first": samples,
                                         "event_ms_groups": medians, "event_ms_median": statistics.median(medians)}
                if a.save_outputs:
                    torch.save(reference, a.out / f"output_{mode}.pt")
                if a.profile:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        reset_output()
                        with torch.profiler.record_function(f"v10_tunable_{mode}"):
                            invoke()
                        torch.cuda.synchronize()
                    prof.export_chrome_trace(str(a.out / f"{mode}.trace.json"))
            result["on_vs_off"] = metrics(torch, firsts["on"], firsts["off"])
            result["on_vs_off"]["bitwise_equal"] = (result["modes"]["on"]["hashes"][0] ==
                                                       result["modes"]["off"]["hashes"][0])
            result["on_speedup_vs_off"] = (result["modes"]["off"]["event_ms_median"] /
                                            result["modes"]["on"]["event_ms_median"])
            # Verify no operator modified A/B, including physical padding storage.
            result["input_storage_sha256_after"] = [
                hashlib.sha256(base.cpu().numpy().tobytes()).hexdigest() for base in bases[:2]]
            result["inputs_unchanged"] = result["input_storage_sha256_after"] == input_hashes[:2]
            if not result["inputs_unchanged"]:
                raise RuntimeError("GEMM modified input tensor storage")
            result["input_logical_sha256_after"] = [
                hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                for t in (raw_a, raw_b)]
            result["all_finite"] = (all(mode["all_finite"] for mode in result["modes"].values())
                                    and result["on_vs_off"]["finite"])
            result["completed"] = result["all_finite"]
            result["status"] = ("worker_complete_pending_signature_assertion" if result["all_finite"]
                                else "failed_nonfinite_output")
            if not result["all_finite"]:
                result["error"] = "Nonfinite GEMM output; raw diagnostics retained, timing invalid"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        result["gpu_audit"] = audit.report()
        result["valid_for_timing"] = bool(result.get("completed") and audit.report()["valid_for_timing"])
        write(a.out / "worker_result.json", result)
    return 0 if result["completed"] else 1


def finalize(a, plan, returncode):
    path = a.out / "worker_result.json"
    result = json.loads(path.read_text()) if path.exists() else dict(plan, completed=False)
    result["worker_exit_code"] = returncode
    wanted = [plan["entry"]["op"], plan["entry"]["signature"]]
    observed = []
    for file in sorted(a.out.glob("*.csv")):
        if file.name == "replay.csv":
            continue
        for row in csv.reader(file.open(newline="")):
            if row and row[0].startswith(("GemmTunableOp_", "GemmStridedBatchedTunableOp_")):
                observed.append({"file": str(file), "key": row[:2], "fields": row})
    result["observed_signature_rows"] = observed
    result["signature_hit"] = any(x["key"] == wanted for x in observed)
    log = a.out / "rocblas_profile.log"
    result["rocblas_log"] = {"path": str(log), "exists": log.exists(),
                            "bytes": log.stat().st_size if log.exists() else 0}
    if log.exists():
        text = log.read_text(errors="replace")
        sid = plan["requested_solution"]
        result["requested_solution_in_rocblas_log"] = (
            bool(re.search(r"solution_index:\s*" + re.escape(sid.rsplit("_", 1)[-1]) + r"(?:[, }]|$)", text))
            if sid.startswith("Gemm_Rocblas_") else None)
    result["completed"] = bool(result.get("completed") and returncode == 0 and result["signature_hit"])
    result["valid_for_timing"] = bool(result["completed"] and result.get("valid_for_timing"))
    result["status"] = "completed_signature_verified" if result["completed"] else "failed_or_signature_not_verified"
    write(a.out / "result.json", result)
    print(json.dumps({k: result.get(k) for k in ("status", "completed", "valid_for_timing",
                                                "signature_hit", "requested_solution",
                                                "requested_solution_in_rocblas_log", "on_speedup_vs_off")}, indent=2))
    return 0 if result["completed"] else 1


def main():
    a = parse()
    if min(a.repeats, a.timing_repeats, a.timing_groups) < 2:
        raise ValueError("At least two repeats and timing groups required")
    plan = prepare(a)
    if a.worker:
        return worker(a, plan)
    if not a.run:
        print(json.dumps(plan, indent=2))
        return 0
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    write(a.out / "plan.json", plan)
    make_private_csv(a, plan)
    env = os.environ.copy()
    for name in ("PYTORCH_TUNABLEOP_ENABLED", "PYTORCH_TUNABLEOP_TUNING", "PYTORCH_TUNABLEOP_FILENAME",
                 "PYTORCH_TUNABLEOP_RECORD_UNTUNED", "PYTORCH_TUNABLEOP_VERBOSE"):
        env.pop(name, None)
    env.update({"PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": str(a.seed),
                "HSA_DISABLE_COREDUMP_ON_EXCEPTION": "1",
                "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1": "0",
                "MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53": "0",
                "PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED": "0",
                "PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE": "0",
                "PYTORCH_TUNABLEOP_UNTUNED_FILENAME": str(a.out / "observed_untuned.csv"),
                "ROCBLAS_LAYER": "4", "ROCBLAS_LOG_PROFILE_PATH": str(a.out / "rocblas_profile.log")})
    with (a.out / "worker.log").open("w") as fp:
        rc = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                             *sys.argv[1:], "--worker"], env=env, stdout=fp, stderr=subprocess.STDOUT).returncode
    return finalize(a, plan, rc)


if __name__ == "__main__":
    raise SystemExit(main())

