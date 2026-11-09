#!/usr/bin/env python3
"""Serialized BF16 / FP8-storage GEMM experiments in the existing gfx1030 env.

This is an operator experiment, never an automatic Comfy dispatch change.
References are native full tensors plus independently computed CPU FP64
selected rows/columns. Timings include conversions unless explicitly prepared.
Accuracy gates are declared below before execution; model RGB gates are separate.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import traceback

import torch
import triton

from gpu_guard_v06 import GpuAudit
from research_comfy_v05 import profile_tables
from rdna2_ops.bf16_gemm_v06 import bf16_gemm, fp16_exact_inputs, fp16_finite_inputs
from rdna2_ops.fp8_weight_gemm_v06 import (
    decode_fp8_weight, fp8_weight_gemm, prepare_int8_weight)


GATES = {
    "bf16": {"normalized_rmse": 0.004, "normalized_max": 0.04},
    "fp8_exact": {"normalized_rmse": 0.002, "normalized_max": 0.02},
    "fp8_int8": {"normalized_rmse": 0.03, "normalized_max": 0.15},
    "rgb": {"max_abs": 2 / 255, "rmse": 0.25 / 255},
}


def tensor_error(value, reference):
    v, r = value.float(), reference.float()
    finite = bool(torch.isfinite(v).all() and torch.isfinite(r).all())
    if not finite:
        return {"finite": False, "bit_equal": False}
    delta = v - r
    rms = float(r.square().mean().sqrt())
    maximum = float(delta.abs().max())
    rmse = float(delta.square().mean().sqrt())
    return {"finite": True, "bit_equal": torch.equal(value, reference),
            "max_abs": maximum, "rmse": rmse, "reference_rms": rms,
            "normalized_rmse": rmse / max(rms, 1e-30),
            "normalized_max": maximum / max(rms, 1e-30)}


def passes(error, gate):
    return (error["finite"] and error["normalized_rmse"] <= gate["normalized_rmse"]
            and error["normalized_max"] <= gate["normalized_max"])


def cpu_reference(a, b, *, weight=False):
    # A full-K dot for selected evenly spaced output coordinates. No GPU GEMM
    # is used as the independent oracle; input storage values are preserved.
    m, k = a.shape[-2:]
    n = b.shape[-2] if weight else b.shape[-1]
    rows = torch.linspace(0, m - 1, min(24, m), dtype=torch.int64)
    cols = torch.linspace(0, n - 1, min(24, n), dtype=torch.int64)
    aa = a[0] if a.ndim == 3 else a
    bb = b[0] if b.ndim == 3 else b
    av = aa.index_select(0, rows.to(a.device)).float().cpu().double()
    if weight:
        bv = bb.float().index_select(0, cols.to(b.device)).cpu().double().T
    else:
        bv = bb.index_select(1, cols.to(b.device)).float().cpu().double()
    with torch.inference_mode():
        ref = av @ bv
    return rows, cols, ref


def sampled_error(value, oracle):
    rows, cols, ref = oracle
    v = value[0] if value.ndim == 3 else value
    v = v.index_select(0, rows.to(v.device)).index_select(1, cols.to(v.device))
    v = v.float().cpu().double()
    delta = v - ref
    rms = float(ref.square().mean().sqrt())
    return {"method": "CPU FP64 dot over full K, 24x24 selected output coordinates, first batch",
            "finite": bool(torch.isfinite(v).all() and torch.isfinite(ref).all()),
            "max_abs": float(delta.abs().max()), "rmse": float(delta.square().mean().sqrt()),
            "normalized_rmse": float(delta.square().mean().sqrt()) / max(rms, 1e-30),
            "reference_rms": rms, "reference_shape": list(ref.shape)}


def timed(fn, repeats):
    torch.cuda.synchronize()
    before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    before.record()
    for _ in range(repeats):
        value = fn()
    after.record()
    after.synchronize()
    return {"wall_ms": (time.perf_counter() - wall) * 1000 / repeats,
            "event_ms": before.elapsed_time(after) / repeats}


def memory(fn):
    gc.collect()
    torch.cuda.synchronize()
    start = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    value = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - start
    return {"additional_peak_allocated_bytes": peak,
            "output_bytes": value.numel() * value.element_size()}


def profile(fn, path):
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True) as prof:
        value = fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    return profile_tables(prof, path)


def run_case(case, methods, baseline, oracle, gate_for, out, audit, rounds, repeats):
    # Compile and validate every candidate before measuring warmed rounds.
    active = {}
    for name, fn in methods.items():
        audit.ensure_clear()
        row = {"method": name}
        case["methods"].append(row)
        try:
            metadata = {}
            value = fn(metadata)
            torch.cuda.synchronize()
            row["error_vs_native"] = tensor_error(value, baseline)
            row["error_vs_cpu_fp64"] = sampled_error(value, oracle)
            row["operator_gate"] = gate_for(name)
            row["operator_pass"] = passes(row["error_vs_native"], row["operator_gate"])
            row["kernel_metadata"] = metadata
            for _ in range(2):
                value = fn(None)
            torch.cuda.synchronize()
            del value
            row["memory"] = memory(lambda: fn(None))
            active[name] = (fn, row)
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc())
            print(json.dumps({"case": case["id"], "method": name, "error": row["error"]}), flush=True)
    names = list(active)
    for trial in range(rounds):
        # Rotate order each round instead of benchmarking one method to completion.
        order = names[trial % len(names):] + names[:trial % len(names)] if names else []
        if trial % 2:
            order.reverse()
        for name in order:
            audit.ensure_clear()
            fn, row = active[name]
            row.setdefault("samples", []).append(timed(lambda: fn(None), repeats))
    for name, (fn, row) in active.items():
        row["median_wall_ms"] = statistics.median(x["wall_ms"] for x in row["samples"])
        row["median_event_ms"] = statistics.median(x["event_ms"] for x in row["samples"])
        row["profile"] = profile(lambda: fn(None), out / (case["id"] + "_" + name + ".trace.json"))
        row["status"] = "measured"
    native = next((r for r in case["methods"] if r["method"] in ("native", "upcast")
                   and r.get("status") == "measured"), None)
    if native:
        for row in case["methods"]:
            if row.get("status") == "measured":
                row["speedup_event"] = native["median_event_ms"] / row["median_event_ms"]
                row["speedup_wall"] = native["median_wall_ms"] / row["median_wall_ms"]
    print(json.dumps({"case": case["id"], "rows": [
        {k: r.get(k) for k in ("method", "status", "median_event_ms", "speedup_event", "operator_pass")}
        for r in case["methods"]]}), flush=True)


def bf16_cases(report, out, audit, cfg):
    for label, batch, m, n, k in (
            ("square", 0, 1024, 1024, 1024),
            ("tall_skinny", 0, 4096, 128, 4096),
            ("vae_pv", 1, 1024, 384, 1024)):
        torch.manual_seed(20261108)
        lead = (batch,) if batch else ()
        a = (torch.randn((*lead, m, k), device="cuda") * 0.5).bfloat16()
        b = (torch.randn((*lead, n, k), device="cuda") * (0.5 / math.sqrt(k))).bfloat16().transpose(-1, -2)
        case = {"id": "bf16_" + label, "kind": "bmm" if batch else "mm",
                "B": batch or 1, "M": m, "N": n, "K": k,
                "a_stride": list(a.stride()), "b_stride": list(b.stride()),
                "input_source": "seeded synthetic BF16; dimensions chosen for square, attention tall-skinny and MOSS VAE-like P.V",
                "methods": []}
        report["cases"].append(case)
        case["fp16_exact_inputs"] = fp16_exact_inputs(a, b)
        case["fp16_finite_inputs"] = fp16_finite_inputs(a, b)
        case["fp16_input_reexpression_error"] = {"a": tensor_error(a.half().bfloat16(), a),
                                                 "b": tensor_error(b.half().bfloat16(), b)}
        case["range_check_timing"] = timed(lambda: fp16_finite_inputs(a, b), cfg.repeats)
        case["exact_check_timing"] = timed(lambda: fp16_exact_inputs(a, b), cfg.repeats)
        try:
            baseline = bf16_gemm(a, b, method="native")
            case["baseline_output"] = "native BF16"
        except Exception as exc:
            case["native_baseline_error"] = f"{type(exc).__name__}: {exc}"
            baseline = bf16_gemm(a, b, method="fp32")
            case["baseline_output"] = "FP32 conversion fallback; native BF16 unsupported, no speedup claim"
        oracle = cpu_reference(a, b)
        methods = {name: (lambda meta, name=name: bf16_gemm(
            a, b, method=name, check_fp16=False, require_exact_fp16=False, metadata=meta))
            for name in ("native", "fp32", "fp16_f32", "fp16_f16", "dot2", "simt_fp32")}
        if not case["fp16_finite_inputs"]:
            raise AssertionError("Unexpected generated input outside finite FP16 range")
        run_case(case, methods, baseline, oracle, lambda name: GATES["bf16"],
                 out, audit, cfg.rounds, cfg.repeats)
        save(report, out)
        del methods, baseline, a, b
        gc.collect()
        torch.cuda.empty_cache()
    range_cases = []
    for label, x in (("overflow", 2.0**80), ("underflow", 2.0**-120)):
        a = torch.full((16, 16), x, dtype=torch.bfloat16, device="cuda")
        b = torch.ones_like(a)
        rejected = False
        try:
            bf16_gemm(a, b, method="dot2")
        except ValueError as exc:
            rejected = "not exactly representable" in str(exc)
        range_cases.append({"case": label, "input": x, "half_guard_rejected": rejected})
    report["range_guard_probes"] = range_cases


def decode_exhaustive(dtype):
    raw = torch.arange(256, device="cuda", dtype=torch.int16).to(torch.uint8).reshape(16, 16)
    fp8 = raw.view(dtype)
    native, meta = fp8.half(), {}
    candidate = decode_fp8_weight(fp8, metadata=meta)
    finite = torch.isfinite(native)
    result = {"dtype": str(dtype), "patterns": 256,
              "finite_bit_equal": torch.equal(native[finite].view(torch.int16), candidate[finite].view(torch.int16)),
              "nan_mask_equal": torch.equal(torch.isnan(native), torch.isnan(candidate)),
              "positive_inf_equal": torch.equal(torch.isposinf(native), torch.isposinf(candidate)),
              "negative_inf_equal": torch.equal(torch.isneginf(native), torch.isneginf(candidate)),
              "kernel_metadata": meta}
    result["pass"] = all(result[k] for k in ("finite_bit_equal", "nan_mask_equal", "positive_inf_equal", "negative_inf_equal"))
    return result


def fp8_cases(report, out, audit, cfg):
    report["fp8_exhaustive_decode"] = [decode_exhaustive(dt) for dt in (torch.float8_e4m3fn, torch.float8_e5m2)]
    if not all(r["pass"] for r in report["fp8_exhaustive_decode"]):
        raise AssertionError("Exhaustive FP8 decoder check failed")
    for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        for label, m, n, k in (("anima_projection", 8192, 2048, 2048),
                               ("anima_mlp", 8192, 8192, 2048),
                               ("encoder_projection", 512, 3072, 1024)):
            torch.manual_seed(20261108)
            a = (torch.randn((m, k), device="cuda") * 0.25).half()
            original = (torch.randn((n, k), device="cuda") / math.sqrt(k)).bfloat16()
            w = original.to(dtype)
            prepared_half, prepared_int8 = w.half(), prepare_int8_weight(w)
            case = {"id": str(dtype).replace("torch.float8_", "fp8_") + "_" + label,
                    "M": m, "N": n, "K": k, "dtype": str(dtype),
                    "a_stride": list(a.stride()), "weight_stride": list(w.stride()),
                    "input_source": "seeded synthetic BF16 weight converted once to unscaled FP8; dimensions from existing Anima diffusion/encoder",
                    "original_bf16_to_fp8_error": tensor_error(w.float(), original.float()),
                    "prepared_fp16_bytes": prepared_half.numel() * prepared_half.element_size(),
                    "prepared_int8_bytes": sum(t.numel() * t.element_size() for t in prepared_int8),
                    "methods": []}
            report["cases"].append(case)
            baseline = fp8_weight_gemm(a, w, method="upcast")
            oracle = cpu_reference(a, w, weight=True)
            for _ in range(2):
                w.half()
                prepare_int8_weight(w)
            case["preparation"] = {
                "fp16": timed(lambda: w.half(), cfg.repeats),
                "int8": timed(lambda: prepare_int8_weight(w), cfg.repeats),
                "semantics": "per-current-weight preparation; prepared methods exclude this cost and retain extra resident storage"}
            methods = {}
            for name in ("upcast", "decode", "prepared_fp16", "fused_dot2", "int8", "prepared_int8"):
                prep = prepared_half if name == "prepared_fp16" else prepared_int8 if name == "prepared_int8" else None
                methods[name] = lambda meta, name=name, prep=prep: fp8_weight_gemm(
                    a, w, method=name, prepared=prep, metadata=meta)
            run_case(case, methods, baseline, oracle,
                     lambda name: GATES["fp8_int8" if "int8" in name else "fp8_exact"],
                     out, audit, cfg.rounds, cfg.repeats)
            save(report, out)
            del methods, baseline, a, w, original, prepared_half, prepared_int8, prep
            gc.collect()
            torch.cuda.empty_cache()


def save(report, out):
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("bf16", "fp8"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    cfg = parser.parse_args()
    if cfg.rounds < 3 or cfg.repeats < 1:
        parser.error("At least three alternating rounds and one repeat required")
    cfg.output.mkdir(parents=True, exist_ok=True)
    if (cfg.output / "results.json").exists():
        raise FileExistsError("Use a new results directory; preserve prior evidence")
    torch.set_num_threads(2)
    report = {"status": "started", "kind": cfg.kind, "gates_declared_before_run": GATES,
              "timing": {"rounds": cfg.rounds, "repeats": cfg.repeats,
                         "conversion_included": "all per-call input/output/quantization; prepared weight setup separately reported",
                         "not_in_timing": "reference, finite/range validation, profiles; no model performance claimed"},
              "software": {"torch": torch.__version__, "hip": torch.version.hip, "triton": triton.__version__},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": []}
    audit = GpuAudit()
    try:
        with audit, torch.inference_mode():
            audit.ensure_clear()
            props = torch.cuda.get_device_properties(0)
            report["device"] = {"name": props.name, "arch": props.gcnArchName, "total_memory": props.total_memory}
            if props.gcnArchName.split(":")[0] != "gfx1030":
                raise RuntimeError("Wrong target GPU")
            (bf16_cases if cfg.kind == "bf16" else fp8_cases)(report, cfg.output, audit, cfg)
            audit.ensure_clear()
            report["status"] = "measured"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        report["gpu_audit"] = audit.report()
        if not report["gpu_audit"]["valid_for_timing"]:
            report["status"] = "invalid_timing_discarded"
            report["timing"]["valid"] = False
        save(report, cfg.output)


if __name__ == "__main__":
    main()
