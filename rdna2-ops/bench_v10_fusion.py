#!/usr/bin/env python3
"""Fixed-seed FP32 operator gates and actual ResBlock AMP/weight_norm contract.

GPU runs are owned by GpuAudit before importing torch. B is the v0.8 selected
direction manifest, B_prime is the same directions plus v1.0 fusion. This is
a pair microbenchmark; operating-path epoch timing is a separate requirement.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
SAFETY = {"MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1": "0",
          "MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53": "0", "HSA_DISABLE_COREDUMP_ON_EXCEPTION": "1"}
GATE = {"finite": True, "nrmse": 0.002, "max_over_reference_rms": 0.02}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pairs(manifest):
    entries = manifest["entries"]
    found = []
    for key1, row1 in entries.items():
        x, w, d, p, g, *tail = json.loads(key1)
        if (x[1] != w[0] or w[1] != x[1] or g != 1 or
            2 * p != d * (w[2] - 1) or "fwd" not in row1["directions"]):
            continue
        key2 = json.dumps([x, w, 1, (w[2] - 1) // 2, g, *tail], separators=(",", ":"))
        row2 = entries.get(key2)
        if row2 is None or "fwd" not in row2["directions"]:
            continue
        found.append(dict(name=f"B{x[0]}_C{x[1]}_T{x[2]}_K{w[2]}_D{d}",
                          key1=key1, key2=key2,
                          pair_key=json.dumps([key1, key2], separators=(",", ":")),
                          shape=x, weight=w, dilation=d, padding=p,
                          directions1=row1["directions"], directions2=row2["directions"]))
    return sorted(found, key=lambda r: (r["shape"][1], r["weight"][2], r["dilation"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--db", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--applio", type=Path, default=Path("~/DevSpace/applio-lab"))
    ap.add_argument("--case", action="append", default=[])
    ap.add_argument("--input-dtypes", default="float16", help="float16 or float16,float32")
    ap.add_argument("--list", action="store_true", help="CPU-only list of eligible pairs")
    ap.add_argument("--seed", type=int, default=63117)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--min-speedup", type=float, default=1.03)
    args = ap.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    cases = pairs(manifest)
    if args.case:
        absent = set(args.case) - {r["name"] for r in cases}
        if absent:
            ap.error("Unknown pair: " + ", ".join(sorted(absent)))
        cases = [r for r in cases if r["name"] in args.case]
    if args.list:
        print(json.dumps(cases, indent=2))
        return
    dtypes = args.input_dtypes.split(",")
    if (not cases or not set(dtypes) <= {"float16", "float32"} or
        args.db is None or args.out is None or min(args.rounds, args.reps) < 1):
        ap.error("Nonempty cases, supported input dtypes, --db, --out and positive timing repetitions required")
    db, out = args.db.resolve(), args.out.resolve()
    if not db.is_relative_to(HERE) or not out.is_relative_to(HERE) or not db.is_dir():
        ap.error("Use a research-copy FindDb and research output path")
    if out.exists():
        ap.error("Refusing to overwrite an existing run")
    out.mkdir(parents=True)
    cache = out / "miopen_cache"
    cache.mkdir()
    os.environ.update(SAFETY)
    os.environ.update(MIOPEN_FIND_MODE="FAST", MIOPEN_USER_DB_PATH=str(db),
                      MIOPEN_CUSTOM_CACHE_DIR=str(cache), APPLIO_RDNA2_DIRECT_CONV="1")
    os.environ.setdefault("TRITON_CACHE_DIR", str(HERE / "results/v10/triton_cache"))
    sys.path.insert(0, str(args.applio.resolve()))
    report = dict(schema="applio_v10_fusion_bench", completed=False, valid_for_timing=False,
                  valid_for_contract=False, seed=args.seed, gate=GATE, safety_env=SAFETY,
                  min_speedup=args.min_speedup, rounds=args.rounds, reps=args.reps,
                  input_dtypes=dtypes, cases=[], source_sha256=sha(HERE / "v10_fused_resblock.py"),
                  benchmark_sha256=sha(__file__), v08_source_sha256=sha(HERE / "v08_direct_conv.py"),
                  v08_manifest=str(manifest_path), v08_manifest_sha256=sha(manifest_path), db=str(db),
                  scope="Synthetic fixed-seed actual ResBlock pairs, FP32 weight_norm masters, no optimizer/epoch",
                  fp32_reference_scope="Single convolution with fixed preactivation and upstream gradient; activation value rounds at the Conv AMP input boundary",
                  contract_reference="v0.8 B ResBlock with identical FP32 master parameters and input/grad-output",
                  memory_scope="Measured allocator peak and kernel/operator counts; static logical tensor traffic is not a hardware DRAM counter")

    def save():
        temp = out / "result.json.tmp"
        temp.write_text(json.dumps(report, indent=2, allow_nan=False))
        temp.replace(out / "result.json")

    # Gates and selection policy are durable before any torch import or GPU work.
    (out / "PREDECLARED_GATES.json").write_text(json.dumps(dict(
        gate=GATE, seed=args.seed, min_pair_speedup=args.min_speedup,
        fp32_reference_scope=report["fp32_reference_scope"],
        contract_reference=report["contract_reference"]), indent=2))
    save()
    from gpu_guard_v06 import GpuAudit
    audit = GpuAudit()
    try:
        with audit:
            audit.ensure_clear()
            import torch
            import torch.nn.functional as F
            import triton
            import v08_direct_conv as base
            import v10_fused_resblock as fusion
            from rvc.lib.algorithm.residuals import ResBlock
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = False
            report.update(torch=torch.__version__, hip=torch.version.hip, triton=triton.__version__,
                          arch=torch.cuda.get_device_properties(0).gcnArchName,
                          device=torch.cuda.get_device_name())
            fusion.load_manifest(manifest_path)

            def error(candidate, reference):
                if candidate is None or reference is None:
                    return dict(screen_pass=False, reason="Missing gradient")
                a, b = candidate.detach().double(), reference.detach().double()
                finite = bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
                delta = a - b
                rms = b.square().mean().sqrt().item()
                rmse, max_abs = delta.square().mean().sqrt().item(), delta.abs().max().item()
                nrmse, maxrms = rmse / max(rms, 1e-30), max_abs / max(rms, 1e-30)
                clean = lambda v: v if math.isfinite(v) else None
                return dict(finite=finite, reference_rms=clean(rms), rmse=clean(rmse),
                            max_abs=clean(max_abs), nrmse=clean(nrmse),
                            max_over_reference_rms=clean(maxrms),
                            candidate_dtype=str(candidate.dtype), reference_dtype=str(reference.dtype),
                            byte_equal=bool(torch.equal(candidate, reference)) if candidate.dtype == reference.dtype else False,
                            screen_pass=finite and nrmse <= GATE["nrmse"] and maxrms <= GATE["max_over_reference_rms"])

            def profile(fn, name):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA],
                                            profile_memory=True, record_shapes=True) as prof:
                    fn()
                    torch.cuda.synchronize()
                trace = out / (name + ".trace.json")
                prof.export_chrome_trace(str(trace))
                totals = defaultdict(lambda: {"calls": 0, "total_ms": 0.0})
                for ev in json.loads(trace.read_text()).get("traceEvents", []):
                    if ev.get("ph") == "X" and ev.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
                        v = totals[(ev["cat"], ev["name"])]
                        v["calls"] += 1
                        v["total_ms"] += float(ev.get("dur", 0)) / 1000
                kernels = sorted([dict(category=k[0], name=k[1], **v) for k, v in totals.items()],
                                 key=lambda v: -v["total_ms"])
                ops = [dict(name=e.key, count=e.count, self_cpu_ms=e.self_cpu_time_total / 1000,
                            self_device_ms=e.self_device_time_total / 1000,
                            self_device_memory_bytes=e.self_device_memory_usage)
                       for e in prof.key_averages()]
                return dict(trace=str(trace), kernels=kernels,
                            kernel_total_ms=sum(v["total_ms"] for v in kernels if v["category"] == "kernel"),
                            kernel_count=sum(v["calls"] for v in kernels if v["category"] == "kernel"),
                            operators=sorted(ops, key=lambda v: -v["self_device_ms"]))

            def operator_check(case, dtype):
                rng = torch.Generator(device="cpu").manual_seed(args.seed)
                x = torch.randn(case["shape"], generator=rng).to(dtype).cuda()
                x.view(-1)[::127] = 0
                w = (torch.randn(case["weight"], generator=rng) / math.sqrt(math.prod(case["weight"][1:]))).half().cuda()
                bias = (torch.randn((w.shape[0],), generator=rng) * .1).half().cuda()
                dy = (torch.randn(case["shape"], generator=rng) * .01).half().cuda()
                skip = (torch.randn(case["shape"], generator=rng) * .01).to(dtype).cuda()
                d, p, dirs = case["dilation"], case["padding"], case["directions1"]
                activated = F.leaky_relu(x, .1).half()
                xf, wf, dyf = activated.float(), w.float(), dy.float()
                reference_y = F.conv1d(xf, wf, bias.float(), 1, p, d)
                rx, rw, _ = torch.ops.aten.convolution_backward.default(
                    dyf, xf, wf, None, [1], [p], [d], False, [0], 1, [True, True, False])
                rx = rx * torch.where(x > 0, 1.0, .1)
                candidate_y = fusion.conv_act_fwd(x, w, bias, dilation=d, padding=p, config=dirs["fwd"])
                cx, cw, cb = fusion._backward_conv(dy, x, w, (d, p, 1, dirs), True, True, True)
                with_skip = fusion.conv_act_fwd(x, w, bias, dilation=d, padding=p,
                                               residual=x, config=dirs["fwd"])
                cx_skip, _, _ = fusion._backward_conv(dy, x, w, (d, p, 1, dirs), True, False, False,
                                                     residual_grad=skip)
                metrics = dict(y=error(candidate_y, reference_y), dx=error(cx, rx), dw=error(cw, rw),
                               db=error(cb, dyf.sum((0, 2))),
                               y_with_residual=error(with_skip, reference_y + x.float()),
                               dx_with_residual=error(cx_skip, rx + skip.float()))
                info = {}
                fusion.conv_act_fwd(x, w, bias, dilation=d, padding=p, config=dirs["fwd"], metadata=info)
                for i, kernel in enumerate(info["kernels"]):
                    asm = kernel.pop("amdgcn")
                    target = out / f"{case['name']}_{str(dtype).split('.')[-1]}_fwd_{i}.s"
                    target.write_text(asm)
                    kernel["assembly_path"] = str(target)
                return dict(metrics=metrics, all_gates_pass=all(v["screen_pass"] for v in metrics.values()),
                            fwd_metadata=info, mask_zero_rule="input > 0; zero uses negative slope",
                            reference="FP32 native Conv with identical rounded AMP Conv input and fixed dy")

            def one(case, dtype_name):
                dtype = {"float16": torch.float16, "float32": torch.float32}[dtype_name]
                operator = operator_check(case, dtype)
                torch.manual_seed(args.seed)
                rng = torch.Generator(device="cpu").manual_seed(args.seed)
                original = ResBlock(case["shape"][1], case["weight"][2], (case["dilation"],))
                with torch.no_grad():
                    for conv in [*original.convs1, *original.convs2]:
                        v = conv.parametrizations.weight.original1
                        values = torch.randn(v.shape, generator=rng) / math.sqrt(math.prod(v.shape[1:]))
                        v.copy_(values)
                        conv.parametrizations.weight.original0.copy_(values.square().sum((1, 2), keepdim=True).sqrt())
                        conv.bias.copy_(torch.randn(conv.bias.shape, generator=rng) * .1)
                models = {arm: copy.deepcopy(original).cuda() for arm in ("B", "B_prime", "native_amp")}
                xc = torch.randn(case["shape"], generator=rng).to(dtype)
                xc.view(-1)[::127] = 0
                xs = {arm: xc.clone().cuda().requires_grad_(True) for arm in models}
                dy = (torch.randn(case["shape"], generator=rng) * .01).to(dtype).cuda()
                all_hits = {"B": {}, "B_prime": {}}

                def context(arm):
                    stack = ExitStack()
                    if arm != "native_amp":
                        stack.enter_context(base.measured_dispatch(manifest_path, hits=all_hits["B"]))
                    if arm == "B_prime":
                        stack.enter_context(fusion.fused_resblocks(manifest_path, hits=all_hits["B_prime"]))
                    return stack

                def invoke(arm):
                    models[arm].zero_grad(set_to_none=True)
                    xs[arm].grad = None
                    with torch.autocast("cuda", dtype=torch.float16):
                        y = models[arm](xs[arm])
                    y.backward(dy)
                    return y

                before = time.perf_counter()
                outputs = {}
                saved_grads = {}
                for arm in models:
                    with context(arm):
                        outputs[arm] = invoke(arm).detach()
                    saved_grads[arm] = {"dx": xs[arm].grad.detach().clone(),
                                        **{"grad:" + n: p.grad.detach().clone() for n, p in models[arm].named_parameters()}}
                torch.cuda.synchronize()
                first_s = time.perf_counter() - before
                metrics = {"y": error(outputs["B_prime"], outputs["B"])}
                metrics.update({name: error(saved_grads["B_prime"][name], value) for name, value in saved_grads["B"].items()})
                native_metrics = {arm: {"y": error(outputs[arm], outputs["native_amp"]),
                                       **{n: error(saved_grads[arm][n], v) for n, v in saved_grads["native_amp"].items()}}
                                  for arm in ("B", "B_prime")}
                masters = all(p.dtype == torch.float32 and p.grad is not None and p.grad.dtype == torch.float32
                              for model in models.values() for p in model.parameters())
                dtypes_match = (outputs["B"].dtype == outputs["B_prime"].dtype == dtype and
                                xs["B"].grad.dtype == xs["B_prime"].grad.dtype == dtype)
                correct_hits = all_hits["B_prime"].get(case["pair_key"], 0) == 1
                row = dict(case=case, input_dtype=dtype_name, operator=operator,
                           contract=dict(metrics=metrics, fp32_master_parameters_and_gradients=masters,
                                         input_output_gradient_dtypes_match=dtypes_match, fusion_hit_once=correct_hits,
                                         all_gates_pass=all(m["screen_pass"] for m in metrics.values()) and masters and dtypes_match and correct_hits),
                           native_amp_diagnostic=native_metrics, first_compile_and_contract_s=first_s,
                           timing={arm: dict(event_ms_rounds=[], wall_ms_rounds=[]) for arm in ("B", "B_prime")})
                del saved_grads, outputs
                for arm in ("B", "B_prime"):
                    with context(arm):
                        for _ in range(3):
                            invoke(arm)
                torch.cuda.synchronize()
                for round_index in range(args.rounds):
                    order = ("B", "B_prime") if round_index % 2 == 0 else ("B_prime", "B")
                    for arm in order:
                        audit.ensure_clear()
                        with context(arm):
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            torch.cuda.synchronize()
                            begin = time.perf_counter()
                            start.record()
                            for _ in range(args.reps):
                                invoke(arm)
                            end.record()
                            end.synchronize()
                            row["timing"][arm]["event_ms_rounds"].append(start.elapsed_time(end) / args.reps)
                            row["timing"][arm]["wall_ms_rounds"].append((time.perf_counter() - begin) * 1000 / args.reps)
                for arm in ("B", "B_prime"):
                    timing = row["timing"][arm]
                    timing.update(event_median_ms=statistics.median(timing["event_ms_rounds"]),
                                  wall_median_ms=statistics.median(timing["wall_ms_rounds"]))
                    models[arm].zero_grad(set_to_none=True)
                    xs[arm].grad = None
                    torch.cuda.synchronize()
                    initial = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    with context(arm):
                        invoke(arm)
                    torch.cuda.synchronize()
                    timing["peak_extra_allocated_bytes"] = torch.cuda.max_memory_allocated() - initial
                    if args.profile:
                        with context(arm):
                            timing["profile"] = profile(lambda: invoke(arm), f"{case['name']}_{dtype_name}_{arm}")
                row["speedup_event"] = row["timing"]["B"]["event_median_ms"] / row["timing"]["B_prime"]["event_median_ms"]
                row["speedup_wall"] = row["timing"]["B"]["wall_median_ms"] / row["timing"]["B_prime"]["wall_median_ms"]
                n = math.prod(case["shape"])
                # All eligible actual training inputs are FP16. FP32 probes have
                # an additional cast boundary and are not assigned this model.
                if dtype == torch.float16:
                    native_activation_remakes = sum("dx" not in ds or "dw" not in ds for ds in (case["directions1"], case["directions2"]))
                    fused_dx = sum("dx" in ds for ds in (case["directions1"], case["directions2"]))
                    row["logical_traffic_model"] = dict(
                        element_bytes=2, elements=n, forward_avoided_read_write_bytes=6 * n * 2,
                        backward_avoided_read_write_bytes=(2 * fused_dx + 2 - 2 * native_activation_remakes) * n * 2,
                        forward_activation_allocation_eliminated_bytes=2 * n * 2,
                        native_backward_activation_remakes=native_activation_remakes,
                        scope="Static logical tensor traffic for eliminated activation/output/gradient intermediates; hardware DRAM bytes unmeasured")
                row["all_gates_pass"] = operator["all_gates_pass"] and row["contract"]["all_gates_pass"]
                row["microbenchmark_candidate"] = row["all_gates_pass"] and row["speedup_event"] >= args.min_speedup and row["speedup_wall"] >= args.min_speedup
                return row

            for case in cases:
                for dtype_name in dtypes:
                    audit.ensure_clear()
                    report["active_case"] = dict(name=case["name"], input_dtype=dtype_name)
                    save()
                    print("BEGIN", case["name"], dtype_name, flush=True)
                    row = one(case, dtype_name)
                    report["cases"].append(row)
                    save()
                    print("RESULT", case["name"], dtype_name, json.dumps(dict(
                        gates=row["all_gates_pass"], event_speedup=row["speedup_event"],
                        wall_speedup=row["speedup_wall"], selected=row["microbenchmark_candidate"])), flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
            report["completed"] = True
    except BaseException as exc:
        report["fatal_error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        report["gpu_audit"] = audit.report()
        report["valid_for_timing"] = report["completed"] and audit.report()["valid_for_timing"] and "fatal_error" not in report
        report["valid_for_contract"] = report["valid_for_timing"] and bool(report["cases"]) and all(r["all_gates_pass"] for r in report["cases"])
        save()
    chosen = [r for r in report["cases"] if r["microbenchmark_candidate"]]
    selected_dtypes = defaultdict(list)
    for row in chosen:
        selected_dtypes[row["case"]["pair_key"]].append("torch." + row["input_dtype"])
    selection = dict(schema="applio_v10_fusion_selection", strict_gate_pass=report["valid_for_timing"],
                     source_sha256=report["source_sha256"], v08_manifest_sha256=report["v08_manifest_sha256"],
                     evidence=str(out / "result.json"), evidence_sha256=sha(out / "result.json"),
                     pair_keys=sorted(selected_dtypes), input_dtypes_by_pair=dict(selected_dtypes),
                     threshold=args.min_speedup,
                     scope="Pair microbenchmark winners only; operating-path epoch A/B and quality decision still required")
    (out / "selection.json").write_text(json.dumps(selection, indent=2))
    print("DONE", out / "result.json", "pairs_selected", len(selection["pair_keys"]), flush=True)
    if not report["valid_for_timing"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
