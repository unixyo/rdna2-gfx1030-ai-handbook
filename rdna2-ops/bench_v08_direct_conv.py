#!/usr/bin/env python3
"""Guarded, direction-separated direct Conv1d measurement for Applio v0.8.

Run native FAST with current or copied seeded research FindDb, then the direct
candidates. --select combines valid reports on CPU and emits ONLY strict-pass
per-direction wins against the faster native baseline. No operating files change.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

SAFETY = {"MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1": "0",
          "MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53": "0"}
GATE = {"finite": True, "nrmse": 0.002, "max_over_reference_rms": 0.02,
        "reference": "FP32 native on identical FP16-rounded X/W/dY",
        "predeclared": True, "scope": "operator gate, not training/audio quality"}

TILES = {
    "fwd": [
        {"bm": 16, "bn": 64, "bk": 32, "num_warps": 4},
        {"bm": 32, "bn": 64, "bk": 32, "num_warps": 4},
        {"bm": 32, "bn": 128, "bk": 32, "num_warps": 4},
        {"bm": 16, "bn": 128, "bk": 64, "num_warps": 4},
        {"bm": 32, "bn": 64, "bk": 64, "num_warps": 4},
        {"bm": 32, "bn": 128, "bk": 64, "num_warps": 8},
    ],
    "dx": [
        {"bm": 16, "bn": 64, "bk": 32, "num_warps": 4},
        {"bm": 32, "bn": 64, "bk": 32, "num_warps": 4},
        {"bm": 32, "bn": 128, "bk": 32, "num_warps": 4},
        {"bm": 16, "bn": 128, "bk": 64, "num_warps": 4},
        {"bm": 32, "bn": 64, "bk": 64, "num_warps": 4},
        {"bm": 32, "bn": 128, "bk": 64, "num_warps": 8},
    ],
    "dw": [
        {"bm": 16, "bn": 32, "bk": 64, "num_warps": 4, "split": 32},
        {"bm": 32, "bn": 32, "bk": 64, "num_warps": 4, "split": 32},
        {"bm": 16, "bn": 32, "bk": 64, "num_warps": 4, "split": 64},
        {"bm": 16, "bn": 32, "bk": 32, "num_warps": 4, "split": 32},
        {"bm": 32, "bn": 32, "bk": 64, "num_warps": 4, "split": 16},
        {"bm": 16, "bn": 16, "bk": 128, "num_warps": 4, "split": 32},
    ],
}


def standard_cases(which):
    # These are real-family shapes, not a claim that every Cartesian combination
    # occurs in voice_model_A. The actual profile can replace them via --cases-json.
    base = [(32, 32, 12800, 3, 1), (64, 64, 6400, 7, 1),
            (128, 128, 3200, 11, 1)]
    if which == "full":
        base = [(c, c, t, r, 1) for c, t in ((32, 12800), (64, 6400),
                    (128, 3200), (192, 298)) for r in (3, 7, 11)]
        base += [(384, 384, 298, 5, 2), (384, 384, 298, 3, 2),
                 (384, 384, 298, 7, 2), (384, 384, 298, 11, 2)]
    rows = []
    for ci, co, t, r, groups in base:
        for d in (1, 3, 5):
            rows.append(dict(name=f"B8_C{ci}_O{co}_T{t}_k{r}_d{d}_g{groups}",
                             shape=[8, ci, t], weight=[co, ci // groups, r],
                             dilation=d, padding=d * (r - 1) // 2, groups=groups,
                             origin="Applio-family fixed-shape probe; actual-use status is separate"))
    return rows


def config_name(cfg):
    return "_".join(f"{k}{v}" for k, v in cfg.items())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key_of(case):
    return json.dumps([case["shape"], case["weight"], case["dilation"], case["padding"],
                       case["groups"], "float16", "NCT", 1], separators=(",", ":"))


def select_reports(paths, output, threshold, *, contract_reports=(), contract_probe=False):
    reports = []
    for source in paths:
        p = Path(source)
        if p.is_dir():
            p = p / "result.json"
        r = json.loads(p.read_text())
        if r.get("schema") == "v08_solver_direction_bench":
            if (not r.get("valid_for_timing") or r.get("mode") != "FAST" or
                r.get("status") != "completed"):
                raise ValueError("Solver evidence must be a completed valid FAST replay: " + str(p))
            if r.get("dtype") != "fp16":
                raise ValueError("Direct FP16 candidates require an FP16 solver baseline: " + str(p))
            converted = []
            for row in r["cases"]:
                c = dict(row["case"])
                if c.get("kind") != "conv1d" or c.get("stride") not in (1, [1]):
                    continue
                for field in ("stride", "padding", "dilation"):
                    if isinstance(c[field], list):
                        c[field] = c[field][0]
                ds = {}
                for direction, data in row["directions"].items():
                    n = dict(data)
                    n["error"] = data["reference_error"]
                    n["bias_included"] = False
                    ds[direction] = dict(native=n, candidates=[])
                converted.append(dict(case=c, directions=ds))
            r = dict(r, schema="applio_v08_direct_conv_bench",
                     source_schema="v08_solver_direction_bench", completed=True,
                     with_bias=False, cases=converted,
                     comparison_note="External fwd baseline excludes bias; a bias-including direct win is conservative")
        if (r.get("schema") != "applio_v08_direct_conv_bench" or
            not r.get("valid_for_timing") or not r.get("completed")):
            raise ValueError("Cannot select from invalid/nonmatching report: " + str(p))
        if r.get("dtype", "fp16") != "fp16":
            raise ValueError("Direct candidate evidence must use FP16 tensors: " + str(p))
        reports.append((p, r))
    native, candidates, cases, boundary_dtypes = {}, {}, {}, {}
    current_source_sha = sha(Path(__file__).with_name("v08_direct_conv.py"))
    contracts, contract_sources = {}, []
    for source in contract_reports:
        path = Path(source)
        if path.is_dir():
            path = path / "result.json"
        report = json.loads(path.read_text())
        if (report.get("schema") != "applio_v08_direct_conv_contract" or
            report.get("completed") is not True or report.get("valid_for_contract") is not True or
            report.get("source_sha256") != current_source_sha or
            report.get("gpu_audit", {}).get("valid_for_timing") is not True or
            report.get("gate", {}).get("nrmse") != GATE["nrmse"] or
            report.get("gate", {}).get("max_over_reference_rms") != GATE["max_over_reference_rms"]):
            raise ValueError("Invalid AMP/weight_norm contract evidence: " + str(path))
        contract_sources.append(dict(path=str(path), sha256=sha(path)))
        for row in report["cases"]:
            if row.get("all_gates_pass") is True and row.get("dispatch_hits", 0) > 0:
                contracts.setdefault(row["key"], []).append(dict(row=row, source=str(path)))
    for path, report in reports:
        for row in report["cases"]:
            # The microbench always creates half tensors. benchmark_dtype alone
            # therefore does not prove that this dtype occurs in training.
            if row["case"].get("benchmark_dtype", "fp16") != "fp16":
                raise ValueError("FP16 direct report has incompatible case dtype: " + str(path))
            key = key_of(row["case"])
            cases[key] = row["case"]
            boundary_dtypes.setdefault(key, set()).update(
                variant.get("input_dtype", "unknown")
                for variant in row["case"].get("logical_boundary_variants", []))
            for direction, data in row.get("directions", {}).items():
                n = data.get("native")
                k = (key, direction)
                if n and n.get("error", {}).get("screen_pass"):
                    native.setdefault(k, []).append(dict(data=n, source=str(path),
                                                        label=report["label"],
                                                        with_bias=report.get("with_bias", False)))
                for candidate in data.get("candidates", []):
                    if report.get("source_sha256") != current_source_sha:
                        raise ValueError("Candidate evidence belongs to different kernel source: " + str(path))
                    if (candidate.get("error", {}).get("screen_pass") and
                        candidate.get("isa_dot2") and candidate.get("event_median_ms") is not None):
                        candidates.setdefault(k, []).append(dict(data=candidate, source=str(path),
                                                                with_bias=report.get("with_bias", False)))
    entries = {}
    decisions = []
    for (key, direction), ns in native.items():
        # Geometry probes without an observed training input remain ineligible.
        # FP32 module inputs need a separate AMP/weight_norm boundary contract.
        if not boundary_dtypes[key]:
            decisions.append(dict(key=key, direction=direction, selected=False,
                                  observed_input_dtypes=[],
                                  reason="No observed training input dtype; boundary validation required"))
            continue
        if any((key, required) not in native for required in ("fwd", "dx", "dw")):
            decisions.append(dict(key=key, direction=direction, selected=False,
                                  reason="Full y/dX/dW native validation required"))
            continue
        baseline = min(ns, key=lambda v: v["data"]["event_median_ms"])
        cs = candidates.get((key, direction), [])
        if not cs:
            decisions.append(dict(key=key, direction=direction, selected=False,
                                  reason="No strict-pass dot2 candidate"))
            continue
        chosen = min(cs, key=lambda v: v["data"]["event_median_ms"])
        n, c = baseline["data"], chosen["data"]
        speed = n["event_median_ms"] / c["event_median_ms"]
        wall_speed = n["wall_median_ms"] / c["wall_median_ms"]
        proofs = []
        missing_dtypes = []
        for dtype in sorted(boundary_dtypes[key] - {"torch.float16"}):
            matching = [proof for proof in contracts.get(key, [])
                        if proof["row"].get("input_dtype") == dtype and
                        proof["row"].get("directions", {}).get(direction) == c["config"]]
            if matching:
                proofs.append(dict(input_dtype=dtype, source=matching[0]["source"]))
            else:
                missing_dtypes.append(dtype)
        accepted = (speed >= threshold and wall_speed >= 1.05 and
                    (not missing_dtypes or contract_probe))
        decision = dict(key=key, direction=direction, selected=accepted,
                        baseline_label=baseline["label"], baseline_source=baseline["source"],
                        candidate_source=chosen["source"], baseline_event_ms=n["event_median_ms"],
                        candidate_event_ms=c["event_median_ms"], speedup=speed,
                        wall_speedup=wall_speed, config=c["config"], error=c["error"],
                        baseline_bias_included=baseline["with_bias"],
                        candidate_bias_included=chosen["with_bias"],
                        observed_input_dtypes=sorted(boundary_dtypes[key]),
                        input_boundary_contracts=proofs,
                        unverified_input_dtypes=missing_dtypes,
                        comparison_scope=("conservative: native excludes bias, candidate includes it"
                                          if direction == "fwd" and not baseline["with_bias"] and chosen["with_bias"]
                                          else "same bias contract or bias-independent backward"))
        if missing_dtypes:
            decision["reason"] = ("Contract probe only; AMP/weight_norm boundary is not validated"
                                  if contract_probe else "AMP/weight_norm input boundary validation required")
        decisions.append(decision)
        if accepted:
            entry = entries.setdefault(key, dict(case=cases[key], strict_gate_pass=True,
                                                directions={}, evidence=[]))
            entry["directions"][direction] = c["config"]
            entry["evidence"].append(decision)
    manifest = dict(schema="applio_v08_direct_conv_dispatch", strict_gate_pass=True,
                    source_sha256=sha(Path(__file__).with_name("v08_direct_conv.py")),
                    status=("contract probe only; not a training selection" if contract_probe else
                            "operator-only; epoch/audio validation still required"),
                    contract_probe_only=contract_probe,
                    selection=dict(event_speedup_min=threshold, wall_speedup_min=1.05,
                                   baseline="fastest accurate FP16 native per shape/direction",
                                   dtype_policy="Observed FP16 input, or explicit matching AMP/weight_norm contract; probe-only exceptions are marked"),
                    entries=entries, decisions=decisions,
                    source_contract_reports=contract_sources,
                    source_reports=[dict(path=str(p), sha256=sha(p)) for p, _ in reports])
    output.mkdir(parents=True, exist_ok=False)
    (output / "dispatch.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(path=str(output / "dispatch.json"), entries=len(entries),
                          directions=sum(len(e["directions"]) for e in entries.values()))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", choices=["representative", "full"], default="representative")
    p.add_argument("--cases-json", type=Path)
    p.add_argument("--case", action="append", default=[])
    p.add_argument("--list", action="store_true")
    p.add_argument("--backend", choices=["native", "both", "direct"], default="both")
    p.add_argument("--label", default="FAST")
    p.add_argument("--db", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--directions", default="fwd,dx,dw")
    p.add_argument("--tile-count", type=int, default=3)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--with-bias", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--select", nargs="+", help="CPU-only selection from existing valid result.json files")
    p.add_argument("--contract-evidence", nargs="+", default=[],
                   help="Completed GPU AMP/weight_norm boundary reports for FP32 module inputs")
    p.add_argument("--contract-probe", action="store_true",
                   help="Build a marked contract-test-only manifest before FP32 input boundary validation")
    p.add_argument("--min-speedup", type=float, default=1.10)
    args = p.parse_args()
    research = Path(__file__).resolve().parent
    out = args.out.resolve()
    if not out.is_relative_to(research):
        p.error("Outputs must remain under the research folder")
    if args.select:
        if args.contract_probe and args.contract_evidence:
            p.error("Contract probe and validated contract evidence are separate selection modes")
        select_reports(args.select, out, args.min_speedup,
                       contract_reports=args.contract_evidence, contract_probe=args.contract_probe)
        return
    cases = (json.loads(args.cases_json.read_text()) if args.cases_json else standard_cases(args.set))
    if isinstance(cases, dict):
        cases = cases["cases"]
    if args.case:
        cases = [c for c in cases if c["name"] in args.case]
    for c in cases:
        if c.get("kind", "conv1d") != "conv1d" or len(c["shape"]) != 3:
            p.error("Use a Conv1d-only case set for this direct candidate")
        for field in ("stride", "padding", "dilation"):
            if isinstance(c.get(field), (list, tuple)):
                if len(c[field]) != 1:
                    p.error("One-dimensional geometry required")
                c[field] = c[field][0]
        c.setdefault("groups", 1)
        c.setdefault("dilation", 1)
        c.setdefault("padding", c["dilation"] * (c["weight"][-1] - 1) // 2)
        if c.get("stride", 1) != 1:
            p.error("This direct candidate supports stride 1 only")
    if args.list:
        print(json.dumps(cases, indent=2))
        return
    if not cases or args.db is None or not (1 <= args.tile_count <= 6):
        p.error("Nonempty cases, --db and tile-count in [1,6] required")
    db = args.db.resolve()
    if not db.is_relative_to(research):
        p.error("Use a research-copy FindDb")
    if out.exists():
        p.error("Refusing to overwrite an existing run")
    out.mkdir(parents=True)
    db.mkdir(parents=True, exist_ok=True)
    cache = db.parent / "miopen_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.update(SAFETY)
    os.environ.update(MIOPEN_FIND_MODE="FAST", MIOPEN_USER_DB_PATH=str(db),
                      MIOPEN_CUSTOM_CACHE_DIR=str(cache))
    os.environ.setdefault("TRITON_CACHE_DIR", str(research / "results/applio_v08/triton_cache"))
    import torch
    import torch.nn.functional as F
    from gpu_guard_v06 import GpuAudit
    from research_comfy_v05 import device_activity_table
    from bench_conv_catalog_v06 import db_inventory
    from v08_direct_conv import conv_fwd, conv_dx, conv_dw
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    directions = args.directions.split(",")
    if set(directions) - {"fwd", "dx", "dw"}:
        p.error("Directions must be fwd,dx,dw")
    report = dict(schema="applio_v08_direct_conv_bench", label=args.label,
                  backend=args.backend, completed=False, torch=torch.__version__, hip=torch.version.hip,
                  triton=__import__("triton").__version__, safety_env=SAFETY, dtype="fp16",
                  source_sha256=sha(research / "v08_direct_conv.py"),
                  benchmark_sha256=sha(Path(__file__)),
                  cudnn_benchmark=torch.backends.cudnn.benchmark, mode="FAST",
                  db=str(db), db_before=db_inventory(db), gate=GATE, cases=[],
                  with_bias=args.with_bias, rounds=args.rounds, reps=args.reps,
                  timing_scope="Separate eager API calls; dW includes allocation, partial and reduction",
                  seed=63117)

    def save():
        (out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False))

    def error(candidate, ref):
        a, b = candidate.double(), ref.double()
        delta = a - b
        rms = b.square().mean().sqrt().item()
        rmse = delta.square().mean().sqrt().item()
        max_abs = delta.abs().max().item()
        finite = bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
        nr, mx = rmse / max(rms, 1e-30), max_abs / max(rms, 1e-30)
        return dict(finite=finite, reference_finite=bool(torch.isfinite(b).all()),
                    reference_rms=rms if math.isfinite(rms) else None,
                    rmse=rmse if math.isfinite(rmse) else None,
                    max_abs=max_abs if math.isfinite(max_abs) else None,
                    nrmse=nr if math.isfinite(nr) else None,
                    max_over_reference_rms=mx if math.isfinite(mx) else None,
                    screen_pass=finite and nr <= .002 and mx <= .02)

    def measure(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        walls, events = [], []
        for _ in range(args.rounds):
            first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            before = time.perf_counter()
            first.record()
            for _ in range(args.reps):
                fn()
            last.record()
            last.synchronize()
            walls.append((time.perf_counter() - before) * 1000 / args.reps)
            events.append(first.elapsed_time(last) / args.reps)
        return dict(wall_ms_rounds=walls, event_ms_rounds=events,
                    wall_median_ms=statistics.median(walls), event_median_ms=statistics.median(events))

    def profile(fn, name):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        target = out / (name + "_trace.json")
        prof.export_chrome_trace(str(target))
        return device_activity_table(target)

    with GpuAudit() as audit:
        try:
            audit.ensure_clear()
            report["device"] = torch.cuda.get_device_name()
            report["arch"] = torch.cuda.get_device_properties(0).gcnArchName
            for case in cases:
                audit.ensure_clear()
                name = case["name"]
                report["active_case"] = name
                print("BEGIN", name, flush=True)
                rng = torch.Generator(device="cpu").manual_seed(report["seed"])
                xc = torch.randn(case["shape"], generator=rng).half()
                wc = (torch.randn(case["weight"], generator=rng) /
                      math.sqrt(math.prod(case["weight"][1:]))).half()
                x, w = xc.cuda(), wc.cuda()
                d, pad, groups = case["dilation"], case["padding"], case["groups"]
                to = x.shape[-1] + 2 * pad - d * (w.shape[-1] - 1)
                dy = (torch.randn((x.shape[0], w.shape[0], to), generator=rng) * .01).half().cuda()
                bias = (torch.randn((w.shape[0],), generator=rng) * .1).half().cuda() if args.with_bias else None
                def native_fwd():
                    return F.conv1d(x, w, bias, 1, pad, d, groups)
                def native_dx():
                    return torch.ops.aten.convolution_backward.default(
                        dy, x, w, None, [1], [pad], [d], False, [0], groups, [True, False, False])[0]
                def native_dw():
                    return torch.ops.aten.convolution_backward.default(
                        dy, x, w, None, [1], [pad], [d], False, [0], groups, [False, True, False])[1]
                native_fns = dict(fwd=native_fwd, dx=native_dx, dw=native_dw)
                xf, wf = x.float(), w.float()
                dyf = dy.float()
                torch.cuda.synchronize()
                ref_start = time.perf_counter()
                refs = dict(fwd=F.conv1d(xf, wf, bias.float() if bias is not None else None,
                                       1, pad, d, groups))
                refs["dx"], refs["dw"], _ = torch.ops.aten.convolution_backward.default(
                    dyf, xf, wf, None, [1], [pad], [d], False, [0], groups, [True, True, False])
                torch.cuda.synchronize()
                row = dict(case=case, fp32_reference_cold_s=time.perf_counter() - ref_start,
                           directions={})
                report["cases"].append(row)
                save()
                for direction in directions:
                    audit.ensure_clear()
                    fn = native_fns[direction]
                    first = time.perf_counter()
                    native_out = fn()
                    torch.cuda.synchronize()
                    cold = time.perf_counter() - first
                    native_row = dict(error=error(native_out, refs[direction]), first_s=cold)
                    data = dict(native=native_row, candidates=[])
                    row["directions"][direction] = data
                    native_row.update(measure(fn))
                    torch.cuda.reset_peak_memory_stats()
                    before_mem = torch.cuda.memory_allocated()
                    native_temp = fn()
                    torch.cuda.synchronize()
                    native_row["peak_extra_bytes"] = torch.cuda.max_memory_allocated() - before_mem
                    del native_temp
                    if args.profile:
                        native_row["device_activities"] = profile(fn, name + "_" + direction + "_native")
                    if args.backend == "native":
                        save()
                        continue
                    for cfg in TILES[direction][:args.tile_count]:
                        audit.ensure_clear()
                        cfg_name = config_name(cfg)
                        info = {}
                        def candidate(metadata=None):
                            kw = dict(dilation=d, padding=pad, groups=groups, config=cfg, metadata=metadata)
                            if direction == "fwd":
                                return conv_fwd(x, w, bias, **kw)
                            if direction == "dx":
                                return conv_dx(dy, w, x.shape, **kw)
                            return conv_dw(x, dy, w.shape, **kw)
                        record = dict(config=cfg)
                        data["candidates"].append(record)
                        try:
                            first = time.perf_counter()
                            yy = candidate(info)
                            torch.cuda.synchronize()
                            record["first_compile_and_call_s"] = time.perf_counter() - first
                            record["error"] = error(yy, refs[direction])
                            record["native_error"] = error(yy, native_out)
                            record["metadata"] = info
                            for kidx, ki in enumerate(info["kernels"]):
                                asm = ki.pop("amdgcn")
                                asm_path = out / (name + "_" + direction + "_" + cfg_name + f"_{kidx}.s")
                                asm_path.write_text(asm)
                                ki["assembly_path"] = str(asm_path)
                            record["isa_dot2"] = bool(info["kernels"][0]["dot2"])
                            if record["error"]["screen_pass"]:
                                record.update(measure(candidate))
                                torch.cuda.synchronize()
                                torch.cuda.reset_peak_memory_stats()
                                before_mem = torch.cuda.memory_allocated()
                                tmp = candidate()
                                torch.cuda.synchronize()
                                record["peak_extra_bytes"] = torch.cuda.max_memory_allocated() - before_mem
                                del tmp
                                record["speedup_same_db"] = (native_row["event_median_ms"] /
                                                            record["event_median_ms"])
                                if args.profile:
                                    record["device_activities"] = profile(
                                        candidate, name + "_" + direction + "_" + cfg_name)
                            del yy
                        except Exception as exc:
                            record["execution_error"] = dict(type=type(exc).__name__, message=str(exc))
                            print("CANDIDATE_ERROR", name, direction, cfg_name, str(exc), flush=True)
                        save()
                        print("CANDIDATE", name, direction, cfg_name,
                              json.dumps({k: record.get(k) for k in
                                          ("event_median_ms", "speedup_same_db", "error", "isa_dot2",
                                           "execution_error")}), flush=True)
                    data["native_recheck"] = measure(fn)
                    save()
                del x, w, dy, bias, xf, wf, dyf, refs, native_out
                gc.collect()
                torch.cuda.empty_cache()
                print("END", name, flush=True)
        except Exception as exc:
            report["fatal_error"] = dict(type=type(exc).__name__, message=str(exc))
            raise
        finally:
            report["gpu_audit"] = audit.report()
            report["valid_for_timing"] = audit.report()["valid_for_timing"] and "fatal_error" not in report
            report["db_after"] = db_inventory(db)
            save()
    report["gpu_audit"] = audit.report()
    report["valid_for_timing"] = audit.report()["valid_for_timing"] and "fatal_error" not in report
    report["completed"] = True
    save()
    print("DONE", out, report["valid_for_timing"], flush=True)


if __name__ == "__main__":
    main()
