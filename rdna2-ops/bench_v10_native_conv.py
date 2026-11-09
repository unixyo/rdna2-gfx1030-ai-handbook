#!/usr/bin/env python3
"""Guarded v1.0 remaining-Conv benchmark. CPU --list/--prepare-only import no torch.

Each run copies its FindDb into a new research result directory. Native
directions are freshly replayed; the optional historical fastest policy
chooses only previously validated single-solver entries. Operator timing,
FP32 error, AMP/weight_norm contract, and dispatch eligibility stay separate.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

from v08_solver_select import (BLOCKED, copy_aux, db_key, discover_db, load_db,
                               normalize_case, output_shape, write_db)

HERE = Path(__file__).resolve().parent
V08 = HERE / "results/applio_v08"
SAFETY = {"MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1": "0",
          "MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53": "0",
          "HSA_DISABLE_COREDUMP_ON_EXCEPTION": "1"}
GATE = {"finite": True, "reference_finite": True, "nrmse": .002,
        "max_over_reference_rms": .02, "seed": 20261108,
        "candidate_event_speedup_min": 1.10, "candidate_wall_speedup_min": 1.05}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def selected_cases(args):
    data = json.loads(args.cases_json.read_text())
    cases = [normalize_case(c) for c in (data["cases"] if isinstance(data, dict) else data)]
    baseline_path = getattr(args, "baseline_dispatch", V08 / "direct_selected_final/dispatch.json")
    baseline_manifest = json.loads(baseline_path.read_text())
    wanted = []
    for c in cases:
        kind, ci, weight = c["kind"], c["shape"][1], c["weight"]
        family = ("remaining1d" if kind == "conv1d" and
                  ((ci == 128 and weight == [128, 128, 11]) or
                   (ci == 192 and weight == [384, 192, 5])) else
                  "transpose" if kind == "conv_transpose1d" else
                  "period" if kind == "conv2d" and weight[-1] == 1 else None)
        if family is None or (args.family != "all" and args.family != family):
            continue
        if family == "period" and args.periods != "all" and c["shape"][-1] not in {int(x) for x in args.periods.split(",")}:
            continue
        if args.case and c["name"] not in args.case:
            continue
        if c["name"] in getattr(args, "exclude_case", []):
            continue
        c["v10_family"] = family
        c["v08_baseline_directions"] = sorted(baseline_directions(c, baseline_manifest))
        wanted.append(c)
    return wanted


def directions_for(c, selection):
    if selection != "auto":
        return selection.split(",")
    if c["v10_family"] != "remaining1d":
        return ["fwd", "dx", "dw"]
    if c["shape"][1] == 192 and not {"fwd", "dx"}.issubset(c.get("v08_baseline_directions", [])):
        return ["fwd", "dx", "dw"]
    return ["dx", "dw"] if c["shape"][1] == 128 and c["dilation"] == [1] else ["dw"]


def prepare_db(source_dir, out, cases, fastest=True):
    source = discover_db(source_dir)
    if source is None:
        raise ValueError("Source FindDb missing")
    target_dir = out / "miopen_db"
    target_dir.mkdir()
    records = load_db(source)
    records = {k: [e for e in es if e["solver"] not in BLOCKED] for k, es in records.items()}
    wanted = {db_key(c, d) for c in cases for d in ("fwd", "dx", "dw")}
    baseline_path = V08 / "solver_selected_final_replay/result.json"
    baseline = json.loads(baseline_path.read_text())
    best = {}
    for row in baseline["cases"]:
        for direction, dr in row["directions"].items():
            key = dr["db_key"]
            if key in wanted and dr["reference_error"]["screen_pass"]:
                best[key] = dict(event_ms=dr["event_median_ms"], evidence=str(baseline_path),
                                 selected="preserved_v08_B_policy")
    used_reports = []
    if fastest:
        paths = set(V08.glob("solver_*/result.json"))
        paths.update(V08.glob("solver_ranks47/rank_*/result.json"))
        paths.update(V08.glob("solver_workspace_ranks24/rank_*/result.json"))
        for path in sorted(paths):
            report = json.loads(path.read_text())
            if (report.get("schema") != "v08_solver_direction_bench" or
                not report.get("valid_for_timing") or report.get("mode") != "FAST" or
                report.get("logging_enabled") or report.get("probe_only")):
                continue
            used_reports.append(dict(path=str(path), sha256=sha(path)))
            for row in report["cases"]:
                for direction, dr in row["directions"].items():
                    key, entries = dr["db_key"], dr.get("db_entries_before", [])
                    if (key not in wanted or not dr.get("reference_error", {}).get("screen_pass") or
                        len(entries) != 1 or entries[0]["solver"] in BLOCKED or
                        not dr.get("db_target_entry_unchanged")):
                        continue
                    ms = dr["event_median_ms"]
                    if ms < best.get(key, {}).get("event_ms", math.inf):
                        records[key] = entries
                        best[key] = dict(event_ms=ms, evidence=str(path),
                                         selected=entries[0]["solver"], entry=entries[0])
    target = target_dir / source.name
    write_db(target, records)
    aux = copy_aux(source, target_dir)
    manifest = dict(source=str(source), source_sha256=sha(source), copied_db=str(target),
                    copied_sha256=sha(target), policy="fastest accurate historical v08 rank per direction, then fresh FAST replay" if fastest else "operating v08 B policy copy",
                    selection_caveat="Historical ordering is a candidate policy; fresh timings are the evidence for v10. Same-solver timing jitter is not an extra optimization.",
                    directions=best, historical_reports=used_reports, auxiliary=aux)
    dump(out / "native_policy.json", manifest)
    return target_dir, manifest


def normal_geometry(c):
    shape, w = c["shape"], c["weight"]
    y = output_shape(c)
    if "transpose" in c["kind"]:
        return shape[0], y[1], shape[1], y[2], 1, w[2], c["stride"][0], c["dilation"][0], c["padding"][0], shape[2]
    return shape[0], shape[1], w[0], shape[2], shape[3] if len(shape) == 4 else 1, w[2], c["stride"][0], c["dilation"][0], c["padding"][0], y[2]


def pow2_up(n):
    return 1 << max(0, int(math.ceil(math.log2(max(n, 1)))))


def configurations(c, direction):
    b, ci, co, h, q, r, s, d, p, ho = normal_geometry(c)
    def tile(bm, bn, bk, split=None, algorithm="implicit_phase"):
        row = dict(algorithm=algorithm, bm=bm, bn=bn, bk=bk, num_warps=4)
        if split is not None:
            row["split"] = split
        return row
    if direction != "dw":
        rows = [tile(32, 64, 32), tile(32, 128, 32), tile(64, 64, 32)]
        if c["v10_family"] == "remaining1d" and direction == "dx":
            rows = [tile(64, 64, 32, algorithm="v08_wide_dx"),
                    tile(32, 64, 64, algorithm="v08_wide_dx"), tile(64, 64, 32)]
        return rows
    rows = []
    for bm, bn, bk in ((16, 16, 64), (32, 64, 64), (32, 32, 64)) if ci * r < 16 else ((32, 64, 64), (32, 32, 64), (16, 32, 64)):
        tiles = math.ceil(co / bm) * math.ceil(ci * r / bn)
        max_useful = max(1, 1 << int(math.log2(max(1, (b * ho * q) // bk))))
        split = min(128, max_useful, pow2_up(256 / tiles))
        rows.append(tile(bm, bn, bk, split, "flat_dw"))
    if c["v10_family"] == "remaining1d":
        # The prior v08 run used split32/64. These bounded small-split candidates
        # test oversplitting directly without rerunning its unchanged cases.
        for split in ((1, 2, 4) if ci == 192 else (4, 8, 16)):
            rows.append(tile(32, 32, 64, split, "v08_small_split_dw"))
    return rows


def roofline(c):
    b, ci, co, h, q, r, s, d, p, ho = normal_geometry(c)
    valid_taps = sum(sum(0 <= oh * s + rr * d - p < h for rr in range(r)) for oh in range(ho))
    nominal = 2 * b * ci * co * r * ho * q
    useful = 2 * b * ci * co * q * valid_taps
    byte_min = 2 * (math.prod(c["shape"]) + math.prod(c["weight"]) + math.prod(output_shape(c)))
    return dict(direct_algorithm_nominal_flops=nominal, valid_nonpadding_flops=useful,
                fp16_compulsory_read_write_bytes=byte_min,
                measured_dot2_tflops=48.7, measured_memory_gbs=436.7,
                optimistic_direct_compute_ms=useful / 48.7e9,
                optimistic_memory_ms=byte_min / 436.7e6,
                optimistic_lower_ms=max(useful / 48.7e9, byte_min / 436.7e6),
                caveat="Compulsory bytes are a lower bound, not measured DRAM traffic. Winograd performs different arithmetic, so effective direct FLOP/s is not ISA utilization.")


def baseline_directions(c, manifest):
    if c["kind"] != "conv1d" or c["stride"] != [1]:
        return {}
    key = json.dumps([c["shape"], c["weight"], c["dilation"][0], c["padding"][0], c["groups"], "float16", "NCT", 1], separators=(",", ":"))
    return {d: dict(algorithm="v08_" + d, **cfg) for d, cfg in manifest.get("entries", {}).get(key, {}).get("directions", {}).items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases-json", type=Path, default=V08 / "conv_cases.json")
    p.add_argument("--family", choices=["all", "remaining1d", "transpose", "period"], default="all")
    p.add_argument("--periods", default="3", help="comma-separated discriminator periods, or all")
    p.add_argument("--case", action="append", default=[])
    p.add_argument("--exclude-case", action="append", default=[],
                   help="Exclude one already completed case name; repeat for multiple cases")
    p.add_argument("--directions", default="auto")
    p.add_argument("--tile-count", type=int, default=3)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261108)
    p.add_argument("--db", type=Path, default=V08 / "solver_selected_final_db")
    p.add_argument("--native-policy", choices=["fastest-v08", "operating-v08"], default="fastest-v08")
    p.add_argument("--baseline-dispatch", type=Path, default=V08 / "direct_selected_final/dispatch.json")
    p.add_argument("--out", type=Path)
    p.add_argument("--list", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--contract", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    GATE["seed"] = args.seed
    cases = selected_cases(args)
    if not cases or not 1 <= args.tile_count <= 6:
        p.error("Nonempty cases and tile-count 1..6 required")
    if args.directions != "auto" and set(args.directions.split(",")) - {"fwd", "dx", "dw"}:
        p.error("Directions must be auto or fwd,dx,dw subset")
    if args.list:
        print(json.dumps([dict(name=c["name"], family=c["v10_family"], directions=directions_for(c, args.directions),
                               shape=c["shape"], weight=c["weight"], configs={d:configurations(c,d)[:args.tile_count] for d in directions_for(c,args.directions)}) for c in cases], indent=2))
        return
    if args.out is None or not args.out.resolve().is_relative_to(HERE / "results/v10"):
        p.error("Use a new --out directory under results/v10")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "PREDECLARED_GATES.json", dict(GATE, phase="saved before GPU import/execution",
        reference="FP32 native on identical FP16-rounded inputs; separate AMP and FP32 weight_norm-master contract"))
    db, policy = prepare_db(args.db, out, cases, args.native_policy == "fastest-v08")
    dump(out / "cases.json", {"cases": cases})
    if args.prepare_only:
        print(json.dumps(dict(out=str(out), cases=len(cases), db=str(db))))
        return
    os.environ.update(SAFETY)
    os.environ.update(MIOPEN_FIND_MODE="FAST", MIOPEN_USER_DB_PATH=str(db),
                      MIOPEN_CUSTOM_CACHE_DIR=str(HERE / "results/v10/miopen_cache"),
                      TRITON_CACHE_DIR=str(HERE / "results/v10/triton_cache"),
                      MIOPEN_ENABLE_LOGGING="0", MIOPEN_LOG_LEVEL="0",
                      PYTORCH_TUNABLEOP_ENABLED="0", PYTORCH_TUNABLEOP_TUNING="0",
                      PYTORCH_TUNABLEOP_RECORD_UNTUNED="0")
    Path(os.environ["MIOPEN_CUSTOM_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    base_manifest = json.loads(args.baseline_dispatch.read_text())
    from gpu_guard_v06 import GpuAudit
    report = dict(schema="applio_v10_remaining_conv_bench", completed=False, valid_for_timing=False,
                  gate=GATE, safety=SAFETY, source_sha256=sha(HERE / "v10_native_conv.py"),
                  benchmark_sha256=sha(__file__), native_policy=str(out / "native_policy.json"),
                  cases=[], timing_scope="Per direction, allocation and all candidate kernels included; no bias in microbenchmark",
                  baseline_dispatch=str(args.baseline_dispatch), baseline_dispatch_sha256=sha(args.baseline_dispatch),
                  selection_args=dict(family=args.family, periods=args.periods, case=args.case,
                                      exclude_case=args.exclude_case, directions=args.directions),
                  tunableop_enabled=False, tunableop_tuning=False,
                  cache_paths={k:os.environ[k] for k in ("MIOPEN_CUSTOM_CACHE_DIR", "TRITON_CACHE_DIR")},
                  rounds=args.rounds, reps=args.reps, warmup=args.warmup)
    save = lambda: dump(out / "result.json", report)
    audit = GpuAudit()
    entries = {}
    try:
        with audit:
            audit.ensure_clear()
            import torch
            import torch.nn.functional as F
            import v10_native_conv as candidate
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = False
            report.update(torch=torch.__version__, hip=torch.version.hip,
                          triton=__import__("triton").__version__, device=torch.cuda.get_device_name())

            def error(a, reference):
                aa, rr = a.double(), reference.double()
                finite = bool(torch.isfinite(aa).all()) and bool(torch.isfinite(rr).all())
                delta = aa - rr
                rms = rr.square().mean().sqrt().item()
                rmse, maximum = delta.square().mean().sqrt().item(), delta.abs().max().item()
                nr, mx = rmse / max(rms, 1e-30), maximum / max(rms, 1e-30)
                clean = lambda v: v if math.isfinite(v) else None
                return dict(finite=finite, reference_finite=bool(torch.isfinite(rr).all()),
                            reference_rms=clean(rms), nrmse=clean(nr), max_over_reference_rms=clean(mx),
                            screen_pass=finite and nr <= GATE["nrmse"] and mx <= GATE["max_over_reference_rms"])

            def measure(fn):
                for _ in range(args.warmup):
                    fn()
                torch.cuda.synchronize()
                walls, events = [], []
                for _ in range(args.rounds):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    begin.record()
                    for _ in range(args.reps):
                        fn()
                    end.record()
                    end.synchronize()
                    walls.append((time.perf_counter() - start) * 1000 / args.reps)
                    events.append(begin.elapsed_time(end) / args.reps)
                return dict(event_ms_rounds=events, wall_ms_rounds=walls,
                            event_median_ms=statistics.median(events), wall_median_ms=statistics.median(walls))

            def profile(fn, name):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                    fn()
                    torch.cuda.synchronize()
                path = out / (name + ".trace.json")
                prof.export_chrome_trace(str(path))
                from research_comfy_v05 import device_activity_table
                return device_activity_table(path)

            def contract(c, configs):
                ci = c["shape"][1]
                trans = "transpose" in c["kind"]
                co = c["weight"][1] if trans else c["weight"][0]
                cls = torch.nn.ConvTranspose1d if trans else torch.nn.Conv2d if len(c["shape"]) == 4 else torch.nn.Conv1d
                kwargs = {k: tuple(c[k]) for k in ("stride", "padding", "dilation")}
                if trans:
                    kwargs["output_padding"] = tuple(c["output_padding"])
                observed = sorted({v["input_dtype"] for v in c.get("logical_boundary_variants", [])} or {"torch.float16"})
                result = dict(scope="same AMP semantics, parametrizations.weight_norm, FP32 bias/g/v masters", variants=[])
                for dtype_name in observed:
                    dtype = getattr(torch, dtype_name.split(".")[-1])
                    torch.manual_seed(GATE["seed"])
                    model = cls(ci, co, tuple(c["weight"][2:]), bias=True, dtype=torch.float32, **kwargs)
                    model = torch.nn.utils.parametrizations.weight_norm(model).cuda()
                    other = copy.deepcopy(model)
                    gen = torch.Generator(device="cpu").manual_seed(GATE["seed"] + 1)
                    xx = torch.randn(c["shape"], generator=gen).to(dtype=dtype, device="cuda").requires_grad_(True)
                    yy = xx.detach().clone().requires_grad_(True)
                    gg = (torch.randn(output_shape(c), generator=gen) * .01).half().cuda()
                    with torch.autocast("cuda", dtype=torch.float16):
                        a = model(xx)
                        z = candidate.apply(yy, other.weight, other.bias, c, configs)
                    named_a, named_z = list(model.named_parameters()), list(other.named_parameters())
                    ga = torch.autograd.grad(a, [xx, *[v for _, v in named_a]], gg)
                    gz = torch.autograd.grad(z, [yy, *[v for _, v in named_z]], gg)
                    metrics = {"y": error(z, a), "dx": error(gz[0], ga[0])}
                    for idx, ((name, _), (other_name, _)) in enumerate(zip(named_a, named_z), 1):
                        if name != other_name:
                            raise RuntimeError("weight_norm parameter mismatch")
                        metrics[name] = error(gz[idx], ga[idx])
                    dtype_checks = dict(
                        outputs_fp16=a.dtype == z.dtype == torch.float16,
                        native_input_gradient_preserved=ga[0].dtype == xx.dtype == dtype,
                        candidate_input_gradient_preserved=gz[0].dtype == yy.dtype == dtype,
                        native_masters_fp32=all(v.dtype == torch.float32 for _, v in named_a),
                        candidate_masters_fp32=all(v.dtype == torch.float32 for _, v in named_z),
                        native_master_gradients_fp32=all(v.dtype == torch.float32 for v in ga[1:]),
                        candidate_master_gradients_fp32=all(v.dtype == torch.float32 for v in gz[1:]))
                    result["variants"].append(dict(input_dtype=dtype_name, output_dtype=str(z.dtype),
                        gradient_dtypes=[str(v.dtype) for v in gz], metrics=metrics, dtype_checks=dtype_checks,
                        strict_pass=all(m["screen_pass"] for m in metrics.values()) and all(dtype_checks.values())))
                    del model, other, xx, yy, gg, a, z, ga, gz, named_a, named_z
                result["strict_pass"] = all(v["strict_pass"] for v in result["variants"])
                return result

            for c in cases:
                audit.ensure_clear()
                print("BEGIN", c["name"], flush=True)
                report["active_case"] = c["name"]
                gen = torch.Generator(device="cpu").manual_seed(GATE["seed"])
                x = torch.randn(c["shape"], generator=gen).half().cuda()
                w = (torch.randn(c["weight"], generator=gen) / math.sqrt(math.prod(c["weight"][1:]))).half().cuda()
                dy = (torch.randn(output_shape(c), generator=gen) * .01).half().cuda()
                kwargs = {k: tuple(c[k]) for k in ("stride", "padding", "dilation")}
                kwargs["groups"] = c["groups"]
                trans = "transpose" in c["kind"]
                if trans:
                    kwargs["output_padding"] = tuple(c["output_padding"])

                def native(direction, xx=x, ww=w, gg=dy):
                    if direction == "fwd":
                        return getattr(F, c["kind"])(xx, ww, None, **kwargs)
                    values = torch.ops.aten.convolution_backward.default(
                        gg, xx, ww, None, c["stride"], c["padding"], c["dilation"], trans,
                        c["output_padding"], c["groups"], [direction == "dx", direction == "dw", False])
                    return values[0] if direction == "dx" else values[1]

                row = dict(case=c, roofline=roofline(c), directions={})
                report["cases"].append(row)
                picked = {}
                for direction in directions_for(c, args.directions):
                    audit.ensure_clear()
                    fn = lambda: native(direction)
                    ref = native(direction, x.float(), w.float(), dy.float())
                    yn = fn()
                    nrow = dict(error=error(yn, ref), **measure(fn))
                    if args.profile:
                        nrow["device_activities"] = profile(fn, c["name"] + "_" + direction + "_native")
                    data = dict(native=nrow, candidates=[])
                    row["directions"][direction] = data
                    for index, cfg in enumerate(configurations(c, direction)[:args.tile_count]):
                        audit.ensure_clear()
                        record = dict(config=cfg)
                        data["candidates"].append(record)
                        info = {}
                        def call(metadata=None):
                            return candidate.run(c, direction, x, w, dy, config=cfg, metadata=metadata)
                        try:
                            start = time.perf_counter()
                            y = call(info)
                            torch.cuda.synchronize()
                            record.update(cold_compile_and_call_s=time.perf_counter()-start,
                                          error=error(y, ref), native_error=error(y, yn))
                            for ki, kernel in enumerate(info.get("kernels", [])):
                                asm = kernel.pop("amdgcn")
                                path = out / (c["name"] + "_" + direction + f"_c{index}_k{ki}.s")
                                path.write_text(asm)
                                kernel["assembly_path"] = str(path)
                            record["metadata"] = info
                            if record["error"]["screen_pass"]:
                                record.update(measure(call))
                                record["event_speedup"] = nrow["event_median_ms"] / record["event_median_ms"]
                                record["wall_speedup"] = nrow["wall_median_ms"] / record["wall_median_ms"]
                                if args.profile:
                                    record["device_activities"] = profile(call, c["name"] + "_" + direction + f"_c{index}")
                                record["strict_operator_win"] = bool(nrow["error"]["screen_pass"] and
                                    record["event_speedup"] >= GATE["candidate_event_speedup_min"] and
                                    record["wall_speedup"] >= GATE["candidate_wall_speedup_min"])
                            del y
                        except Exception as exc:
                            record["execution_error"] = dict(type=type(exc).__name__, message=str(exc))
                        save()
                        print("CANDIDATE", c["name"], direction, index,
                              json.dumps({k:record.get(k) for k in ("event_median_ms", "event_speedup", "error", "execution_error")}), flush=True)
                    data["native_recheck"] = measure(fn)
                    baseline_event = min(nrow["event_median_ms"], data["native_recheck"]["event_median_ms"])
                    baseline_wall = min(nrow["wall_median_ms"], data["native_recheck"]["wall_median_ms"])
                    data["selection_baseline"] = dict(event_ms=baseline_event, wall_ms=baseline_wall,
                        policy="faster of native initial/recheck, chosen independently for event and wall")
                    for record in data["candidates"]:
                        if record.get("event_median_ms") is None:
                            continue
                        record["event_speedup"] = baseline_event / record["event_median_ms"]
                        record["wall_speedup"] = baseline_wall / record["wall_median_ms"]
                        record["strict_operator_win"] = bool(nrow["error"]["screen_pass"] and
                            record["event_speedup"] >= GATE["candidate_event_speedup_min"] and
                            record["wall_speedup"] >= GATE["candidate_wall_speedup_min"])
                    winners = [r for r in data["candidates"] if r.get("strict_operator_win")]
                    if winners:
                        best = min(winners, key=lambda r:r["event_median_ms"])
                        picked[direction] = best["config"]
                        data["selected_candidate"] = data["candidates"].index(best)
                    del ref, yn
                    save()
                merged = baseline_directions(c, base_manifest)
                merged.update(picked)
                if picked and args.contract:
                    row["contract"] = contract(c, merged)
                    if row["contract"]["strict_pass"]:
                        entries[candidate.case_key(c)] = dict(case=c, directions=merged,
                            new_directions=picked, strict_gate_pass=True, contract_pass=True,
                            evidence=str(out / "result.json"))
                elif picked:
                    row["contract"] = dict(strict_pass=False, not_measured="--no-contract; no dispatch eligibility")
                save()
                del x, w, dy
                gc.collect()
                print("END", c["name"], "operator_wins", list(picked), flush=True)
            report["completed"] = True
    except BaseException as exc:
        report["fatal_error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        report["gpu_audit"] = audit.report()
        report["valid_for_timing"] = bool(report["completed"] and report["gpu_audit"]["valid_for_timing"])
        report["selected_entries"] = len(entries)
        report["selected_new_directions"] = sum(len(r["new_directions"]) for r in entries.values())
        save()
        dump(out / "dispatch.json", dict(schema="applio_v10_native_dispatch",
            strict_gate_pass=report["valid_for_timing"], source_sha256=report["source_sha256"],
            baseline_v08_source_sha256=base_manifest.get("source_sha256"),
            status="operator and AMP/weight_norm contract only; operating epoch/audio still required",
            entries=entries if report["valid_for_timing"] else {}, evidence=str(out / "result.json"), gate=GATE))
    print("DONE", out, report["valid_for_timing"], "entries", len(entries), flush=True)


if __name__ == "__main__":
    main()
