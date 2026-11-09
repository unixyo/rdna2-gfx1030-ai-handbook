#!/usr/bin/env python3
"""Direction-specific MIOpen FindDb screening for the existing gfx1030 runtime.

This is an opt-in research tool, never an operating launcher. CPU subcommands
prepare ranked, single-entry FindDb copies and merge only measured valid wins.
GPU bench requires the parent's serial lease, uses gpu_guard_v06, and measures
forward, input-gradient and weight-gradient independently. Benchmark timings do
not include the independent FP32 reference. Cold costs and warm costs are separate.
"""
from __future__ import annotations
import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
SAFETY = {
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1": "0",
    "MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53": "0",
}
BLOCKED = {"ConvHipImplicitGemmV4R1Fwd", "ConvOclBwdWrW53"}
DIRECTIONS = ("fwd", "dx", "dw")
GATE = {"finite": True, "reference_finite": True,
        "nrmse": 0.002, "max_over_reference_rms": 0.02}
KEY_DIRECTION = re.compile(r"-(F|B|W)(?:_g(\d+))?$")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def owned(path):
    p = Path(path).resolve()
    if not p.is_relative_to(HERE):
        raise ValueError("Only research paths under " + str(HERE) + " are allowed")
    return p


def fresh_dir(path):
    p = owned(path)
    if p.exists():
        raise ValueError("Use a new output directory: " + str(p))
    p.mkdir(parents=True)
    return p


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")


def entry(raw):
    solver, colon, tail = raw.partition(":")
    fields = tail.split(",")
    if not colon or len(fields) != 3:
        raise ValueError("Unrecognized FindDb entry: " + raw)
    ms, workspace = float(fields[0]), int(fields[1])
    if not math.isfinite(ms) or ms < 0 or workspace < 0:
        raise ValueError("Invalid FindDb time/workspace: " + raw)
    return dict(solver=solver, recorded_ms=ms, workspace_bytes=workspace,
                algorithm=fields[2], raw=raw)


def load_db(path):
    path = Path(path)
    records = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        key, equal, values = line.partition("=")
        if not equal or not KEY_DIRECTION.search(key):
            raise ValueError(f"Unrecognized FindDb line {path}:{number}")
        if key in records:
            raise ValueError(f"Duplicate FindDb key {path}:{number}: {key}")
        records[key] = [entry(x) for x in values.split(";") if x]
    return records


def write_db(path, records):
    data = "".join(k + "=" + ";".join(e["raw"] for e in es) + "\n"
                   for k, es in records.items() if es)
    Path(path).write_text(data)


def discover_db(folder):
    files = sorted(Path(folder).glob("*.ufdb.txt"))
    if len(files) > 1:
        raise ValueError("Pass an isolated directory with one device/runtime FindDb")
    return files[0] if files else None


def copy_aux(source, out):
    """Copy bounded sibling tuning/time files, never operating cache contents."""
    copied = []
    source = Path(source)
    for p in source.parent.iterdir():
        if (p.is_file() and p != source and
            (p.name == source.name + ".time" or
             p.name.endswith((".udb", ".db", ".udb.txt", ".db.txt")))):
            q = Path(out) / p.name
            shutil.copy2(p, q)
            copied.append({"source": str(p), "output": str(q), "bytes": p.stat().st_size,
                           "sha256": sha(p)})
    return copied


def shape_tuple(value, n, default):
    if value is None:
        return (default,) * n
    if isinstance(value, int):
        return (value,) * n
    out = tuple(int(x) for x in value)
    if len(out) != n:
        raise ValueError("Spatial argument rank mismatch")
    return out


def normalize_case(c):
    c = dict(c)
    c["shape"] = list(c["shape"])
    c["weight"] = list(c["weight"])
    n = len(c["shape"]) - 2
    if n not in (1, 2):
        raise ValueError("This Applio selector supports Conv1D/2D and ConvTranspose1D/2D")
    c.setdefault("kind", f"conv{n}d")
    c.setdefault("groups", 1)
    for k, default in (("stride", 1), ("padding", 0), ("dilation", 1),
                       ("output_padding", 0)):
        c[k] = list(shape_tuple(c.get(k), n, default))
    if len(c["weight"]) != len(c["shape"]):
        raise ValueError("Input and weight ranks must match")
    c.setdefault("name", "conv_" + hashlib.sha256(
        json.dumps(c, sort_keys=True).encode()).hexdigest()[:12])
    return c


def output_shape(c):
    trans = "transpose" in c["kind"]
    n, ci, *spatial = c["shape"]
    co = c["weight"][1] * c["groups"] if trans else c["weight"][0]
    result = [n, co]
    for t, k, s, p, d, op in zip(spatial, c["weight"][2:], c["stride"],
                               c["padding"], c["dilation"], c["output_padding"]):
        result.append((t - 1) * s - 2 * p + d * (k - 1) + op + 1 if trans
                      else (t + 2 * p - d * (k - 1) - 1) // s + 1)
    if min(result) <= 0:
        raise ValueError("Invalid output shape")
    return result


def db_key(c, direction, dtype="FP16"):
    """MIOpen NCHW key order, verified against v0.6 normal/grouped/transpose DB."""
    c = normalize_case(c)
    x, y = c["shape"], output_shape(c)
    trans = "transpose" in c["kind"]
    if trans:
        code = {"fwd": "B", "dx": "F", "dw": "W"}[direction]
        a, b = (y, x) if direction == "dx" else (x, y)
    else:
        code = {"fwd": "F", "dx": "B", "dw": "W"}[direction]
        a, b = (x, y) if direction == "fwd" else (y, x)
    def hw(s):
        return (1, s[2]) if len(s) == 3 else tuple(s[2:])
    def pair(v, default):
        return (default, v[0]) if len(v) == 1 else tuple(v)
    ah, aw = hw(a); bh, bw = hw(b)
    filt = pair(c["weight"][2:], 1)
    padding = pair(c["padding"], 0)
    stride = pair(c["stride"], 1)
    dilation = pair(c["dilation"], 1)
    px = lambda v: "x".join(str(z) for z in v)
    key = f"{a[1]}-{ah}-{aw}-{px(filt)}-{b[1]}-{bh}-{bw}-{a[0]}-{px(padding)}-{px(stride)}-{px(dilation)}-0-NCHW-{dtype}-{code}"
    if c["groups"] != 1:
        key += "_g" + str(c["groups"])
    return key


def case_id(c):
    c = normalize_case(c)
    return json.dumps({k: c[k] for k in ("shape", "weight", "kind", "groups",
                      "stride", "padding", "dilation", "output_padding")},
                      sort_keys=True)


def command_inventory(args):
    source = owned(args.source)
    db = load_db(source)
    result = {"source": str(source), "sha256": sha(source), "keys": len(db),
              "directions": {}, "blocked_entries": [], "entries": []}
    for key, es in db.items():
        d = KEY_DIRECTION.search(key).group(1)
        result["directions"][d] = result["directions"].get(d, 0) + 1
        for e in es:
            row = {"key": key, **e}
            if e["solver"] in BLOCKED:
                result["blocked_entries"].append(row)
            if args.full:
                result["entries"].append(row)
    if args.out:
        dump(owned(args.out), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_prepare(args):
    source = owned(args.source)
    before = sha(source)
    db = load_db(source)
    out = fresh_dir(args.out)
    eligible = {k: sorted((e for e in es if e["solver"] not in BLOCKED),
                          key=lambda e: (e["recorded_ms"], e["solver"]))
                for k, es in db.items() if "-FP16-" in k}
    max_ranks = max((len(es) for es in eligible.values()), default=0)
    count = max_ranks if args.ranks == 0 else min(args.ranks, max_ranks)
    manifest = {"schema": "v08_ranked_finddb", "source": str(source),
                "source_sha256": before, "gate": GATE, "safety": SAFETY,
                "blocked_solvers": sorted(BLOCKED), "ranks": [],
                "status": "candidates_require_replay_validation",
                "all_recorded_candidates_enumerated": count == max_ranks,
                "note": "Seed record timings order candidates, not measured final wins."}
    # Each rank is a fresh process-wide policy. Never switch FindMode per Conv.
    for rank in range(count):
        d = out / f"rank_{rank:02d}"
        d.mkdir()
        selected = {k: [es[rank]] for k, es in eligible.items() if len(es) > rank}
        target = d / source.name
        write_db(target, selected)
        aux = copy_aux(source, d)
        manifest["ranks"].append({"rank": rank, "db": str(d),
                                 "file": str(target), "sha256": sha(target),
                                 "keys": len(selected), "auxiliary_files": aux})
    manifest["source_unchanged"] = sha(source) == before
    dump(out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def error_metrics(candidate, reference):
    import torch
    a = candidate.detach().float().cpu().double()
    b = reference.detach().float().cpu().double()
    finite = bool(torch.isfinite(a).all())
    ref_finite = bool(torch.isfinite(b).all())
    if not finite or not ref_finite:
        return {"finite": finite, "reference_finite": ref_finite,
                "screen_pass": False, "nrmse": None,
                "max_over_reference_rms": None, "max_abs": None,
                "rmse": None, "reference_rms": None}
    diff = a - b
    rms = float(b.square().mean().sqrt())
    rmse = float(diff.square().mean().sqrt())
    maximum = float(diff.abs().max())
    nr, mx = rmse / max(rms, 1e-30), maximum / max(rms, 1e-30)
    return {"finite": finite, "reference_finite": ref_finite,
            "screen_pass": nr <= GATE["nrmse"] and mx <= GATE["max_over_reference_rms"],
            "nrmse": nr, "max_over_reference_rms": mx,
            "max_abs": maximum, "rmse": rmse, "reference_rms": rms}


@contextlib.contextmanager
def capture_fd2(path):
    """Capture native MIOpen C++ stderr without a shell or pipe buffering."""
    sys.stderr.flush()
    saved = os.dup(2)
    with open(path, "wb", buffering=0) as f:
        os.dup2(f.fileno(), 2)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)


def command_bench(args):
    # This function is the only GPU entry. CPU subcommands do not import torch.
    out = fresh_dir(args.out)
    dbdir = owned(args.db)
    dbdir.mkdir(parents=True, exist_ok=True)
    os.environ.update(SAFETY)
    os.environ["MIOPEN_FIND_MODE"] = args.mode
    os.environ["MIOPEN_USER_DB_PATH"] = str(dbdir)
    cache = owned(args.cache) if args.cache else out.parent / "miopen_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = str(cache)
    # Logging/profiling adds overhead. Only --log runs may claim a solver name
    # observed in native logs; do not use their wall time to choose a winner.
    os.environ["MIOPEN_ENABLE_LOGGING"] = "1" if args.log else "0"
    os.environ["MIOPEN_LOG_LEVEL"] = "6" if args.log else "0"
    if args.cases:
        data = json.loads(owned(args.cases).read_text())
        selected = data["cases"] if isinstance(data, dict) else data
    else:
        from bench_conv_catalog_v06 import cases
        selected = cases(args.set)
    selected = [normalize_case(c) for c in selected]
    if args.case:
        selected = [c for c in selected if c["name"] == args.case]
    if not selected:
        raise ValueError("No cases selected")
    if len({case_id(c) for c in selected}) != len(selected):
        raise ValueError("Duplicate tensor/descriptor cases must be collapsed before timing")
    import torch
    import torch.nn.functional as F
    from gpu_guard_v06 import GpuAudit
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    source = discover_db(dbdir)
    before = load_db(source) if source else {}
    source_sha = sha(source) if source else None
    report = {"schema": "v08_solver_direction_bench", "label": args.label,
              "torch": torch.__version__, "hip": torch.version.hip,
              "mode": args.mode, "benchmark": False, "dtype": "fp16",
              "gate": GATE, "safety": SAFETY, "seed": args.seed,
              "db": str(dbdir), "db_file_before": str(source) if source else None,
              "db_sha256_before": source_sha, "cases": [],
              "warmup": args.warmup, "rounds": args.rounds, "reps": args.reps,
              "logging_enabled": args.log, "probe_only": args.probe_only,
              "timing_scope": "each direction alone; no bias, allocation included; GPU event and wall separate",
              "reference": "FP32 MIOpen on exactly FP16-rounded input/weight/grad-output; both safety blockers active",
              "status": "running"}
    audit = GpuAudit()
    try:
        with audit, capture_fd2(out / "miopen_stderr.log"):
            audit.ensure_clear()
            report["device"] = torch.cuda.get_device_name()
            for c in selected:
                audit.ensure_clear()
                print("BEGIN", c["name"], flush=True)
                gen = torch.Generator(device="cpu").manual_seed(args.seed)
                xcpu = torch.randn(c["shape"], generator=gen).half()
                wcpu = (torch.randn(c["weight"], generator=gen) /
                        math.sqrt(math.prod(c["weight"][1:]))).half()
                dycpu = (torch.randn(output_shape(c), generator=gen) * 0.01).half()
                x, w, dy = xcpu.cuda(), wcpu.cuda(), dycpu.cuda()
                trans = "transpose" in c["kind"]
                kw = {k: tuple(c[k]) for k in ("stride", "padding", "dilation")}
                kw["groups"] = c["groups"]
                if trans:
                    kw["output_padding"] = tuple(c["output_padding"])
                native = getattr(F, c["kind"])
                def run(direction, xx=x, ww=w, gg=dy):
                    if direction == "fwd":
                        return native(xx, ww, None, **kw)
                    mask = [direction == "dx", direction == "dw", False]
                    vals = torch.ops.aten.convolution_backward.default(
                        gg, xx, ww, None, c["stride"], c["padding"], c["dilation"],
                        trans, c["output_padding"], c["groups"], mask)
                    return vals[0] if direction == "dx" else vals[1]
                row = {"case": c, "directions": {}}
                outputs = {}
                for direction in DIRECTIONS:
                    torch.cuda.synchronize()
                    print("V08_DIRECTION_BEGIN", c["name"], direction, file=sys.stderr, flush=True)
                    cold = time.perf_counter()
                    outputs[direction] = run(direction)
                    torch.cuda.synchronize()
                    cold_s = time.perf_counter() - cold
                    print("V08_DIRECTION_READY", c["name"], direction, file=sys.stderr, flush=True)
                    for _ in range(0 if args.probe_only else args.warmup):
                        run(direction)
                    torch.cuda.synchronize()
                    event_ms, wall_ms = [], []
                    for _ in range(0 if args.probe_only else args.rounds):
                        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        a.record()
                        for _ in range(args.reps):
                            run(direction)
                        b.record(); b.synchronize()
                        event_ms.append(a.elapsed_time(b) / args.reps)
                        wall_ms.append((time.perf_counter() - start) * 1000 / args.reps)
                    key = db_key(c, direction)
                    dr = {"cold_s": cold_s, "event_ms_rounds": event_ms,
                          "wall_ms_rounds": wall_ms,
                          "event_median_ms": statistics.median(event_ms) if event_ms else None,
                          "wall_median_ms": statistics.median(wall_ms) if wall_ms else None,
                          "db_key": key, "db_entries_before": before.get(key, []),
                          "requested_single_solver": before[key][0]["solver"]
                              if len(before.get(key, [])) == 1 else None}
                    if args.profile and not args.probe_only:
                        with torch.profiler.profile(activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA]) as prof:
                            run(direction)
                            torch.cuda.synchronize()
                        trace = out / (c["name"] + "_" + direction + ".trace.json")
                        prof.export_chrome_trace(str(trace))
                        from research_comfy_v05 import device_activity_table
                        dr["device_activities"] = device_activity_table(trace)
                        dr["trace"] = str(trace)
                    row["directions"][direction] = dr
                if args.probe_only:
                    # A native-log probe performs exactly one convolution per
                    # direction; no FP32 reference convolutions contaminate it.
                    for direction in DIRECTIONS:
                        row["directions"][direction]["reference_error"] = {
                            "screen_pass": None, "finite": bool(torch.isfinite(outputs[direction]).all()),
                            "reference_finite": None, "nrmse": None,
                            "max_over_reference_rms": None, "not_measured": "native-log probe only"}
                    row["all_gates_pass"] = None
                    report["reference"] = "Not measured in native-log-only probe; use separately validated timed replay."
                else:
                    xf, wf, dyf = x.float(), w.float(), dy.float()
                    for direction in DIRECTIONS:
                        reference = run(direction, xf, wf, dyf)
                        row["directions"][direction]["reference_error"] = error_metrics(outputs[direction], reference)
                        del reference
                    row["all_gates_pass"] = all(row["directions"][d]["reference_error"]["screen_pass"] for d in DIRECTIONS)
                    del xf, wf, dyf
                report["cases"].append(row)
                dump(out / "result.json", report)
                print("RESULT", c["name"], json.dumps({
                    d: {"ms": row["directions"][d]["event_median_ms"],
                        "pass": row["directions"][d]["reference_error"]["screen_pass"]}
                    for d in DIRECTIONS}), flush=True)
                del x, w, dy, xcpu, wcpu, dycpu, outputs
                gc.collect()
            report["status"] = "completed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["gpu_audit"] = audit.report()
        report["valid_for_timing"] = (report["status"] == "completed" and
                                      audit.report()["valid_for_timing"] and not args.log and not args.probe_only)
        after_file = discover_db(dbdir)
        after = load_db(after_file) if after_file else {}
        report["db_file_after"] = str(after_file) if after_file else None
        report["db_sha256_after"] = sha(after_file) if after_file else None
        for row in report["cases"]:
            for dr in row["directions"].values():
                key = dr["db_key"]
                dr["db_entries_after"] = after.get(key, [])
                dr["db_target_entry_unchanged"] = dr["db_entries_before"] == after.get(key, [])
        dump(out / "result.json", report)
    print("DONE", out, report["valid_for_timing"], flush=True)


def kernel_fingerprint(direction):
    rows = direction.get("device_activities", {}).get("rows", [])
    return sorted((r.get("category"), r["name"], int(r["count"])) for r in rows)


def command_merge(args):
    baseline_path = owned(args.baseline)
    baseline = json.loads(baseline_path.read_text())
    if not baseline.get("valid_for_timing"):
        raise ValueError("Baseline timing must be valid")
    if baseline["gate"] != GATE or baseline["mode"] != "FAST":
        raise ValueError("Baseline gate/mode mismatch")
    source = owned(args.base_db)
    records = load_db(source)
    records = {k: [e for e in es if e["solver"] not in BLOCKED] for k, es in records.items()}
    before = sha(source)
    reports = []
    for path in args.candidate:
        path = owned(path)
        data = json.loads(path.read_text())
        if (not data.get("valid_for_timing") or data.get("gate") != GATE or
            data.get("mode") != "FAST" or data.get("seed") != baseline.get("seed")):
            raise ValueError("Candidate audit/gate/mode/seed mismatch: " + str(path))
        if data["torch"] != baseline["torch"] or data["hip"] != baseline["hip"]:
            raise ValueError("Candidate runtime mismatch")
        reports.append((path, data, {case_id(r["case"]): r for r in data["cases"]}))
    decisions = []
    candidate_reviews = []
    for base in baseline["cases"]:
        cid = case_id(base["case"])
        for direction in DIRECTIONS:
            dr = base["directions"][direction]
            if not dr["reference_error"]["screen_pass"]:
                raise ValueError("Baseline failed numerical gate: " + base["case"]["name"] + "/" + direction)
            winner = None
            best = dr["event_median_ms"]
            for path, data, rows in reports:
                if cid not in rows:
                    continue
                cand = rows[cid]["directions"][direction]
                entries = cand["db_entries_before"]
                if (not cand["reference_error"]["screen_pass"] or len(entries) != 1 or
                    entries[0]["solver"] in BLOCKED or not cand["db_target_entry_unchanged"]):
                    continue
                ms = cand["event_median_ms"]
                event_gain = dr["event_median_ms"] / ms
                wall_gain = dr["wall_median_ms"] / cand["wall_median_ms"]
                native_fp, candidate_fp = kernel_fingerprint(dr), kernel_fingerprint(cand)
                changed = bool(native_fp and candidate_fp and native_fp != candidate_fp)
                reason = ("missing_actual_kernel_trace" if not native_fp or not candidate_fp else
                          "same_kernel_names_and_launch_counts_treat_as_noise" if not changed else
                          "event_or_wall_gain_below_threshold" if min(event_gain, wall_gain) < args.min_speedup else
                          "eligible_measured_changed_kernel")
                candidate_reviews.append({"case": base["case"]["name"], "direction": direction,
                                          "evidence": str(path), "event_speedup": event_gain,
                                          "wall_speedup": wall_gain, "reason": reason})
                if changed and min(event_gain, wall_gain) >= args.min_speedup and ms < best:
                    best = ms
                    winner = (path, cand)
            decision = {"case": base["case"]["name"], "direction": direction,
                        "db_key": dr["db_key"], "baseline_ms": dr["event_median_ms"],
                        "selected_ms": best, "speedup": dr["event_median_ms"] / best,
                        "selected": "preserve_baseline"}
            if winner is not None:
                path, cand = winner
                records[dr["db_key"]] = cand["db_entries_before"]
                decision.update(selected=cand["db_entries_before"][0]["solver"],
                                evidence=str(path), gate=cand["reference_error"])
            decisions.append(decision)
    out = fresh_dir(args.out)
    target = out / source.name
    write_db(target, records)
    auxiliary = copy_aux(source, out)
    manifest = {"schema": "v08_direction_selected_finddb",
                "status": "operator_screened_requires_fresh_FAST_replay_and_model_AB",
                "baseline": str(baseline_path), "base_db": str(source),
                "base_db_sha256": before, "base_db_unchanged": sha(source) == before,
                "output_db": str(target), "output_sha256": sha(target),
                "untested_keys": "preserve baseline, removing both known blocked solver names",
                "gate": GATE, "safety": SAFETY, "min_speedup": args.min_speedup,
                "candidate_results": [str(p) for p, _, _ in reports],
                "auxiliary_files": auxiliary, "decisions": decisions,
                "candidate_reviews": candidate_reviews,
                "selection_policy": "Both event and wall speedup >= threshold, actual kernel/name launch-count change; same-kernel few-percent differences preserve baseline",
                "changed_directions": sum(x["selected"] != "preserve_baseline" for x in decisions)}
    dump(out / "selection_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sp = p.add_subparsers(dest="command", required=True)
    q = sp.add_parser("inventory")
    q.add_argument("--source", type=Path, required=True)
    q.add_argument("--out", type=Path)
    q.add_argument("--full", action="store_true")
    q.set_defaults(function=command_inventory)
    q = sp.add_parser("prepare")
    q.add_argument("--source", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--ranks", type=int, default=3, help="0: all recorded candidate ranks")
    q.set_defaults(function=command_prepare)
    q = sp.add_parser("bench")
    q.add_argument("--cases", type=Path, help="JSON list (or {'cases': [...]}) using v0.6 case fields")
    q.add_argument("--set", default="resblock")
    q.add_argument("--case")
    q.add_argument("--db", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--cache", type=Path)
    q.add_argument("--label", default="unspecified")
    q.add_argument("--mode", choices=["FAST", "NORMAL"], default="FAST")
    q.add_argument("--seed", type=int, default=63117)
    q.add_argument("--rounds", type=int, default=3)
    q.add_argument("--reps", type=int, default=7)
    q.add_argument("--warmup", type=int, default=3)
    q.add_argument("--profile", action="store_true")
    q.add_argument("--log", action="store_true", help="Native solver logging; timing marked invalid to avoid logging overhead bias")
    q.add_argument("--probe-only", action="store_true", help="With --log: exactly one FP16 convolution per direction, no warm/timed/reference convolutions")
    q.set_defaults(function=command_bench)
    q = sp.add_parser("merge")
    q.add_argument("--baseline", type=Path, required=True)
    q.add_argument("--candidate", type=Path, nargs="+", required=True)
    q.add_argument("--base-db", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--min-speedup", type=float, default=1.05)
    q.set_defaults(function=command_merge)
    args = p.parse_args()
    if getattr(args, "probe_only", False) and not args.log:
        p.error("--probe-only requires --log and never produces valid timing")
    if getattr(args, "min_speedup", 1.05) < 1.05:
        p.error("Use at least 1.05; smaller gains need separate repeated holdout evidence")
    if getattr(args, "ranks", 0) < 0:
        p.error("ranks must be nonnegative")
    if any(getattr(args, k, 1) < 1 for k in ("rounds", "reps", "warmup")):
        p.error("timing counts must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
