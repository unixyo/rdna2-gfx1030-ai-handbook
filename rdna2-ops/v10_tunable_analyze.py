"""CPU-only analysis of v1.0 TunableOp probe/model JSON and Chrome traces.

No torch, ROCm, image decoder or GPU calls. Missing evidence remains pending.
Model input syntax: ROLE:LABEL=DIR, with ROLE original/filter/off.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path, sources, missing):
    if not path.exists():
        missing.append(str(path))
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        missing.append(f"{path}: {type(exc).__name__}: {exc}")
        return None
    sources[str(path)] = digest(path)
    return data


def equal_known(a, b):
    return None if a is None or b is None else a == b


def hash_stats(hashes, expected=None):
    values = [x for x in hashes if x]
    enough = len(values) >= 2 and (expected is None or len(values) == expected)
    return {"samples": len(values), "distinct": len(set(values)), "hashes": values,
            "status": "pending" if not enough else ("bitwise_equal" if len(set(values)) == 1 else "different")}


def audit_gate(a):
    if not a or "valid_for_timing" not in a:
        return None
    return bool(a.get("available") and a["valid_for_timing"] and
                not a.get("foreign_clients") and not a.get("errors"))


def trace_kernels(path, sources, missing):
    data = read_json(path, sources, missing)
    if data is None:
        return {"status": "pending", "path": str(path)}
    kernels = collections.Counter(e.get("name", "") for e in data.get("traceEvents", [])
                                  if e.get("cat") == "kernel")
    tensile = [k for k in kernels if "Cijk_" in k]
    return {"status": "observed" if kernels else "pending", "path": str(path),
            "reason": None if kernels else "trace_has_no_kernel_events",
            "kernel_launches": sum(kernels.values()), "kernels": dict(kernels),
            "tensile_names": tensile,
            "canonical_tensile_names": [k.removesuffix(".kd") for k in tensile],
            "name_normalization": "Only the terminal AMD kernel descriptor suffix .kd is removed; raw names are retained."}


def expected_kernel(inventory, op, signature, solution):
    found = {e["tensile"]["name"] for c in inventory.get("csvs", []) for e in c["entries"]
             if e.get("tensile") and (e["op"], e["signature"], e["solution"]) == (op, signature, solution)}
    return sorted(found)


def probe(label, directory, inventory, sources):
    missing = []
    r = read_json(directory / "result.json", sources, missing)
    if r is None:
        return {"label": label, "directory": str(directory), "status": "pending", "missing": missing}
    e = r.get("entry", {})
    key = [e.get("op"), e.get("signature")]
    loaded = r.get("loaded_target_rows", [])
    selected = r.get("requested_solution")
    loaded_exact = len(loaded) == 1 and loaded[0][:3] == [*key, selected]
    observed = r.get("pre_replay_signature_rows")
    inputs = r.get("input_storage_sha256")
    after = r.get("input_storage_sha256_after")
    input_unchanged = (inputs[:2] == after) if inputs and after else None
    checks = {"completed": r.get("completed"), "reported_timing_valid": r.get("valid_for_timing"),
              "audit": audit_gate(r.get("gpu_audit")),
              "signature_hit": r.get("signature_hit"),
              "pre_replay_signature_exact": observed == [key] if observed else None,
              "loaded_solution_exact": loaded_exact, "inputs_unchanged": input_unchanged,
              "all_finite": r.get("all_finite")}
    modes = {}
    for name in ("on", "off"):
        m = r.get("modes", {}).get(name, {})
        samples = m.get("repeat_vs_first", [])
        times = m.get("event_ms_groups", [])
        modes[name] = {**hash_stats(m.get("hashes", []), r.get("repeats")),
                       "all_finite": m.get("all_finite"),
                       "max_abs_vs_first": max((x["max_abs"] for x in samples if x.get("max_abs") is not None), default=None),
                       "event_ms_median": statistics.median(times) if times else None,
                       "event_groups": times,
                       "trace": trace_kernels(directory / f"{name}.trace.json", sources, missing)}
    expected = expected_kernel(inventory, *key, selected)
    actual = (modes["on"]["trace"].get("canonical_tensile_names")
              if modes["on"]["trace"].get("status") == "observed" else None)
    kernel_match = any(x in actual for x in expected) if expected and actual is not None else None
    trace_status = ("matched" if kernel_match else "mismatch") if kernel_match is not None else "pending"
    valid = all(v is True for v in checks.values())
    return {"label": label, "directory": str(directory), "status": "verified" if valid else
            ("invalid" if any(v is False for v in checks.values()) else "pending"),
            "checks": checks, "op": key[0], "signature": key[1], "solution": selected,
            "seed": r.get("seed"), "alpha": r.get("alpha"), "beta": r.get("beta"),
            "layouts": r.get("layouts"), "actual_tensors": r.get("actual_tensors"),
            "input_storage_sha256": inputs, "runtime": r.get("runtime"),
            "modes": modes, "expected_selected_tensile_names": expected,
            "selected_trace_kernel_match": kernel_match,
            "selected_trace_status": trace_status,
            "verification_scope": "Measurement checks; selected-kernel trace confirmation is separate and may remain pending.",
            "solution_index_in_rocblas_log": r.get("requested_solution_in_rocblas_log"),
            "trace_note": "Default/reserved IDs need direct trace interpretation; no expected Tensile name is assumed.",
            "on_vs_off": r.get("on_vs_off"), "missing": missing, "error": r.get("error")}


def argument(args, name):
    return args[args.index(name) + 1] if name in args and args.index(name) + 1 < len(args) else None


def snapshot_hash(s):
    return s.get("sha256") if isinstance(s, dict) else None


def model(role, label, directory, sources, warmup, inventory):
    missing = []
    p = read_json(directory / "protocol.json", sources, missing) or {}
    runs = read_json(directory / "runs.json", sources, missing)
    runs = runs if isinstance(runs, list) else p.get("runs", [])
    capture = read_json(directory / "conditioning_capture.json", sources, missing) or {}
    samplers = capture.get("sampler_calls", [])
    clips = capture.get("conditioning_calls", [])
    snapshots = [row.get(field, {}) for row in clips for field in ("tokens", "output")]
    snapshots += [row.get(field, {}) for row in samplers
                  for field in ("positive", "negative", "latent_input", "latent_output")]
    finite_evidence = [snapshot.get("all_tensor_leaves_finite") for snapshot in snapshots]
    capture_finite = (False if False in finite_evidence else
                      True if finite_evidence and all(x is True for x in finite_evidence) else None)
    image_rows = collections.defaultdict(list)
    for image in p.get("images", []):
        path = Path(image["path"])
        match = re.match(r"^(?:anima|rehearsal)_(\d+)_(.+?)_\d+_?\.png$", path.name)
        if match:
            integrity = path.exists() and digest(path) == image.get("file_sha256")
            image_rows[int(match[1])].append({"sha256": image.get("rgb_sha256") if integrity else None,
                                             "path": str(path), "integrity": integrity})
    checks = {"completed": p.get("completed"), "reported_timing_valid": p.get("valid_for_timing"),
              "audit": audit_gate(p.get("gpu_audit")),
              "source_csv_unchanged": p.get("input_csv_unchanged"), "runner_unchanged": p.get("runner_unchanged"),
              "workflow_unchanged": p.get("workflow_unchanged"),
              "runtime_isolated": p.get("runtime_directories_isolated"),
              "capture_installed": capture.get("installed"), "capture_finite": capture_finite,
              "capture_complete": len(samplers) == sum(bool(r.get("success")) for r in runs) if capture else None}
    summaries = {}
    first_seen = set()
    successful_seen = 0
    excluded = []
    for row in runs:
        mode, idx = row.get("mode"), row.get("idx")
        if not row.get("success"):
            continue
        if mode not in summaries:
            summaries[mode] = {"all_indices": [], "warm_indices": [], "denoise_s": [],
                               "request_wall_s": [], "rgb": [], "positive": [], "negative": [],
                               "latent_input": [], "latent_output": [], "seeds": [], "clip_calls_seen": []}
        dst = summaries[mode]
        dst["all_indices"].append(idx)
        cold = mode == "warm" or (warmup == "first-per-mode" and mode not in first_seen) or (
            warmup == "first-request" and successful_seen == 0)
        first_seen.add(mode)
        successful_seen += 1
        sec = row.get("wall", {}).get("denoise")
        if not cold and isinstance(sec, (int, float)) and math.isfinite(sec):
            dst["warm_indices"].append(idx)
            dst["denoise_s"].append(sec)
            dst["request_wall_s"].append(row.get("request_wall_s"))
        else:
            excluded.append({"idx": idx, "mode": mode, "reason": "warmup" if cold else "missing_time"})
        images = sorted(image_rows.get(idx, []), key=lambda x: x["path"])
        # One digest for the ordered RGB image list emitted by this request.
        dst["rgb"].append(hashlib.sha256(json.dumps([x["sha256"] for x in images]).encode()).hexdigest()
                          if images and all(x["integrity"] and x["sha256"] for x in images) else None)
        sampler = next((s for s in samplers if s.get("index") == idx and s.get("mode") == mode), {})
        for field in ("positive", "negative", "latent_input", "latent_output"):
            dst[field].append(snapshot_hash(sampler.get(field)))
        dst["seeds"].append(sampler.get("seed"))
        dst["clip_calls_seen"].append(sampler.get("clip_calls_seen"))
    for mode, dst in summaries.items():
        dst["warm_denoise_median_s"] = statistics.median(dst["denoise_s"]) if dst["denoise_s"] else None
        dst["warm_samples"] = len(dst["denoise_s"])
        for field in ("rgb", "positive", "negative", "latent_input", "latent_output"):
            dst[field] = hash_stats(dst[field], len(dst["all_indices"]))
    valid = all(v is True for v in checks.values())
    stage_evidence = []
    for clip in clips:
        trace = (trace_kernels(Path(clip["trace"]), sources, missing) if clip.get("trace")
                 else {"status": "pending"})
        actual = trace.get("canonical_tensile_names") if trace.get("status") == "observed" else None
        hits = []
        for hit in clip.get("csv", {}).get("hits", []):
            expected = expected_kernel(inventory, hit.get("op"), hit.get("signature"), hit.get("loaded_solution"))
            hit_match = any(x in actual for x in expected) if expected and actual is not None else None
            hits.append({**hit, "expected_tensile_names": expected,
                         "selected_kernel_present_in_stage": hit_match,
                         "trace_status": ("matched" if hit_match else "mismatch") if hit_match is not None else "pending"})
        stage_evidence.append({"index": clip.get("index"), "mode": clip.get("mode"),
                               "exact_csv_hits": hits, "misses": clip.get("csv", {}).get("misses", []),
                               "hit_logging_enabled": clip.get("csv", {}).get("enabled"),
                               "trace": trace, "link_scope": "Same CLIP region, not individual-call correlation."})
    return {"role": role, "label": label, "directory": str(directory), "model": p.get("model"),
            "status": "verified" if valid else ("invalid" if p.get("error") or p.get("exit_code", 0) not in (0, None)
                                                      or ("exit_code" in p and any(v is False for v in checks.values())) else "pending"),
            "checks": checks, "speed_eligible": bool(valid and p.get("valid_for_speed_comparison")),
            "timing_scope": p.get("timing_scope"), "warmup_policy": warmup, "excluded": excluded,
            "tunable_setting": p.get("tunable"),
            "csv_sha256": p.get("csv_sha256"), "runner_sha256": p.get("runner_sha256"),
            "workflow_sha256": p.get("workflow_sha256"), "runtime_torch": p.get("runtime_torch"),
            "seed": argument(p.get("args", []), "--seed"), "steps": argument(p.get("args", []), "--steps"),
            "modes": summaries, "clip_call_count": len(clips), "sampler_call_count": len(samplers),
            "clip_token_sha256": [snapshot_hash(x.get("tokens")) for x in clips],
            "clip_output_sha256": [snapshot_hash(x.get("output")) for x in clips],
            "conditioning_csv_evidence": stage_evidence,
            "sampler_csv_evidence": [{"index": row.get("index"), "mode": row.get("mode"),
                                      "tunable_enabled": row.get("tunable_enabled"),
                                      "hit_logging_enabled": row.get("csv", {}).get("enabled"),
                                      "exact_csv_hits": row.get("csv", {}).get("hits", []),
                                      "misses": row.get("csv", {}).get("misses", [])}
                                     for row in samplers],
            "missing": missing, "error": p.get("error")}


def same_fields(a, b, fields):
    return {field: equal_known(a.get(field), b.get(field)) for field in fields}


def probe_pairs(items):
    pairs = []
    for a, b in itertools.combinations(items, 2):
        controls = same_fields(a, b, ("op", "signature", "seed", "alpha", "beta", "layouts",
                                      "actual_tensors", "input_storage_sha256", "runtime"))
        compatible = all(x is True for x in controls.values())
        rows = {}
        for mode in ("on", "off"):
            ah = a.get("modes", {}).get(mode, {}).get("hashes", [])
            bh = b.get("modes", {}).get(mode, {}).get("hashes", [])
            rows[mode] = {"same_first_output": ah[0] == bh[0] if ah and bh and compatible else None,
                          "combined_distinct_outputs": len(set(ah + bh)) if ah and bh and compatible else None}
        pairs.append({"a": a["label"], "b": b["label"], "controls": controls,
                      "status": "comparable" if compatible and a["status"] == b["status"] == "verified" else "pending_or_mismatch",
                      "same_solution": equal_known(a.get("solution"), b.get("solution")), "outputs": rows})
    return pairs


def model_pairs(items):
    pairs, speeds = [], []
    controls_fields = ("model", "seed", "steps", "runner_sha256", "workflow_sha256", "runtime_torch")
    for a, b in itertools.combinations(items, 2):
        controls = same_fields(a, b, controls_fields)
        compatible = all(x is True for x in controls.values())
        modes = {}
        for mode in set(a.get("modes", {})) & set(b.get("modes", {})):
            fields = {}
            for field in ("positive", "negative", "latent_input", "latent_output", "rgb"):
                ah, bh = a["modes"][mode][field]["hashes"], b["modes"][mode][field]["hashes"]
                fields[field] = {"shared_hashes": len(set(ah) & set(bh)) if ah and bh and compatible else None,
                                 "combined_distinct": len(set(ah + bh)) if ah and bh and compatible else None}
            modes[mode] = fields
        pairs.append({"a": a["label"], "b": b["label"], "controls": controls,
                      "status": "comparable" if compatible and a["status"] == b["status"] == "verified" else "pending_or_mismatch",
                      "kind": "process_replay" if a.get("csv_sha256") and a.get("csv_sha256") == b.get("csv_sha256") and a["role"] == b["role"] else "intervention",
                      "same_clip_tokens": equal_known(a.get("clip_token_sha256") or None, b.get("clip_token_sha256") or None),
                      "modes": modes})
    for base in items:
        if base["role"] != "off":
            continue
        for cand in items:
            if cand["role"] not in ("original", "filter"):
                continue
            if base.get("model") and cand.get("model") and base["model"] != cand["model"]:
                continue
            fields = same_fields(base, cand, controls_fields)
            mode = {"anima": "cand", "qwen": "tbase"}.get(cand.get("model"))
            bm, cm = base.get("modes", {}).get("base", {}), cand.get("modes", {}).get(mode, {})
            ready = (all(x is True for x in fields.values()) and base["speed_eligible"] and cand["speed_eligible"]
                     and bm.get("warm_samples", 0) >= 2 and cm.get("warm_samples", 0) >= 2)
            speeds.append({"baseline": base["label"] + ":base", "candidate": cand["label"] + ":" + (mode or "pending-mode"),
                           "status": "verified" if ready else "pending",
                           "controls": fields, "baseline_warm_n": bm.get("warm_samples", 0),
                           "candidate_warm_n": cm.get("warm_samples", 0),
                           "speedup": bm["warm_denoise_median_s"] / cm["warm_denoise_median_s"] if ready else None,
                           "note": "Needs completed exclusive uninstrumented runs and at least 2 warm samples per arm."})
    return pairs, speeds


def parse_item(value, model_item=False):
    left, sep, path = value.partition("=")
    if not sep or not left or not path:
        raise argparse.ArgumentTypeError("Use LABEL=DIR, or ROLE:LABEL=DIR for model")
    if model_item:
        role, sep, label = left.partition(":")
        if not sep or role not in ("original", "filter", "off") or not label:
            raise argparse.ArgumentTypeError("Model role must be original/filter/off")
        return role, label, Path(path).resolve()
    return left, Path(path).resolve()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", action="append", default=[])
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--inventory", type=Path, default=ROOT / "results/v10/tunable_cpu/inventory.json")
    ap.add_argument("--warmup", choices=["first-per-mode", "first-request", "none"], default="first-per-mode")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sources, missing = {}, []
    inventory = read_json(args.inventory, sources, missing) or {}
    probes = [probe(*parse_item(x), inventory, sources) for x in args.probe]
    models = [model(*parse_item(x, True), sources, args.warmup, inventory) for x in args.model]
    mpairs, speeds = model_pairs(models)
    result = {"cpu_only": True, "torch_imported": False, "gpu_execution": False,
              "helper_sha256": digest(Path(__file__)), "source_sha256": sources,
              "warmup_policy": args.warmup, "probes": probes, "probe_comparisons": probe_pairs(probes),
              "models": models, "model_comparisons": mpairs, "speed_comparisons": speeds,
              "missing_inventory": missing,
              "limits": ["Hash metadata is read without loading .pt tensors.",
                         "No cause is inferred from GSU or differing outputs alone.",
                         "Missing captures/traces/protocols remain pending; old runs.json alone cannot validate timing."]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# TunableOp CPU 결과 요약", "", "GPU를 실행하지 않고 저장된 JSON/trace를 읽은 결과다. 누락은 pending이다.", "",
             f"warm 제외 규칙: {args.warmup}. 각 속도 비율에는 arm별 warm 2회 이상이 필요하다.", "",
             "| 프로브 | 상태 | signature | solution | on 출력 종류 | off 출력 종류 | 선택 kernel 일치 |",
             "|---|---|---|---|---:|---:|---|"]
    for r in probes:
        lines.append(f"| {r['label']} | {r['status']} | {r.get('signature','pending')} | {r.get('solution','pending')} | "
                     f"{r.get('modes',{}).get('on',{}).get('distinct','pending')} | "
                     f"{r.get('modes',{}).get('off',{}).get('distinct','pending')} | {r.get('selected_trace_status','pending')} |")
    lines += ["", "| 모델/역할 | 상태 | mode | warm n | denoise 중앙값 s | RGB 반복 | 조건 반복 | 속도 판정 가능 |",
              "|---|---|---|---:|---:|---|---|---|"]
    for r in models:
        for mode, s in r["modes"].items():
            lines.append(f"| {r['label']}/{r['role']} | {r['status']} | {mode} | {s['warm_samples']} | "
                         f"{s['warm_denoise_median_s'] if s['warm_denoise_median_s'] is not None else 'pending'} | {s['rgb']['status']} | "
                         f"{s['positive']['status']}/{s['negative']['status']} | {r['speed_eligible']} |")
        if not r["modes"]:
            lines.append(f"| {r['label']}/{r['role']} | pending | pending | 0 | pending | pending | pending | False |")
    lines += ["", "| 속도 기준 | 후보 | 상태 | 기준 warm n | 후보 warm n | 배율 |",
              "|---|---|---|---:|---:|---:|"]
    for s in speeds:
        lines.append(f"| {s['baseline']} | {s['candidate']} | {s['status']} | {s['baseline_warm_n']} | "
                     f"{s['candidate_warm_n']} | {s['speedup'] if s['speedup'] is not None else 'pending'} |")
    lines += ["", "프로브 상태는 측정 관문이며 선택 kernel trace의 확인 상태는 별도다.",
              "비교·관문·SHA·stage별 실제 CSV hit·trace 이름과 누락 경로는 analysis.json에 있다.", ""]
    (args.out / "SUMMARY_KO.md").write_text("\n".join(lines))
    print(json.dumps({"out": str(args.out), "probes": {r["label"]: r["status"] for r in probes},
                      "models": {r["label"]: r["status"] for r in models},
                      "speed_comparisons": speeds}, indent=2))


if __name__ == "__main__":
    main()

