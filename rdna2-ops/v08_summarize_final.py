#!/usr/bin/env python3
"""Summarize the declared 100-epoch A/A-prime/B runs using only the stdlib.

Exit 0: all recorded training checks pass; exit 2: results are still pending;
exit 1: a run or validation failed. Perceptual quality always remains pending_user.
No training modules, model tensors, GPU APIs, or subprocesses are loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/applio_v08"
LOSSES = ("mel", "fm", "kl", "adv", "loss_g", "loss_d")
ARMS = {"A": "final100_A", "A_prime": "final100_A_prime", "B": "final100_B"}
FROZEN_SOURCES = (
    "applio_epoch_v08.py", "v08_run_epoch_series.py", "v08_direct_conv.py",
    "v08_nonconv_opt.py", "results/applio_v08/direct_selected_final/dispatch.json",
)


def owned(value):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Research path required: " + str(path))
    return path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path):
    try:
        data = path.read_bytes()
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value, hashlib.sha256(data).hexdigest(), None
    except (OSError, ValueError) as error:
        return None, None, f"{type(error).__name__}: {error}"


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sha256_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def close(left, right):
    return finite(left) and finite(right) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-9)


def summarize_run(result, coordinator, *, expected_epochs=100, tail_epochs=25,
                  steps_per_epoch=31, require_reserved=True):
    """Pure arithmetic/record checks; parameterization is for the short CPU check only."""
    checks = {}
    epochs = result.get("epochs", [])
    steps = result.get("steps", [])
    if not isinstance(epochs, list) or not all(isinstance(e, dict) for e in epochs):
        epochs = []
    if not isinstance(steps, list) or not all(isinstance(s, dict) for s in steps):
        steps = []
    expected_steps = expected_epochs * steps_per_epoch
    checks["completed"] = result.get("completed") is True
    checks["schema"] = result.get("schema") == "applio_epoch_v08"
    checks["coordinator_completed"] = (coordinator.get("status") == "completed"
        and coordinator.get("exit_code") == 0 and coordinator.get("valid") is True)
    checks["timing_valid"] = (result.get("valid_for_timing") is True
        and result.get("valid_for_epoch_timing") is True)
    checks["epoch_sequence"] = [e.get("epoch") for e in epochs] == list(range(1, expected_epochs + 1))
    checks["step_sequence"] = [s.get("step") for s in steps] == list(range(1, expected_steps + 1))
    checks["step_epoch_alignment"] = (len(steps) == expected_steps and all(
        s.get("epoch") == i // steps_per_epoch + 1 for i, s in enumerate(steps)))
    checks["steps_per_epoch"] = (result.get("steps_per_epoch") == steps_per_epoch
        and len(epochs) == expected_epochs
        and all(e.get("steps") == steps_per_epoch for e in epochs))
    bad_losses = {key: sum(not finite(s.get(key)) for s in steps) for key in LOSSES}
    checks["six_losses_finite"] = len(steps) == expected_steps and not any(bad_losses.values())
    checks["epoch_nonfinite_loss_counts"] = (len(epochs) == expected_epochs
        and all(e.get("nonfinite_losses") == 0 for e in epochs))
    seconds = [e.get("seconds") for e in epochs]
    checks["epoch_times"] = len(seconds) == expected_epochs and all(finite(s) and s > 0 for s in seconds)
    timing = {"first_epoch_s": None, "steady_epoch_median_s": None,
              "training_epoch_sum_s": None, "external_whole_wall_s": coordinator.get("elapsed_wall_s"),
              "process_after_import_s": result.get("total_process_after_import_s")}
    if checks["epoch_times"]:
        timing.update(first_epoch_s=seconds[0], steady_epoch_median_s=statistics.median(seconds[1:]),
                      training_epoch_sum_s=sum(seconds))
    checks["stored_epoch_sum_matches"] = close(timing["training_epoch_sum_s"], result.get("training_epoch_sum_s"))
    checks["stored_steady_median_matches"] = close(timing["steady_epoch_median_s"], result.get("steady_epoch_median_s"))
    checks["external_wall_recorded"] = finite(timing["external_whole_wall_s"]) and timing["external_whole_wall_s"] > 0
    checks["coordinator_timing_matches"] = all(close(timing[key], coordinator.get(key))
        for key in ("first_epoch_s", "steady_epoch_median_s", "training_epoch_sum_s"))
    peaks = {}
    for kind in ("allocated", "reserved"):
        values = [e.get(f"peak_{kind}_bytes") for e in epochs]
        available = len(values) == expected_epochs and all(
            isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in values)
        peaks[f"max_{kind}_bytes"] = max(values) if available else None
        peaks[f"max_{kind}_gib"] = max(values) / (1 << 30) if available else None
        if kind == "allocated" or require_reserved:
            checks[f"peak_{kind}_recorded"] = available
    scale_valid = len(steps) == expected_steps and all(
        finite(s.get("scale_before")) and s["scale_before"] > 0
        and finite(s.get("scale_after")) and s["scale_after"] > 0 for s in steps)
    checks["scales_recorded"] = scale_valid
    reductions = sum(s["scale_after"] < s["scale_before"] for s in steps) if scale_valid else None
    if scale_valid:
        per_epoch = Counter(s["epoch"] for s in steps if s["scale_after"] < s["scale_before"])
        checks["scale_reduction_flags_match"] = all(
            s.get("scale_reduced") is (s["scale_after"] < s["scale_before"]) for s in steps)
        checks["epoch_scale_reductions_match"] = all(
            e.get("scale_reductions") == per_epoch[e.get("epoch")] for e in epochs)
        checks["coordinator_scale_reductions_match"] = reductions == coordinator.get("scale_reductions")
    grad_counts = {}
    for role in ("d", "g"):
        key, marker = f"grad_norm_{role}", f"grad_norm_{role}_nonfinite"
        valid = lambda s: (key in s and marker in s and (
            (finite(s[key]) and s[marker] is None)
            or (s[key] is None and s[marker] in ("nan", "inf", "-inf"))))
        checks[f"grad_norm_{role}_records"] = len(steps) == expected_steps and all(valid(s) for s in steps)
        grad_counts[role] = sum(s.get(marker) is not None for s in steps)
    grad_counts["either"] = sum(s.get("grad_norm_d_nonfinite") is not None
        or s.get("grad_norm_g_nonfinite") is not None for s in steps)
    tail_start = expected_epochs - tail_epochs + 1
    tail = [s for s in steps if isinstance(s.get("epoch"), int) and tail_start <= s["epoch"] <= expected_epochs]
    checks["tail_step_count"] = len(tail) == tail_epochs * steps_per_epoch
    medians = {key: statistics.median(s[key] for s in tail)
        if tail and all(finite(s.get(key)) for s in tail) else None for key in LOSSES}
    checks["stored_tail_medians_match"] = all(close(value, result.get("last_quarter_loss_medians", {}).get(key))
        for key, value in medians.items())
    checks["coordinator_tail_medians_match"] = all(close(value, coordinator.get("last_quarter_loss_medians", {}).get(key))
        for key, value in medians.items())
    audit = result.get("gpu_audit", {})
    checks["gpu_guard"] = (audit.get("available") is True and audit.get("valid_for_timing") is True
        and finite(audit.get("sample_count")) and audit["sample_count"] > 0
        and audit.get("foreign_clients") == [] and audit.get("errors") == [])
    protection = result.get("operating_protection", {})
    checks["operating_1003_unchanged"] = (protection.get("legacy16_count") == 16
        and protection.get("additional_count") == 987 and protection.get("changed") == [])
    before, after = result.get("protected_before", {}), result.get("protected_after", {})
    checks["research_protected_16_unchanged"] = (result.get("protected_unchanged") is True
        and isinstance(before, dict) and len(before) == 16 and before == after
        and all(sha256_string(value) for value in before.values()))
    initial = result.get("initial_state_sha256")
    checks["initial_g_d_sha_recorded"] = (isinstance(initial, list) and len(initial) == 2
        and all(sha256_string(value) for value in initial))
    return {"checks": checks, "observed_epochs": len(epochs), "observed_steps": len(steps),
        "expected_epochs": expected_epochs, "expected_steps": expected_steps,
        "nonfinite_loss_counts": bad_losses, "timing": timing, "memory": peaks,
        "amp": {"scale_reductions": reductions, "initial_scale": steps[0].get("scale_before") if steps else None,
                "final_scale": steps[-1].get("scale_after") if steps else None,
                "nonfinite_grad_step_counts": grad_counts},
        "tail": {"first_epoch": tail_start, "last_epoch": expected_epochs, "epochs": tail_epochs,
                 "steps": len(tail), "loss_medians": medians},
        "initial_state_sha256": initial, "final_state_sha256": result.get("final_state_sha256"),
        "gpu_audit": {key: audit.get(key) for key in ("available", "valid_for_timing", "sample_count",
            "foreign_clients", "errors", "method", "limitation")},
        "operating_protection": protection, "model_export": result.get("model_export"),
        "direct_dispatch_hits": result.get("direct_dispatch_hits", {}),
        "db_unchanged": result.get("db_before") == result.get("db_after"),
        "db_before": result.get("db_before"), "db_after": result.get("db_after")}


def difference(baseline, variant):
    delta = variant - baseline
    relative = delta / abs(baseline) if baseline != 0 else None
    return {"A": baseline, "variant": variant, "delta_variant_minus_A": delta,
            "relative_delta": relative, "percent_delta": relative * 100 if relative is not None else None}


def compare_runs(a, variant):
    losses = {key: difference(a["tail"]["loss_medians"][key], variant["tail"]["loss_medians"][key])
              for key in LOSSES}
    timing = {}
    for key in ("first_epoch_s", "steady_epoch_median_s", "training_epoch_sum_s", "external_whole_wall_s"):
        av, bv = a["timing"][key], variant["timing"][key]
        timing[key] = {**difference(av, bv), "speedup_A_over_variant": av / bv}
    return {"tail_loss_medians": losses, "timing": timing}


def build_summary(series_path):
    status_path = series_path.with_name(series_path.stem + "_status.json")
    summary = {"schema": "applio_v08_final_training_summary", "created_utc": datetime.now(timezone.utc).isoformat(),
        "summarizer_sha256": digest(Path(__file__)), "series": str(series_path), "status_file": str(status_path),
        "completed": False, "training_checks_pass": False, "quality_decision": "pending_user",
        "quality_note": "Training statistics do not establish perceptual equivalence; listening remains separate.",
        "expected": {"arms": ARMS, "epochs_each": 100, "steps_per_epoch": 31, "steps_each": 3100,
                     "tail_epochs": [76, 100], "tail_steps_each": 775, "protected_operating_files": 1003},
        "definitions": {"steady_epoch_median_s": "Median of epoch durations 2 through 100, including epoch 100 save work.",
            "training_epoch_sum_s": "Sum of all 100 recorded epoch durations; no extrapolation.",
            "external_whole_wall_s": "Coordinator monotonic child-process wall time, including imports, final saves and export.",
            "tail_loss_medians": "Per-loss median over all 775 steps in epochs 76 through 100, not median of epoch medians.",
            "loss_g": "Logged generator total loss; do not sum component medians.",
            "loss_d": "Logged discriminator total loss.",
            "loss_deltas": "Difference of pooled-step medians: variant minus A; relative delta divides by abs(A).",
            "protection": "Per-run recorded hashes/checks; this tool does not rescan the 1003 operating files.",
            "source_freeze": "Current frozen-file hashes plus per-run harness, dispatch and nonconv hashes where recorded."},
        "issues": [], "pending": [], "runs": {}, "comparisons": {}}
    issues, pending = summary["issues"], summary["pending"]
    plan, plan_sha, error = load_json(series_path)
    summary["series_sha256"] = plan_sha
    if error:
        summary.update(status="failed", issues=["Series plan: " + error])
        return summary
    runs = plan.get("runs", [])
    if not isinstance(runs, list) or not all(isinstance(row, dict) for row in runs):
        runs = []
    if [row.get("name") for row in runs] != list(ARMS.values()) or any(row.get("epochs") != 100 for row in runs):
        summary.update(status="failed", issues=["Plan must declare final100_A, final100_A_prime, final100_B, 100 epochs each, in order."])
        return summary
    frozen = plan.get("source_sha256", {})
    if not isinstance(frozen, dict):
        frozen = {}
    source_rows = {}
    for name in sorted(set(FROZEN_SOURCES) | set(frozen)):
        row = {"expected": frozen.get(name), "observed": None, "matches": False}
        try:
            row["observed"] = digest(owned(name))
            row["matches"] = sha256_string(row["expected"]) and row["observed"] == row["expected"]
        except (OSError, ValueError) as error:
            row["error"] = str(error)
        source_rows[name] = row
        if not row["matches"]:
            issues.append("Frozen source mismatch or missing hash: " + name)
    summary["source_freeze"] = source_rows
    state, state_sha, state_error = load_json(status_path)
    summary["status_sha256"] = state_sha
    if state_error:
        pending.append("Coordinator status unavailable: " + state_error)
        state = {}
    elif state.get("schema") != "applio_v08_epoch_series" or owned(state.get("plan", "")) != series_path:
        issues.append("Coordinator schema or plan path mismatch")
    status_runs = state.get("runs", [])
    if not isinstance(status_runs, list) or not all(isinstance(row, dict) for row in status_runs):
        issues.append("Malformed coordinator runs")
        status_runs = []
    names = [row.get("name") for row in status_runs]
    if len(names) != len(set(names)) or any(name not in ARMS.values() for name in names):
        issues.append("Duplicate or unexpected coordinator run names")
    by_name = {row.get("name"): row for row in status_runs}
    for arm, run in zip(ARMS, runs):
        name = run["name"]
        coordinator = by_name.get(name, {})
        result_path = owned(run["out"]) / "result.json"
        result, result_sha, result_error = load_json(result_path)
        record = {"name": name, "result": str(result_path), "result_sha256": result_sha,
                  "coordinator_status": coordinator.get("status", "not_started")}
        summary["runs"][arm] = record
        failed = (coordinator.get("status", "").startswith("failed")
                  or coordinator.get("exit_code") not in (None, 0) or bool((result or {}).get("error")))
        if failed:
            issues.append(f"{arm}: training/coordinator failure")
            record.update(status="failed", error=(result or {}).get("error"), read_error=result_error)
            continue
        if result_error or (result or {}).get("completed") is not True or coordinator.get("status") != "completed":
            if state.get("completed") is True or coordinator.get("status") == "completed":
                issues.append(f"{arm}: declared completed but result is missing, unreadable, or incomplete")
                record["status"] = "failed"
            else:
                pending.append(f"{arm}: result/coordinator completion pending")
                record["status"] = "waiting"
            record.update(read_error=result_error, observed_epochs=len((result or {}).get("epochs", [])),
                          observed_steps=len((result or {}).get("steps", [])))
            continue
        record.update(summarize_run(result, coordinator))
        checks = record["checks"]
        checks["harness_frozen"] = result.get("harness_sha256") == frozen.get("applio_epoch_v08.py")
        checks["expected_eager_training"] = (result.get("compile") == "" and result.get("mode") == "FAST"
            and result.get("profile_at") == 0 and result.get("batch_size") == 8
            and result.get("dataset") == "voice_model_A")
        checks["coordinator_out_matches"] = owned(coordinator.get("out", "")) == result_path.parent
        if arm == "B":
            manifest = result.get("direct_dispatch_manifest") or {}
            nonconv = result.get("nonconv") or {}
            checks["dispatch_frozen"] = manifest.get("sha256") == frozen.get(FROZEN_SOURCES[-1])
            checks["direct_dispatch_used"] = bool(record["direct_dispatch_hits"]) and all(
                isinstance(n, int) and n > 0 for n in record["direct_dispatch_hits"].values())
            checks["nonconv_frozen"] = nonconv.get("source_sha256") == frozen.get("v08_nonconv_opt.py")
            checks["selected_nonconv_settings"] = (nonconv.get("adamw_by_optimizer") == {"G": "native", "D": "single"}
                and nonconv.get("slice_gather") is True and nonconv.get("grad_norm") == "foreach"
                and nonconv.get("discriminator_weight_norm_cache") is False
                and nonconv.get("skip_empty_cache") is True and not nonconv.get("snapshot_dir"))
            record["nonconv"] = nonconv
        else:
            checks["native_control"] = result.get("direct_dispatch_manifest") is None and "nonconv" not in result
        record["status"] = "completed" if all(checks.values()) else "failed"
        issues.extend(f"{arm}: {key}" for key, passed in checks.items() if not passed)
    if state.get("completed") is not True:
        pending.append("Coordinator series is not completed")
    ready = all(row.get("status") == "completed" for row in summary["runs"].values())
    if ready:
        records = summary["runs"]
        initial = [row["initial_state_sha256"] for row in records.values()]
        summary["same_initial_g_d_sha256"] = all(value == initial[0] for value in initial[1:])
        if not summary["same_initial_g_d_sha256"]:
            issues.append("Initial G/D state hashes differ across A/A_prime/B")
        summary["comparisons"] = {arm + "_minus_A": compare_runs(records["A"], records[arm])
                                  for arm in ("A_prime", "B")}
        summary["all_arms_total"] = {key: sum(row["timing"][key] for row in records.values())
            for key in ("training_epoch_sum_s", "external_whole_wall_s")}
    summary["status"] = "failed" if issues else "waiting" if pending else "completed"
    summary["completed"] = summary["training_checks_pass"] = summary["status"] == "completed"
    summary["comparisons_valid_for_training_summary"] = summary["completed"]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, default=RESULTS / "epoch_series_100.json")
    args = parser.parse_args()
    series_path = owned(args.series)
    output = series_path.parent / "final_training_summary.json"
    try:
        summary = build_summary(series_path)
    except (OSError, ValueError, TypeError, KeyError) as error:
        summary = {"schema": "applio_v08_final_training_summary", "status": "failed", "completed": False,
            "training_checks_pass": False, "quality_decision": "pending_user",
            "issues": [f"Invalid input: {type(error).__name__}: {error}"], "series": str(series_path)}
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "status": summary["status"],
        "completed": summary["completed"], "quality_decision": summary["quality_decision"],
        "issues": summary.get("issues", []), "pending": summary.get("pending", [])}, ensure_ascii=False))
    return {"completed": 0, "waiting": 2, "failed": 1}[summary["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
