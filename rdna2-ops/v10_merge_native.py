#!/usr/bin/env python3
"""CPU-only merge of completed v10 remaining-convolution measurements.

Requires disjoint cases, complete expected coverage, one source/benchmark/gate,
valid GPU audits, and matching raw operator + dtype/weight_norm contracts.
No torch/ROCm import and no GPU execution. Existing output directories are
never overwritten. The merged dispatch remains an epoch/audio candidate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import bench_v10_native_conv as bench
from v08_solver_select import case_id, db_key

HERE = Path(__file__).resolve().parent
V10 = HERE / "results/v10"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric_pass(metric, gate):
    return (metric.get("screen_pass") is True and metric.get("finite") is True and
            metric.get("reference_finite") is True and
            metric.get("nrmse") is not None and metric["nrmse"] <= gate["nrmse"] and
            metric.get("max_over_reference_rms") is not None and
            metric["max_over_reference_rms"] <= gate["max_over_reference_rms"])


def contract_pass(row, gate):
    contract = row.get("contract", {})
    if contract.get("strict_pass") is not True or not contract.get("variants"):
        return False
    observed = {v["input_dtype"] for v in row["case"].get("logical_boundary_variants", [])} or {"torch.float16"}
    if observed != {v["input_dtype"] for v in contract["variants"]}:
        return False
    names = {"outputs_fp16", "native_input_gradient_preserved", "candidate_input_gradient_preserved",
             "native_masters_fp32", "candidate_masters_fp32", "native_master_gradients_fp32",
             "candidate_master_gradients_fp32"}
    return all(v.get("strict_pass") is True and set(v.get("dtype_checks", {})) == names and
               all(v["dtype_checks"].values()) and v.get("metrics") and
               all(numeric_pass(m, gate) for m in v["metrics"].values()) for v in contract["variants"])


def classify_native(data):
    kernels = data.get("device_activities", {}).get("rows", [])
    names = [str(k["name"]) for k in kernels if k.get("category") == "kernel"]
    if not names:
        return "unattributed", names
    lower = " ".join(names).lower()
    if any(token in lower for token in ("winograd", "miopensp3asmconv", "f2x3_stride", "f3x2_stride")):
        return "winograd", names
    if "fft" in lower:
        return "fft", names
    if any(token in lower for token in ("cijk", "im2", "col2", "gemm", "convolution")):
        return "direct_or_gemm", names
    return "unattributed", names


def measured_roof(roof, event_ms, *, interpret_direct=True, scratch_bytes=0):
    if event_ms is None or event_ms <= 0:
        return None
    # This is a conditional capacity model, not a DRAM-counter measurement or
    # proof that all time above it can be removed. Inputs may already be cached.
    result = dict(compulsory_tensor_rate_gbs=roof["fp16_compulsory_read_write_bytes"] / (event_ms * 1e6),
                  conditional_dram_only_floor_ms=roof["optimistic_memory_ms"],
                  scratch_allocation_bytes=scratch_bytes,
                  nominal_scratch_write_read_bytes=2 * scratch_bytes)
    if interpret_direct:
        result.update(effective_useful_tflops=roof["valid_nonpadding_flops"] / (event_ms * 1e9),
                      effective_useful_fraction_of_dot2_cap=roof["optimistic_direct_compute_ms"] / event_ms,
                      conditional_roof_fraction=roof["optimistic_lower_ms"] / event_ms)
    else:
        result["compute_model_excluded"] = "Winograd/FFT/unattributed arithmetic must not be divided by direct-Conv FLOPs"
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=V10 / "native_selected")
    p.add_argument("--expected-cases", type=Path, default=bench.V08 / "conv_cases.json")
    p.add_argument("--baseline-dispatch", type=Path, default=bench.V08 / "direct_selected_final/dispatch.json")
    args = p.parse_args()
    out = args.out.resolve()
    require(out.is_relative_to(V10), "Output must be under results/v10")
    require(not out.exists(), "Use a fresh output directory")
    kernel_sha = sha(HERE / "v10_native_conv.py")
    benchmark_sha = sha(HERE / "bench_v10_native_conv.py")
    baseline_sha = sha(HERE / "v08_direct_conv.py")
    expected_args = argparse.Namespace(cases_json=args.expected_cases, baseline_dispatch=args.baseline_dispatch,
                                      family="all", periods="all", case=[], exclude_case=[])
    expected_cases = bench.selected_cases(expected_args)
    expected = {(case_id(c), d) for c in expected_cases for d in bench.directions_for(c, "auto")}
    seen_cases, seen_directions = set(), set()
    reports, entries, directions = [], {}, []
    gate = None
    runtime = None
    counts = dict(cases=0, directions=0, candidates=0, candidate_operator_pass=0,
                  candidate_execution_error=0, native_operator_fail=0, selected_new_directions=0,
                  selected_entries=0, contract_variants=0, contract_metrics=0)
    for input_path in args.results:
        result_path = input_path.resolve()
        require(result_path.is_relative_to(V10), "Inputs must be v10 research results")
        data = json.loads(result_path.read_text())
        manifest_path = result_path.with_name("dispatch.json")
        manifest = json.loads(manifest_path.read_text())
        policy_path = Path(data["native_policy"])
        policy = json.loads(policy_path.read_text())
        require(data.get("completed") is True and data.get("valid_for_timing") is True,
                "Incomplete or invalid timing: " + str(result_path))
        audit = data.get("gpu_audit", {})
        require(audit.get("available") is True and audit.get("valid_for_timing") is True and
                not audit.get("foreign_clients") and not audit.get("errors"), "GPU audit failed")
        require(data.get("source_sha256") == kernel_sha == manifest.get("source_sha256"), "Kernel SHA mismatch")
        require(data.get("benchmark_sha256") == benchmark_sha, "Benchmark SHA mismatch")
        require(manifest.get("baseline_v08_source_sha256") == baseline_sha, "Baseline v08 source SHA mismatch")
        require(manifest.get("schema") == "applio_v10_native_dispatch" and manifest.get("strict_gate_pass") is True,
                "Dispatch schema/gate mismatch")
        require(data.get("baseline_dispatch_sha256") == sha(args.baseline_dispatch), "Baseline dispatch mismatch")
        current_runtime = {k:data.get(k) for k in ("torch", "hip", "triton", "device")}
        if gate is None:
            gate, runtime = data["gate"], current_runtime
        require(gate == data["gate"] == manifest["gate"] and runtime == current_runtime, "Gate/runtime mismatch")
        require(data.get("tunableop_enabled") is False and data.get("tunableop_tuning") is False,
                "TunableOp state is not fixed off")
        reports.append(dict(path=str(result_path), sha256=sha(result_path),
                            dispatch_path=str(manifest_path), dispatch_sha256=sha(manifest_path),
                            native_policy_path=str(policy_path), native_policy_sha256=sha(policy_path)))
        raw_by_key = {}
        for row in data["cases"]:
            case = row["case"]
            key = case_id(case)
            require(key not in seen_cases, "Duplicate case across reports: " + case["name"])
            seen_cases.add(key)
            raw_by_key[key] = row
            roof = bench.roofline(case)
            for field in ("direct_algorithm_nominal_flops", "valid_nonpadding_flops", "fp16_compulsory_read_write_bytes",
                          "optimistic_direct_compute_ms", "optimistic_memory_ms", "optimistic_lower_ms"):
                require(math.isclose(roof[field], row["roofline"][field], rel_tol=1e-12, abs_tol=1e-12),
                        "Roofline arithmetic mismatch: " + case["name"] + "/" + field)
            for direction, values in row["directions"].items():
                pair = (key, direction)
                require(pair not in seen_directions and pair in expected, "Duplicate/unexpected direction")
                seen_directions.add(pair)
                native = values["native"]
                native_pass = numeric_pass(native["error"], gate)
                counts["native_operator_fail"] += int(not native_pass)
                ne = min(native["event_median_ms"], values["native_recheck"]["event_median_ms"])
                nw = min(native["wall_median_ms"], values["native_recheck"]["wall_median_ms"])
                require(math.isclose(ne, values["selection_baseline"]["event_ms"], rel_tol=1e-12), "Native event comparator mismatch")
                require(math.isclose(nw, values["selection_baseline"]["wall_ms"], rel_tol=1e-12), "Native wall comparator mismatch")
                algorithm, native_names = classify_native(native)
                valid = [c for c in values["candidates"] if numeric_pass(c.get("error", {}), gate) and c.get("event_median_ms")]
                best = min(valid, key=lambda c:c["event_median_ms"]) if valid else None
                chosen_index = values.get("selected_candidate")
                chosen = values["candidates"][chosen_index] if chosen_index is not None else None
                if chosen is not None:
                    require(native_pass and numeric_pass(chosen["error"], gate), "Chosen operator failed accuracy")
                    require(ne / chosen["event_median_ms"] >= gate["candidate_event_speedup_min"] and
                            nw / chosen["wall_median_ms"] >= gate["candidate_wall_speedup_min"], "Chosen operator failed timing")
                counts["candidates"] += len(values["candidates"])
                counts["candidate_operator_pass"] += sum(numeric_pass(c.get("error", {}), gate) for c in values["candidates"])
                counts["candidate_execution_error"] += sum("execution_error" in c for c in values["candidates"])
                entry = manifest["entries"].get(key)
                accepted = bool(entry is not None and direction in entry.get("new_directions", {}))
                if accepted:
                    require(chosen is not None and entry["new_directions"][direction] == chosen["config"], "Dispatch/raw choice mismatch")
                directions.append(dict(case_name=case["name"], case_key=key, family=case["v10_family"],
                    input_shape=case["shape"], weight_shape=case["weight"], direction=direction,
                    logical_input_dtypes=sorted({v["input_dtype"] for v in case.get("logical_boundary_variants", [])}),
                    native_event_initial_ms=native["event_median_ms"], native_event_recheck_ms=values["native_recheck"]["event_median_ms"],
                    native_event_ms=ne, native_wall_ms=nw, native_error=native["error"],
                    native_kernel_family=algorithm, native_kernel_names=native_names,
                    native_policy_choice=policy.get("directions", {}).get(db_key(case, direction)),
                    best_accurate_candidate_event_ms=best["event_median_ms"] if best else None,
                    best_accurate_candidate_speedup=ne / best["event_median_ms"] if best else None,
                    best_accurate_candidate_config=best["config"] if best else None,
                    operator_selected=chosen is not None, dispatch_selected=accepted,
                    selection_reason="operator_and_contract_pass" if accepted else
                        "contract_failed_or_unmeasured" if chosen is not None else "accuracy_or_predeclared_speed_gate_not_met",
                    selected_event_ms=chosen["event_median_ms"] if accepted else None,
                    selected_speedup=ne / chosen["event_median_ms"] if accepted else None,
                    selected_config=chosen["config"] if accepted else None,
                    selected_error=chosen["error"] if accepted else None,
                    selected_metadata=chosen.get("metadata") if accepted else None,
                    roofline=roof,
                    native_capacity_model=measured_roof(roof, ne, interpret_direct=algorithm == "direct_or_gemm"),
                    selected_capacity_model=measured_roof(roof, chosen["event_median_ms"],
                        scratch_bytes=chosen.get("metadata", {}).get("scratch_bytes", 0)) if accepted else None,
                    raw_result=str(result_path), raw_result_sha256=sha(result_path)))
        require(len(manifest["entries"]) == data["selected_entries"], "Manifest entry count mismatch")
        require(sum(len(e["new_directions"]) for e in manifest["entries"].values()) == data["selected_new_directions"], "Manifest direction count mismatch")
        for key, entry in manifest["entries"].items():
            require(key in raw_by_key and key == case_id(entry["case"]), "Dispatch case missing from raw data")
            row = raw_by_key[key]
            require(entry.get("strict_gate_pass") is True and entry.get("contract_pass") is True and contract_pass(row, gate),
                    "Dispatch AMP/weight_norm/dtype contract failed")
            require(entry.get("new_directions") and set(entry["new_directions"]).issubset(entry["directions"]), "Missing new directions")
            require(key not in entries, "Duplicate dispatch entry")
            entries[key] = entry
            counts["contract_variants"] += len(row["contract"]["variants"])
            counts["contract_metrics"] += sum(len(v["metrics"]) for v in row["contract"]["variants"])
    require(seen_directions == expected, "Expected coverage missing: " + str(len(expected - seen_directions)))
    require("torch" not in sys.modules, "CPU merge imported torch")
    counts.update(cases=len(seen_cases), directions=len(seen_directions), selected_entries=len(entries),
                  selected_new_directions=sum(len(e["new_directions"]) for e in entries.values()))
    counts["preserved_v08_direct_directions"] = sum(len(e["directions"]) - len(e["new_directions"]) for e in entries.values())
    family_counts = {}
    for row in directions:
        rec = family_counts.setdefault(row["family"], dict(directions=0, selected=0))
        rec["directions"] += 1
        rec["selected"] += int(row["dispatch_selected"])
    summary = dict(schema="applio_v10_native_merged_summary", completed=True, coverage_complete=True,
        source_sha256=kernel_sha, benchmark_sha256=benchmark_sha, merge_source_sha256=sha(__file__),
        runtime=runtime, gate=gate, counts=counts, families=family_counts, source_reports=reports,
        native_baseline="fresh fastest historical-accurate policy; each comparator is the faster of initial/recheck event and wall separately",
        roofline_review=dict(arithmetic_recomputed_all_cases=True,
            compute_capacity="48.7 TFLOP/s measured dot2 calibration, useful direct-Conv mathematical FLOPs",
            memory_capacity="436.7 GB/s measured DRAM calibration applied conditionally to compulsory tensor bytes",
            caveats=["No DRAM hardware counters were recorded. Warm/cached tensor reads need not cross DRAM; this is not a strict physical time floor.",
                     "Direct useful FLOPs exclude masked/tile-padding work, address instructions, launch latency, and register traffic.",
                     "Winograd, FFT, and unattributed native kernels have no direct-FLOP compute-utilization estimate.",
                     "Time above a conditional roofline is not proof that the whole difference can be removed.",
                     "A microbenchmark win is not an operating epoch or perceptual adoption result."]),
        status="operator and AMP/weight_norm validated candidate; operating B/B-prime and audio are separate")
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "dispatch.json", dict(schema="applio_v10_native_dispatch", strict_gate_pass=True,
        source_sha256=kernel_sha, baseline_v08_source_sha256=baseline_sha, entries=entries,
        gate=gate, source_reports=reports, status=summary["status"], summary_path=str(out / "summary.json")))
    dump(out / "summary.json", summary)
    dump(out / "direction_summary.json", directions)
    fields = ["case_name", "family", "direction", "logical_input_dtypes", "native_kernel_family",
              "native_event_ms", "native_wall_ms", "best_accurate_candidate_event_ms",
              "best_accurate_candidate_speedup", "operator_selected", "dispatch_selected",
              "selected_event_ms", "selected_speedup", "selection_reason", "raw_result"]
    with (out / "direction_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in directions:
            writer.writerow({k:json.dumps(row[k], ensure_ascii=False) if isinstance(row[k], (list, dict)) else row[k] for k in fields})
    print(json.dumps(dict(out=str(out), counts=counts, families=family_counts,
                          dispatch_sha256=sha(out / "dispatch.json")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
