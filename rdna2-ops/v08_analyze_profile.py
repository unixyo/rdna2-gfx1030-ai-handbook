#!/usr/bin/env python3
"""Analyze v0.8 PyTorch/ROCm profile JSON using only Python standard libraries.

No torch, numpy, ROCm, subprocess, or GPU-device API is imported or called.
CPU operator self-device totals, device kernels, transfers, and annotation
ranges are separate accounting systems. They must never be summed together.
"""
from __future__ import annotations
import argparse
import ast
import bisect
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
CONV_PREFIX = "V08_CONV_"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    with path.open() as f:
        return json.load(f)


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def union_us(intervals):
    return sum(b - a for a, b in merge_intervals(intervals))


def gaps_in(window, merged):
    a, b = window
    gaps = []
    cursor = a
    for start, end in merged:
        if end <= a or start >= b:
            continue
        start, end = max(a, start), min(b, end)
        if start > cursor:
            gaps.append([cursor, start])
        cursor = max(cursor, end)
    if cursor < b:
        gaps.append([cursor, b])
    return gaps


def aggregate_rows(rows):
    grouped = {}
    for row in rows:
        x = grouped.setdefault(row["name"], dict(name=row["name"], count=0,
            self_gpu_us=0.0, gpu_us=0.0, self_cpu_us=0.0, cpu_us=0.0))
        for key in ("count", "self_gpu_us", "gpu_us", "self_cpu_us", "cpu_us"):
            x[key] += row.get(key, 0)
    return grouped


def rank_rows(rows, value, denominator, limit=10):
    out = []
    for row in sorted(rows, key=lambda r: r.get(value, 0), reverse=True)[:limit]:
        out.append(dict(row, share_percent=100 * row.get(value, 0) / denominator if denominator else 0))
    return out


def tensor_shapes(event):
    return event.get("args", {}).get("Input Dims", [])


def concrete(event, index):
    values = event.get("args", {}).get("Concrete Inputs", [])
    if index >= len(values):
        return None
    value = values[index]
    if not isinstance(value, str) or not value:
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def backward_descriptor(event):
    shapes = tensor_shapes(event)
    return dict(direction="backward_combined_dX_dW", op=event["name"],
                grad_output=shapes[0] if len(shapes) > 0 else [],
                input=shapes[1] if len(shapes) > 1 else [],
                weight=shapes[2] if len(shapes) > 2 else [],
                stride=concrete(event, 4), padding=concrete(event, 5),
                dilation=concrete(event, 6), transposed=concrete(event, 7),
                output_padding=concrete(event, 8), groups=concrete(event, 9),
                output_mask=concrete(event, 10),
                input_types=event.get("args", {}).get("Input type", [])[:3])


def build_contexts(events):
    """Link a GPU External id to its CPU launch owner and enclosing conv/Adam."""
    per_thread = defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and event.get("cat") in {"cpu_op", "user_annotation", "cuda_runtime"}:
            per_thread[(event.get("pid"), event.get("tid"))].append(event)
    by_external = {}
    backward_specs = {}
    forward_counts = Counter()
    host_wait_owners = defaultdict(lambda: dict(count=0, cpu_us=0.0))
    for thread, cpu in per_thread.items():
        stack = []
        for e in sorted(cpu, key=lambda x: (x["ts"], -x.get("dur", 0))):
            end = e["ts"] + e.get("dur", 0)
            # Events on the same thread are nested. Cross-boundary ranges are not
            # accepted as parents. A 0.1 us tolerance covers JSON timestamp rounding.
            while stack and (e["ts"] >= stack[-1][0] - 0.0001 or end > stack[-1][0] + 0.1):
                stack.pop()
            parent = stack[-1][2] if stack else (None, None, None, None)
            desc, back, opt, miopen = parent
            name = e["name"]
            if name.startswith(CONV_PREFIX):
                desc = name[len(CONV_PREFIX):]
                forward_counts[desc] += 1
            if name == "aten::convolution_backward":
                spec = backward_descriptor(e)
                key = json.dumps(spec, sort_keys=True, separators=(",", ":"))
                back = hashlib.sha256(key.encode()).hexdigest()[:12]
                backward_specs[back] = spec
            if name.startswith("Optimizer.step#"):
                opt = name
            if name in ("aten::miopen_convolution", "aten::miopen_convolution_transpose"):
                miopen = name
            context = (desc, back, opt, miopen)
            ext = e.get("args", {}).get("External id")
            if ext is not None:
                by_external[ext] = (context, name)
            if e.get("cat") == "cuda_runtime" and re.search(r"Synchronize|Memcpy|StreamWait", name):
                parents = [frame[1] for frame in stack if frame[3] != "cuda_runtime"][-3:]
                key = (name, tuple(parents))
                host_wait_owners[key]["count"] += 1
                host_wait_owners[key]["cpu_us"] += e.get("dur", 0)
            stack.append((end, name, context, e.get("cat")))
    return by_external, backward_specs, forward_counts, host_wait_owners


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--diagnostic-only", action="store_true")
    p.add_argument("--diagnostic-reason", default="")
    args = p.parse_args()
    start = time.perf_counter()
    research = Path(__file__).resolve().parent
    run = args.run.resolve()
    out = (args.out or run / "profile_summary.json").resolve()
    if not run.is_relative_to(research) or not out.is_relative_to(research):
        p.error("Analysis input/output must be inside rdna2-ops")
    if out.exists():
        p.error("Choose a new output; preserving earlier analysis")
    if args.steps < 1:
        p.error("Positive profiled step count required")
    paths = {key: run / name for key, name in {
        "rows": "profile_rows.json", "trace": "profile_trace.json", "result": "result.json"
    }.items()}
    rows = load_json(paths["rows"])
    result = load_json(paths["result"])
    trace = load_json(paths["trace"])
    events = trace["traceEvents"]
    complete = [e for e in events if e.get("ph") == "X" and "ts" in e and "dur" in e]
    cpu_op_names = {e["name"] for e in complete if e.get("cat") == "cpu_op"}
    cpu_all = aggregate_rows(r for r in rows if r["device_type"] == "DeviceType.CPU")
    cpu_ops = {k: v for k, v in cpu_all.items() if k in cpu_op_names}
    cpu_self_gpu_total = sum(x["self_gpu_us"] for x in cpu_ops.values())
    cpu_all_self_gpu_total = sum(x["self_gpu_us"] for x in cpu_all.values())
    runtime = aggregate_rows(r for r in rows if r["device_type"] == "DeviceType.CPU"
                             and (r["name"].startswith("hip") or r["name"].startswith("cuda")))
    kernels = [e for e in complete if e.get("cat") == "kernel"]
    gpu_events = [e for e in complete if e.get("cat") in GPU_CATEGORIES]
    gpu_annotations = [e for e in complete if e.get("cat") == "gpu_user_annotation"]
    cpu_annotations = [e for e in complete if e.get("cat") == "user_annotation"]
    spans = [e for e in complete if e.get("cat") == "Trace"]
    if len(spans) == 1:
        window = (spans[0]["ts"], spans[0]["ts"] + spans[0]["dur"])
        window_basis = "Trace/PyTorch Profiler X range"
    else:
        measured = [e for e in complete if e.get("cat") != "gpu_user_annotation"]
        window = (min(e["ts"] for e in measured), max(e["ts"] + e["dur"] for e in measured))
        window_basis = "min/max of recorded CPU/runtime/GPU complete events"
    wall_us = window[1] - window[0]
    intervals = [(e["ts"], e["ts"] + e["dur"]) for e in gpu_events]
    merged = merge_intervals(intervals)
    idle = gaps_in(window, merged)
    kernel_sum = sum(e["dur"] for e in kernels)
    kernel_by_name = {}
    for e in kernels:
        row = kernel_by_name.setdefault(e["name"], dict(name=e["name"], calls=0, gpu_us=0.0))
        row["calls"] += 1
        row["gpu_us"] += e["dur"]
    registry = []
    kernel_ids = {}
    for i, name in enumerate(sorted(kernel_by_name)):
        kid = f"k{i:04d}"
        kernel_ids[name] = kid
        registry.append(dict(id=kid, **kernel_by_name[name]))

    contexts, backward_specs, forward_counts, host_owners = build_contexts(events)
    forward_usage = defaultdict(lambda: defaultdict(lambda: dict(calls=0, gpu_us=0.0, miopen_gpu_us=0.0)))
    backward_usage = defaultdict(lambda: defaultdict(lambda: dict(calls=0, gpu_us=0.0)))
    optimizer_kernel = defaultdict(lambda: dict(calls=0, gpu_us=0.0))
    unmatched = Counter()
    for e in kernels:
        context = contexts.get(e.get("args", {}).get("External id"))
        if context is None:
            unmatched[e["name"]] += 1
            continue
        (desc, back, opt, miopen), owner = context
        kid = kernel_ids[e["name"]]
        if desc:
            cell = forward_usage[desc][kid]
            cell["calls"] += 1; cell["gpu_us"] += e["dur"]
            if miopen:
                cell["miopen_gpu_us"] += e["dur"]
        if back:
            cell = backward_usage[back][kid]
            cell["calls"] += 1; cell["gpu_us"] += e["dur"]
        if opt:
            optimizer_kernel[kid]["calls"] += 1
            optimizer_kernel[kid]["gpu_us"] += e["dur"]

    forward_rows = []
    for desc, counts in forward_usage.items():
        usage = sorted([dict(kernel_id=kid, **x) for kid, x in counts.items()],
                       key=lambda x: x["gpu_us"], reverse=True)
        total = sum(x["gpu_us"] for x in usage)
        conv = sum(x["miopen_gpu_us"] for x in usage)
        im2col = sum(x["gpu_us"] for x in usage if re.search(r"im2.*col", registry[int(x["kernel_id"][1:])]["name"], re.I))
        forward_rows.append(dict(id=desc, descriptor=result.get("conv_calls", {}).get(desc),
            profile_calls=forward_counts[desc], all_forward_kernel_us=total,
            miopen_forward_kernel_us=conv, im2col_us=im2col,
            all_forward_us_per_call=total / forward_counts[desc] if forward_counts[desc] else None,
            kernel_usage=usage))
    forward_rows.sort(key=lambda x: x["miopen_forward_kernel_us"], reverse=True)
    backward_rows = []
    for key, counts in backward_usage.items():
        usage = sorted([dict(kernel_id=kid, **x) for kid, x in counts.items()],
                       key=lambda x: x["gpu_us"], reverse=True)
        backward_rows.append(dict(id=key, descriptor=backward_specs[key],
                                  gpu_us=sum(x["gpu_us"] for x in usage), kernel_usage=usage))
    backward_rows.sort(key=lambda x: x["gpu_us"], reverse=True)

    scalar_names = {"aten::item", "aten::_local_scalar_dense", "aten::is_nonzero"}
    scalars = [v for k, v in cpu_all.items() if k in scalar_names]
    sync_runtime = [v for k, v in runtime.items() if re.search(r"Synchronize|Memcpy|StreamWait", k)]
    owner_rows = [dict(runtime=name, ancestors=list(parents), **v)
                  for (name, parents), v in host_owners.items()]
    owner_rows.sort(key=lambda x: x["cpu_us"], reverse=True)
    cpu_sync_events = [e for e in complete if e.get("cat") == "cuda_runtime"
                       and re.search(r"Synchronize", e["name"])]
    activity_by_device = {}
    for device in {e.get("args", {}).get("device", e.get("pid")) for e in gpu_events}:
        selected = [e for e in gpu_events if e.get("args", {}).get("device", e.get("pid")) == device]
        activity_by_device[str(device)] = dict(events=len(selected), union_us=union_us(
            (e["ts"], e["ts"] + e["dur"]) for e in selected))
    largest_idle = []
    for a, b in sorted(idle, key=lambda x: x[1] - x[0], reverse=True)[:20]:
        overlaps = []
        for e in complete:
            if e.get("cat") not in {"cpu_op", "cuda_runtime", "user_annotation"}:
                continue
            duration = max(0, min(b, e["ts"] + e["dur"]) - max(a, e["ts"]))
            if duration:
                overlaps.append(dict(name=e["name"], cat=e["cat"], overlap_us=duration,
                                     event_duration_us=e["dur"]))
        overlaps.sort(key=lambda x: x["overlap_us"], reverse=True)
        largest_idle.append(dict(start_relative_us=a - window[0], duration_us=b - a,
                                 overlapping_cpu_ranges=overlaps[:5]))
    histogram = []
    for lo, hi in ((0, 5), (5, 10), (10, 50), (50, 100), (100, 1000), (1000, float("inf"))):
        durations = [b - a for a, b in idle if lo < b - a <= hi]
        histogram.append(dict(lower_us=lo, upper_us=None if hi == float("inf") else hi,
                              count=len(durations), total_us=sum(durations)))
    category_durations = {
        cat: dict(count=sum(e.get("cat") == cat for e in complete),
                  sum_us=sum(e["dur"] for e in complete if e.get("cat") == cat))
        for cat in ("kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation")
    }
    dl = [e for e in cpu_annotations if e["name"].startswith("enumerate(DataLoader)")]
    profile_epoch = sorted({r["epoch"] for r in result.get("steps", [])
                            if result.get("profile_at", 0) <= r.get("step", -1) <
                            result.get("profile_at", 0) + args.steps})
    total_gpu_union = sum(b - a for a, b in merged)
    summary = dict(
        schema="applio_profile_summary_v08", analysis_seconds=time.perf_counter() - start,
        source_files={k: dict(path=str(v), bytes=v.stat().st_size, sha256=sha256(v)) for k, v in paths.items()},
        parser_sha256=sha256(Path(__file__)),
        diagnostic_only=args.diagnostic_only, diagnostic_reason=args.diagnostic_reason,
        valid_for_catalog_shares=bool(result.get("valid_for_timing") and not args.diagnostic_only),
        gpu_guard_valid=bool(result.get("gpu_audit", {}).get("valid_for_timing")),
        protected_unchanged=result.get("protected_unchanged"),
        operating_protection=result.get("operating_protection"),
        effective_cudnn_benchmark=result.get("effective_cudnn_benchmark"),
        dataset=dict(name=result.get("dataset"), filelist_rows=result.get("dataset_rows"),
                     sampler_samples=result.get("dataset_sampler_samples"),
                     steps_per_epoch=result.get("steps_per_epoch"), batch_size=result.get("batch_size")),
        profile=dict(start_step=result.get("profile_at"), steps=args.steps, dataloader_ranges=len(dl),
                     epoch=profile_epoch, window_us=wall_us, window_basis=window_basis,
                     profiled_wall_us_per_step=wall_us / args.steps,
                     epoch_times_are_speed_baseline=False,
                     epoch_time_warning="Profiler collection/export is inside the epoch wall timer. Do not use profile epoch seconds or steady_epoch_median_s for speedup.",
                     trace_gpu_device_ids=activity_by_device,
                     device_id_warning="ROCm trace device IDs are retained verbatim; do not index deviceProperties with them to infer a different GPU."),
        accounting=dict(
            cpu_operator_self_gpu_us=cpu_self_gpu_total,
            all_cpu_rows_self_gpu_us=cpu_all_self_gpu_total,
            kernel_sum_us=kernel_sum,
            kernel_union_us=union_us((e["ts"], e["ts"] + e["dur"]) for e in kernels),
            gpu_activity_union_us=total_gpu_union,
            no_recorded_gpu_activity_us=sum(b-a for a,b in idle),
            gpu_activity_coverage_percent=100 * total_gpu_union / wall_us,
            kernel_share_of_profile_wall_percent=100 * kernel_sum / wall_us,
            categories=category_durations,
            cpu_vs_gpu_row_rule="CPU self-device attribution and device kernel durations describe the same work; never add them.",
            annotation_rule="GPU/CPU annotation spans overlap their kernels and include gaps. Excluded from kernel/union totals.",
            idle_rule="Idle means no recorded kernel/copy/memset in this trace. It is not a hardware-utilization measurement or proof of a particular CPU bottleneck.",
            cpu_vs_kernel_residual_us=cpu_self_gpu_total-kernel_sum),
        cpu_operator_top10=rank_rows(cpu_ops.values(),"self_gpu_us",cpu_self_gpu_total),
        cpu_operator_top40=rank_rows(cpu_ops.values(),"self_gpu_us",cpu_self_gpu_total,40),
        device_kernel_top10=rank_rows(kernel_by_name.values(),"gpu_us",kernel_sum),
        kernel_registry=registry,
        host_sync_top10=rank_rows(sync_runtime,"cpu_us",wall_us),
        host_scalar_rows=scalars,
        host_sync_owner_top20=owner_rows[:20],
        host_synchronize_interval_union_us=union_us((e["ts"],e["ts"]+e["dur"]) for e in cpu_sync_events),
        host_sync_rule="Runtime sync CPU time includes waiting for GPU work, and scalar ops contain runtime calls. These rows must not be summed with one another or treated as removable idle.",
        dataloader=dict(count=len(dl), sum_cpu_us=sum(e["dur"] for e in dl),
                        max_cpu_us=max((e["dur"] for e in dl),default=0)),
        idle_gap_histogram=histogram, largest_idle_gaps=largest_idle,
        forward_mapping=dict(method="GPU kernel External id -> enclosing CPU V08_CONV annotation",
                             descriptors=len(forward_rows), annotation_calls=sum(forward_counts.values()),
                             gpu_kernels_without_cpu_external_id=sum(unmatched.values()),
                             unmatched_kernel_names=dict(unmatched),
                             note="Module input/weight dtype metadata can precede autocast; MIOpen call kernels are separated from normalization/casts inside the module range."),
        forward_shapes=forward_rows,
        backward_shapes=backward_rows,
        backward_rule="Backward rows combine dX and dW. Kernel name alone is not used to claim a separate direction or solver ID.",
        adamw_kernel_usage=[dict(kernel_id=k,**v) for k,v in sorted(optimizer_kernel.items(),key=lambda kv:kv[1]["gpu_us"],reverse=True)],
        adamw_kernel_us=sum(v["gpu_us"] for v in optimizer_kernel.values()),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps(dict(out=str(out), diagnostic_only=args.diagnostic_only,
        profile_wall_s=wall_us / 1e6, kernel_sum_s=kernel_sum / 1e6,
        gpu_activity_union_s=total_gpu_union / 1e6,
        no_recorded_gpu_activity_s=sum(b-a for a,b in idle)/1e6,
        cpu_operator_self_gpu_s=cpu_self_gpu_total / 1e6,
        forward_shapes=len(forward_rows), backward_shapes=len(backward_rows),
        unmatched_kernel_count=sum(unmatched.values())), indent=2))


if __name__ == "__main__":
    main()
