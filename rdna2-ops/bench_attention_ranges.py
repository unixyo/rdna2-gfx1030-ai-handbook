#!/usr/bin/env python3
"""Measure exact FP32 attention ranges and forced native SDPA backends.

Use the existing ROCm environment. Inputs are seeded random tensors with the
model's BSHD-to-BHSD layout, not captured activations. --packed-v additionally
models V as the third view of a packed QKV projection. No graph timing is used.
Backend probes include FP32-to-target QKV casts and output-to-FP32 conversion
inside each timed call. Image tolerances are recorded, not evaluated here.
"""
import argparse
import datetime as dt
import json
from pathlib import Path
import statistics
import time
import warnings

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from rdna2_ops.attention import scaled_dot_product_attention


LENGTHS = [128, 256, 512, 768, 1024, 1061, 1536, 2048, 3072, 4096]


def inputs(length, packed_v=False):
    q, k = [torch.randn(1, length, 24, 128, device="cuda", dtype=torch.float32).transpose(1, 2)
            for _ in range(2)]
    if packed_v:
        v = torch.randn(1, length, 3, 24, 128, device="cuda", dtype=torch.float32)[:, :, 2].transpose(1, 2)
    else:
        v = torch.randn(1, length, 24, 128, device="cuda", dtype=torch.float32).transpose(1, 2)
    return q, k, v


def layout(tensors):
    return {name: {"shape": list(t.shape), "stride": list(t.stride()),
                   "storage_offset": t.storage_offset(), "dtype": str(t.dtype)}
            for name, t in zip(("q", "k", "v"), tensors)}


def errors(actual, expected):
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    same = (actual.shape == expected.shape and actual.dtype == expected.dtype
            and torch.equal(actual.contiguous().view(torch.uint8),
                            expected.contiguous().view(torch.uint8)))
    delta = actual.double() - expected.double()
    return {"finite": finite, "byte_equal": same,
            "max_abs": delta.abs().max().item() if finite else None,
            "rmse": delta.square().mean().sqrt().item() if finite else None}


def timed_call(fn, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    begin = time.perf_counter()
    start.record()
    for _ in range(iterations):
        output = fn()
    end.record()
    end.synchronize()
    return {"wall_ms": (time.perf_counter() - begin) * 1000 / iterations,
            "event_ms": start.elapsed_time(end) / iterations}


def paired_timings(native, candidate, cfg):
    functions = {"native_fp32": native, "candidate": candidate}
    for _ in range(cfg.warmup):
        for fn in functions.values():
            fn()
    torch.cuda.synchronize()
    samples = {name: [] for name in functions}
    for trial in range(cfg.rounds):
        order = list(functions) if trial % 2 == 0 else list(reversed(functions))
        for name in order:
            samples[name].append(timed_call(functions[name], cfg.iterations))
    medians = {name: {metric: statistics.median(s[metric] for s in values)
                      for metric in ("wall_ms", "event_ms")}
               for name, values in samples.items()}
    return {"samples": samples, "medians": medians,
            "speedup_wall": medians["native_fp32"]["wall_ms"] / medians["candidate"]["wall_ms"],
            "speedup_event": medians["native_fp32"]["event_ms"] / medians["candidate"]["event_ms"]}


def exact_range(length, cfg):
    tensors = inputs(length, cfg.packed_v)
    row = {"length": length, "layout": layout(tensors), "status": "started"}
    try:
        native = lambda: F.scaled_dot_product_attention(*tensors)
        candidate = lambda: scaled_dot_product_attention(*tensors, backend="rdna2")
        row["accuracy"] = errors(candidate(), native())
        if not row["accuracy"]["finite"] or not row["accuracy"]["byte_equal"]:
            row["status"] = "accuracy_failed"
            return row
        row.update(paired_timings(native, candidate, cfg))
        row["status"] = "ok"
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return row


def capability(backend_name, tensors):
    queries = {"FLASH_ATTENTION": "can_use_flash_attention",
               "EFFICIENT_ATTENTION": "can_use_efficient_attention",
               "CUDNN_ATTENTION": "can_use_cudnn_attention"}
    query = queries.get(backend_name)
    if query is None:
        return {"status": "no_capability_query"}
    try:
        params = torch.backends.cuda.SDPAParams(*tensors, None, 0.0, False, False)
        return {"query": query, "can_use": bool(getattr(torch.backends.cuda, query)(params, debug=True)),
                "note": "A capability result is not proof of successful kernel execution."}
    except Exception as exc:
        return {"query": query, "error": f"{type(exc).__name__}: {exc}"}


def backend_probe(backend_name, dtype, tensors, fp32_reference, cfg):
    backend = getattr(SDPBackend, backend_name)
    converted = tuple(t.to(dtype) for t in tensors)
    row = {"backend": backend_name, "dtype": str(dtype),
           "input_layout": layout(tensors), "converted_layout": layout(converted),
           "status": "started"}

    def candidate():
        # Conversion/allocation is part of the production FP32 boundary cost.
        lowered = tuple(t.to(dtype) for t in tensors)
        with sdpa_kernel([backend]):
            return F.scaled_dot_product_attention(*lowered).to(torch.float32)

    native = lambda: F.scaled_dot_product_attention(*tensors)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        try:
            with sdpa_kernel([backend]):
                row["capability"] = capability(backend_name, converted)
            # Always attempt the forced backend, even if its capability API says
            # false, so the actual rejection is recorded instead of inferred.
            row["forced_call_attempted"] = True
            actual = candidate()
            torch.cuda.synchronize()
            with sdpa_kernel([SDPBackend.MATH]):
                dtype_reference = F.scaled_dot_product_attention(*converted).to(torch.float32)
            row["vs_fp32_native"] = errors(actual, fp32_reference)
            row["vs_dtype_math"] = errors(actual, dtype_reference)
            if not row["vs_fp32_native"]["finite"] or not row["vs_dtype_math"]["finite"]:
                row["status"] = "nonfinite_output"
            else:
                row.update(paired_timings(native, candidate, cfg))
                row["status"] = "executed"
        except Exception as exc:
            row.update(status="rejected_or_failed", error=f"{type(exc).__name__}: {exc}")
        row["warnings"] = list(dict.fromkeys(str(w.message) for w in recorded))
    return row


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True, help="New JSON path; existing files are not overwritten.")
    ap.add_argument("--lengths", nargs="+", type=int, default=LENGTHS)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--packed-v", action="store_true")
    ap.add_argument("--only", choices=("all", "ranges", "probes"), default="all")
    cfg = ap.parse_args()
    if cfg.output.exists():
        ap.error("--output already exists; choose a new result path")
    if min(cfg.rounds, cfg.warmup, cfg.iterations) < 1 or any(not 1 <= s <= 4096 for s in cfg.lengths):
        ap.error("rounds, warmup and iterations must be positive; lengths must be 1..4096")
    report = {
        "schema_version": 1, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seed": 20261105, "rounds": cfg.rounds, "iterations": cfg.iterations,
        "warmup_calls_per_method": cfg.warmup, "packed_v": cfg.packed_v,
        "inputs": "Seeded normal FP32 BSHD-to-BHSD tensors; not captured model activations.",
        "timing": "Complete callable, including checks, allocations and required dtype conversions. "
                  "Wall time includes final synchronization; events include host submission gaps. "
                  "Input creation and backend warmup are excluded. Pair order alternates; no graphs.",
        "probe_timing": "FP32 QKV -> target-dtype casts -> single forced SDPA backend -> FP32 output. "
                        "The single-backend context is included in the callable cost.",
        "acceptance_declared_before_measurement": {
            "exact_fp32_path": "Finite outputs and identical contiguous uint8 views.",
            "subsequent_real_images": {"finite_rgb": True, "max_abs": 2 / 255, "rmse": 0.25 / 255},
            "image_validation_performed_here": False,
            "note": "Random operator error is diagnostic; it cannot accept a real-image precision change.",
        },
        "ranges": [], "backend_probes": [], "status": "started",
    }
    exit_status = 1
    try:
        if not torch.cuda.is_available() or not torch.version.hip:
            raise RuntimeError("This harness requires the existing ROCm GPU environment")
        torch.set_num_threads(2)
        torch.manual_seed(report["seed"])
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        report["environment"] = {"torch": torch.__version__, "torch_git": torch.version.git_version,
                                 "hip": torch.version.hip, "gpu": properties.name,
                                 "arch": getattr(properties, "gcnArchName", None)}
        preferred = getattr(torch.backends.cuda, "preferred_rocm_fa_library", None)
        report["environment"]["rocm_fa_preference"] = str(preferred()) if preferred else None
        reduced = getattr(torch.backends.cuda, "fp16_bf16_reduction_math_sdp_allowed", None)
        report["environment"]["math_sdp_reduced_precision"] = bool(reduced()) if reduced else None
        save(cfg.output, report)
        with torch.inference_mode():
            if cfg.only != "probes":
                for length in cfg.lengths:
                    row = exact_range(length, cfg)
                    report["ranges"].append(row)
                    save(cfg.output, report)
                    print(json.dumps({k: v for k, v in row.items() if k not in ("layout", "samples")}), flush=True)
            if cfg.only != "ranges":
                tensors = inputs(1061, cfg.packed_v)
                reference = F.scaled_dot_product_attention(*tensors)
                backends = ["MATH", "FLASH_ATTENTION", "EFFICIENT_ATTENTION"]
                if hasattr(SDPBackend, "CUDNN_ATTENTION"):
                    backends.append("CUDNN_ATTENTION")
                for dtype in (torch.float32, torch.float16, torch.bfloat16):
                    for backend in backends:
                        row = backend_probe(backend, dtype, tensors, reference, cfg)
                        report["backend_probes"].append(row)
                        save(cfg.output, report)
                        print(json.dumps({k: v for k, v in row.items()
                                          if k not in ("input_layout", "converted_layout", "samples")}), flush=True)
        report["range_exactness_pass"] = all(r["status"] == "ok" for r in report["ranges"]) if report["ranges"] else None
        report["status"] = "completed"
        # Backend rejection is a result of this probe, not a harness failure.
        exit_status = 1 if report["range_exactness_pass"] is False else 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        report["exit_status"] = exit_status
        save(cfg.output, report)
        print(f"JSON: {cfg.output.resolve()}", flush=True)
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
