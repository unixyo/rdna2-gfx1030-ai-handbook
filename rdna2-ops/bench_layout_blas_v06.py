#!/usr/bin/env python3
"""Bounded actual-gfx1030 layout map for FP16/FP32 mm/bmm.

Twelve shape/dtype cases x four storage layouts. Shapes come from actual
Anima/Qwen/MOSS PV operations plus two tall-skinny and one projection mm.
Tensor values are deterministic synthetic data; whole-model evidence is
separate. Existing Python/Torch/rocBLAS only. No environment changes.
"""
from __future__ import annotations
import argparse
import datetime as dt
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch
from torch.profiler import profile, ProfilerActivity

from gpu_guard_v06 import GpuAudit
from rdna2_ops.layout_blas_v06 import LAYOUTS, prepare_operands, multiply_prepared, matmul_layout
from research_comfy_v05 import profile_tables

CASES = {
    "anima_pv_self": {"batch": 32, "m": 4096, "n": 128, "k": 4096,
                      "origin": "Anima B2*H16, S4096, D128 PV (packed math path)"},
    "qwen_pv_prefix": {"batch": 32, "m": 2400, "n": 128, "k": 4964,
                       "origin": "Qwen B1*H32 prefix-cache PV"},
    "moss_pv": {"batch": 24, "m": 1061, "n": 128, "k": 1061,
                "origin": "MOSS B1*H24 S1061 PV"},
    "mm_n64_k4096": {"batch": None, "m": 4096, "n": 64, "k": 4096,
                     "origin": "Representative tall-skinny mm; synthetic shape"},
    "mm_n128_k8192": {"batch": None, "m": 4096, "n": 128, "k": 8192,
                      "origin": "Representative tall-skinny mm; synthetic shape"},
    "mm_projection": {"batch": None, "m": 1061, "n": 3072, "k": 3072,
                      "origin": "MOSS observed projection dimensions; floating point test"},
}
TOL = {"float16": {"rtol": 5e-3, "atol": 5e-3},
       "float32": {"rtol": 3e-4, "atol": 3e-5}}


def tensor_info(x):
    return {"shape": list(x.shape), "stride": list(x.stride()),
            "dtype": str(x.dtype), "contiguous": x.is_contiguous(),
            "storage_bytes": x.untyped_storage().nbytes()}


def errors(value, reference, tol):
    value, reference = value.double(), reference.double()
    finite = bool(torch.isfinite(value).all())
    if not finite:
        return {"finite": False, "allclose": False, "byte_equal": False}
    delta = value - reference
    rms = delta.square().mean().sqrt().item()
    ref_rms = reference.square().mean().sqrt().item()
    return {"finite": True, "allclose": bool(torch.allclose(value, reference, **tol)),
            "value_equal": bool(torch.equal(value, reference)),
            "max_abs": delta.abs().max().item(), "rmse": rms,
            "relative_rmse": rms / max(ref_rms, 1e-30)}


def fp64_sample(a, b):
    m, n = a.shape[-2], b.shape[-1]
    rows, cols = sorted({0, m // 3, m // 2, m - 1}), sorted({0, n // 3, n // 2, n - 1})
    if a.ndim == 3:
        batches = sorted({0, a.shape[0] - 1})
        av = a[batches][:, rows, :].cpu().double()
        bv = b[batches][:, :, cols].cpu().double()
        ref = torch.bmm(av, bv)
    else:
        batches = None
        ref = a[rows, :].cpu().double() @ b[:, cols].cpu().double()
    return ref, {"rows": rows, "cols": cols, "batches": batches}


def select_sample(cpu_output, selection):
    if selection["batches"] is not None:
        return cpu_output[selection["batches"]][:, selection["rows"]][:, :, selection["cols"]]
    return cpu_output[selection["rows"]][:, selection["cols"]]


def measure(fn, repeat):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    start_event.record()
    for _ in range(repeat):
        result = fn()
        del result
    stop_event.record()
    torch.cuda.synchronize()
    return {"wall_ms": (time.perf_counter() - started) * 1000 / repeat,
            "event_ms": start_event.elapsed_time(stop_event) / repeat}


def peak(fn):
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    increment = torch.cuda.max_memory_allocated() - before
    del result
    return increment


def call_profile(fn):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        result = fn()
        torch.cuda.synchronize()
    del result
    table = profile_tables(prof)
    # Complete grouped device rows keep exact selected kernel names, copies,
    # count and raw duration. GPU annotations are excluded by schema v2.
    return {"device_activities": table["device_activities"],
            "operator_attribution": table["operator_attribution"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    ap.add_argument("--dtypes", nargs="+", choices=("float16", "float32"), default=["float16", "float32"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3)
    cfg = ap.parse_args()
    if min(cfg.rounds, cfg.repeat, cfg.warmup) < 1:
        ap.error("Positive rounds/repeat/warmup required")
    if cfg.output.exists():
        ap.error("Choose a new output path; evidence is never overwritten")
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    result = {"status": "started", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "contract": "C=A@B, logical values unchanged across layouts, explicit mm/bmm, inference",
              "layout_labels": {"NN": "row-major A and B",
                                "NT": "row-major A, column-major logical B via transpose-contiguous-transpose",
                                "TN": "column-major logical A, row-major B",
                                "STRIDED": "bmm B batch-interleaved [N,B*N,1]; mm B column stride 2"},
              "label_note": "PyTorch storage labels, NOT raw rocBLAS transA/transB flags; exact input strides and kernel names govern",
              "data": "Synthetic deterministic values at observed or explicitly representative shapes; model-output A/B is separate",
              "reference": "Full native NN result plus selected independent CPU FP64 dot products",
              "acceptance": TOL, "rounds": cfg.rounds, "repeat": cfg.repeat, "warmup_per_layout_round": cfg.warmup,
              "timing": "Alternating layout order and prepared/copy-inclusive phase order per round; profiler excluded",
              "memory": "PyTorch peak allocated increment per call; prepared storage also reported; driver allocations not included",
              "sources": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                          ("bench_layout_blas_v06.py", "rdna2_ops/layout_blas_v06.py", "gpu_guard_v06.py", "research_comfy_v05.py")},
              "cases": []}
    def save():
        cfg.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    save()
    audit = GpuAudit()
    try:
        with audit, torch.inference_mode():
            audit.ensure_clear()
            props = torch.cuda.get_device_properties(0)
            if not torch.version.hip or props.gcnArchName.split(":")[0] != "gfx1030":
                raise RuntimeError("Actual gfx1030 HIP runtime required")
            result["environment"] = {"python": sys.executable, "torch": str(torch.__version__),
                                     "hip": torch.version.hip, "gpu": props.name, "arch": props.gcnArchName,
                                     "preferred_blas": str(torch.backends.cuda.preferred_blas_library()),
                                     "float32_matmul_precision": torch.get_float32_matmul_precision(),
                                     "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                                     "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
            for case_index, name in enumerate(cfg.cases):
                spec = CASES[name]
                for dtype_name in cfg.dtypes:
                    audit.ensure_clear()
                    torch.manual_seed(20261107 + case_index)
                    dtype = getattr(torch, dtype_name)
                    prefix = () if spec["batch"] is None else (spec["batch"],)
                    a = torch.randn((*prefix, spec["m"], spec["k"]), dtype=dtype, device="cuda") * 0.125
                    b = torch.randn((*prefix, spec["k"], spec["n"]), dtype=dtype, device="cuda") * 0.125
                    native = multiply_prepared(a, b).cpu()
                    fp64, selection = fp64_sample(a, b)
                    row = {"name": name, **spec, "dtype": dtype_name,
                           "base_a": tensor_info(a), "base_b": tensor_info(b),
                           "fp64_sample_selection": selection,
                           "native_vs_fp64_sample": errors(select_sample(native, selection), fp64, TOL[dtype_name]),
                           "layouts": {}}
                    result["cases"].append(row)
                    for round_index in range(cfg.rounds):
                        order = list(LAYOUTS) if round_index % 2 == 0 else list(reversed(LAYOUTS))
                        for position, layout in enumerate(order):
                            audit.ensure_clear()
                            aa, bb = prepare_operands(a, b, layout)
                            entry = row["layouts"].setdefault(layout, {"a": tensor_info(aa), "b": tensor_info(bb),
                                                                       "prepared_samples": [], "copy_inclusive_samples": []})
                            prepared = lambda: multiply_prepared(aa, bb)
                            inclusive = lambda: matmul_layout(a, b, layout)
                            if round_index == 0:
                                entry["logical_inputs_equal"] = bool(torch.equal(aa, a) and torch.equal(bb, b))
                                value = prepared().cpu()
                                incl_value = inclusive().cpu()
                                entry["vs_native_nn"] = errors(value, native, TOL[dtype_name])
                                entry["vs_fp64_sample"] = errors(select_sample(value, selection), fp64, TOL[dtype_name])
                                entry["copy_inclusive_vs_prepared"] = errors(incl_value, value, {"rtol": 0.0, "atol": 0.0})
                                entry["byte_equal_to_native"] = bool(torch.equal(value.view(torch.uint8), native.view(torch.uint8)))
                                entry["copy_inclusive_byte_equal_to_prepared"] = bool(torch.equal(incl_value.view(torch.uint8), value.view(torch.uint8)))
                                entry["prepared_extra_storage_bytes"] = sum(x.untyped_storage().nbytes() for x, base in ((aa, a), (bb, b))
                                                                             if x.untyped_storage().data_ptr() != base.untyped_storage().data_ptr())
                                del value, incl_value
                            phases = [("prepared", prepared), ("copy_inclusive", inclusive)]
                            if round_index % 2:
                                phases.reverse()
                            for phase, fn in phases:
                                for _ in range(cfg.warmup):
                                    y = fn()
                                    del y
                                sample = measure(fn, cfg.repeat)
                                sample.update(round=round_index, position=position)
                                entry[phase + "_samples"].append(sample)
                            if round_index == 0:
                                entry["prepared_peak_increment_bytes"] = peak(prepared)
                                entry["copy_inclusive_peak_increment_bytes"] = peak(inclusive)
                                entry["prepared_profile"] = call_profile(prepared)
                            del prepared, inclusive, aa, bb
                            save()
                        audit.ensure_clear()
                    for entry in row["layouts"].values():
                        for phase in ("prepared", "copy_inclusive"):
                            entry[phase + "_median_wall_ms"] = statistics.median(x["wall_ms"] for x in entry[phase + "_samples"])
                            entry[phase + "_median_event_ms"] = statistics.median(x["event_ms"] for x in entry[phase + "_samples"])
                            entry[phase + "_speedup_vs_nn"] = (statistics.median(x["wall_ms"] for x in row["layouts"]["NN"][phase + "_samples"])
                                                               / entry[phase + "_median_wall_ms"])
                        entry["operator_gate"] = (entry["logical_inputs_equal"] and entry["vs_native_nn"]["allclose"]
                                                  and entry["vs_fp64_sample"]["allclose"]
                                                  and entry["copy_inclusive_byte_equal_to_prepared"])
                    print("LAYOUT_CASE " + json.dumps({"name": name, "dtype": dtype_name,
                          "layouts": {layout: {"prepared_ms": entry["prepared_median_wall_ms"],
                                               "copy_inclusive_ms": entry["copy_inclusive_median_wall_ms"],
                                               "gate": entry["operator_gate"]} for layout, entry in row["layouts"].items()}}), flush=True)
                    del a, b, native, fp64
                    gc.collect()
                    save()
            audit.ensure_clear()
        result["gpu_audit"] = audit.report()
        result["status"] = "measured" if result["gpu_audit"]["valid_for_timing"] else "discarded_overlap"
        result["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()
        if result["status"] != "measured":
            raise RuntimeError("Discarded timing due to overlap or unavailable audit")
    except Exception:
        result["status"] = "discarded_overlap" if not audit.report()["valid_for_timing"] else "failed"
        result["gpu_audit"] = audit.report()
        result["error"] = traceback.format_exc()
        save()
        raise


if __name__ == "__main__":
    main()
