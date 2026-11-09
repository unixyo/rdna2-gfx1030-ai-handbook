#!/usr/bin/env python3
"""Actual-QKV timing/error of FP16 GEMM islands; no model/config mutation."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from rdna2_ops.mixed_attention import mixed_attention


def profile_gemm_paths(q, k, v, scale):
    """One bounded backend attribution probe; operand conversions are excluded.

    The main benchmark includes conversions. This optional phase only identifies
    the kernels chosen for FP32, FP16->FP16 and FP16->FP32 BMM on actual strides.
    """
    from research_comfy_v05 import profile_tables
    b, h, length, dim = q.shape
    value = dim ** -.5 if scale is None else float(scale)
    root = math.sqrt(abs(value))
    qs = (q * (-root if value < 0 else root)).reshape(b*h, length, dim)
    ks = (k * root).reshape(b*h, k.shape[-2], dim).transpose(1, 2)
    probability = torch.bmm(qs, ks).softmax(dim=-1)
    result = []
    for stage, a, bb in (("QK", qs, ks), ("PV", probability, v.reshape(b*h, v.shape[-2], dim))):
        ah, bh = a.half(), bb.half()
        functions = {"fp32": lambda: torch.bmm(a, bb),
                     "fp16_output": lambda: torch.bmm(ah, bh),
                     "fp32_output": lambda: torch.bmm(ah, bh, out_dtype=torch.float32)}
        for name, fn in functions.items():
            for _ in range(3): fn()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True) as profiler:
                for _ in range(3): output = fn()
                torch.cuda.synchronize()
            result.append({"stage": stage, "path": name, "output_dtype": str(output.dtype),
                           "input_strides": [list(t.stride()) for t in ((a, bb) if name == "fp32" else (ah, bh))],
                           "profile": profile_tables(profiler)})
            del output
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile-gemm", action="store_true")
    a = ap.parse_args()
    path = Path(a.out)
    if path.exists(): raise FileExistsError(path)
    data = torch.load(a.input, map_location="cpu", weights_only=True)
    recorded_strides = data.get("strides", data.get("metadata", {}).get("strides", {}))
    tensors = []
    for index, name in enumerate(("q", "k", "v")):
        cpu = data[name]
        stride = recorded_strides.get(name) if isinstance(recorded_strides, dict) else recorded_strides[index]
        if stride is None:
            gpu = cpu.to("cuda")
        else:
            # .cpu() may compress the packed-QKV storage gaps. Reconstruct the
            # recorded layout before benchmarking the actual MOSS V access path.
            gpu = torch.empty_strided(cpu.shape, stride, device="cuda", dtype=cpu.dtype)
            gpu.copy_(cpu)
        tensors.append(gpu)
    q, k, v = tensors
    scale = data.get("scale")
    if q.ndim != 4: raise ValueError("MOSS BHSD capture required")
    funcs = {"native_math": lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale),
             "control_fp32": lambda: mixed_attention(q, k, v, scale=scale),
             "qk_half": lambda: mixed_attention(q, k, v, qk="half", scale=scale),
             "pv_half": lambda: mixed_attention(q, k, v, pv="half", scale=scale),
             "both_half": lambda: mixed_attention(q, k, v, qk="half", pv="half", scale=scale),
             "qk_split3": lambda: mixed_attention(q, k, v, qk="split", scale=scale),
             "pv_split3": lambda: mixed_attention(q, k, v, pv="split", scale=scale)}
    flag = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    report = {"input": a.input, "shape": list(q.shape), "strides": [list(t.stride()) for t in (q,k,v)],
              "source": "actual captured QKV with original strides restored",
              "capture_metadata": data.get("metadata", {}), "scale": scale,
              "input_stats": {name: {"finite": bool(torch.isfinite(data[name]).all()),
                                      "max_abs": data[name].abs().max().item(),
                                      "half_finite": bool(torch.isfinite(data[name].half()).all())}
                              for name in ("q", "k", "v")},
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "acceptance": "operator diagnostic; final model RGB max<=2/255, RMSE<=.25/255 is decisive",
              "timing": "ordinary wall; 3 warmups then 3 alternating rounds x3 calls; allocations/conversions included",
              "rows": []}
    try:
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            ref = funcs["native_math"]()
            timings = {n: [] for n in funcs}
            for name, fn in funcs.items():
                actual = fn()
                delta = actual.double() - ref.double()
                error = {"finite": bool(torch.isfinite(actual).all()),
                         "byte_equal": bool(torch.equal(actual.view(torch.uint8), ref.contiguous().view(torch.uint8))),
                         "max_abs": delta.abs().max().item(), "rmse": delta.square().mean().sqrt().item(),
                         "relative_rmse": (delta.square().mean()/ref.double().square().mean()).sqrt().item()}
                del actual, delta
                for _ in range(3): fn()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                y = fn(); torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() - base
                del y
                report["rows"].append({"mode": name, "error": error, "extra_peak_bytes": peak})
            names = list(funcs)
            for index in range(3):
                for name in names if index % 2 == 0 else list(reversed(names)):
                    torch.cuda.synchronize(); start = time.perf_counter()
                    for _ in range(3): y = funcs[name]()
                    torch.cuda.synchronize()
                    timings[name].append((time.perf_counter()-start)*1000/3)
                    del y
            baseline = statistics.median(timings["native_math"])
            for row in report["rows"]:
                row["round_ms"] = timings[row["mode"]]
                row["median_ms"] = statistics.median(row["round_ms"])
                row["speedup"] = baseline/row["median_ms"]
                print(json.dumps(row), flush=True)
            if a.profile_gemm:
                report["gemm_profile_scope"] = "3 warmed BMM calls per path; conversions excluded; diagnostic GPU activity totals"
                report["gemm_profiles"] = profile_gemm_paths(q, k, v, scale)
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = flag
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__": main()
