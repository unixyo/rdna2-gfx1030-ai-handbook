#!/usr/bin/env python3
"""Correctness and end-to-end timings for rdna2_ops.group_norm_silu.

No packages are installed or settings persisted. The FP64 reference computes
centered variance independently and rounds GroupNorm to the input dtype before
SiLU, matching the two-operator contract. Tolerances are fixed below.
HIP GroupNorm is eager-only: graph timing is reported as policy_disabled.
Use bench_compiled.py for native PyTorch and compiled graph timings.
"""
import argparse
import datetime as dt
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F

from rdna2_ops import group_norm_silu


TOLERANCES = {
    torch.float32: {"rtol": 3e-4, "atol": 3e-5},
    torch.float16: {"rtol": 3e-3, "atol": 3e-3},
    torch.bfloat16: {"rtol": 3e-2, "atol": 3e-2},
}
BENCH_SHAPES = [(1, 320, 64, 64), (2, 640, 32, 32),
                (1, 128, 128, 128), (1, 1280, 16, 16)]


def dtype_name(dtype):
    return str(dtype).removeprefix("torch.")


def fp64_reference(x, groups, weight, bias, eps):
    grouped = x.double().reshape(x.shape[0], groups, -1)
    centered = grouped - grouped.mean(dim=-1, keepdim=True)
    variance = centered.square().mean(dim=-1, keepdim=True)
    norm = (centered * torch.rsqrt(variance + eps)).reshape(x.shape)
    affine_shape = (1, x.shape[1]) + (1,) * (x.ndim - 2)
    if weight is not None:
        norm = norm * weight.double().reshape(affine_shape)
    if bias is not None:
        norm = norm + bias.double().reshape(affine_shape)
    rounded = norm.to(x.dtype).double()
    return (rounded * torch.sigmoid(rounded)).to(x.dtype)


def error_metrics(actual, expected, dtype):
    tol = TOLERANCES[dtype]
    a, b = actual.detach().double(), expected.detach().double()
    error = (a - b).abs()
    finite = torch.isfinite(a) & torch.isfinite(b)
    limit = tol["atol"] + tol["rtol"] * b.abs()
    bad = (~finite) | (error > limit)
    is_finite = bool(finite.all().item())
    return {
        "pass": not bool(bad.any().item()),
        "finite": is_finite,
        "failed_elements": int(bad.sum().item()),
        "max_abs": float(error.max().item()) if is_finite else None,
        "max_scaled_error": float((error / limit).max().item()) if is_finite else None,
        "rtol": tol["rtol"], "atol": tol["atol"],
    }


def offset_random(shape, dtype, offset=0):
    flat = torch.randn(math.prod(shape) + offset, device="cuda", dtype=dtype)
    return flat[offset:].view(shape)


def make_inputs(case, dtype):
    shape = case["shape"]
    offset = case.get("offset", 0)
    x = offset_random(shape, dtype, offset)
    kind = case.get("kind", "random")
    if kind == "constant":
        x.fill_(3.0)
    elif kind == "high_mean":
        mean, sigma = {
            torch.float32: (10000.0, 0.125),
            torch.float16: (64.0, 0.125),
            torch.bfloat16: (16.0, 0.25),
        }[dtype]
        x.copy_(torch.randn(shape, device="cuda", dtype=torch.float32) * sigma + mean)
    weight = (offset_random((shape[1],), dtype, int(offset > 0)).mul_(0.2).add_(1)
              if case.get("weight", True) else None)
    bias = (offset_random((shape[1],), dtype, int(offset > 0)).mul_(0.1)
            if case.get("bias", True) else None)
    return x, weight, bias


def forward_cases():
    cases = [{"name": f"workload_{i}", "shape": shape, "groups": 32}
             for i, shape in enumerate(BENCH_SHAPES)]
    cases += [
        {"name": "tiny_constant_bias_only", "shape": (2, 12, 1, 1),
         "groups": 4, "kind": "constant", "weight": False, "eps": 1e-3},
        {"name": "tails_storage_offset_weight_only", "shape": (2, 15, 7, 9),
         "groups": 5, "offset": 3, "bias": False},
        {"name": "group_elements_8191", "shape": (1, 32, 1, 8191), "groups": 32},
        {"name": "group_elements_8192_no_affine", "shape": (1, 32, 1, 8192),
         "groups": 32, "weight": False, "bias": False},
        {"name": "group_elements_8193_other_stream", "shape": (1, 32, 1, 8193),
         "groups": 32, "other_stream": True},
        {"name": "large_mean_small_variance", "shape": (2, 32, 32, 67),
         "groups": 8, "kind": "high_mean"},
    ]
    return cases


def validate_forward(case, dtype):
    row = {**case, "dtype": dtype_name(dtype), "backend": "hip"}
    stream = torch.cuda.Stream() if case.get("other_stream") else torch.cuda.current_stream()
    with torch.no_grad(), torch.cuda.stream(stream):
        x, weight, bias = make_inputs(case, dtype)
        args = (x, case["groups"], weight, bias, case.get("eps", 1e-5))
        reference = fp64_reference(*args)
        actual = group_norm_silu(*args, backend="hip")
        if actual.shape != x.shape or actual.dtype != x.dtype or actual.device != x.device:
            raise AssertionError("HIP output must preserve input shape, dtype, and device")
        # An immediate consumer on the caller's stream checks stream ordering.
        if case.get("other_stream"):
            actual = actual.clone()
        native = F.silu(F.group_norm(*args))
    stream.synchronize()
    row["hip_vs_fp64"] = error_metrics(actual, reference, dtype)
    row["native_vs_fp64"] = error_metrics(native, reference, dtype)
    row["hip_vs_native"] = error_metrics(actual, native, dtype)
    row["pass"] = row["hip_vs_fp64"]["pass"]
    return row


def validate_fallback(noncontiguous=False):
    dtype = torch.float32
    shape = (2, 12, 7, 9) if noncontiguous else (2, 8, 3, 5)
    groups = 3 if noncontiguous else 4
    x = torch.randn(shape, device="cuda", dtype=dtype)
    if noncontiguous:
        x = x.transpose(-1, -2)
    else:
        x.requires_grad_()
    weight = torch.randn(shape[1], device="cuda", dtype=dtype,
                         requires_grad=not noncontiguous)
    bias = torch.randn_like(weight, requires_grad=not noncontiguous)
    actual = group_norm_silu(x, groups, weight, bias, backend="auto")
    if actual.shape != x.shape or actual.dtype != x.dtype or actual.device != x.device:
        raise AssertionError("Auto output must preserve input shape, dtype, and device")
    native_x = x.detach().clone().requires_grad_(not noncontiguous)
    native_w = weight.detach().clone().requires_grad_(not noncontiguous)
    native_b = bias.detach().clone().requires_grad_(not noncontiguous)
    native = F.silu(F.group_norm(native_x, groups, native_w, native_b))
    with torch.no_grad():
        reference = fp64_reference(x, groups, weight, bias, 1e-5)
    row = {
        "name": "noncontiguous_auto_fallback" if noncontiguous else "autograd_auto_fallback",
        "dtype": dtype_name(dtype), "shape": list(x.shape), "groups": groups,
        "backend": "auto", "is_contiguous": x.is_contiguous(),
        "auto_vs_fp64": error_metrics(actual, reference, dtype),
        "auto_vs_native": error_metrics(actual, native, dtype),
    }
    row["pass"] = row["auto_vs_fp64"]["pass"] and row["auto_vs_native"]["pass"]
    if not noncontiguous:
        grad = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, (x, weight, bias), grad)
        native_grads = torch.autograd.grad(native, (native_x, native_w, native_b), grad)
        row["gradients"] = {key: error_metrics(a, b, dtype) for key, a, b in
                            zip(("input", "weight", "bias"), actual_grads, native_grads)}
        row["pass"] &= all(v["pass"] for v in row["gradients"].values())
    return row


def timed_call(fn, iterations):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    begin = time.perf_counter_ns()
    start.record()
    for _ in range(iterations):
        last = fn()
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter_ns() - begin) / 1e6 / iterations
    return {"wall_ms": wall_ms, "event_ms": start.elapsed_time(end) / iterations}


def validate_autocast():
    x = torch.randn(2, 32, 7, 9, device="cuda", dtype=torch.float16)
    w = torch.ones(32, device="cuda", dtype=torch.float16)
    b = torch.zeros_like(w)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        actual = group_norm_silu(x, 8, w, b)
        native = F.silu(F.group_norm(x, 8, w, b))
    same = actual.dtype == native.dtype and torch.equal(actual, native)
    return {"name": "autocast_auto_fallback", "pass": same,
            "actual_dtype": str(actual.dtype), "native_dtype": str(native.dtype)}


def validate_forward_ad():
    x = torch.randn(2, 8, 3, 5, device="cuda")
    tangent = torch.randn_like(x)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(x, tangent)
        actual = torch.autograd.forward_ad.unpack_dual(group_norm_silu(dual, 4))
        expected = torch.autograd.forward_ad.unpack_dual(F.silu(F.group_norm(dual, 4)))
        ok = actual.tangent is not None and torch.equal(actual.primal, expected.primal) and torch.equal(actual.tangent, expected.tangent)
    return {"name": "forward_ad_auto_fallback", "pass": bool(ok)}


def validate_func_jvp():
    x = torch.randn(2, 8, 3, 5, device="cuda")
    tangent = torch.randn_like(x)
    actual = torch.func.jvp(lambda y: group_norm_silu(y, 4), (x,), (tangent,))
    expected = torch.func.jvp(lambda y: F.silu(F.group_norm(y, 4)), (x,), (tangent,))
    return {"name": "torch_func_jvp_auto_fallback", "pass": all(torch.equal(a,b) for a,b in zip(actual,expected))}


def capture_callable(fn, calls):
    """Capture a native/compiled callable; HIP GroupNorm is eager-only."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            last = fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(calls):
            last = fn()
    # Retain the final output to preserve the graph's allocation lifetime.
    return graph, last


def benchmark(shape, dtype, args):
    case = {"shape": shape, "groups": 32}
    x, weight, bias = make_inputs(case, dtype)
    functions = {backend: (lambda b=backend: group_norm_silu(
        x, 32, weight, bias, 1e-5, backend=b)) for backend in ("torch", "hip")}
    for _ in range(args.warmup):
        functions["torch"]()
        functions["hip"]()
    methods = {name: {"wall_ms": [], "event_ms": []} for name in functions}
    for round_index in range(args.rounds):
        order = ("torch", "hip") if round_index % 2 == 0 else ("hip", "torch")
        for name in order:
            sample = timed_call(functions[name], args.iterations)
            for key, value in sample.items():
                methods[name][key].append(value)
    row = {"shape": shape, "dtype": dtype_name(dtype), "groups": 32,
           "iterations": args.iterations, "rounds": args.rounds, "methods": methods}
    for timing in ("wall_ms", "event_ms"):
        for method in methods.values():
            method["median_" + timing] = statistics.median(method[timing])
        row["speedup_" + timing.removesuffix("_ms")] = (
            methods["torch"]["median_" + timing] / methods["hip"]["median_" + timing])
    row["graph"] = {
        "status": "policy_disabled", "backend": "hip",
        "requested": not args.no_graphs,
        "reason": "HIP GroupNorm is eager-only; no HIP graph warmup, capture, or replay. "
                  "Use bench_compiled.py for native/compiled graph timings.",
    }
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("validation_benchmark.json"))
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--no-graphs", action="store_true",
                        help="Retained for compatibility; HIP graph timing is always disabled by policy.")
    args = parser.parse_args()
    if min(args.iterations, args.warmup, args.rounds) < 1:
        parser.error("iterations, warmup, and rounds must be positive")
    report = {
        "schema_version": 1, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reference": "FP64 centered variance and affine; round to input dtype before FP64 SiLU",
        "timing": "Full call includes validation/allocation; wall includes final synchronization. "
                  "Events include host submission gaps. HIP graph timing is policy_disabled. "
                  "Inputs, first compilation, and warmup are excluded; order alternates each round.",
        "correctness": [], "benchmarks": [],
    }
    status = 1
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU available to this PyTorch environment")
        torch.manual_seed(20261105)
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        report["environment"] = {
            "python": sys.version.split()[0], "torch": torch.__version__, "hip": torch.version.hip,
            "device": props.name, "architecture": getattr(props, "gcnArchName", None),
            "total_memory_bytes": props.total_memory,
        }
        print(f"Device: {props.name}; arch={getattr(props, 'gcnArchName', '?')}; "
              f"torch={torch.__version__}; HIP={torch.version.hip}", flush=True)
        for dtype in TOLERANCES:
            for case in forward_cases():
                try:
                    row = validate_forward(case, dtype)
                except Exception as exc:
                    row = {**case, "dtype": dtype_name(dtype), "pass": False,
                           "error": f"{type(exc).__name__}: {exc}"}
                report["correctness"].append(row)
                if not row["pass"]:
                    detail = row.get("error", row.get("hip_vs_fp64"))
                    print(f"FAIL {dtype_name(dtype)} {case['name']}: {detail}", flush=True)
        for noncontiguous in (True, False):
            try:
                row = validate_fallback(noncontiguous)
            except Exception as exc:
                row = {"name": "noncontiguous_auto_fallback" if noncontiguous else
                       "autograd_auto_fallback", "pass": False,
                       "error": f"{type(exc).__name__}: {exc}"}
            report["correctness"].append(row)
            if not row["pass"]:
                print(f"FAIL {row['name']}: {row.get('error', row)}", flush=True)
        report["correctness"].append(validate_autocast())
        report["correctness"].append(validate_forward_ad())
        report["correctness"].append(validate_func_jvp())
        passed = sum(row["pass"] for row in report["correctness"])
        report["correctness_pass"] = passed == len(report["correctness"])
        report["native_reference_differences"] = [
            f"{row['dtype']}:{row['name']}" for row in report["correctness"]
            if "native_vs_fp64" in row and not row["native_vs_fp64"]["pass"]]
        print(f"Correctness: {passed}/{len(report['correctness'])}; "
              f"native/reference differences: {report['native_reference_differences']}", flush=True)
        if report["correctness_pass"] and not args.validate_only:
            with torch.no_grad():
                for dtype in TOLERANCES:
                    for shape in BENCH_SHAPES:
                        row = benchmark(shape, dtype, args)
                        report["benchmarks"].append(row)
                        graph = row.get("graph", {})
                        graph_text = graph.get("status", "not_requested")
                        print(f"BENCH {dtype_name(dtype):8s} {shape}: "
                              f"wall={row['speedup_wall']:.3f}x; "
                              f"events={row['speedup_event']:.3f}x; graph={graph_text}", flush=True)
        elif not args.validate_only:
            report["benchmark_skipped"] = "Correctness gate failed; fixed tolerances unchanged."
        status = 0 if report["correctness_pass"] else 1
    except Exception as exc:
        report["fatal_error"] = f"{type(exc).__name__}: {exc}"
        print(f"ERROR {report['fatal_error']}", flush=True)
    finally:
        report["exit_status"] = status
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"JSON: {args.output.resolve()}", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
