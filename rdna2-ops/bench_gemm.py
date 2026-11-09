#!/usr/bin/env python3
"""Bounded rocBLAS/Tensile versus Triton FP16 GEMM comparison on gfx1030.

Example (use the existing GPU environment's Python):
  python bench_gemm.py --probe-hipblaslt --output results/gemm.json

Defaults: three M,N,K shapes, transposed weight views, one fixed Triton tile.
The benchmark process alone sets ROCBLAS_USE_HIPBLASLT=0 before importing
PyTorch and selects its 'cublas' enum, which maps to hipBLAS/rocBLAS on ROCm.
The separately spawned hipBLASLt probe preserves the caller's rocBLAS settings.
No package installation, external build, model patch, or global configuration.

Timing includes ordinary eager dispatch and output allocation, after warmup;
GPU events and synchronized CPU wall time are both reported. Memory figures
cover PyTorch's allocator only, not all allocations inside external libraries.
"""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import warnings


# Fixed before measurement, including for full-matrix agreement and FP64 checks.
TOLERANCE = {"atol": 2e-3, "rtol": 2e-3, "relative_l2": 1e-3}
ENV_KEYS = (
    "ROCBLAS_USE_HIPBLASLT", "ROCBLAS_USE_HIPBLASLT_BATCHED",
    "TORCH_BLAS_PREFER_HIPBLASLT", "TORCH_BLAS_PREFER_CUBLASLT",
    "HSA_OVERRIDE_GFX_VERSION", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
)
torch = None  # Imported only after this process's backend environment is set.


def dimensions(text):
    try:
        result = tuple(int(v) for v in text.lower().replace(",", "x").split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use MxNxK, for example 1024x1024x1024") from exc
    if len(result) != 3 or min(result) <= 0:
        raise argparse.ArgumentTypeError("Three positive M,N,K dimensions are required")
    return result


def errors(actual, reference):
    actual = actual.detach().to(device="cpu", dtype=torch.float64)
    reference = reference.detach().to(device="cpu", dtype=torch.float64)
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    if not finite:
        return {"finite": False, "passed": False}
    diff = actual - reference
    relative_l2 = (torch.linalg.vector_norm(diff)
                   / torch.linalg.vector_norm(reference).clamp_min(1e-30)).item()
    normalized_max = (diff.abs() / (TOLERANCE["atol"]
                                    + TOLERANCE["rtol"] * reference.abs())).max().item()
    return {
        "finite": True,
        "max_absolute": diff.abs().max().item(),
        "rms_error": diff.square().mean().sqrt().item(),
        "reference_rms": reference.square().mean().sqrt().item(),
        "relative_l2": relative_l2,
        "max_normalized_error": normalized_max,
        "passed": normalized_max <= 1.0 and relative_l2 <= TOLERANCE["relative_l2"],
    }


def inputs(shape, layout, device):
    m, n, k = shape
    a = torch.randn((m, k), device=device, dtype=torch.float16)
    if layout == "transposed":
        b = torch.randn((n, k), device=device, dtype=torch.float16).T
    else:
        b = torch.randn((k, n), device=device, dtype=torch.float16)
    return a, b


@contextmanager
def rocblas_backend():
    """Keep backend/precision changes confined to this standalone process."""
    preferred = torch.backends.cuda.preferred_blas_library
    before = preferred()
    matmul = torch.backends.cuda.matmul
    attributes = ("allow_fp16_reduced_precision_reduction", "allow_fp16_accumulation")
    saved = {}
    try:
        preferred("cublas")
        if not str(preferred()).lower().endswith("cublas"):
            raise RuntimeError("PyTorch did not accept the rocBLAS ('cublas') baseline")
        for name in attributes:
            if hasattr(matmul, name):
                saved[name] = getattr(matmul, name)
                setattr(matmul, name, False)
        yield {
            "preferred_before": str(before), "preferred_during": str(preferred()),
            "precision_before": saved.copy(),
            "precision_during": {name: getattr(matmul, name) for name in saved},
            "selection": "PyTorch cublas enum -> hipBLAS/rocBLAS; rocBLAS Tensile forced in this process",
        }
    finally:
        for name, value in saved.items():
            setattr(matmul, name, value)
        preferred(before)


def measure_once(fn, iterations, device):
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    for _ in range(iterations):
        fn()  # Release each result; no retained output from a preceding iteration.
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - wall) * 1000 / iterations
    return {
        "wall_ms": wall_ms, "event_ms": start.elapsed_time(end) / iterations,
        "baseline_allocated_bytes": allocated, "baseline_reserved_bytes": reserved,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "peak_extra_allocated_bytes": torch.cuda.max_memory_allocated(device) - allocated,
        "peak_extra_reserved_bytes": torch.cuda.max_memory_reserved(device) - reserved,
    }


def timings(functions, args, device, shape):
    for _ in range(args.warmup):
        for fn in functions.values():
            fn()
    torch.cuda.synchronize(device)
    samples = {name: [] for name in functions}
    names = list(functions)
    for trial in range(args.rounds):
        for name in (names if trial % 2 == 0 else names[::-1]):
            samples[name].append(measure_once(functions[name], args.iterations, device))
    result = {}
    flops = 2 * math.prod(shape)
    for name, rows in samples.items():
        event_ms = statistics.median(r["event_ms"] for r in rows)
        wall_ms = statistics.median(r["wall_ms"] for r in rows)
        result[name] = {
            "event_ms": event_ms, "wall_ms": wall_ms,
            "event_tflops": flops / (event_ms * 1e9),
            "wall_tflops": flops / (wall_ms * 1e9),
            "peak_extra_allocated_bytes": max(r["peak_extra_allocated_bytes"] for r in rows),
            "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows),
            "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in rows),
            "samples": rows,
        }
    return result


def small_reference(fp16_gemm, args, device):
    results = []
    # All M/N/K tails plus both supported common layouts, not a large test grid.
    for layout in ("contiguous", "transposed"):
        a, b = inputs((37, 53, 79), layout, device)
        reference = a.cpu().double() @ b.cpu().double()
        native = torch.mm(a, b)
        candidate = fp16_gemm(a, b, tile=args.tile, num_warps=args.num_warps)
        row = {"shape_mnk": [37, 53, 79], "rhs_layout": layout,
               "rocblas_vs_fp64": errors(native, reference),
               "triton_vs_fp64": errors(candidate, reference)}
        results.append(row)
        if not all(row[key]["passed"] for key in ("rocblas_vs_fp64", "triton_vs_fp64")):
            raise AssertionError("Small FP64 reference failed: " + json.dumps(row))
    return results


def benchmark_shape(fp16_gemm, args, device, shape):
    m, n, k = shape
    a, b = inputs(shape, args.rhs_layout, device)
    native = lambda: torch.mm(a, b)
    candidate = lambda: fp16_gemm(a, b, tile=args.tile, num_warps=args.num_warps)
    metadata = {}
    torch.cuda.synchronize(device)
    initial = time.perf_counter()
    expected = native()
    torch.cuda.synchronize(device)
    rocblas_first = (time.perf_counter() - initial) * 1000
    initial = time.perf_counter()
    actual = fp16_gemm(a, b, tile=args.tile, num_warps=args.num_warps, metadata=metadata)
    torch.cuda.synchronize(device)
    triton_first = (time.perf_counter() - initial) * 1000
    if metadata["matrix_instructions_present"]:
        raise AssertionError("Unexpected MFMA/WMMA in a gfx1030 SIMD kernel")

    # A small output block, spanning edges/interior, uses every K element.
    # This tests accumulation at the measured K without a huge CPU FP64 GEMM.
    rows = torch.linspace(0, m - 1, min(32, m), dtype=torch.int64, device=device)
    cols = torch.linspace(0, n - 1, min(32, n), dtype=torch.int64, device=device)
    oracle = (a.index_select(0, rows).cpu().double()
              @ b.index_select(1, cols).cpu().double())
    checks = {
        "triton_vs_rocblas_full": errors(actual, expected),
        "rocblas_vs_fp64_sample": errors(expected[rows[:, None], cols[None, :]], oracle),
        "triton_vs_fp64_sample": errors(actual[rows[:, None], cols[None, :]], oracle),
    }
    row = {
        "shape_mnk": list(shape), "rhs_layout": args.rhs_layout,
        "a_stride": list(a.stride()), "b_stride": list(b.stride()),
        "dtype": "float16", "triton_accumulator": "float32",
        "fp64_sample_shape": [len(rows), len(cols)], "checks": checks,
        "kernel": metadata,
        "first_call_wall_ms_including_jit_or_library_init": {
            "rocblas": rocblas_first, "triton": triton_first},
    }
    del actual, expected, oracle, rows, cols
    if not all(check["passed"] for check in checks.values()):
        row["status"] = "accuracy_failed"
        return row
    row["timings"] = timings({"rocblas": native, "triton": candidate}, args, device, shape)
    row["event_speedup"] = row["timings"]["rocblas"]["event_ms"] / row["timings"]["triton"]["event_ms"]
    row["wall_speedup"] = row["timings"]["rocblas"]["wall_ms"] / row["timings"]["triton"]["wall_ms"]
    row["status"] = "passed"
    return row


def hipblaslt_child(device_index):
    """One explicitly preferred torch.mm; success alone is not proof of no fallback."""
    global torch
    import torch as imported_torch
    torch = imported_torch
    result = {"requested": "cublaslt (hipBLASLt on ROCm)",
              "environment": {key: os.environ.get(key) for key in ENV_KEYS},
              "direct_hipblaslt_support_confirmed": False}
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if not torch.version.hip:
                raise RuntimeError("The installed PyTorch is not built for ROCm")
            torch.cuda.set_device(device_index)
            result["torch_version"] = torch.__version__
            result["hip_version"] = torch.version.hip
            result["preferred_before"] = str(torch.backends.cuda.preferred_blas_library())
            torch.backends.cuda.preferred_blas_library("cublaslt")
            result["preferred_after_request"] = str(torch.backends.cuda.preferred_blas_library())
            with torch.inference_mode():
                a = torch.ones((64, 64), device="cuda", dtype=torch.float16)
                output = torch.mm(a, a)
                torch.cuda.synchronize()
                result["numeric_check"] = bool(torch.equal(output.cpu(), torch.full((64, 64), 64., dtype=torch.float16)))
            result["preferred_after_mm"] = str(torch.backends.cuda.preferred_blas_library())
            kept = result["preferred_after_mm"].lower().endswith("cublaslt")
            result["status"] = "preferred_mm_succeeded_support_unconfirmed" if kept else "preference_fell_back"
            result["reason"] = (
                "A successful torch.mm does not establish that its internal library path avoided fallback; "
                "do not label hipBLASLt supported from this result alone."
            )
    except Exception as exc:
        result.update(status="error", error_type=type(exc).__name__, error=str(exc))
    result["warnings"] = [str(w.message) for w in caught]
    print(json.dumps(result), flush=True)
    return 0


def hipblaslt_probe(args):
    env = os.environ.copy()
    env["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
    env["TORCH_BLAS_PREFER_CUBLASLT"] = "1"
    command = [sys.executable, str(Path(__file__).resolve()), "--_hipblaslt-child",
               "--device", str(args.device)]
    try:
        child = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
        lines = child.stdout.strip().splitlines()
        try:
            result = json.loads(lines[-1]) if lines else {"status": "no_json_result"}
        except json.JSONDecodeError:
            result = {"status": "invalid_json_result"}
        result.update(returncode=child.returncode, stderr_tail=child.stderr[-8000:],
                      stdout_tail=child.stdout[-8000:])
        return result
    except subprocess.TimeoutExpired as exc:
        return {"status": "timeout", "timeout_seconds": 60, "error": str(exc),
                "direct_hipblaslt_support_confirmed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", type=dimensions, nargs="+", default=[(1024, 1024, 1024), (2048, 2048, 2048), (1061, 3072, 3072)])
    parser.add_argument("--rhs-layout", choices=("contiguous", "transposed"), default="transposed")
    parser.add_argument("--tile", type=dimensions, default=(32, 64, 32))
    parser.add_argument("--num-warps", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--probe-hipblaslt", action="store_true")
    parser.add_argument("--_hipblaslt-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=Path("results/gemm.json"))
    args = parser.parse_args()
    if args._hipblaslt_child:
        return hipblaslt_child(args.device)
    if args.output.exists():
        parser.error("choose a new --output path; existing evidence is not overwritten")
    if min(args.warmup, args.rounds, args.iterations) < 1:
        parser.error("warmup, rounds and iterations must be positive")
    report = {
        "status": "running", "seed": args.seed, "tolerance": TOLERANCE,
        "error_definition": "max_normalized_error = max(abs(error)/(atol+rtol*abs(reference))); also require relative L2 <= fixed threshold",
        "environment_before": {key: os.environ.get(key) for key in ENV_KEYS},
        "protocol": {"warmup": args.warmup, "rounds": args.rounds,
                     "iterations_per_round": args.iterations, "alternating_order": True,
                     "allocation_included": True, "graph_capture": False,
                     "memory_scope": "PyTorch allocator; external library allocations may be absent"},
        "rows": [],
    }
    # Probe first, before this benchmark's rocBLAS environment override or GPU init.
    report["hipblaslt_probe"] = hipblaslt_probe(args) if args.probe_hipblaslt else {"status": "not_requested"}
    original_internal_backend = os.environ.get("ROCBLAS_USE_HIPBLASLT")
    os.environ["ROCBLAS_USE_HIPBLASLT"] = "0"
    try:
        global torch
        import torch as imported_torch
        torch = imported_torch
        import triton
        from rdna2_ops.gemm import fp16_gemm
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("Use the existing ROCm/PyTorch GPU environment")
        torch.cuda.set_device(args.device)
        device = torch.device("cuda", args.device)
        properties = torch.cuda.get_device_properties(device)
        if properties.gcnArchName.split(":")[0] != "gfx1030":
            raise RuntimeError("This comparison is specifically for gfx1030")
        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        report.update(torch_version=torch.__version__, hip_version=torch.version.hip,
                      triton_version=triton.__version__, gpu=properties.name,
                      architecture=properties.gcnArchName, python=sys.version,
                      environment_during={key: os.environ.get(key) for key in ENV_KEYS})
        with torch.inference_mode(), rocblas_backend() as selection:
            report["blas_selection"] = selection
            report["small_fp64_reference"] = small_reference(fp16_gemm, args, device)
            for shape in args.shapes:
                row = benchmark_shape(fp16_gemm, args, device, shape)
                report["rows"].append(row)
                print(json.dumps(row), flush=True)
        report["preferred_blas_after_restore"] = str(torch.backends.cuda.preferred_blas_library())
        report["status"] = "passed" if all(row["status"] == "passed" for row in report["rows"]) else "accuracy_failed"
    except Exception as exc:
        report.update(status="error", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc(limit=8))
    finally:
        if original_internal_backend is None:
            os.environ.pop("ROCBLAS_USE_HIPBLASLT", None)
        else:
            os.environ["ROCBLAS_USE_HIPBLASLT"] = original_internal_backend
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "error": report.get("error")}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
