#!/usr/bin/env python3
"""Validate/measure explicit gfx1030 tiled attention; no production patching.

Use the existing ROCm environment. The native baseline is forced MATH with
FP16/BF16 reduction disabled, restored on exit. Timings include validation,
launch and output allocation, excluding compilation/warmup and input creation.
Memory is PyTorch allocator peak allocated bytes above resident inputs; compiler
or driver scratch allocations outside that allocator are not measured.
"""
import argparse
import datetime as dt
import json
import math
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from rdna2_ops.tiled_attention import tiled_attention


TILES = {(16, 32), (32, 32), (32, 64)}
# Fixed before execution, not modified in response to test failures.
TOLERANCES = {"fp32": {"rtol": 3e-4, "atol": 3e-5},
              "fp16_operand_or_output": {"rtol": 1e-2, "atol": 3e-3}}


def make_inputs(batch, heads, length, keys, dim, dtype, layout):
    if layout == "contiguous":
        shapes = ((batch, heads, length, dim), (batch, heads, keys, dim))
        return tuple(torch.randn(shape, device="cuda", dtype=dtype)
                     for shape in (shapes[0], shapes[1], shapes[1]))
    # A nonzero offset and interleaved heads exercise real strided reads.
    q = torch.randn(batch, length + 1, heads, dim, device="cuda", dtype=dtype)[:, 1:].transpose(1, 2)
    k = torch.randn(batch, keys + 1, heads, dim, device="cuda", dtype=dtype)[:, 1:].transpose(1, 2)
    if layout == "packed_v":
        v = torch.randn(batch, keys + 1, 3, heads, dim, device="cuda", dtype=dtype)[:, 1:, 2].transpose(1, 2)
    else:
        v = torch.randn(batch, keys + 1, heads, dim, device="cuda", dtype=dtype)[:, 1:].transpose(1, 2)
    return q, k, v


def layouts(tensors):
    return {name: {"shape": list(t.shape), "stride": list(t.stride()),
                   "offset": t.storage_offset(), "dtype": str(t.dtype)}
            for name, t in zip(("q", "k", "v"), tensors)}


def reference64(tensors, causal, scale=None, rows=None):
    """Independent CPU FP64 formula; no SDPA or online-softmax implementation."""
    q, k, v = (t.detach().to(device="cpu", dtype=torch.float64) for t in tensors)
    positions = torch.arange(q.shape[-2]) if rows is None else rows
    if rows is not None:
        q = q.index_select(2, rows)
    scores = q @ k.transpose(-2, -1)
    scores *= 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    if causal:
        scores.masked_fill_(torch.arange(k.shape[-2])[None, :] > positions[:, None], -float("inf"))
    return torch.softmax(scores, dim=-1) @ v


def errors(actual, reference, tolerance, rows=None):
    value = actual.detach().to(device="cpu", dtype=torch.float64)
    if rows is not None:
        value = value.index_select(2, rows)
    finite = bool(torch.isfinite(value).all() and torch.isfinite(reference).all())
    delta = value - reference
    return {"finite": finite,
            "passed": finite and bool(torch.allclose(value, reference, **tolerance)),
            "max_abs": delta.abs().max().item() if finite else None,
            "rmse": delta.square().mean().sqrt().item() if finite else None}


def timed(fn, iterations):
    begin_event, end_event = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    torch.cuda.synchronize()
    begin = time.perf_counter()
    begin_event.record()
    for _ in range(iterations):
        value = fn()
    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter() - begin) * 1000 / iterations
    del value
    return {"wall_ms": wall_ms, "event_ms": begin_event.elapsed_time(end_event) / iterations}


def peak_memory(fn):
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    value = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    result = {"baseline_allocated_bytes": baseline, "peak_allocated_bytes": peak,
              "incremental_peak_bytes": peak - baseline,
              "output_bytes": value.numel() * value.element_size()}
    del value
    return result


def measure(functions, cfg):
    for fn in functions.values():
        for _ in range(cfg.warmup):
            value = fn()
            del value
    torch.cuda.synchronize()
    samples = {name: [] for name in functions}
    orders = []
    for trial in range(cfg.rounds):
        order = list(functions) if trial % 2 == 0 else list(reversed(functions))
        orders.append(order)
        for name in order:
            samples[name].append(timed(functions[name], cfg.iterations))
    result = {}
    for name, fn in functions.items():
        result[name] = {"samples": samples[name],
                        "median": {metric: statistics.median(s[metric] for s in samples[name])
                                   for metric in ("wall_ms", "event_ms")},
                        "memory": peak_memory(fn)}
    if "native_math" in result:
        native = result["native_math"]
        for name, row in result.items():
            if name != "native_math":
                row["speedup_wall"] = native["median"]["wall_ms"] / row["median"]["wall_ms"]
                row["peak_allocated_ratio_native_over_tiled"] = (
                    native["memory"]["incremental_peak_bytes"] /
                    max(1, row["memory"]["incremental_peak_bytes"]))
    return {"round_order": orders, "methods": result}


def correctness(cfg, dtype, compute_dtype, tolerance):
    cases = [(1, 2, 1, 1, 64, False, None),
             (1, 2, 17, 17, 64, True, None),
             (1, 2, 31, 47, 64, False, -0.125),
             (1, 2, 31, 47, 64, True, None),
             (1, 2, 33, 17, 128, False, 0.0),
             (1, 2, 33, 17, 128, True, None),
             (2, 2, 63, 63, 128, False, None),
             (1, 2, 65, 65, 128, True, None)]
    results = []
    for batch, heads, length, keys, dim, causal, scale in cases:
        tensors = make_inputs(batch, heads, length, keys, dim, dtype, cfg.layout)
        reference = reference64(tensors, causal, scale)
        for block_q, block_k in cfg.tiles:
            row = {"shape": [batch, heads, length, dim], "key_length": keys,
                   "causal": causal, "scale": scale, "tile": [block_q, block_k]}
            try:
                actual = tiled_attention(*tensors, is_causal=causal, scale=scale,
                                         block_q=block_q, block_k=block_k,
                                         compute_dtype=compute_dtype)
                row.update(errors(actual, reference, tolerance))
                del actual
            except Exception as exc:
                row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
            results.append(row)
            if not row["passed"]:
                return results
    return results


def bench_case(length, dim, causal, cfg, dtype, compute_dtype, tolerance):
    tensors = make_inputs(cfg.batch, cfg.heads, length, length, dim, dtype, cfg.layout)
    rows = torch.linspace(0, length - 1, min(8, length)).round().long().unique()
    sampled_reference = reference64(tensors, causal, rows=rows)
    row = {"length": length, "head_dim": dim, "causal": causal,
           "layout": layouts(tensors), "fp64_query_rows": rows.tolist(), "accuracy": {}}
    functions = {}

    def native():
        with sdpa_kernel([SDPBackend.MATH]):
            return F.scaled_dot_product_attention(*tensors, is_causal=causal)

    # Conservative planning bound, not a measured allocation. Native MATH
    # materializes scores, probabilities and its safe-softmax intermediates.
    score_bytes = cfg.batch * cfg.heads * length * length * 4
    estimated = 4 * score_bytes + (5 * length * length if causal else 0)
    estimated += 4 * sum(t.numel() for t in tensors)
    free, _ = torch.cuda.mem_get_info()
    available = free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    budget = min(int(cfg.max_native_gib * 2**30), int(0.8 * available))
    row["native_budget"] = {"estimated_peak_bytes": estimated, "budget_bytes": budget,
                            "one_fp32_score_matrix_bytes": score_bytes}
    native_reference = None
    if estimated <= budget:
        try:
            native_value = native()
            native_reference = native_value.detach().to(device="cpu", dtype=torch.float64)
            row["native_status"] = "executed"
            row["accuracy"]["native_math_vs_fp64_rows"] = errors(native_value, sampled_reference, tolerance, rows)
            del native_value
            functions["native_math"] = native
        except torch.cuda.OutOfMemoryError as exc:
            row["native_status"] = "out_of_memory"
            row["native_error"] = str(exc)
            # Recover only after a real OOM; steady-state rounds do not flush
            # the allocator cache or change the other method's memory policy.
            torch.cuda.empty_cache()
    else:
        row["native_status"] = "budget_skipped"

    for block_q, block_k in cfg.tiles:
        name = f"tiled_{block_q}x{block_k}"

        def candidate(bq=block_q, bk=block_k):
            return tiled_attention(*tensors, is_causal=causal,
                                   block_q=bq, block_k=bk, compute_dtype=compute_dtype)

        try:
            actual = candidate()
            accuracy = {"vs_fp64_rows": errors(actual, sampled_reference, tolerance, rows)}
            if native_reference is not None:
                accuracy["vs_native_math"] = errors(actual, native_reference, tolerance)
            row["accuracy"][name] = accuracy
            del actual
            if all(value["passed"] for value in accuracy.values()):
                functions[name] = candidate
        except Exception as exc:
            row["accuracy"][name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    valid_tiled = [name for name in functions if name != "native_math"]
    row["passed"] = len(valid_tiled) == len(cfg.tiles)
    if valid_tiled:
        row.update(measure(functions, cfg))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--only", choices=("all", "correctness", "bench"), default="all")
    ap.add_argument("--lengths", nargs="+", type=int, default=[4097, 8192, 16384])
    ap.add_argument("--head-dims", nargs="+", type=int, default=[64, 128])
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--causal", choices=("both", "yes", "no"), default="both")
    ap.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    ap.add_argument("--compute-dtype", choices=("input", "fp16", "fp32"), default="input")
    ap.add_argument("--layout", choices=("interleaved", "contiguous", "packed_v"), default="interleaved")
    ap.add_argument("--tiles", nargs="+", default=["16x32"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-native-gib", type=float, default=10.0)
    cfg = ap.parse_args()
    if cfg.output.exists():
        ap.error("choose a new --output path; existing evidence is not overwritten")
    try:
        cfg.tiles = [tuple(int(n) for n in value.split("x")) for value in cfg.tiles]
    except ValueError:
        ap.error("tiles must be 16x32, 32x32 or 32x64")
    if not cfg.tiles or len(set(cfg.tiles)) != len(cfg.tiles) or any(t not in TILES for t in cfg.tiles):
        ap.error("use one to three distinct supported tiles")
    if min(cfg.batch, cfg.heads, cfg.rounds, cfg.iterations, cfg.warmup, *cfg.lengths) < 1:
        ap.error("shape and iteration arguments must be positive")
    if any(dim not in (64, 128) for dim in cfg.head_dims) or not math.isfinite(cfg.max_native_gib) or cfg.max_native_gib < 0:
        ap.error("head dimensions must be 64/128 and native budget finite/nonnegative")
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[cfg.dtype]
    compute_dtype = dtype if cfg.compute_dtype == "input" else {"fp16": torch.float16, "fp32": torch.float32}[cfg.compute_dtype]
    tolerance = TOLERANCES["fp32" if dtype == compute_dtype == torch.float32 else "fp16_operand_or_output"]
    report = {"schema_version": 1, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "seed": 20261106, "input_dtype": str(dtype), "compute_dtype": str(compute_dtype),
              "explicit_input_downcast": dtype == torch.float32 and compute_dtype == torch.float16,
              "tolerance_fixed_before_execution": tolerance, "tiles": cfg.tiles,
              "rounds": cfg.rounds, "iterations": cfg.iterations, "warmup": cfg.warmup,
              "native_math_reduced_precision": False,
              "memory_metric": "PyTorch peak allocated minus resident baseline; excludes non-allocator driver scratch.",
              "algorithm": "Online softmax; FP32 statistics/output accumulator; no global LxS score tensor.",
              "timing": "Complete callable, alternating warm rounds; no graphs; no compile/warmup/input creation.",
              "correctness": [], "benchmarks": [], "status": "started"}

    def save():
        cfg.output.parent.mkdir(parents=True, exist_ok=True)
        cfg.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    previous = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    code = 1
    try:
        if not torch.cuda.is_available() or not torch.version.hip:
            raise RuntimeError("use the existing ROCm environment on gfx1030")
        torch.set_num_threads(2)
        torch.manual_seed(report["seed"])
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        report["environment"] = {"torch": torch.__version__, "torch_git": torch.version.git_version,
                                 "hip": torch.version.hip, "gpu": props.name,
                                 "arch": getattr(props, "gcnArchName", None),
                                 "previous_native_math_reduced_precision": previous}
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        save()
        with torch.inference_mode():
            if cfg.only != "bench":
                report["correctness"] = correctness(cfg, dtype, compute_dtype, tolerance)
                save()
                passed = all(row["passed"] for row in report["correctness"])
                print(json.dumps({"correctness_passed": passed, "cases": len(report["correctness"])}), flush=True)
                if not passed:
                    raise RuntimeError("small-shape correctness/compilation failed; see the final correctness row")
            if cfg.only != "correctness":
                causal_modes = [False, True] if cfg.causal == "both" else [cfg.causal == "yes"]
                for dim in cfg.head_dims:
                    for length in cfg.lengths:
                        for causal in causal_modes:
                            row = bench_case(length, dim, causal, cfg, dtype, compute_dtype, tolerance)
                            report["benchmarks"].append(row)
                            save()
                            print(json.dumps({"length": length, "dim": dim, "causal": causal,
                                              "passed": row["passed"], "native_status": row["native_status"],
                                              "methods": {name: {"median": value["median"],
                                                                  "peak_bytes": value["memory"]["incremental_peak_bytes"]}
                                                          for name, value in row.get("methods", {}).items()}}), flush=True)
        passed = all(row["passed"] for row in report["benchmarks"])
        report["status"] = "passed" if passed else "accuracy_failed"
        code = 0 if passed else 1
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(report["error"], flush=True)
    finally:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous)
        report["exit_status"] = code
        save()
        print(f"JSON: {cfg.output.resolve()}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
