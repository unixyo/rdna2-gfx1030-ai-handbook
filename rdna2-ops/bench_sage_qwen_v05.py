"""CPU-prepared, explicit Sage-inspired Qwen benchmark; never installs a dispatch.

Run only with the research GPU lease. All candidate times include K centering,
row quantization, output allocation, and restoration to Qwen's [B,L,H*D] layout.
The optional FP32 comparator must be chosen from measured priority-2 results.
Operator acceptance cannot substitute for the final RGB workflow gate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import triton

from rdna2_ops.sage_attention_v05 import sage_attention
from rdna2_ops.fp32_attention import fp32_attention
from bench_fp32_attention import (
    TILES, kernel_info, load_capture, params, qwen_baselines, rocblas_query_tiled,
)
from bench_quantized_attention import measure_interleaved
from bench_tiled_attention import reference64


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def errors(value, reference):
    value = value.detach().to(device="cpu", dtype=torch.float64)
    reference = reference.detach().to(device="cpu", dtype=torch.float64)
    finite = bool(torch.isfinite(value).all() and torch.isfinite(reference).all())
    if not finite:
        return dict(finite=False, max_abs=None, rmse=None, relative_rmse=None,
                    cosine=None, operator_gate=False)
    delta = value - reference
    rmse = delta.square().mean().sqrt().item()
    norm = reference.square().mean().sqrt().item()
    denom = value.norm().item() * reference.norm().item()
    cosine = ((value * reference).sum().item() / denom if denom else
              float(bool(torch.equal(value, reference))))
    maximum = delta.abs().max().item()
    relative = rmse / max(norm, 1e-12)
    return dict(finite=True, max_abs=maximum, rmse=rmse,
                relative_rmse=relative, cosine=cosine,
                operator_gate=maximum <= .02 and relative <= .03 and cosine >= .999)


def small_correctness():
    # Rectangular/tail indexing newly differs from v0.4; one independent FP64
    # check covers B/H addressing, both supported D values, and partial tiles.
    checks = []
    torch.manual_seed(20261106)
    for batch, dim in ((1, 128), (2, 64)):
        q = torch.randn(batch, 2, 17, dim, device="cuda") * .25
        k = torch.randn(batch, 2, 37, dim, device="cuda") * .25
        v = torch.randn_like(k) * .25
        reference = reference64((q, k, v), False)
        for smooth in (False, True):
            y = sage_attention(q, k, v, smooth_k=smooth)
            checks.append(dict(shape_q=list(q.shape), shape_k=list(k.shape),
                               smooth_k=smooth, output_dtype=str(y.dtype),
                               accuracy=errors(y, reference)))
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captured", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--tiles", nargs="+", choices=("16x32", "32x32"), default=["16x32", "32x32"])
    ap.add_argument("--num-warps", type=int, choices=(4, 8), default=4)
    ap.add_argument("--fp32-tile", choices=tuple(TILES))
    ap.add_argument("--query-block", type=int)
    ap.add_argument("--fp32-evidence", type=Path)
    ap.add_argument("--qwen-runtime", type=Path,
                    default=Path("~/DevSpace/qwen-image-21-rdna2-lab01/runtime/ComfyUI"))
    ap.add_argument("--qwen-rehearsal", type=Path,
                    default=Path("~/DevSpace/rdna2-pro-research-20261102/rehearsal_qwen_20261106"))
    cfg = ap.parse_args()
    if cfg.repeat <= 0 or (cfg.query_block is not None and cfg.query_block <= 0):
        ap.error("Positive repeat/query block required")
    if cfg.output.exists():
        ap.error("Output exists; choose a new evidence path")
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        status="started", implementation="independent Sage-inspired port; not official SageAttention",
        input="actual captured Qwen FP32 Q/K/V, same data for every method",
        contract="unmasked noncausal D64/128; INT8 QK + FP16 PV, direct FP32 output",
        quantization="per-token absmax/127, clamp [-127,127], nearest ties away from zero; optional K mean removal",
        acceptance=dict(max_abs=.02, relative_rmse=.03, cosine_min=.999,
                        final_rgb_max=2/255, final_rgb_rmse=.25/255,
                        final_rgb_status="not measured in this operator benchmark"),
        timing="alternating forward/reverse order; includes validation, mean, quantization, allocation, layout restore; no graph",
        memory="separate phase after timing; peak allocated increment including all temporaries/output, excluding resident inputs/references",
        isa="static textual instruction counts in compiled main kernel, not dynamic executed-operation counts",
        repeat=cfg.repeat, methods={}, sources={}, comparison_selection={},
    )
    def save():
        cfg.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        report["runtime"] = dict(torch=torch.__version__, hip=torch.version.hip,
                                triton=triton.__version__,
                                device=torch.cuda.get_device_name(),
                                arch=torch.cuda.get_device_properties(0).gcnArchName,
                                matmul_precision=torch.get_float32_matmul_precision())
        root = Path(__file__).resolve().parent
        source_paths = [Path(__file__), root/"rdna2_ops/sage_attention_v05.py",
                        root/"rdna2_ops/quantized_attention.py",
                        root/"bench_fp32_attention.py", root/"bench_quantized_attention.py",
                        root/"bench_tiled_attention.py", root/"rdna2_ops/fp32_attention.py",
                        cfg.qwen_runtime/"comfy/ldm/modules/attention.py",
                        cfg.qwen_rehearsal/"scripts/candidates.py", cfg.captured]
        if cfg.fp32_evidence:
            source_paths.append(cfg.fp32_evidence)
        report["sources"] = {str(p): digest(p) for p in source_paths}
        report["small_rectangular_checks"] = small_correctness()
        save()
        if not all(r["accuracy"]["operator_gate"] for r in report["small_rectangular_checks"]):
            raise RuntimeError("New rectangular/tail correctness check failed")

        data, native_inputs = load_capture(cfg.captured)
        metadata = data.get("metadata", {})
        if metadata.get("mask") is not None:
            raise ValueError("Captured masks are unsupported")
        heads = metadata.get("heads")
        if heads != 32 or any(t.ndim != 3 or t.dtype != torch.float32 for t in native_inputs):
            raise ValueError("This driver requires actual Qwen H32 FP32 [B,L,H*D] capture")
        q3, k3, v3 = native_inputs
        q, k, v = [t.reshape(t.shape[0], t.shape[1], heads, -1).transpose(1, 2)
                   for t in native_inputs]
        scale = data.get("scale")
        if scale is not None and float(scale) != q.shape[-1]**-.5:
            raise ValueError("Native split reference here supports the captured default scale only")
        if not all(bool(torch.isfinite(t).all()) for t in (q, k, v)):
            raise ValueError("Actual input is nonfinite")
        maxima = [t.abs().max().item() for t in (q, k, v)]
        if maxima[2] > 65504:
            raise ValueError("Actual V would overflow FP16 PV operands")
        report["capture"] = dict(path=str(cfg.captured), metadata=metadata,
                                 shapes=[list(t.shape) for t in (q, k, v)],
                                 strides=[list(t.stride()) for t in (q, k, v)],
                                 maximum_abs_qkv=maxima, scale=scale,
                                 full_score_fp32_bytes=q.shape[0]*heads*q.shape[-2]*k.shape[-2]*4)
        split, split_alpha = qwen_baselines(cfg.qwen_runtime, cfg.qwen_rehearsal)
        functions = dict(attention_split=lambda: split(q3, k3, v3, heads),
                         split_alpha=lambda: split_alpha(q3, k3, v3, heads))
        def finish(y):
            return y.transpose(1, 2).reshape(q.shape[0], q.shape[-2], heads*q.shape[-1])
        def restore(y):
            return y.reshape(q.shape[0], q.shape[-2], heads, q.shape[-1]).transpose(1, 2)

        native_ref = restore(functions["attention_split"]()).detach().cpu()
        alpha_ref = restore(functions["split_alpha"]()).detach().cpu()
        rows = torch.linspace(0, q.shape[-2]-1, min(4, q.shape[-2])).round().long().unique()
        fp64 = reference64((q[:, :2], k[:, :2], v[:, :2]), False, rows=rows)
        report["fp64_scope"] = dict(heads=[0, 1], query_rows=rows.tolist())
        report["comparison_selection"] = dict(
            always=["attention_split", "split_alpha"],
            fp32_tile=cfg.fp32_tile, query_block=cfg.query_block,
            evidence=str(cfg.fp32_evidence) if cfg.fp32_evidence else None,
            note="An optional additional comparator is not called best without measured evidence.")
        if cfg.fp32_tile:
            functions["fp32_"+cfg.fp32_tile] = lambda: finish(
                fp32_attention(q, k, v, scale_mode="post", **params(cfg.fp32_tile)))
        if cfg.query_block:
            functions["rocblas_query_"+str(cfg.query_block)] = lambda: finish(
                rocblas_query_tiled(q, k, v, cfg.query_block, "post"))
        for name, fn in functions.items():
            y = restore(fn())
            report["methods"][name] = dict(
                status="available", vs_native=errors(y, native_ref),
                vs_split_alpha=errors(y, alpha_ref),
                vs_fp64_rows=errors(y[:, :2].index_select(2, rows.to(y.device)), fp64))
            del y
        save()
        for smooth in (True, False):
            for tile in cfg.tiles:
                bq, bk = map(int, tile.split("x"))
                name = f"sage_smooth{int(smooth)}_q{bq}k{bk}w{cfg.num_warps}"
                row = dict(status="started", smooth_k=smooth, block_q=bq, block_k=bk,
                           num_warps=cfg.num_warps)
                report["methods"][name] = row
                save()
                try:
                    y, compiled = sage_attention(q, k, v, smooth_k=smooth,
                        block_q=bq, block_k=bk, num_warps=cfg.num_warps, return_kernel=True)
                    row.update(status="available", output_dtype=str(y.dtype),
                               vs_native=errors(y, native_ref),
                               vs_split_alpha=errors(y, alpha_ref),
                               vs_fp64_rows=errors(y[:, :2].index_select(2, rows.to(y.device)), fp64),
                               kernel=kernel_info(compiled))
                    row["operator_gate"] = all(row[x]["operator_gate"]
                                              for x in ("vs_native", "vs_split_alpha", "vs_fp64_rows"))
                    if row["vs_native"]["finite"]:
                        functions[name] = lambda smooth=smooth, bq=bq, bk=bk: finish(
                            sage_attention(q, k, v, smooth_k=smooth, block_q=bq,
                                           block_k=bk, num_warps=cfg.num_warps))
                    del y
                except Exception as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                save()
        measurements, order = measure_interleaved(functions, cfg.repeat)
        report["timing_round_order"] = order
        for name, timing in measurements.items():
            row = report["methods"][name]
            row.update(status="measured", timing=timing,
                       speedup_wall={base: measurements[base]["median_wall_ms"]/timing["median_wall_ms"]
                                     for base in ("attention_split", "split_alpha")})
        report["status"] = "measured"
        save()
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    with torch.inference_mode():
        main()
