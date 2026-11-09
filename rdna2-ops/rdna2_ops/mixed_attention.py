"""Explicit FP16 GEMM islands with FP32 attention/quantization boundaries.

This diagnostic keeps Q/K prescaling, softmax, output storage, and the surrounding
model FP32. Only chosen BMM operands are converted to FP16; BMM accumulates/stores
FP32 using PyTorch's out_dtype contract. Upcasting a half OUTPUT is never used.
The optional three-product split is a separate approximate arithmetic candidate.
It is NOT a production automatic-dispatch path and still materializes scores.
"""
from contextlib import contextmanager
import math
import torch


def _gemm(a, b, mode):
    if mode == "float":
        return torch.bmm(a, b)
    ah, bh = a.half(), b.half()
    hh = torch.bmm(ah, bh, out_dtype=torch.float32)
    if mode == "half":
        return hh
    if mode != "split":
        raise ValueError(mode)
    # Scale residuals to retain subnormal-sized remainders in the half format.
    al = ((a - ah.float()) * 4096.0).half()
    bl = ((b - bh.float()) * 4096.0).half()
    cross = torch.bmm(ah, bl, out_dtype=torch.float32)
    cross = cross + torch.bmm(al, bh, out_dtype=torch.float32)
    return hh + cross * (1.0 / 4096.0)


def mixed_attention(q, k, v, *, qk="float", pv="float", scale=None):
    if any(t.dtype != torch.float32 or t.ndim != 4 for t in (q, k, v)):
        raise ValueError("FP32 BHSD input required")
    if k.shape[:2] != q.shape[:2] or v.shape != k.shape or k.shape[-1] != q.shape[-1]:
        raise ValueError("Q/K/V shape mismatch")
    b, h, length, dim = q.shape
    scale = dim ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("Finite scale required")
    root = math.sqrt(abs(scale))
    # Match math SDPA's FP32 Q/K prescaling before each selected GEMM island.
    qs = (q * (-root if scale < 0 else root)).reshape(b*h, length, dim)
    ks = (k * root).reshape(b*h, k.shape[-2], dim).transpose(1, 2)
    scores = _gemm(qs, ks, qk)
    probabilities = scores.softmax(dim=-1)
    probabilities = torch.where(torch.isneginf(scores).all(dim=-1, keepdim=True), 0.0, probabilities)
    result = _gemm(probabilities, v.reshape(b*h, v.shape[-2], dim), pv)
    if result.dtype != torch.float32:
        raise RuntimeError("FP32 output boundary was not preserved")
    return result.reshape(b, h, length, dim)


@contextmanager
def protected_gemm_islands(owner, *, qk="float", pv="float", scope="long", boundary_refs=None, record_reference=False):
    """Replace the owned process's SDPA callable; retain every INT8 boundary FP32.

    The original model continues to run norm, RoPE, ConvRot, quantization,
    dequantization, and residual updates in FP32. A quantizer input check records
    this actual invariant. The context cannot restore values lost inside an island.
    """
    from comfy_kitchen.backends.eager import quantization
    original = owner.scaled_dot_product_attention
    original_quant = quantization.quantize_int8_rowwise
    flag = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    stats = {"qk": qk, "pv": pv, "scope": scope, "hits": 0, "fallbacks": 0,
             "quantizer_calls": 0, "quantizer_non_fp32": 0,
             "output_dtype": "torch.float32", "full_score_storage": True,
             "boundary_probes": [], "timing_includes_two_boundary_probes": boundary_refs is not None}

    def quantize(x, *a, **kw):
        index = stats["quantizer_calls"]
        stats["quantizer_calls"] += 1
        stats["quantizer_non_fp32"] += int(x.dtype != torch.float32)
        if x.dtype != torch.float32:
            raise RuntimeError("Quantizer input boundary became non-FP32")
        with torch.autocast("cuda", enabled=False):
            result = original_quant(x, *a, **kw)
        # Two bounded native references cover the first QKV and output projection
        # boundaries. These CPU probes are diagnostic, not a speed measurement.
        if boundary_refs is not None and index < 2:
            current = {"input": x.detach().cpu(), "codes": result[0].detach().cpu(),
                       "scales": result[1].detach().cpu()}
            if record_reference:
                boundary_refs[index] = current
            elif index in boundary_refs:
                ref = boundary_refs[index]
                if ref["input"].shape != current["input"].shape:
                    raise RuntimeError("Quantizer probe alignment changed")
                dx = current["input"].double() - ref["input"].double()
                dc = current["codes"].to(torch.int16) - ref["codes"].to(torch.int16)
                dq = (current["codes"].float()*current["scales"].float()).double() - (ref["codes"].float()*ref["scales"].float()).double()
                rms = dx.square().mean().sqrt().item()
                qrms = dq.square().mean().sqrt().item()
                stats["boundary_probes"].append({"index": index, "shape": list(x.shape),
                    "input_rmse": rms, "input_max_abs": dx.abs().max().item(),
                    "changed_codes": int(torch.count_nonzero(dc)),
                    "changed_rows": int(torch.count_nonzero(dc.any(dim=-1))),
                    "scale_max_abs": (current["scales"]-ref["scales"]).abs().max().item(),
                    "dequant_rmse": qrms, "dequant_max_abs": dq.abs().max().item(),
                    "local_rms_amplification": qrms/rms if rms else None})
        return result

    def attention(q, k, v, *a, **kw):
        names = ("attn_mask", "dropout_p", "is_causal")
        opts = dict(zip(names, (None, 0.0, False)))
        if len(a) > 3 or set(kw) - set(names) - {"scale", "enable_gqa"}:
            stats["fallbacks"] += 1
            return original(q, k, v, *a, **kw)
        opts.update(zip(names, a)); opts.update(kw)
        if (opts["attn_mask"] is not None or opts["dropout_p"] != 0.0 or opts["is_causal"]
                or opts.get("enable_gqa", False) or q.dtype != torch.float32
                or (scope == "long" and q.shape[-2] < 128)):
            stats["fallbacks"] += 1
            return original(q, k, v, *a, **kw)
        with torch.autocast("cuda", enabled=False):
            out = mixed_attention(q, k, v, qk=qk, pv=pv, scale=opts.get("scale"))
        stats["hits"] += 1
        return out

    owner.scaled_dot_product_attention = attention
    quantization.quantize_int8_rowwise = quantize
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    try:
        yield stats
    finally:
        owner.scaled_dot_product_attention = original
        quantization.quantize_int8_rowwise = original_quant
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = flag
