"""Experimental gfx1030 INT8 QK + streaming softmax + floating-point PV.

Independently implemented online-softmax recurrence. SageAttention motivates
centering K and quantizing QK; this prototype uses per-token symmetric INT8
scales, not Sage's per-block quantizer or NVIDIA tensor-core PV kernel.
No full score/probability matrix is allocated. No automatic dispatch is changed.
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _quant_rows(X, Mean, Out, Scale, H: tl.constexpr, S: tl.constexpr,
                D: tl.constexpr, XB: tl.constexpr, XH: tl.constexpr,
                XS: tl.constexpr, XD: tl.constexpr, CENTER: tl.constexpr):
    r = tl.program_id(0)
    d = tl.arange(0, D)
    b, h, s = r // (H * S), (r // S) % H, r % S
    x = tl.load(X + b * XB + h * XH + s * XS + d * XD).to(tl.float32)
    if CENTER:
        x -= tl.load(Mean + (b * H + h) * D + d)
    step = tl.maximum(tl.max(tl.abs(x), 0) / 127.0, 1.0e-20)
    # Symmetric nearest rounding, ties away from zero; no vendor intrinsics.
    scaled = x / step
    z = tl.where(scaled >= 0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5))
    z = tl.minimum(tl.maximum(z, -127.0), 127.0).to(tl.int8)
    tl.store(Out + r * D + d, z)
    tl.store(Scale + r, step)


@triton.jit
def _int8_stream(Q, K, V, QS, KS, O, S: tl.constexpr, D: tl.constexpr,
                 H: tl.constexpr, VB: tl.constexpr, VH: tl.constexpr,
                 VS: tl.constexpr, VD: tl.constexpr, SM: tl.constexpr,
                 BQ: tl.constexpr, BK: tl.constexpr, PV_HALF: tl.constexpr,
                 CAUSAL: tl.constexpr):
    qb, bh = tl.program_id(0), tl.program_id(1)
    rows = qb * BQ + tl.arange(0, BQ)
    cols0 = tl.arange(0, BK)
    dims = tl.arange(0, D)
    q = tl.load(Q + bh * S * D + rows[:, None] * D + dims[None, :],
                mask=rows[:, None] < S, other=0)
    qs = tl.load(QS + bh * S + rows, mask=rows < S, other=0)
    mx = tl.full((BQ,), -float("inf"), tl.float32)
    denom = tl.zeros((BQ,), tl.float32)
    numerator = tl.zeros((BQ, D), tl.float32)
    stop = S
    if CAUSAL:
        stop = tl.minimum(S, (qb + 1) * BQ)
    for begin in range(0, stop, BK):
        cols = begin + cols0
        k = tl.load(K + bh * S * D + dims[:, None] + cols[None, :] * D,
                    mask=cols[None, :] < S, other=0)
        ks = tl.load(KS + bh * S + cols, mask=cols < S, other=0)
        integer_scores = tl.dot(q, k, out_dtype=tl.int32)
        scores = integer_scores.to(tl.float32) * (qs[:, None] * ks[None, :]) * SM
        valid = cols[None, :] < S
        if CAUSAL:
            valid = valid & (cols[None, :] <= rows[:, None])
        scores = tl.where(valid, scores, -float("inf"))
        next_mx = tl.maximum(mx, tl.max(scores, 1))
        alpha = tl.exp(mx - next_mx)
        p = tl.exp(scores - next_mx[:, None])
        denom = denom * alpha + tl.sum(p, 1)
        numerator = numerator * alpha[:, None]
        v = tl.load(V + (bh // H) * VB + (bh % H) * VH
                    + cols[:, None] * VS + dims[None, :] * VD,
                    mask=cols[:, None] < S, other=0)
        if PV_HALF:
            pv = tl.dot(p.to(tl.float16), v.to(tl.float16), out_dtype=tl.float32)
        else:
            pv = tl.dot(p, v.to(tl.float32), input_precision="ieee", out_dtype=tl.float32)
        numerator += pv
        mx = next_mx
    y = numerator / denom[:, None]
    tl.store(O + bh * S * D + rows[:, None] * D + dims[None, :], y,
             mask=rows[:, None] < S)


def prepare_qk(q, k, *, smooth_k=True):
    """Quantize current tensors; never cache by pointer or mutate inputs."""
    b, h, s, d = q.shape
    mean = k.float().mean(dim=-2).contiguous() if smooth_k else torch.empty(1, device=q.device)
    outputs = []
    for x, center in ((q, False), (k, smooth_k)):
        out = torch.empty(x.shape, dtype=torch.int8, device=x.device)
        scale = torch.empty((b, h, s), dtype=torch.float32, device=x.device)
        _quant_rows[(b * h * s,)](x, mean, out, scale, h, s, d,
                                  *x.stride(), center, num_warps=4)
        outputs.extend((out, scale))
    return outputs[0], outputs[2], outputs[1], outputs[3]


def _validate(q, k, v, pv_dtype, block_q, block_k):
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Requires equal B,H,S,D Q/K/V shapes")
    if q.shape[-1] not in (64, 128) or min(q.shape) <= 0:
        raise ValueError("Positive dimensions and head_dim 64/128 required")
    if any(t.dtype not in (torch.float16, torch.float32) for t in (q, k, v)):
        raise TypeError("FP16/FP32 tensors required")
    if any(t.device != q.device for t in (k, v)) or q.device.type != "cuda":
        raise ValueError("Q/K/V must share a HIP GPU")
    if not torch.version.hip or torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0] != "gfx1030":
        raise ValueError("This experiment is validated only for gfx1030")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q,k,v)):
        raise ValueError("Forward inference only; no custom backward")
    if torch.is_autocast_enabled("cuda") or torch.compiler.is_compiling():
        raise ValueError("Explicit eager dtype required")
    if torch.jit.is_tracing() or torch.jit.is_scripting() or torch._C._are_functorch_transforms_active():
        raise ValueError("Eager inference only")
    if getattr(torch.autograd.forward_ad, "_current_level", -1) >= 0 or torch.cuda.is_current_stream_capturing():
        raise ValueError("No forward AD or graph capture")
    if q.numel() >= 2**31 or q.shape[0]*q.shape[1] > 65535:
        raise ValueError("Tensor/grid exceeds supported indexing range")
    if any(sum((n-1)*s for n,s in zip(t.shape,t.stride())) >= 2**31
           or any(s < 0 for s in t.stride()) for t in (q,k,v)):
        raise ValueError("Nonnegative, signed-32-bit address spans required")
    if pv_dtype not in ("fp16", "fp32") or block_q not in (16, 32) or block_k not in (32, 64):
        raise ValueError("PV fp16/fp32, BQ16/32 and BK32/64 required")


def quantized_attention(q, k, v, *, smooth_k=True, pv_dtype="fp32",
                        block_q=16, block_k=32, is_causal=False,
                        scale=None, return_kernel=False):
    """Approximate experimental attention; includes K centering and QK quantization.

    FP32 PV avoids a probability/value FP16 boundary. FP16 PV uses FP32
    accumulation, a different path from Sage's FP16 tensor-core accumulator.
    Inputs must be finite; callers must validate their workload's dynamic range.
    """
    _validate(q,k,v,pv_dtype,block_q,block_k)
    if is_causal and smooth_k:
        raise ValueError("Strict causal INT8 quantization requires smooth_k=False; a future-dependent mean changes rounding")
    q8,k8,qs,ks = prepare_qk(q,k,smooth_k=smooth_k)
    b,h,s,d = q.shape
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    sm = 1/math.sqrt(d) if scale is None else float(scale)
    if not math.isfinite(sm):
        raise ValueError("Finite attention scale required")
    compiled = _int8_stream[(triton.cdiv(s,block_q),b*h)](
        q8,k8,v,qs,ks,out,s,d,h,*v.stride(),sm,block_q,block_k,
        pv_dtype=="fp16",bool(is_causal),num_warps=4,num_stages=1)
    return (out,compiled) if return_kernel else out
