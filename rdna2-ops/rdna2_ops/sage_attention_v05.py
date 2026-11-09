"""Explicit Sage-inspired gfx1030 attention for FP32 rectangular Q/K/V.

Independent extension of this project's v0.4 online-softmax prototype, not the
official SageAttention package. QK uses per-token symmetric INT8 + INT32 dot4;
softmax and output accumulators are FP32; PV operands are FP16 (dot2).
The FP32 input tensors are never modified, and the result is directly FP32.
Quantization, optional K mean removal and all allocations occur on each call.
No global Q_length x K_length score/probability tensor or cache is allocated.

Contract: finite inference-only gfx1030 FP32 [B,H,L,D] Q and [B,H,S,D] K/V,
D=64/128; unmasked noncausal attention without dropout/GQA/graph capture.
V must be representable in finite FP16 for this PV path. Callers validate their
actual dynamic range and final-model quality; this module changes no dispatch.
"""
import math
import torch
import triton
import triton.language as tl
from .quantized_attention import _quant_rows


@triton.jit
def _sage_rectangular(Q, K, V, QS, KS, O, L: tl.constexpr, S: tl.constexpr,
                      D: tl.constexpr, H: tl.constexpr,
                      VB: tl.constexpr, VH: tl.constexpr,
                      VS: tl.constexpr, VD: tl.constexpr, SM: tl.constexpr,
                      BQ: tl.constexpr, BK: tl.constexpr):
    qb, bh = tl.program_id(0), tl.program_id(1)
    rows = qb * BQ + tl.arange(0, BQ)
    cols0 = tl.arange(0, BK)
    dims = tl.arange(0, D)
    q = tl.load(Q + bh * L * D + rows[:, None] * D + dims[None, :],
                mask=rows[:, None] < L, other=0)
    qs = tl.load(QS + bh * L + rows, mask=rows < L, other=0)
    mx = tl.full((BQ,), -float("inf"), tl.float32)
    denom = tl.zeros((BQ,), tl.float32)
    numerator = tl.zeros((BQ, D), tl.float32)
    for begin in range(0, S, BK):
        cols = begin + cols0
        k = tl.load(K + bh * S * D + dims[:, None] + cols[None, :] * D,
                    mask=cols[None, :] < S, other=0)
        ks = tl.load(KS + bh * S + cols, mask=cols < S, other=0)
        integer_scores = tl.dot(q, k, out_dtype=tl.int32)
        scores = integer_scores.to(tl.float32) * (qs[:, None] * ks[None, :]) * SM
        scores = tl.where(cols[None, :] < S, scores, -float("inf"))
        next_mx = tl.maximum(mx, tl.max(scores, 1))
        alpha = tl.exp(mx - next_mx)
        p = tl.exp(scores - next_mx[:, None])
        denom = denom * alpha + tl.sum(p, 1)
        numerator = numerator * alpha[:, None]
        v = tl.load(V + (bh // H) * VB + (bh % H) * VH
                    + cols[:, None] * VS + dims[None, :] * VD,
                    mask=cols[:, None] < S, other=0)
        numerator += tl.dot(p.to(tl.float16), v.to(tl.float16),
                            out_dtype=tl.float32)
        mx = next_mx
    y = numerator / denom[:, None]
    tl.store(O + bh * L * D + rows[:, None] * D + dims[None, :], y,
             mask=rows[:, None] < L)


def prepare_qk(q, k, *, smooth_k=True):
    """Prepare both sequence lengths independently, preserving current inputs."""
    b, h, _, d = q.shape
    mean = (k.mean(dim=-2).contiguous() if smooth_k
            else torch.empty(1, device=q.device, dtype=torch.float32))
    outputs = []
    for x, center in ((q, False), (k, smooth_k)):
        length = x.shape[-2]
        packed = torch.empty(x.shape, device=x.device, dtype=torch.int8)
        scale = torch.empty((b, h, length), device=x.device, dtype=torch.float32)
        _quant_rows[(b * h * length,)](
            x, mean, packed, scale, h, length, d, *x.stride(), center,
            num_warps=4)
        outputs.extend((packed, scale))
    return outputs[0], outputs[2], outputs[1], outputs[3]


def _validate(q, k, v, block_q, block_k, num_warps):
    if any(t.ndim != 4 for t in (q, k, v)):
        raise ValueError("Requires Q[B,H,L,D], K/V[B,H,S,D]")
    if k.shape != v.shape or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Matching B/H/D and equal K/V shapes required; GQA unsupported")
    if q.shape[-1] not in (64, 128) or min(*q.shape, *k.shape) <= 0:
        raise ValueError("Positive dimensions and head_dim 64/128 required")
    if any(t.dtype != torch.float32 for t in (q, k, v)):
        raise TypeError("Explicit FP32 Q/K/V required")
    if q.device.type != "cuda" or any(t.device != q.device for t in (k, v)):
        raise ValueError("Q/K/V must share a HIP GPU")
    if not torch.version.hip or torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0] != "gfx1030":
        raise ValueError("This prototype targets gfx1030")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise ValueError("Inference forward only")
    if torch.is_autocast_enabled("cuda") or torch.compiler.is_compiling():
        raise ValueError("Explicit eager dtype required")
    if torch.jit.is_tracing() or torch.jit.is_scripting() or torch._C._are_functorch_transforms_active():
        raise ValueError("Eager inference only")
    if getattr(torch.autograd.forward_ad, "_current_level", -1) >= 0 or torch.cuda.is_current_stream_capturing():
        raise ValueError("No forward AD or graph capture")
    if q.shape[0] * q.shape[1] > 65535 or any(t.numel() >= 2**31 for t in (q, k, v)):
        raise ValueError("Tensor/grid exceeds supported indexing range")
    if any(sum((n-1)*s for n, s in zip(t.shape, t.stride())) >= 2**31
           or any(s < 0 for s in t.stride()) for t in (q, k, v)):
        raise ValueError("Nonnegative signed-32-bit address spans required")
    if block_q not in (16, 32) or block_k not in (32, 64) or num_warps not in (4, 8):
        raise ValueError("BQ16/32, BK32/64, warps4/8 required")


def sage_attention(q, k, v, *, smooth_k=True, scale=None, block_q=16,
                   block_k=32, num_warps=4, return_kernel=False):
    """INT8 QK + FP16 PV, both exposed at a directly stored FP32 boundary.

    The output is approximate despite its FP32 dtype. Final RGB acceptance is a
    separate whole-workflow measurement, never inferred from an operator gate.
    """
    _validate(q, k, v, block_q, block_k, num_warps)
    b, h, length, d = q.shape
    keys = k.shape[-2]
    sm = d**-0.5 if scale is None else float(scale)
    if not math.isfinite(sm):
        raise ValueError("Finite attention scale required")
    q8, k8, qs, ks = prepare_qk(q, k, smooth_k=smooth_k)
    out = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    kernel = _sage_rectangular[(triton.cdiv(length, block_q), b*h)](
        q8, k8, v, qs, ks, out, length, keys, d, h, *v.stride(), sm,
        block_q, block_k, num_warps=num_warps, num_stages=1)
    return (out, kernel) if return_kernel else out
