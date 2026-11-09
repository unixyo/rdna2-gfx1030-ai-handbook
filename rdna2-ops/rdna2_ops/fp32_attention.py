"""Explicit gfx1030 FP32 streaming attention experiments.

The public function never changes Comfy/PyTorch dispatch. Query/key dot, softmax,
PV dot and the output all stay FP32. No complete L x S score tensor is allocated.
Output-channel tiling trades repeated QK work for fewer live registers; fixed
split-K combines independent online-softmax partials in a second kernel.
These are measured candidates, not a promise that any tile beats rocBLAS.
"""
from __future__ import annotations

import math
import numbers

import torch
import triton
import triton.language as tl


@triton.jit
def _fp32_stream(
    Q, K, V, O, Partials, Stats,
    L: tl.constexpr, S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    QB: tl.constexpr, QH: tl.constexpr, QS: tl.constexpr,
    KB: tl.constexpr, KH: tl.constexpr, KS: tl.constexpr,
    VB: tl.constexpr, VH: tl.constexpr, VS: tl.constexpr,
    SCALE: tl.constexpr, Q_SCALE: tl.constexpr, K_SCALE: tl.constexpr,
    PRE_SCALE: tl.constexpr, CAUSAL: tl.constexpr, UNROLL_D: tl.constexpr,
    BQ: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr, BP: tl.constexpr,
    SPLITS: tl.constexpr, KEYS_PER_SPLIT: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
):
    qt, bh, part_v = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if SPLITS == 1:
        vtile = part_v
        part = 0
    else:
        vtile = part_v % (D // BV)
        part = part_v // (D // BV)
    b, h = bh // H, bh % H
    qr = qt * BQ + tl.arange(0, BQ)
    dk = tl.arange(0, BD)
    dv = vtile * BV + tl.arange(0, BV)
    kc = tl.arange(0, BK)
    mx = tl.full((BQ,), -float("inf"), tl.float32)
    den = tl.zeros((BQ,), tl.float32)
    num = tl.zeros((BQ, BV), tl.float32)
    first = part * KEYS_PER_SPLIT
    stop = tl.minimum(first + KEYS_PER_SPLIT, S)
    if CAUSAL:
        stop = tl.minimum(stop, (qt + 1) * BQ)

    for begin in range(first, stop, BK):
        kr = begin + kc
        scores = tl.zeros((BQ, BK), tl.float32)
        # Shorter K reduction operands lower their live footprint. The same
        # FP32 accumulator spans all D chunks; no operand is rounded to FP16.
        if UNROLL_D:
            for dc in tl.static_range(D // BD):
                dims = dc * BD + dk
                q = tl.load(Q + b * QB + h * QH + qr[:, None] * QS + dims[None, :],
                            mask=qr[:, None] < L, other=0.0)
                k = tl.load(K + b * KB + h * KH + dims[:, None] + kr[None, :] * KS,
                            mask=kr[None, :] < S, other=0.0)
                if PRE_SCALE:
                    q = q * Q_SCALE
                    k = k * K_SCALE
                scores = tl.dot(q, k, scores, input_precision="ieee")
        else:
            # A real inner loop prevents the complete D128 operands from
            # being kept live across the outer K/V streaming loop.
            for dc in tl.range(0, D, BD, loop_unroll_factor=1):
                dims = dc + dk
                q = tl.load(Q + b * QB + h * QH + qr[:, None] * QS + dims[None, :],
                            mask=qr[:, None] < L, other=0.0)
                k = tl.load(K + b * KB + h * KH + dims[:, None] + kr[None, :] * KS,
                            mask=kr[None, :] < S, other=0.0)
                if PRE_SCALE:
                    q = q * Q_SCALE
                    k = k * K_SCALE
                scores = tl.dot(q, k, scores, input_precision="ieee")
        if not PRE_SCALE:
            scores = scores * SCALE
        visible = kr[None, :] < S
        if CAUSAL:
            visible = visible & (kr[None, :] <= qr[:, None])
        scores = tl.where(visible, scores, -float("inf"))
        new_mx = tl.maximum(mx, tl.max(scores, 1))
        # A causal split can contain no visible keys. Keep it a neutral partial.
        exponent_mx = tl.where(new_mx == -float("inf"), 0.0, new_mx)
        old_weight = tl.exp(mx - exponent_mx)
        p = tl.exp(scores - exponent_mx[:, None])
        den = den * old_weight + tl.sum(p, 1)
        num = num * old_weight[:, None]
        if BP == BK:
            v = tl.load(V + b * VB + h * VH + kr[:, None] * VS + dv[None, :],
                        mask=kr[:, None] < S, other=0.0)
            num = tl.dot(p, v, num, input_precision="ieee")
        else:
            for pv_begin in tl.range(0, BK, BP, loop_unroll_factor=1):
                pi = pv_begin + tl.arange(0, BP)
                take = tl.broadcast_to(pi[None, :], (BQ, BP))
                p_chunk = tl.gather(p, take, 1)
                v_rows = begin + pi
                v = tl.load(V + b * VB + h * VH + v_rows[:, None] * VS + dv[None, :],
                            mask=v_rows[:, None] < S, other=0.0)
                num = tl.dot(p_chunk, v, num, input_precision="ieee")
        mx = new_mx

    out_index = (bh * L + qr[:, None]) * D + dv[None, :]
    if SPLITS == 1:
        tl.store(O + out_index, num / den[:, None], mask=qr[:, None] < L)
    else:
        tl.store(Partials + part * TOTAL_ROWS * D + out_index, num,
                 mask=qr[:, None] < L)
        if vtile == 0:
            st = part * (2 * TOTAL_ROWS) + bh * L + qr
            tl.store(Stats + st, mx, mask=qr < L)
            tl.store(Stats + st + TOTAL_ROWS, den, mask=qr < L)


@triton.jit
def _merge_partials(Partials, Stats, O, ROWS: tl.constexpr, D: tl.constexpr,
                    SPLITS: tl.constexpr):
    row = tl.program_id(0)
    part = tl.arange(0, SPLITS)
    dims = tl.arange(0, D)
    mx = tl.load(Stats + part * (2 * ROWS) + row)
    den = tl.load(Stats + part * (2 * ROWS) + ROWS + row)
    weight = tl.exp(mx - tl.max(mx, 0))
    value = tl.load(Partials + part[:, None] * ROWS * D + row * D + dims[None, :])
    numerator = tl.sum(value * weight[:, None], 0)
    denominator = tl.sum(den * weight, 0)
    tl.store(O + row * D + dims, numerator / denominator)


def fp32_attention(q, k, v, *, scale=None, scale_mode="post",
                   is_causal=False, block_q=16, block_k=32, block_d=128,
                   block_v=128, num_warps=8, split_k=1, unroll_d=True,
                   block_pv=0, return_kernels=False):
    """Forward-only FP32 BHLD attention with linear storage for fixed split_k.

    Q[B,H,L,D], K/V[B,H,S,D], D64/128, positive lengths and channel stride1.
    Nonnegative outer strides and storage offsets are read without copies.
    Output is contiguous BHLD FP32. `post` scales the QK result (Comfy split);
    `sdpa` multiplies Q and K by sqrt(abs(scale)) before the dot (math SDPA).
    Both modes change reduction order relative to BLAS/native softmax, so final
    workload quality must be measured. No masks, dropout, GQA or backward.
    `is_causal` uses upper-left alignment, not a KV-cache position offset.

    No full score/probability storage: split_k1 allocates only output;
    split_k>1 additionally allocates split_k*B*H*L*(D+2) FP32 elements.
    D, split_k and tile dimensions are bounded independent of sequence length.
    Finite inputs and non-overflowing dot products are required.
    """
    if torch.compiler.is_compiling() or torch.jit.is_tracing() or torch.jit.is_scripting():
        raise ValueError("Explicit eager FP32 inference only")
    if torch._C._are_functorch_transforms_active() or getattr(torch.autograd.forward_ad, "_current_level", -1) >= 0:
        raise ValueError("Automatic differentiation is not implemented")
    ts = (q, k, v)
    if any(not isinstance(t, torch.Tensor) or t.ndim != 4 for t in ts):
        raise ValueError("Expected four-dimensional BHLD tensors")
    if any(t.dtype != torch.float32 for t in ts):
        raise ValueError("Q/K/V must remain FP32")
    if q.device.type != "cuda" or not torch.version.hip or any(t.device != q.device for t in ts):
        raise ValueError("Q/K/V must share a ROCm GPU")
    if min(*q.shape, *k.shape, *v.shape) <= 0 or q.shape[-1] not in (64, 128):
        raise ValueError("Positive dimensions, D64/128 required")
    if k.shape[:2] != q.shape[:2] or v.shape != k.shape or k.shape[-1] != q.shape[-1]:
        raise ValueError("Q/K/V must share B/H/D and K/V shape")
    if any(t.stride(-1) != 1 or any(s < 0 for s in t.stride()) for t in ts):
        raise ValueError("Unit channel stride and nonnegative outer strides required")
    if any(sum((n-1)*s for n,s in zip(t.shape,t.stride())) >= 2**31 for t in ts):
        raise ValueError("Address spans must fit signed 32-bit element offsets")
    if torch.is_grad_enabled() and any(t.requires_grad for t in ts):
        raise ValueError("Use no_grad/inference_mode; backward is not implemented")
    if torch.is_autocast_enabled("cuda"):
        raise ValueError("Disable autocast; this operator stays FP32")
    if type(is_causal) is not bool or type(unroll_d) is not bool or scale_mode not in ("post", "sdpa"):
        raise ValueError("is_causal/unroll_d must be bool, scale_mode post or sdpa")
    if (block_q not in (16,32,64) or block_k not in (16,32,64)
            or block_d not in (16,32,64,128) or block_d > q.shape[-1]
            or block_v not in (32,64,128) or block_v > q.shape[-1]
            or block_pv not in (0,16,32) or block_pv > block_k
            or num_warps not in (4,8,16) or split_k not in (1,2,4,8)):
        raise ValueError("Unsupported bounded tile or split count")
    sm = 1 / math.sqrt(q.shape[-1]) if scale is None else scale
    if not isinstance(sm, numbers.Real) or not math.isfinite(sm) or abs(sm) > torch.finfo(torch.float32).max:
        raise ValueError("Finite FP32-range scale required")
    b, h, length, dim = q.shape
    if b*h > 65535 or q.numel() * split_k >= 2**31:
        raise ValueError("Output/partial indexing exceeds supported bounds")
    root = math.sqrt(abs(sm))
    with torch.cuda.device(q.device):
        if torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0] != "gfx1030":
            raise ValueError("This experiment is restricted to gfx1030")
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("Graph capture is not an implemented contract")
        out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
        if split_k > 1:
            partials = torch.empty((split_k, b, h, length, dim), device=q.device, dtype=torch.float32)
            stats = torch.empty((split_k, 2, b*h*length), device=q.device, dtype=torch.float32)
        else:
            partials = stats = out  # constexpr branch never reads these aliases.
        span = triton.cdiv(k.shape[-2], block_k*split_k) * block_k
        stream = _fp32_stream[(triton.cdiv(length,block_q), b*h, split_k*(dim//block_v))](
            q,k,v,out,partials,stats,length,k.shape[-2],h,dim,
            *q.stride()[:3],*k.stride()[:3],*v.stride()[:3],
            float(sm), -root if sm < 0 else root, root, scale_mode == "sdpa", is_causal,unroll_d,
            block_q,block_k,block_d,block_v,block_pv or block_k,split_k,span,b*h*length,
            num_warps=num_warps,num_stages=1)
        kernels = [stream]
        if split_k > 1:
            kernels.append(_merge_partials[(b*h*length,)](
                partials,stats,out,b*h*length,dim,split_k,num_warps=4,num_stages=1))
    return (out,kernels) if return_kernels else out
