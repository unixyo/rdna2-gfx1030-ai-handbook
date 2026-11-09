"""Experimental forward-only online attention for gfx1030.

Explicit operator: this module does not alter PyTorch/Comfy automatic dispatch.
Only the output is allocated; each program keeps one query tile, a score tile,
row statistics and an output accumulator while streaming through K/V.

The online-softmax recurrence follows the FlashAttention algorithm. This is an
independent small implementation using ordinary Triton dot lowering, without
MFMA/WMMA intrinsics, descriptors, architecture overrides or autotuning.
Primary references:
https://arxiv.org/abs/2205.14135
https://triton-lang.org/main/python-api/generated/triton.language.dot.html

Unlike the existing exact math operator, this changes reduction order and is
not bitwise equivalent to native SDPA. Validate the consuming workload before
adoption. Nonfinite inputs and overflowing scores are outside this contract.
"""
from __future__ import annotations

import math
import numbers

import torch
import triton
import triton.language as tl


@triton.jit
def _tiled_forward(
    Q, K, V, O,
    query_length, key_length, heads,
    qb, qh, qs, kb, kh, ks, vb, vh, vs,
    scale_log2,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CAUSAL: tl.constexpr,
    HALF_COMPUTE: tl.constexpr,
):
    query_tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head % heads
    rows = query_tile * BLOCK_Q + tl.arange(0, BLOCK_Q)
    channels = tl.arange(0, HEAD_DIM)
    columns = tl.arange(0, BLOCK_K)

    query = tl.load(
        Q + batch * qb + head * qh
        + rows[:, None] * qs + channels[None, :],
        mask=rows[:, None] < query_length, other=0.0,
    )
    if HALF_COMPUTE:
        query = query.to(tl.float16)
    else:
        query = query.to(tl.float32)

    running_max = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_Q,), tl.float32)
    numerator = tl.zeros((BLOCK_Q, HEAD_DIM), tl.float32)
    last_key = key_length
    if CAUSAL:
        # Upper-left causal alignment, also for rectangular L x S attention.
        last_key = tl.minimum(last_key, (query_tile + 1) * BLOCK_Q)

    for first_key in range(0, last_key, BLOCK_K):
        key_rows = first_key + columns
        key = tl.load(
            K + batch * kb + head * kh
            + channels[:, None] + key_rows[None, :] * ks,
            mask=key_rows[None, :] < key_length, other=0.0,
        )
        if HALF_COMPUTE:
            key = key.to(tl.float16)
        else:
            key = key.to(tl.float32)

        logits = tl.dot(query, key, input_precision="ieee", out_dtype=tl.float32)
        logits = logits * scale_log2
        visible = key_rows[None, :] < key_length
        if CAUSAL:
            visible = visible & (key_rows[None, :] <= rows[:, None])
        logits = tl.where(visible, logits, -float("inf"))

        next_max = tl.maximum(running_max, tl.max(logits, axis=1))
        old_weight = tl.exp2(running_max - next_max)
        probability = tl.exp2(logits - next_max[:, None])
        denominator = denominator * old_weight + tl.sum(probability, axis=1)
        numerator = numerator * old_weight[:, None]

        value = tl.load(
            V + batch * vb + head * vh
            + key_rows[:, None] * vs + channels[None, :],
            mask=key_rows[:, None] < key_length, other=0.0,
        )
        if HALF_COMPUTE:
            value = value.to(tl.float16)
            probability_dot = probability.to(tl.float16)
        else:
            value = value.to(tl.float32)
            probability_dot = probability
        numerator = tl.dot(
            probability_dot, value, numerator,
            input_precision="ieee", out_dtype=tl.float32,
        )
        running_max = next_max

    output = numerator / denominator[:, None]
    output_offsets = ((batch_head * query_length + rows[:, None]) * HEAD_DIM
                      + channels[None, :])
    tl.store(O + output_offsets, output, mask=rows[:, None] < query_length)


def tiled_attention(q, k, v, *, is_causal=False, scale=None,
                    block_q=16, block_k=32, compute_dtype=None):
    """Forward attention without a global L x S score/probability tensor.

    Q: [B,H,L,D], K/V: [B,H,S,D], D in {64,128}; positive B/H/L/S,
    same device/dtype, innermost stride 1. Other nonnegative strides, including
    BSHD interleaving and packed-QKV views, are read directly. Output is a new
    contiguous [B,H,L,D] tensor in Q's dtype. Causal means key_index <=
    query_index (upper-left alignment, not a KV-cache position offset).

    compute_dtype=None uses the input dtype. FP16 computation converts Q/K/V
    and the probability tile to FP16; both dot accumulators, softmax statistics
    and the running output remain FP32. FP32 computation uses IEEE FP32 dot
    operands, including the probability tile. FP32 input is lowered to FP16
    only when compute_dtype=torch.float16 is explicitly supplied. Conversion
    happens per tile, without allocating full converted input tensors.

    This experimental gfx1030 operator requires eager inference, finite inputs
    and finite dot products. No mask, dropout, GQA, backward, compilation or
    graph-capture API is supplied. Unsupported metadata raises rather than
    silently allocating a quadratic native fallback. No speedup is promised.
    """
    if torch.compiler.is_compiling() or torch.jit.is_tracing() or torch.jit.is_scripting():
        raise ValueError("tiled_attention is an eager forward-only experiment")
    if torch._C._are_functorch_transforms_active() or getattr(torch.autograd.forward_ad, "_current_level", -1) >= 0:
        raise ValueError("tiled_attention does not implement automatic differentiation")
    tensors = (q, k, v)
    if any(not isinstance(t, torch.Tensor) or t.ndim != 4 for t in tensors):
        raise ValueError("Q/K/V must be four-dimensional BHLD tensors")
    if q.device.type != "cuda" or not torch.version.hip or any(t.device != q.device for t in tensors):
        raise ValueError("tiled_attention requires Q/K/V on the same ROCm device")
    if q.dtype not in (torch.float16, torch.float32) or any(t.dtype != q.dtype for t in tensors):
        raise ValueError("Q/K/V must share FP16 or FP32 dtype")
    if any(any(n <= 0 for n in t.shape) for t in tensors):
        raise ValueError("all dimensions must be positive")
    if (k.shape[:2] != q.shape[:2] or v.shape != k.shape
            or q.shape[-1] != k.shape[-1] or q.shape[-1] not in (64, 128)):
        raise ValueError("require shared B/H, K/V shape, and head dimension 64 or 128")
    if any(t.stride(-1) != 1 or any(s < 0 for s in t.stride()) for t in tensors):
        raise ValueError("require unit channel stride and nonnegative strides")
    # Pointer arithmetic remains signed 32-bit in this initial SIMD kernel.
    if any(sum((n - 1) * s for n, s in zip(t.shape, t.stride())) >= 2**31 for t in tensors):
        raise ValueError("input address span must fit signed 32-bit element offsets")
    if q.numel() >= 2**31 or q.shape[0] * q.shape[1] > 65535:
        raise ValueError("output indexing or launch grid exceeds the supported range")
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        raise ValueError("use torch.no_grad()/inference_mode; backward is not implemented")
    if torch.is_autocast_enabled("cuda"):
        raise ValueError("disable autocast and select input/compute dtype explicitly")
    if not isinstance(is_causal, bool):
        raise TypeError("is_causal must be bool")
    if (block_q, block_k) not in ((16, 32), (32, 32), (32, 64)):
        raise ValueError("supported tiles are (16,32), (32,32), (32,64)")
    selected_dtype = q.dtype if compute_dtype is None else compute_dtype
    if selected_dtype not in (torch.float16, torch.float32):
        raise ValueError("compute_dtype must be torch.float16 or torch.float32")
    actual_scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    if not isinstance(actual_scale, numbers.Real) or not math.isfinite(actual_scale):
        raise ValueError("scale must be a finite real number")
    if abs(actual_scale) * math.log2(math.e) > torch.finfo(torch.float32).max:
        raise ValueError("scale exceeds the FP32 softmax range")

    with torch.cuda.device(q.device):
        arch = torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0]
        if arch != "gfx1030":
            raise ValueError("this experimental kernel is restricted to gfx1030")
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("tiled_attention requires eager execution, not graph capture")
        output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
        _tiled_forward[(triton.cdiv(q.shape[-2], block_q), q.shape[0] * q.shape[1])](
            q, k, v, output, q.shape[-2], k.shape[-2], q.shape[1],
            *q.stride()[:3], *k.stride()[:3], *v.stride()[:3],
            float(actual_scale) * math.log2(math.e),
            HEAD_DIM=q.shape[-1], BLOCK_Q=block_q, BLOCK_K=block_k,
            CAUSAL=is_causal, HALF_COMPUTE=selected_dtype == torch.float16,
            num_warps=4, num_stages=1,
        )
    return output
