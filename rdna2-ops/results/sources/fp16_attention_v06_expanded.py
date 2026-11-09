"""Explicit gfx1030 FP16 materialized Attention experiments.

Baseline manual_vt matches the existing Anima direct baddbmm(alpha)/softmax/
transposed-V recipe. Other methods isolate output-buffer reuse, native Q/K
pre-scaling, fused scale+packing, and an in-place Triton row softmax.
No global dispatch is modified. Q/K/V and scores must remain finite; inference
only, unmasked, no dropout/GQA. Final-model accuracy is a separate gate.
"""
import math
import torch
import triton
import triton.language as tl

METHODS = ("manual_nn", "manual_vt", "post_out_vt", "post_inplace_vt", "prescale_native",
           "prescale_vt", "fused_prescale_vt", "fused_prescale_inplace_vt",
           "prescale4d_native", "prescale4d_vt")


@triton.jit
def _pack_scaled(X, O, COUNT: tl.constexpr, H: tl.constexpr, L: tl.constexpr,
                 D: tl.constexpr, XB: tl.constexpr, XH: tl.constexpr,
                 XL: tl.constexpr, XD: tl.constexpr, SCALE: tl.constexpr,
                 BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = i % D
    row = (i // D) % L
    head = (i // (D * L)) % H
    batch = i // (D * L * H)
    x = tl.load(X + batch * XB + head * XH + row * XL + d * XD,
                mask=i < COUNT, other=0).to(tl.float32)
    tl.store(O + i, x * SCALE, mask=i < COUNT)


@triton.jit
def _softmax_inplace(S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(S + row * N + col, mask=col < N,
                other=-float("inf")).to(tl.float32)
    maximum = tl.max(x, 0)
    p = tl.exp(x - maximum)
    denom = tl.sum(p, 0)
    y = tl.where(maximum == -float("inf"), 0., p / denom)
    tl.store(S + row * N + col, y, mask=col < N)


def _pack(x, scale, kernels):
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    b, h, length, dim = x.shape
    kernel = _pack_scaled[(triton.cdiv(x.numel(), 1024),)](
        x, out, x.numel(), h, length, dim, *x.stride(), float(scale), 1024,
        num_warps=4, enable_fp_fusion=False)
    kernels.append(kernel)
    return out.reshape(b*h, length, dim)


def _validate(q, k, v, method, scale):
    if method not in METHODS:
        raise ValueError("Unknown FP16 Attention method: " + str(method))
    if any(not isinstance(t, torch.Tensor) or t.ndim != 4 for t in (q, k, v)):
        raise ValueError("Requires Tensor Q[B,H,L,D], K/V[B,H,S,D]")
    if any(t.dtype != torch.float16 or t.device != q.device for t in (q, k, v)):
        raise ValueError("Same-device FP16 Q/K/V required")
    if k.shape != v.shape or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Matching B/H/D, equal K/V required")
    if min(*q.shape, *k.shape) <= 0 or q.shape[-1] not in (64, 128):
        raise ValueError("Positive dimensions and D64/128 required")
    if q.device.type != "cuda" or not torch.version.hip:
        raise ValueError("HIP GPU required")
    if torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0] != "gfx1030":
        raise ValueError("Only gfx1030 has been measured")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise ValueError("Inference forward only")
    if (torch.is_autocast_enabled("cuda") or torch.compiler.is_compiling()
            or torch.jit.is_tracing() or torch.jit.is_scripting()
            or torch.cuda.is_current_stream_capturing()
            or torch._C._are_functorch_transforms_active()
            or getattr(torch.autograd.forward_ad, "_current_level", -1) >= 0):
        raise ValueError("Explicit eager inference only")
    if any(any(s < 0 for s in t.stride()) or
           sum((n-1)*s for n, s in zip(t.shape, t.stride())) >= 2**31
           for t in (q, k, v)):
        raise ValueError("Nonnegative signed-int32 address span required")
    if q.shape[0]*q.shape[1]*q.shape[-2]*k.shape[-2] >= 2**31:
        raise ValueError("Score index exceeds int32")
    if not math.isfinite(scale):
        raise ValueError("Finite scale required")


def fp16_attention(q, k, v, *, method="manual_vt", scale=None,
                   return_kernels=False):
    """Return FP16 BHLD output; conversion/packing allocations are per-call.

    prescale_native aims to preserve native reduced-precision math SDPA's
    arithmetic and layout. All other paths require explicit error validation.
    The in-place softmax owns its scores buffer and never overwrites inputs.
    """
    scale = q.shape[-1]**-.5 if scale is None else float(scale)
    _validate(q, k, v, method, scale)
    b, h, length, dim = q.shape
    keys = k.shape[-2]
    kernels = []
    if method.startswith("prescale4d_"):
        root = math.sqrt(abs(scale))
        scores = torch.matmul(q * (-root if scale < 0 else root),
                              k.transpose(-2, -1) * root)
        probability = torch.softmax(scores, dim=-1)
        if method == "prescale4d_native":
            from ._softmax import repair_all_negative_inf_rows_
            repair_all_negative_inf_rows_(scores, probability)
            values = v
        else:
            values = v.transpose(-2, -1).contiguous().transpose(-2, -1)
        result = torch.matmul(probability, values)
        return (result, kernels) if return_kernels else result
    prescale = "prescale" in method
    if prescale:
        root = math.sqrt(abs(scale))
        qs = -root if scale < 0 else root
        if method.startswith("fused_"):
            q3, k3 = _pack(q, qs, kernels), _pack(k, root, kernels)
        else:
            q3 = (q * qs).reshape(b*h, length, dim)
            k3 = (k * root).reshape(b*h, keys, dim)
        scores = torch.bmm(q3, k3.transpose(1, 2))
    else:
        q3 = q.contiguous().reshape(b*h, length, dim)
        k3 = k.contiguous().reshape(b*h, keys, dim)
        buffer = torch.empty((b*h, length, keys), device=q.device, dtype=q.dtype)
        if method in ("manual_nn", "manual_vt"):
            scores = torch.baddbmm(buffer, q3, k3.transpose(1, 2), beta=0, alpha=scale)
        else:
            scores = torch.baddbmm(buffer, q3, k3.transpose(1, 2), beta=0,
                                   alpha=scale, out=buffer)
        del buffer
    v3 = v.reshape(b*h, keys, dim)
    if "inplace" in method:
        block = triton.next_power_of_2(keys)
        if block > 16384:
            raise ValueError("In-place row softmax measured only up to 16384 keys")
        kernels.append(_softmax_inplace[(b*h*length,)](
            scores, keys, block, num_warps=4 if block <= 4096 else 8,
            enable_fp_fusion=False))
        probability = scores
    else:
        probability = torch.softmax(scores, dim=-1)
        if method == "prescale_native":
            from ._softmax import repair_all_negative_inf_rows_
            repair_all_negative_inf_rows_(scores, probability)
    if method not in ("prescale_native", "manual_nn"):
        v3 = v3.transpose(1, 2).contiguous().transpose(1, 2)
    result = torch.bmm(probability, v3).reshape(b, h, length, dim)
    return (result, kernels) if return_kernels else result
