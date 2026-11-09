"""Research-only row normalization and AdaLN for gfx1030/Triton 3.8.

No global installation, compilation at import, implicit copies, or autograd.
LayerNorm uses centered FP32 variance. ``rounding='eager'`` retains dtype
boundaries in norm(x) * (1 + scale) + shift; ``'fp32'`` is a distinct numerical
candidate matching the arithmetic intent of a fused kitchen implementation.
The reduction order still differs from PyTorch: model A/B is mandatory.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_DTYPE = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16,
          torch.float32: tl.float32}


@triton.jit
def _modulate_v06(Z, S, T, O, N: tl.constexpr, D: tl.constexpr,
                  SG: tl.constexpr, TG: tl.constexpr,
                  SS: tl.constexpr, TS: tl.constexpr,
                  STYPE: tl.constexpr, PTYPE: tl.constexpr,
                  RTYPE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = index // D, index % D
    z = tl.load(Z + index, index < N, 0).to(tl.float32)
    s = tl.load(S + (row // SG) * SS + col, index < N, 0).to(tl.float32)
    t = tl.load(T + (row // TG) * TS + col, index < N, 0).to(tl.float32)
    factor = (1.0 + s).to(STYPE).to(tl.float32)
    product = (z * factor).to(PTYPE).to(tl.float32)
    value = (product + t).to(RTYPE)
    tl.store(O + index, value, index < N)


@triton.jit
def _row_norm_v06(X, W, B, S, T, PS, O, D: tl.constexpr, EPS: tl.constexpr,
                  SG: tl.constexpr, TG: tl.constexpr, PSG: tl.constexpr,
                  SS: tl.constexpr, TS: tl.constexpr, PSS: tl.constexpr,
                  ROWS_PER_BATCH: tl.constexpr, PREFIX: tl.constexpr,
                  RMS: tl.constexpr, AFFINE: tl.constexpr, BIAS: tl.constexpr,
                  MODULATE: tl.constexpr, EAGER: tl.constexpr,
                  XTYPE: tl.constexpr, STYPE: tl.constexpr,
                  PTYPE: tl.constexpr, RTYPE: tl.constexpr,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    keep = col < D
    x = tl.load(X + row * D + col, keep, 0).to(tl.float32)
    if RMS:
        var = tl.sum(x * x, 0) / D
        z = x * tl.rsqrt(var + EPS)
    else:
        mean = tl.sum(x, 0) / D
        centered = tl.where(keep, x - mean, 0.0)
        var = tl.sum(centered * centered, 0) / D
        z = (x - mean) * tl.rsqrt(var + EPS)
    if AFFINE:
        w = tl.load(W + col, keep, 0).to(tl.float32)
        z = z * w
    if BIAS:
        b = tl.load(B + col, keep, 0).to(tl.float32)
        z = z + b
    if MODULATE:
        s = tl.load(S + (row // SG) * SS + col, keep, 0).to(tl.float32)
        if PREFIX > 0:
            ps = tl.load(PS + (row // PSG) * PSS + col, keep, 0).to(tl.float32)
            s = tl.where(row % ROWS_PER_BATCH < PREFIX, ps, s)
        shift = tl.load(T + (row // TG) * TS + col, keep, 0).to(tl.float32)
        if EAGER:
            z = z.to(XTYPE).to(tl.float32)
            factor = (1.0 + s).to(STYPE).to(tl.float32)
            z = (z * factor).to(PTYPE).to(tl.float32)
            z = (z + shift).to(RTYPE)
        else:
            z = z * (1.0 + s) + shift
    tl.store(O + row * D + col, z, keep)


def _validate_x(x, eps):
    if x.device.type != "cuda" or x.dtype not in _DTYPE:
        raise ValueError("Expected GPU FP16/BF16/FP32 input")
    if not x.is_contiguous() or x.ndim < 2 or x.numel() == 0:
        raise ValueError("Expected nonempty contiguous rows; no implicit copy")
    if x.numel() >= 2**31:
        raise ValueError("Supported addressing is signed 32-bit elements")
    if x.shape[-1] > 8192 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("D <= 8192 and finite positive eps required")
    if torch.is_grad_enabled() and x.requires_grad:
        raise ValueError("Inference-only kernel")
    if torch.is_autocast_enabled("cuda") or torch.compiler.is_compiling():
        raise ValueError("Autocast and compiler paths retain native normalization")


def _modulation_rows(t, x):
    """Return a view of distinct rows and their contiguous broadcast group.

    [B,T,1,1,D] for [B,T,H,W,D], [B,1,D] for [B,S,D], and
    a global [D] row need no expansion. Alternating broadcast patterns such
    as [1,T,D] for B>1 are deliberately declined rather than silently copied.
    Chunked scale tensors may have row gaps; the row stride is preserved.
    """
    if t.device != x.device or t.dtype not in _DTYPE or t.ndim > x.ndim:
        raise ValueError("Invalid modulation device, dtype or dimensions")
    if t.shape[-1] != x.shape[-1] or t.stride(-1) != 1:
        raise ValueError("Modulation requires a dense final D dimension")
    if any(s < 0 for s in t.stride()) or sum((n - 1) * s for n, s in zip(t.shape, t.stride())) >= 2**31:
        raise ValueError("Modulation requires nonnegative signed-32-bit addressing")
    if torch.is_grad_enabled() and t.requires_grad:
        raise ValueError("Inference-only modulation")
    shape = (1,) * (x.ndim - t.ndim) + tuple(t.shape)
    broadcast_seen = False
    group = 1
    for got, want in zip(shape[:-1], x.shape[:-1]):
        if got not in (1, want):
            raise ValueError("Modulation does not broadcast to x")
        if got == 1 and want != 1:
            broadcast_seen = True
            group *= want
        elif want != 1 and broadcast_seen:
            raise ValueError("Noncontiguous broadcast groups are unsupported")
    flat = t.view(-1, x.shape[-1])
    if flat.shape[0] * group != x.numel() // x.shape[-1]:
        raise ValueError("Distinct-row broadcast mismatch")
    return flat, group


def row_norm(x, weight=None, bias=None, eps=1e-6, *, rms=False, backend="triton"):
    """LayerNorm/RMSNorm over the final dimension, optional same-dtype affine.

    Bare native RMSNorm and kitchen RMSNorm+RoPE are different baselines.
    This function does not replace the latter.
    """
    if backend == "torch":
        if rms:
            if bias is not None:
                raise ValueError("RMSNorm has no bias")
            return F.rms_norm(x, (x.shape[-1],), weight, eps)
        return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    if backend != "triton":
        raise ValueError(backend)
    _validate_x(x, eps)
    if rms and bias is not None:
        raise ValueError("RMSNorm has no bias")
    for p in (weight, bias):
        if p is not None and (p.shape != (x.shape[-1],) or not p.is_contiguous()
                              or p.dtype != x.dtype or p.device != x.device
                              or (torch.is_grad_enabled() and p.requires_grad)):
            raise ValueError("Affine tensors must be dense same-dtype D-vectors")
    out = torch.empty_like(x)
    d = x.shape[-1]
    _row_norm_v06[(x.numel() // d,)](
        x, weight if weight is not None else x, bias if bias is not None else x,
        x, x, x, out, d, float(eps), 1, 1, 1, d, d, d, 1, 0,
        rms, weight is not None, bias is not None, False, False,
        _DTYPE[x.dtype], _DTYPE[x.dtype], _DTYPE[x.dtype], _DTYPE[x.dtype],
        triton.next_power_of_2(d), num_warps=4, enable_fp_fusion=False)
    return out


def modulate(normalized, scale, shift, *, out_dtype=None):
    """Keep the native normalization, fuse only its eager modulation/cast.

    Explicit conversions preserve scale-add, product, and final-add rounding.
    Fusion is disabled in the compiler to retain separate FP32 multiply/add.
    Unlike a replacement reduction, this candidate can be bitwise equivalent.
    """
    x = normalized
    _validate_x(x, 1e-6)
    s, sg = _modulation_rows(scale, x)
    t, tg = _modulation_rows(shift, x)
    ptype = torch.promote_types(x.dtype, scale.dtype)
    rtype = torch.promote_types(ptype, shift.dtype)
    odtype = out_dtype or rtype
    if odtype not in _DTYPE:
        raise ValueError("Unsupported output dtype")
    out = torch.empty(x.shape, device=x.device, dtype=odtype)
    _modulate_v06[(triton.cdiv(x.numel(), 1024),)](
        x, s, t, out, x.numel(), x.shape[-1], sg, tg, s.stride(0), t.stride(0),
        _DTYPE[scale.dtype], _DTYPE[ptype], _DTYPE[rtype], 1024,
        num_warps=4, enable_fp_fusion=False)
    return out


def adaln(x, scale, shift, eps=1e-6, *, rms=False, rounding="eager",
          out_dtype=None, prefix_scale=None, prefix_len=0, backend="triton"):
    """Normalize and modulate, optionally choosing a separate prefix scale.

    ``out_dtype`` can absorb the following Anima FP16 cast. The residual input
    remains FP32. Prefix selection supports Qwen [B,S,D] without calculating
    prefix rows twice. This is still opt-in, not a production dispatcher.
    """
    if rounding not in ("eager", "fp32"):
        raise ValueError(rounding)
    if backend == "torch":
        if rounding != "eager":
            raise ValueError("torch reference implements eager rounding only")
        norm = F.rms_norm if rms else F.layer_norm
        z = norm(x, (x.shape[-1],), eps=eps)
        out = z * (1 + scale) + shift
        if prefix_len:
            if prefix_scale is None or x.ndim != 3:
                raise ValueError("Prefix selection requires [B,S,D] and prefix_scale")
            out[:, :prefix_len] = z[:, :prefix_len] * (1 + prefix_scale) + shift
        return out.to(out_dtype) if out_dtype is not None else out
    if backend != "triton":
        raise ValueError(backend)
    _validate_x(x, eps)
    s, sg = _modulation_rows(scale, x)
    t, tg = _modulation_rows(shift, x)
    ps, psg = s, sg
    if prefix_len:
        if x.ndim != 3 or not 0 <= prefix_len <= x.shape[1] or prefix_scale is None:
            raise ValueError("Prefix selection requires [B,S,D] and valid prefix")
        ps, psg = _modulation_rows(prefix_scale, x)
        if prefix_scale.dtype != scale.dtype:
            raise ValueError("Prefix/target scale dtype mismatch")
    elif prefix_scale is not None:
        raise ValueError("prefix_scale without a positive prefix_len")
    ptype = torch.promote_types(x.dtype, scale.dtype)
    rtype = torch.promote_types(ptype, shift.dtype)
    odtype = out_dtype or (rtype if rounding == "eager" else x.dtype)
    if odtype not in _DTYPE:
        raise ValueError("Unsupported output dtype")
    out = torch.empty(x.shape, device=x.device, dtype=odtype)
    d = x.shape[-1]
    _row_norm_v06[(x.numel() // d,)](
        x, x, x, s, t, ps, out, d, float(eps), sg, tg, psg,
        s.stride(0), t.stride(0), ps.stride(0), x.shape[1] if x.ndim == 3 else 1,
        int(prefix_len), rms, False, False, True, rounding == "eager",
        _DTYPE[x.dtype], _DTYPE[scale.dtype], _DTYPE[ptype], _DTYPE[rtype],
        triton.next_power_of_2(d), num_warps=4, enable_fp_fusion=False)
    return out
