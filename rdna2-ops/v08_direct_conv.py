"""Research-only NCT Conv1d implicit GEMM for RDNA2 (gfx1030), Triton 3.8.

No im2col tensor is materialized. fwd/dX use FP16 dot inputs with FP32
accumulators. dW reduces B*T in deterministic split chunks into FP32 scratch,
then sums those chunks in one reduction kernel. There are no FP16 partial sums
or atomics. This module does not install hooks or enable candidates on import.

The opt-in context requires APPLIO_RDNA2_DIRECT_CONV=1 AND an explicit measured
manifest. Native fallback is retained independently for every direction.
"""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_NATIVE_CONV1D = F.conv1d


@triton.jit
def _fwd(X, W, BIAS, Y, B: tl.constexpr, CI: tl.constexpr, CO: tl.constexpr,
         T: tl.constexpr, R: tl.constexpr, D: tl.constexpr, P: tl.constexpr,
         TO: tl.constexpr, G: tl.constexpr, HAS_BIAS: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    C: tl.constexpr = CI // G
    O: tl.constexpr = CO // G
    pid = tl.program_id(0)
    batch = tl.program_id(1)
    group = tl.program_id(2)
    om = (pid % tl.cdiv(O, BM)) * BM + tl.arange(0, BM)
    tt = (pid // tl.cdiv(O, BM)) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(tl.cdiv(C * R, BK)):
        cr = step * BK + kk
        ic = cr // R
        rr = cr % R
        xv_t = tt[None, :] + rr[:, None] * D - P
        av = tl.load(W + (group * O + om[:, None]) * (C * R) + cr[None, :],
                     (om[:, None] < O) & (cr[None, :] < C * R), 0.0)
        bv = tl.load(X + (batch * CI + group * C + ic[:, None]) * T + xv_t,
                     (ic[:, None] < C) & (tt[None, :] < TO) &
                     (xv_t >= 0) & (xv_t < T), 0.0)
        acc = tl.dot(av, bv, acc, input_precision="ieee", out_dtype=tl.float32)
    # Match the native AMP boundary: convolution output is FP16 before bias add.
    result = acc.to(tl.float16)
    if HAS_BIAS:
        bias = tl.load(BIAS + group * O + om, om < O, 0.0).to(tl.float32)
        result = (result.to(tl.float32) + bias[:, None]).to(tl.float16)
    tl.store(Y + (batch * CO + group * O + om[:, None]) * TO + tt[None, :],
             result, (om[:, None] < O) & (tt[None, :] < TO))


@triton.jit
def _dx(DY, W, DX, B: tl.constexpr, CI: tl.constexpr, CO: tl.constexpr,
        T: tl.constexpr, R: tl.constexpr, D: tl.constexpr, P: tl.constexpr,
        TO: tl.constexpr, G: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    C: tl.constexpr = CI // G
    O: tl.constexpr = CO // G
    pid = tl.program_id(0)
    batch = tl.program_id(1)
    group = tl.program_id(2)
    ic = (pid % tl.cdiv(C, BM)) * BM + tl.arange(0, BM)
    tt = (pid // tl.cdiv(C, BM)) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(tl.cdiv(O * R, BK)):
        oor = step * BK + kk
        oc = oor // R
        rr = oor % R
        yv_t = tt[None, :] + P - rr[:, None] * D
        av = tl.load(W + (group * O + oc[None, :]) * (C * R) +
                     ic[:, None] * R + rr[None, :],
                     (ic[:, None] < C) & (oc[None, :] < O), 0.0)
        bv = tl.load(DY + (batch * CO + group * O + oc[:, None]) * TO + yv_t,
                     (oc[:, None] < O) & (tt[None, :] < T) &
                     (yv_t >= 0) & (yv_t < TO), 0.0)
        acc = tl.dot(av, bv, acc, input_precision="ieee", out_dtype=tl.float32)
    tl.store(DX + (batch * CI + group * C + ic[:, None]) * T + tt[None, :],
             acc.to(tl.float16), (ic[:, None] < C) & (tt[None, :] < T))


@triton.jit
def _dw_partial(X, DY, PART, B: tl.constexpr, CI: tl.constexpr, CO: tl.constexpr,
                T: tl.constexpr, R: tl.constexpr, D: tl.constexpr, P: tl.constexpr,
                TO: tl.constexpr, G: tl.constexpr, SPLIT: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    C: tl.constexpr = CI // G
    O: tl.constexpr = CO // G
    STEPS: tl.constexpr = tl.cdiv(B * TO, SPLIT * BK)
    pid = tl.program_id(0)
    rr = tl.program_id(1)
    split_group = tl.program_id(2)
    group = split_group % G
    split = split_group // G
    oc = (pid % tl.cdiv(O, BM)) * BM + tl.arange(0, BM)
    ic = (pid // tl.cdiv(O, BM)) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(STEPS):
        bt = (split * STEPS + step) * BK + inner
        batch = bt // TO
        tt = bt % TO
        xt = tt + rr * D - P
        av = tl.load(DY + (batch[None, :] * CO + group * O + oc[:, None]) * TO +
                     tt[None, :], (oc[:, None] < O) & (bt[None, :] < B * TO), 0.0)
        bv = tl.load(X + (batch[:, None] * CI + group * C + ic[None, :]) * T +
                     xt[:, None], (ic[None, :] < C) & (bt[:, None] < B * TO) &
                     (xt[:, None] >= 0) & (xt[:, None] < T), 0.0)
        acc = tl.dot(av, bv, acc, input_precision="ieee", out_dtype=tl.float32)
    offsets = split * (CO * C * R) + ((group * O + oc[:, None]) * C +
                                     ic[None, :]) * R + rr
    tl.store(PART + offsets, acc, (oc[:, None] < O) & (ic[None, :] < C))


@triton.jit
def _dw_reduce(PART, DW, N: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLIT)
    vals = tl.load(PART + splits[:, None] * N + cols[None, :],
                   cols[None, :] < N, 0.0)
    total = tl.sum(vals, axis=0)
    tl.store(DW + cols, total.to(tl.float16), cols < N)


@lru_cache(None)
def _is_gfx1030(index):
    return bool(torch.version.hip) and (
        torch.cuda.get_device_properties(index).gcnArchName.split(":")[0] == "gfx1030")


def _check(x, w, dilation, padding, groups):
    if x.ndim != 3 or w.ndim != 3 or x.shape[1] != w.shape[1] * groups:
        raise ValueError("Expected X[B,CI,T], W[CO,CI/groups,R]")
    if groups < 1 or w.shape[0] % groups:
        raise ValueError("Invalid groups")
    if x.device.type != "cuda" or w.device != x.device or not _is_gfx1030(x.device.index):
        raise ValueError("This research kernel requires ROCm gfx1030")
    if x.dtype != torch.float16 or w.dtype != torch.float16:
        raise ValueError("Explicit FP16 tensors required")
    if not x.is_contiguous() or not w.is_contiguous():
        raise ValueError("Contiguous NCT/OIR storage required")
    if dilation < 1 or padding < 0:
        raise ValueError("Invalid dilation/padding")
    b, ci, t = x.shape
    co, _, r = w.shape
    to = t + 2 * padding - dilation * (r - 1)
    if min(b, ci, t, co, r, to) < 1 or max(x.numel(), w.numel(), b * co * to) >= 2**31:
        raise ValueError("Nonempty tensors and 32-bit element offsets required")
    return b, ci, co, t, r, dilation, padding, to, groups


def _cfg(config, default=(32, 64, 32, 4)):
    cfg = dict(config or {})
    bm, bn, bk, nw = (int(cfg.get(k, v)) for k, v in
                      zip(("bm", "bn", "bk", "num_warps"), default))
    if any(v not in (16, 32, 64, 128) for v in (bm, bn, bk)) or nw not in (1, 2, 4, 8):
        raise ValueError("Unsupported tile/warp configuration")
    return bm, bn, bk, nw


def _metadata(kernels, info, config):
    if info is None:
        return
    info.update(config=dict(config or {}), kernels=[])
    for kernel in kernels:
        asm = kernel.asm.get("amdgcn", "")
        counts = Counter(re.findall(r"^\s*(v_[a-z0-9_]+)\s", asm, re.MULTILINE))
        info["kernels"].append(dict(
            name=getattr(kernel, "name", None),
            amdgcn=asm,
            assembly_sha256=hashlib.sha256(asm.encode()).hexdigest(),
            instruction_counts={k: v for k, v in counts.items()
                                if k.startswith(("v_dot", "v_fma", "v_mfma", "v_wmma"))},
            dot2=any(k.startswith("v_dot2") for k in counts),
            registers=getattr(kernel, "n_regs", None), spills=getattr(kernel, "n_spills", None),
            shared_bytes=getattr(kernel.metadata, "shared", None)))


def conv_fwd(x, w, bias=None, *, dilation=1, padding=0, groups=1, config=None, metadata=None):
    """Explicit FP16 forward; caller owns AMP casting, autograd and synchronization."""
    dims = _check(x, w, dilation, padding, groups)
    b, ci, co, t, r, d, p, to, g = dims
    if bias is not None and (bias.dtype != torch.float16 or bias.device != x.device or
                             bias.shape != (co,) or not bias.is_contiguous()):
        raise ValueError("Bias must be contiguous FP16 [CO] on the input device")
    bm, bn, bk, nw = _cfg(config)
    out = torch.empty((b, co, to), device=x.device, dtype=torch.float16)
    kernel = _fwd[(triton.cdiv(co // g, bm) * triton.cdiv(to, bn), b, g)](
        x, w, bias if bias is not None else x, out, *dims, bias is not None,
        bm, bn, bk, num_warps=nw, num_stages=1, matrix_instr_nonkdim=0)
    _metadata([kernel], metadata, config)
    return out


def conv_dx(dy, w, input_shape, *, dilation=1, padding=0, groups=1, config=None, metadata=None):
    """Explicit FP16 dX; input_shape is the original NCT shape."""
    b, ci, t = tuple(input_shape)
    co, c, r = w.shape
    to = t + 2 * padding - dilation * (r - 1)
    if (dy.shape != (b, co, to) or dy.dtype != torch.float16 or w.dtype != torch.float16
        or not dy.is_contiguous() or not w.is_contiguous() or ci != c * groups
        or co % groups or dy.device.type != "cuda" or dy.device != w.device
        or not _is_gfx1030(dy.device.index)):
        raise ValueError("Incompatible contiguous FP16 dY/W/input shape")
    bm, bn, bk, nw = _cfg(config)
    out = torch.empty((b, ci, t), device=dy.device, dtype=torch.float16)
    dims = (b, ci, co, t, r, dilation, padding, to, groups)
    kernel = _dx[(triton.cdiv(ci // groups, bm) * triton.cdiv(t, bn), b, groups)](
        dy, w, out, *dims, bm, bn, bk, num_warps=nw, num_stages=1, matrix_instr_nonkdim=0)
    _metadata([kernel], metadata, config)
    return out


def conv_dw(x, dy, weight_shape, *, dilation=1, padding=0, groups=1, config=None, metadata=None):
    """Explicit FP16 dW, FP32 split partials, fixed-order reduction, no atomics."""
    b, ci, t = x.shape
    co, c, r = tuple(weight_shape)
    to = t + 2 * padding - dilation * (r - 1)
    if (dy.shape != (b, co, to) or dy.dtype != torch.float16 or x.dtype != torch.float16
        or not dy.is_contiguous() or not x.is_contiguous() or ci != c * groups
        or co % groups or x.device.type != "cuda" or dy.device != x.device
        or not _is_gfx1030(x.device.index)):
        raise ValueError("Incompatible contiguous FP16 X/dY/weight shape")
    bm, bn, bk, nw = _cfg(config, default=(16, 32, 64, 4))
    split = int((config or {}).get("split", 32))
    if split not in (1, 2, 4, 8, 16, 32, 64, 128):
        raise ValueError("dW split must be a power of two up to 128")
    out = torch.empty((co, c, r), device=x.device, dtype=torch.float16)
    partial = torch.empty((split, co, c, r), device=x.device, dtype=torch.float32)
    dims = (b, ci, co, t, r, dilation, padding, to, groups)
    k1 = _dw_partial[(triton.cdiv(co // groups, bm) * triton.cdiv(c, bn), r, split * groups)](
        x, dy, partial, *dims, split, bm, bn, bk, num_warps=nw, num_stages=1,
        matrix_instr_nonkdim=0)
    k2 = _dw_reduce[(triton.cdiv(out.numel(), 128),)](
        partial, out, out.numel(), split, 128, num_warps=4, num_stages=1)
    _metadata([k1, k2], metadata, config)
    if metadata is not None:
        metadata["scratch_bytes"] = partial.numel() * partial.element_size()
    return out


def shape_key(input_shape, weight_shape, dilation=1, padding=0, groups=1):
    return json.dumps([list(input_shape), list(weight_shape), int(dilation), int(padding),
                       int(groups), "float16", "NCT", 1], separators=(",", ":"))


class _MeasuredConv(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
    def forward(ctx, x, w, bias, dilation, padding, groups, directions):
        ctx.save_for_backward(x, w)
        ctx.dilation, ctx.padding, ctx.groups = dilation, padding, groups
        ctx.has_bias = bias is not None
        ctx.directions = directions
        if "fwd" in directions:
            return conv_fwd(x, w, bias, dilation=dilation, padding=padding, groups=groups,
                            config=directions["fwd"])
        return _NATIVE_CONV1D(x, w, bias, 1, padding, dilation, groups)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        dy = grad_output.contiguous()
        d, p, g = ctx.dilation, ctx.padding, ctx.groups
        need_x, need_w, need_b = ctx.needs_input_grad[:3]
        dx = dw = db = None
        native_x = need_x and "dx" not in ctx.directions
        native_w = need_w and "dw" not in ctx.directions
        if native_x or native_w:
            dx0, dw0, _ = torch.ops.aten.convolution_backward.default(
                dy, x, w, None, [1], [p], [d], False, [0], g,
                [native_x, native_w, False])
            if native_x:
                dx = dx0
            if native_w:
                dw = dw0
        if need_x and "dx" in ctx.directions:
            dx = conv_dx(dy, w, x.shape, dilation=d, padding=p, groups=g,
                         config=ctx.directions["dx"])
        if need_w and "dw" in ctx.directions:
            dw = conv_dw(x, dy, w.shape, dilation=d, padding=p, groups=g,
                         config=ctx.directions["dw"])
        if need_b and ctx.has_bias:
            db = dy.sum((0, 2))
        return dx, dw, db, None, None, None, None


@contextmanager
def measured_dispatch(manifest_path, *, hits=None):
    """Install only an explicit, validated per-shape manifest in this process.

    Disabled unless APPLIO_RDNA2_DIRECT_CONV=1. The manifest must state
    schema="applio_v08_direct_conv_dispatch", strict_gate_pass=true and contain
    a nonempty directions map for each accepted shape. No built-in winners.
    """
    if os.environ.get("APPLIO_RDNA2_DIRECT_CONV", "0") != "1":
        yield
        return
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("schema") != "applio_v08_direct_conv_dispatch":
        raise ValueError("Unknown dispatch manifest schema")
    if manifest.get("strict_gate_pass") is not True:
        raise ValueError("Strict operator validation has not passed")
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if manifest.get("source_sha256") != source_sha:
        raise ValueError("Dispatch manifest belongs to a different kernel source; revalidate")
    entries = manifest.get("entries", {})
    by_shape = {}
    for key, row in entries.items():
        if (row.get("strict_gate_pass") is not True or not row.get("directions") or
            set(row["directions"]) - {"fwd", "dx", "dw"}):
            raise ValueError("Each dispatch entry requires measured valid directions")
        parts = json.loads(key)
        if len(parts) != 8 or parts[5:] != ["float16", "NCT", 1]:
            raise ValueError("Unsupported manifest layout/dtype/stride")
        lookup = (tuple(parts[0]), tuple(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
        by_shape[lookup] = (key, row)
    original = torch.nn.Conv1d._conv_forward
    observed = hits if hits is not None else {}

    def hook(module, x, w, bias):
        if (x.device.type != "cuda" or x.ndim != 3 or w.ndim != 3 or
            module.padding_mode != "zeros" or module.stride != (1,) or
            not isinstance(module.padding, tuple) or len(module.padding) != 1 or
            not x.is_contiguous() or not w.is_contiguous() or
            not _is_gfx1030(x.device.index)):
            return original(module, x, w, bias)
        lookup = (tuple(x.shape), tuple(w.shape), module.dilation[0],
                  module.padding[0], module.groups)
        selected = by_shape.get(lookup)
        if selected is None:
            return original(module, x, w, bias)
        key, row = selected
        amp = torch.is_autocast_enabled("cuda")
        if amp:
            supported_dtype = (torch.get_autocast_dtype("cuda") == torch.float16 and
                               x.dtype in (torch.float16, torch.float32) and
                               w.dtype in (torch.float16, torch.float32) and
                               (bias is None or bias.dtype in (torch.float16, torch.float32)))
        else:
            supported_dtype = (x.dtype == w.dtype == torch.float16 and
                               (bias is None or bias.dtype == torch.float16))
        if not supported_dtype:
            return original(module, x, w, bias)
        observed[key] = observed.get(key, 0) + 1
        return _MeasuredConv.apply(x, w, bias, module.dilation[0], module.padding[0],
                                   module.groups, row["directions"])
    torch.nn.Conv1d._conv_forward = hook
    try:
        yield
    finally:
        torch.nn.Conv1d._conv_forward = original
