"""Research-only LeakyReLU/Conv/Conv/residual fusion on top of v0.8 B.

The two convolutions still have independent dot2 launches. Activation values
are computed during their input loads; the second convolution adds the skip
connection before its store. Backward applies the activation mask in dX's
store and merges the first dX with the residual gradient. dW recomputes the
activation while loading its input, with the original FP16 rounding boundary.

Only pairs whose two forward shapes are present in the measured v0.8 manifest
are eligible. Its direction choices are preserved, including native dW/dX.
Weight normalization runs outside the custom Function, in its original graph.
No source, module method, environment variable or GPU state changes on import.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

import v08_direct_conv as base


@triton.jit
def _act(v, SLOPE: tl.constexpr):
    # leaky_relu is not autocast: round to the source dtype, then the Conv
    # autocast rounds to FP16. This also preserves the FP32-input boundary.
    vf = v.to(tl.float32)
    return tl.where(vf > 0.0, vf, vf * SLOPE).to(v.dtype).to(tl.float16)


@triton.jit
def _fwd_act_res(X, W, BIAS, RES, Y,
                 B: tl.constexpr, CI: tl.constexpr, CO: tl.constexpr,
                 T: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
                 P: tl.constexpr, TO: tl.constexpr, G: tl.constexpr,
                 HAS_BIAS: tl.constexpr, HAS_RES: tl.constexpr,
                 SLOPE: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    C: tl.constexpr = CI // G
    O: tl.constexpr = CO // G
    pid, batch, group = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    om = (pid % tl.cdiv(O, BM)) * BM + tl.arange(0, BM)
    tt = (pid // tl.cdiv(O, BM)) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(tl.cdiv(C * R, BK)):
        cr = step * BK + kk
        ic, rr = cr // R, cr % R
        xt = tt[None, :] + rr[:, None] * D - P
        av = tl.load(W + (group * O + om[:, None]) * (C * R) + cr[None, :],
                     (om[:, None] < O) & (cr[None, :] < C * R), 0.0)
        bv = tl.load(X + (batch * CI + group * C + ic[:, None]) * T + xt,
                     (ic[:, None] < C) & (tt[None, :] < TO) &
                     (xt >= 0) & (xt < T), 0.0)
        bv = _act(bv, SLOPE)
        acc = tl.dot(av, bv, acc, input_precision="ieee", out_dtype=tl.float32)
    result = acc.to(tl.float16)
    if HAS_BIAS:
        bias = tl.load(BIAS + group * O + om, om < O, 0.0).to(tl.float32)
        result = (result.to(tl.float32) + bias[:, None]).to(tl.float16)
    offsets = (batch * CO + group * O + om[:, None]) * TO + tt[None, :]
    valid = (om[:, None] < O) & (tt[None, :] < TO)
    if HAS_RES:
        skip = tl.load(RES + offsets, valid, 0.0).to(tl.float32)
        total = result.to(tl.float32) + skip
        tl.store(Y + offsets, total, valid)
    else:
        tl.store(Y + offsets, result, valid)


@triton.jit
def _dx_mask_res(DY, W, X, RES_GRAD, DX,
                 B: tl.constexpr, CI: tl.constexpr, CO: tl.constexpr,
                 T: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
                 P: tl.constexpr, TO: tl.constexpr, G: tl.constexpr,
                 HAS_RES: tl.constexpr, SLOPE: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    C: tl.constexpr = CI // G
    O: tl.constexpr = CO // G
    pid, batch, group = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ic = (pid % tl.cdiv(C, BM)) * BM + tl.arange(0, BM)
    tt = (pid // tl.cdiv(C, BM)) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(tl.cdiv(O * R, BK)):
        oor = step * BK + kk
        oc, rr = oor // R, oor % R
        yt = tt[None, :] + P - rr[:, None] * D
        av = tl.load(W + (group * O + oc[None, :]) * (C * R) +
                     ic[:, None] * R + rr[None, :],
                     (ic[:, None] < C) & (oc[None, :] < O), 0.0)
        bv = tl.load(DY + (batch * CO + group * O + oc[:, None]) * TO + yt,
                     (oc[:, None] < O) & (tt[None, :] < T) &
                     (yt >= 0) & (yt < TO), 0.0)
        acc = tl.dot(av, bv, acc, input_precision="ieee", out_dtype=tl.float32)
    offsets = (batch * CI + group * C + ic[:, None]) * T + tt[None, :]
    valid = (ic[:, None] < C) & (tt[None, :] < T)
    xv = tl.load(X + offsets, valid, 0.0)
    # Conv backward -> FP16, autocast backward -> input dtype, activation
    # backward -> input dtype, and only then the residual accumulation.
    grad = acc.to(tl.float16).to(tl.float32)
    grad = tl.where(xv > 0.0, grad, grad * SLOPE).to(xv.dtype)
    if HAS_RES:
        skip = tl.load(RES_GRAD + offsets, valid, 0.0).to(tl.float32)
        total = grad.to(tl.float32) + skip
        tl.store(DX + offsets, total, valid)
    else:
        tl.store(DX + offsets, grad, valid)


@triton.jit
def _dw_act_partial(X, DY, PART,
                    B: tl.constexpr, CI: tl.constexpr, CO: tl.constexpr,
                    T: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
                    P: tl.constexpr, TO: tl.constexpr, G: tl.constexpr,
                    SPLIT: tl.constexpr, SLOPE: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    C: tl.constexpr = CI // G
    O: tl.constexpr = CO // G
    STEPS: tl.constexpr = tl.cdiv(B * TO, SPLIT * BK)
    pid, rr, split_group = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    group, split = split_group % G, split_group // G
    oc = (pid % tl.cdiv(O, BM)) * BM + tl.arange(0, BM)
    ic = (pid // tl.cdiv(O, BM)) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(STEPS):
        bt = (split * STEPS + step) * BK + inner
        batch, tt = bt // TO, bt % TO
        xt = tt + rr * D - P
        av = tl.load(DY + (batch[None, :] * CO + group * O + oc[:, None]) * TO +
                     tt[None, :], (oc[:, None] < O) & (bt[None, :] < B * TO), 0.0)
        bv = tl.load(X + (batch[:, None] * CI + group * C + ic[None, :]) * T +
                     xt[:, None], (ic[None, :] < C) & (bt[:, None] < B * TO) &
                     (xt[:, None] >= 0) & (xt[:, None] < T), 0.0)
        bv = _act(bv, SLOPE)
        acc = tl.dot(av, bv, acc, input_precision="ieee", out_dtype=tl.float32)
    offsets = split * (CO * C * R) + ((group * O + oc[:, None]) * C + ic[None, :]) * R + rr
    tl.store(PART + offsets, acc, (oc[:, None] < O) & (ic[None, :] < C))


@triton.jit
def _native_mask_res(DX0, X, RES_GRAD, DX, N: tl.constexpr,
                     HAS_RES: tl.constexpr, SLOPE: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    xv = tl.load(X + idx, idx < N, 0.0)
    grad = tl.load(DX0 + idx, idx < N, 0.0).to(tl.float32)
    grad = tl.where(xv > 0.0, grad, grad * SLOPE).to(xv.dtype)
    if HAS_RES:
        skip = tl.load(RES_GRAD + idx, idx < N, 0.0).to(tl.float32)
        total = grad.to(tl.float32) + skip
        tl.store(DX + idx, total, idx < N)
    else:
        tl.store(DX + idx, grad, idx < N)


def _dims(x, w, dilation, padding, groups):
    if (x.ndim != 3 or w.ndim != 3 or x.dtype not in (torch.float16, torch.float32)
        or w.dtype != torch.float16 or not x.is_contiguous() or not w.is_contiguous()
        or x.device.type != "cuda" or x.device != w.device or not base._is_gfx1030(x.device.index)):
        raise ValueError("Contiguous gfx1030 FP16/FP32 X and FP16 W required")
    b, ci, t = x.shape
    co, c, r = w.shape
    to = t + 2 * padding - dilation * (r - 1)
    if (groups < 1 or ci != c * groups or co % groups or dilation < 1 or padding < 0
        or min(b, ci, t, co, c, r, to) < 1 or max(x.numel(), w.numel(), b * co * to) >= 2**31):
        raise ValueError("Invalid stride-one convolution geometry")
    return b, ci, co, t, r, dilation, padding, to, groups


def conv_act_fwd(x, w, bias=None, *, dilation=1, padding=0, groups=1,
                 residual=None, slope=0.1, config=None, metadata=None):
    dims = _dims(x, w, dilation, padding, groups)
    b, ci, co, t, r, d, p, to, g = dims
    if bias is not None and (bias.dtype != torch.float16 or bias.shape != (co,)
                            or bias.device != x.device or not bias.is_contiguous()):
        raise ValueError("Contiguous FP16 bias required")
    if residual is not None and (residual.shape != (b, co, to)
                                 or residual.dtype not in (torch.float16, torch.float32)
                                 or residual.device != x.device or not residual.is_contiguous()):
        raise ValueError("Residual shape/device/layout mismatch")
    out_dtype = torch.float16 if residual is None else residual.dtype
    out = torch.empty((b, co, to), device=x.device, dtype=out_dtype)
    bm, bn, bk, nw = base._cfg(config)
    kernel = _fwd_act_res[(triton.cdiv(co // g, bm) * triton.cdiv(to, bn), b, g)](
        x, w, bias if bias is not None else x, residual if residual is not None else x, out,
        *dims, bias is not None, residual is not None, slope, bm, bn, bk,
        num_warps=nw, num_stages=1, matrix_instr_nonkdim=0, enable_fp_fusion=False)
    base._metadata([kernel], metadata, config)
    return out


def conv_act_dx(dy, w, x, *, dilation=1, padding=0, groups=1,
                residual_grad=None, slope=0.1, config=None, metadata=None):
    dims = _dims(x, w, dilation, padding, groups)
    b, ci, co, t, r, d, p, to, g = dims
    if dy.shape != (b, co, to) or dy.dtype != torch.float16 or not dy.is_contiguous():
        raise ValueError("Contiguous FP16 dY required")
    if residual_grad is not None and (residual_grad.shape != x.shape or
                                      residual_grad.dtype != x.dtype or not residual_grad.is_contiguous()):
        raise ValueError("Residual gradient must match input")
    out = torch.empty_like(x)
    bm, bn, bk, nw = base._cfg(config)
    kernel = _dx_mask_res[(triton.cdiv(ci // g, bm) * triton.cdiv(t, bn), b, g)](
        dy, w, x, residual_grad if residual_grad is not None else x, out, *dims,
        residual_grad is not None, slope, bm, bn, bk,
        num_warps=nw, num_stages=1, matrix_instr_nonkdim=0, enable_fp_fusion=False)
    base._metadata([kernel], metadata, config)
    return out


def conv_act_dw(x, dy, weight_shape, *, dilation=1, padding=0, groups=1,
                slope=0.1, config=None, metadata=None):
    b, ci, t = x.shape
    co, c, r = tuple(weight_shape)
    to = t + 2 * padding - dilation * (r - 1)
    if (dy.shape != (b, co, to) or dy.dtype != torch.float16
        or x.dtype not in (torch.float16, torch.float32) or not dy.is_contiguous()
        or not x.is_contiguous() or ci != c * groups or co % groups
        or dy.device != x.device or x.device.type != "cuda" or not base._is_gfx1030(x.device.index)):
        raise ValueError("Incompatible activation/dY geometry")
    bm, bn, bk, nw = base._cfg(config, default=(16, 32, 64, 4))
    split = int((config or {}).get("split", 32))
    if split not in (1, 2, 4, 8, 16, 32, 64, 128):
        raise ValueError("dW split must be a power of two <= 128")
    out = torch.empty((co, c, r), dtype=torch.float16, device=x.device)
    partial = torch.empty((split, co, c, r), dtype=torch.float32, device=x.device)
    dims = (b, ci, co, t, r, dilation, padding, to, groups)
    k1 = _dw_act_partial[(triton.cdiv(co // groups, bm) * triton.cdiv(c, bn), r, split * groups)](
        x, dy, partial, *dims, split, slope, bm, bn, bk,
        num_warps=nw, num_stages=1, matrix_instr_nonkdim=0, enable_fp_fusion=False)
    k2 = base._dw_reduce[(triton.cdiv(out.numel(), 128),)](
        partial, out, out.numel(), split, 128, num_warps=4, num_stages=1)
    base._metadata([k1, k2], metadata, config)
    if metadata is not None:
        metadata["scratch_bytes"] = partial.numel() * partial.element_size()
    return out


def _backward_conv(dy, x, w, info, need_x, need_w, need_b, *, residual_grad=None, slope=0.1):
    d, p, g, directions = info
    dx = dw = db = None
    native_x, native_w = need_x and "dx" not in directions, need_w and "dw" not in directions
    if native_x or native_w:
        # Native dW needs its activation tensor. Recompute it only for these
        # measured fallback directions; no activation is saved in forward.
        activated = F.leaky_relu(x, slope).to(torch.float16)
        dx0, dw0, _ = torch.ops.aten.convolution_backward.default(
            dy, activated, w, None, [1], [p], [d], False, [0], g,
            [native_x, native_w, False])
        if native_w:
            dw = dw0
        if native_x:
            dx = torch.empty_like(x)
            _native_mask_res[(triton.cdiv(x.numel(), 256),)](
                dx0, x, residual_grad if residual_grad is not None else x, dx, x.numel(),
                residual_grad is not None, slope, 256, num_warps=4, enable_fp_fusion=False)
    if need_x and "dx" in directions:
        dx = conv_act_dx(dy, w, x, dilation=d, padding=p, groups=g,
                         residual_grad=residual_grad, slope=slope, config=directions["dx"])
    if need_w and "dw" in directions:
        dw = conv_act_dw(x, dy, w.shape, dilation=d, padding=p, groups=g,
                         slope=slope, config=directions["dw"])
    if need_b:
        # custom_fwd intentionally preserves autocast for FP32 preactivation.
        # Explicit dtype keeps v08's disabled-autocast FP16 bias reduction.
        db = dy.sum((0, 2), dtype=torch.float16)
    return dx, dw, db


class _FusedPair(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, w1, b1, w2, b2, info1, info2, slope):
        # Do not cast x before LeakyReLU: FP32 input is a distinct boundary.
        ctx.parameter_dtypes = tuple(v.dtype if v is not None else None for v in (w1, b1, w2, b2))
        w1h, w2h = w1.to(torch.float16), w2.to(torch.float16)
        b1h = b1.to(torch.float16) if b1 is not None else None
        b2h = b2.to(torch.float16) if b2 is not None else None
        d1, p1, g1, dirs1 = info1
        d2, p2, g2, dirs2 = info2
        h = conv_act_fwd(x, w1h, b1h, dilation=d1, padding=p1, groups=g1,
                          slope=slope, config=dirs1["fwd"])
        y = conv_act_fwd(h, w2h, b2h, dilation=d2, padding=p2, groups=g2,
                          residual=x, slope=slope, config=dirs2["fwd"])
        ctx.save_for_backward(x, h, w1h, w2h)
        ctx.info1, ctx.info2, ctx.slope = info1, info2, slope
        return y

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    @once_differentiable
    def backward(ctx, grad_output):
        x, h, w1, w2 = ctx.saved_tensors
        need_x, need_w1, need_b1, need_w2, need_b2 = ctx.needs_input_grad[:5]
        need_h = need_x or need_w1 or need_b1
        grad_skip = grad_output.contiguous()
        dy = grad_skip.to(torch.float16)
        dh, dw2, db2 = _backward_conv(dy, h, w2, ctx.info2, need_h, need_w2, need_b2,
                                     slope=ctx.slope)
        dx = dw1 = db1 = None
        if need_h:
            dx, dw1, db1 = _backward_conv(dh, x, w1, ctx.info1, need_x, need_w1, need_b1,
                                         residual_grad=grad_skip if need_x else None, slope=ctx.slope)
        grads = [dw1, db1, dw2, db2]
        grads = [v.to(dtype) if v is not None else None for v, dtype in zip(grads, ctx.parameter_dtypes)]
        return dx, *grads, None, None, None


def load_manifest(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    expected = hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()
    if (manifest.get("schema") != "applio_v08_direct_conv_dispatch"
        or manifest.get("strict_gate_pass") is not True or manifest.get("source_sha256") != expected):
        raise ValueError("A validated v0.8 manifest matching the source is required")
    by_shape = {}
    for key, entry in manifest["entries"].items():
        parts = json.loads(key)
        if (entry.get("strict_gate_pass") is not True or not entry.get("directions")
            or set(entry["directions"]) - {"fwd", "dx", "dw"}
            or len(parts) != 8 or parts[5:] != ["float16", "NCT", 1]):
            raise ValueError("Malformed v0.8 entry")
        lookup = (tuple(parts[0]), tuple(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
        by_shape[lookup] = (key, entry)
    return manifest, by_shape


def _lookup(module, input_shape, by_shape):
    if (not isinstance(module, torch.nn.Conv1d) or module.padding_mode != "zeros"
        or module.stride != (1,) or not isinstance(module.padding, tuple)
        or len(module.padding) != 1 or module._forward_hooks or module._forward_pre_hooks
        or module._backward_hooks or module.__class__.__name__.endswith("FoldConv1d")):
        return None
    wshape = (module.out_channels, module.in_channels // module.groups, module.kernel_size[0])
    row = by_shape.get((tuple(input_shape), wshape, module.dilation[0], module.padding[0], module.groups))
    return row if row is not None and "fwd" in row[1]["directions"] else None


def pair_key(key1, key2):
    return json.dumps([key1, key2], separators=(",", ":"))


@contextmanager
def fused_resblocks(manifest_path, *, selection=None, hits=None):
    """Temporarily fuse eligible ResBlock pairs in this process only.

    ``selection`` may be a validated bench selection JSON path or an explicit
    iterable of pair keys for a research probe. None probes every eligible
    pair. The original forward method is restored even after an exception.
    """
    from rvc.lib.algorithm.residuals import ResBlock, LRELU_SLOPE
    from rvc.lib.algorithm import fold_conv
    if fold_conv.ENABLED:
        raise ValueError("v1.0 fusion is not defined for APPLIO_DILATION_FOLD=1")
    manifest, by_shape = load_manifest(manifest_path)
    if isinstance(selection, (str, Path)):
        selected = json.loads(Path(selection).read_text())
        if (selected.get("schema") != "applio_v10_fusion_selection"
            or selected.get("strict_gate_pass") is not True
            or selected.get("source_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            or selected.get("v08_manifest_sha256") != hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()):
            raise ValueError("Fusion selection is not validated for this source/manifest")
        allowed = set(selected["pair_keys"])
        allowed_dtypes = selected.get("input_dtypes_by_pair", {})
        if any(key not in allowed_dtypes for key in allowed):
            raise ValueError("Fusion selection must identify the validated input dtype")
    else:
        allowed = set(selection) if selection is not None else None
        allowed_dtypes = None
    original = ResBlock.forward
    observed = hits if hits is not None else Counter()

    def forward(module, x, x_mask=None):
        amp = torch.is_autocast_enabled("cuda")
        if (x_mask is not None or x.device.type != "cuda" or x.ndim != 3 or not x.is_contiguous()
            or not base._is_gfx1030(x.device.index)
            or (amp and torch.get_autocast_dtype("cuda") != torch.float16)
            or x.dtype not in ((torch.float16, torch.float32) if amp else (torch.float16,))):
            return original(module, x, x_mask)
        for c1, c2 in zip(module.convs1, module.convs2):
            first = _lookup(c1, x.shape, by_shape)
            second = _lookup(c2, x.shape, by_shape)
            pk = pair_key(first[0], second[0]) if first is not None and second is not None else None
            if (pk is None or (allowed is not None and pk not in allowed)
                or (allowed_dtypes is not None and str(x.dtype) not in allowed_dtypes.get(pk, []))):
                x = c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU_SLOPE)), LRELU_SLOPE)) + x
                continue
            # Access parametrized weights once, in the outer autograd graph.
            w1, w2, b1, b2 = c1.weight, c2.weight, c1.bias, c2.bias
            dtype_ok = all(v is None or v.dtype in ((torch.float16, torch.float32) if amp else (torch.float16,))
                           for v in (w1, w2, b1, b2))
            if not dtype_ok or not all(v is None or v.is_contiguous() for v in (w1, w2, b1, b2)):
                x = c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU_SLOPE)), LRELU_SLOPE)) + x
                continue
            info1 = (c1.dilation[0], c1.padding[0], c1.groups, first[1]["directions"])
            info2 = (c2.dilation[0], c2.padding[0], c2.groups, second[1]["directions"])
            observed[pk] = observed.get(pk, 0) + 1
            x = _FusedPair.apply(x, w1, b1, w2, b2, info1, info2, LRELU_SLOPE)
        return x

    ResBlock.forward = forward
    try:
        yield observed
    finally:
        ResBlock.forward = original
