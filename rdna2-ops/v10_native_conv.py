"""Research-only remaining Applio convolution candidates for gfx1030.

The normal convolution view is [B, CI, H, Q], with a [CO, CI, R, 1]
filter. Q=1 covers Conv1d; discriminator period Conv2d keeps its NCHW
storage and period Q. ConvTranspose1d is the adjoint of the same operator.

Changes versus v08: dW tiles the flattened CI*R output dimension and uses
shape-sized deterministic splits; split=1 writes the FP16 output directly.
The backward-data kernel partitions output height by stride phase, so a
stride-10 transposed convolution only evaluates its 1 or 2 live filter taps.
No im2col tensor, atomics, hooks, persistent tensors, or automatic selection.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from v08_direct_conv import _cfg, _is_gfx1030, _metadata


@triton.jit
def _forward(X, W, BIAS, Y, B: tl.constexpr, CI: tl.constexpr,
             CO: tl.constexpr, H: tl.constexpr, Q: tl.constexpr,
             R: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
             P: tl.constexpr, HO: tl.constexpr, HAS_BIAS: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    batch = tl.program_id(1)
    oc = (pid % tl.cdiv(CO, BM)) * BM + tl.arange(0, BM)
    pos = (pid // tl.cdiv(CO, BM)) * BN + tl.arange(0, BN)
    oh, q = pos // Q, pos % Q
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(tl.cdiv(CI * R, BK)):
        cr = step * BK + kk
        ic, rr = cr // R, cr % R
        ih = oh[None, :] * S + rr[:, None] * D - P
        a = tl.load(W + oc[:, None] * (CI * R) + cr[None, :],
                    (oc[:, None] < CO) & (cr[None, :] < CI * R), 0.0)
        b = tl.load(X + ((batch * CI + ic[:, None]) * H + ih) * Q + q[None, :],
                    (ic[:, None] < CI) & (pos[None, :] < HO * Q) &
                    (ih >= 0) & (ih < H), 0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=tl.float32)
    result = acc.to(tl.float16)
    if HAS_BIAS:
        bias = tl.load(BIAS + oc, oc < CO, 0.0).to(tl.float32)
        result = (result.to(tl.float32) + bias[:, None]).to(tl.float16)
    tl.store(Y + (batch * CO + oc[:, None]) * (HO * Q) + pos[None, :],
             result, (oc[:, None] < CO) & (pos[None, :] < HO * Q))


@triton.jit
def _backward_data_phase(DY, W, BIAS, DX, B: tl.constexpr,
                         CI: tl.constexpr, CO: tl.constexpr,
                         H: tl.constexpr, Q: tl.constexpr,
                         R: tl.constexpr, S: tl.constexpr,
                         P: tl.constexpr, HO: tl.constexpr,
                         HAS_BIAS: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Dilation=1 only. A program owns one input-height residue modulo S.
    pid = tl.program_id(0)
    batch = tl.program_id(1)
    phase = tl.program_id(2)
    ic = (pid % tl.cdiv(CI, BM)) * BM + tl.arange(0, BM)
    pos = (pid // tl.cdiv(CI, BM)) * BN + tl.arange(0, BN)
    ih, q = (pos // Q) * S + phase, pos % Q
    first_r = (phase + P) % S
    RP: tl.constexpr = tl.cdiv(R, S)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(tl.cdiv(CO * RP, BK)):
        cr = step * BK + kk
        oc = cr // RP
        rr = first_r + (cr % RP) * S
        oh = (ih[None, :] + P - rr[:, None]) // S
        a = tl.load(W + (oc[None, :] * CI + ic[:, None]) * R + rr[None, :],
                    (ic[:, None] < CI) & (oc[None, :] < CO) & (rr[None, :] < R), 0.0)
        b = tl.load(DY + ((batch * CO + oc[:, None]) * HO + oh) * Q + q[None, :],
                    (oc[:, None] < CO) & (rr[:, None] < R) &
                    (ih[None, :] < H) & (oh >= 0) & (oh < HO), 0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=tl.float32)
    result = acc.to(tl.float16)
    if HAS_BIAS:
        bias = tl.load(BIAS + ic, ic < CI, 0.0).to(tl.float32)
        result = (result.to(tl.float32) + bias[:, None]).to(tl.float16)
    tl.store(DX + ((batch * CI + ic[:, None]) * H + ih[None, :]) * Q + q[None, :],
             result, (ic[:, None] < CI) & (ih[None, :] < H))


@triton.jit
def _backward_weight_flat(X, DY, OUT, B: tl.constexpr,
                          CI: tl.constexpr, CO: tl.constexpr,
                          H: tl.constexpr, Q: tl.constexpr,
                          R: tl.constexpr, S: tl.constexpr,
                          D: tl.constexpr, P: tl.constexpr, HO: tl.constexpr,
                          SPLIT: tl.constexpr,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    oc = (pid % tl.cdiv(CO, BM)) * BM + tl.arange(0, BM)
    cr = (pid // tl.cdiv(CO, BM)) * BN + tl.arange(0, BN)
    ic, rr = cr // R, cr % R
    kk = tl.arange(0, BK)
    STEPS: tl.constexpr = tl.cdiv(B * HO * Q, SPLIT * BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(STEPS):
        bp = (split * STEPS + step) * BK + kk
        batch = bp // (HO * Q)
        pos = bp % (HO * Q)
        oh, q = pos // Q, pos % Q
        ih = oh[:, None] * S + rr[None, :] * D - P
        a = tl.load(DY + (batch[None, :] * CO + oc[:, None]) * (HO * Q) + pos[None, :],
                    (oc[:, None] < CO) & (bp[None, :] < B * HO * Q), 0.0)
        b = tl.load(X + ((batch[:, None] * CI + ic[None, :]) * H + ih) * Q + q[:, None],
                    (cr[None, :] < CI * R) & (bp[:, None] < B * HO * Q) &
                    (ih >= 0) & (ih < H), 0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=tl.float32)
    offset = split * CO * CI * R + oc[:, None] * (CI * R) + cr[None, :]
    if SPLIT == 1:
        tl.store(OUT + offset, acc.to(tl.float16),
                 (oc[:, None] < CO) & (cr[None, :] < CI * R))
    else:
        tl.store(OUT + offset, acc, (oc[:, None] < CO) & (cr[None, :] < CI * R))


@triton.jit
def _reduce_weight(PART, DW, N: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    split = tl.arange(0, SPLIT)
    values = tl.load(PART + split[:, None] * N + col[None, :], col[None, :] < N, 0.0)
    tl.store(DW + col, tl.sum(values, axis=0).to(tl.float16), col < N)


def _scalar(value, default):
    return int(value[0] if isinstance(value, (list, tuple)) else value if value is not None else default)


def geometry(case):
    """Return normal-convolution geometry and original output shape (CPU only)."""
    shape, weight = tuple(case["shape"]), tuple(case["weight"])
    trans = "transpose" in case["kind"]
    if case.get("groups", 1) != 1 or len(shape) not in (3, 4):
        raise ValueError("Only ungrouped 1D or kernel-width-one 2D is supported")
    s, d, p = (_scalar(case.get(k), default) for k, default in
               (("stride", 1), ("dilation", 1), ("padding", 0)))
    op = _scalar(case.get("output_padding"), 0)
    if min(s, d) < 1 or p < 0 or op != 0:
        raise ValueError("Positive stride/dilation, nonnegative padding, output_padding=0 required")
    b, ci, h = shape[:3]
    q = shape[3] if len(shape) == 4 else 1
    r = weight[2]
    if len(shape) == 4:
        if trans or weight[3] != 1:
            raise ValueError("2D supports normal convolution with kernel width 1")
        for key, val in (("stride", 1), ("dilation", 1), ("padding", 0)):
            if len(case[key]) != 2 or int(case[key][1]) != val:
                raise ValueError("Width stride/dilation/padding must be 1/1/0")
    if trans:
        if len(shape) != 3 or weight[0] != ci or d != 1:
            raise ValueError("Transposed candidate supports contiguous 1D, dilation 1")
        co = weight[1]
        ho = (h - 1) * s - 2 * p + r
        normal = (b, co, ci, ho, 1, r, s, 1, p, h)
        out = (b, co, ho)
    else:
        if weight[1] != ci:
            raise ValueError("Input channel/weight mismatch")
        co = weight[0]
        ho = (h + 2 * p - d * (r - 1) - 1) // s + 1
        normal = (b, ci, co, h, q, r, s, d, p, ho)
        out = (b, co, ho, q) if len(shape) == 4 else (b, co, ho)
    if min(normal[:7]) < 1 or ho < 1:
        raise ValueError("Empty/invalid convolution")
    return normal, out


def _validate(x, w, dy, bias, case, direction):
    normal, out = geometry(case)
    for tensor in (x, w, dy, bias):
        if tensor is not None and (tensor.device != x.device or
                                   tensor.dtype != torch.float16 or not tensor.is_contiguous()):
            raise ValueError("Explicit contiguous FP16 tensors on one device required")
    if x.device.type != "cuda" or not _is_gfx1030(x.device.index):
        raise ValueError("ROCm gfx1030 required")
    if tuple(x.shape) != tuple(case["shape"]) or tuple(w.shape) != tuple(case["weight"]):
        raise ValueError("Case/tensor shape mismatch")
    if direction != "fwd" and (dy is None or tuple(dy.shape) != out):
        raise ValueError("Backward requires matching grad-output")
    if bias is not None and tuple(bias.shape) != (out[1],):
        raise ValueError("Bias shape mismatch")
    if max(x.numel(), w.numel(), math.prod(out)) >= 2**31:
        raise ValueError("32-bit element offsets required")
    return normal, out


def run(case, direction, x, w, dy=None, bias=None, *, config=None, metadata=None):
    """Run one explicit FP16 direction; AMP/autograd ownership remains with caller."""
    if direction not in ("fwd", "dx", "dw"):
        raise ValueError("Unknown direction")
    normal, output_shape = _validate(x, w, dy, bias, case, direction)
    trans = "transpose" in case["kind"]
    cfg = dict(config or {})
    algorithm = cfg.get("algorithm", "flat_dw" if direction == "dw" else "implicit_phase")
    if algorithm.startswith("v08_"):
        if len(x.shape) != 3 or trans or _scalar(case.get("stride"), 1) != 1:
            raise ValueError("v08 tile candidates require normal stride-one Conv1d")
        from v08_direct_conv import conv_fwd, conv_dw, conv_dx
        kw = dict(dilation=normal[7], padding=normal[8], groups=1, config=cfg, metadata=metadata)
        if algorithm in ("v08_small_split_dw", "v08_dw") and direction == "dw":
            result = conv_dw(x, dy, w.shape, **kw)
        elif algorithm in ("v08_wide_dx", "v08_dx") and direction == "dx":
            result = conv_dx(dy, w, x.shape, **kw)
        elif algorithm == "v08_fwd" and direction == "fwd":
            result = conv_fwd(x, w, bias, **kw)
        else:
            raise ValueError("v08 candidate/direction mismatch")
        if metadata is not None:
            metadata["algorithm"] = algorithm
        return result
    b, ci, co, h, q, r, s, d, p, ho = normal
    bm, bn, bk, nw = _cfg(cfg)
    launch = dict(num_warps=nw, num_stages=1, matrix_instr_nonkdim=0)
    scratch = 0
    if direction == "dw":
        if algorithm != "flat_dw":
            raise ValueError("dW requires flat_dw")
        split = int(cfg.get("split", 1))
        if split not in (1, 2, 4, 8, 16, 32, 64, 128):
            raise ValueError("Split must be a power of two up to 128")
        result = torch.empty_like(w)
        part = result if split == 1 else torch.empty((split, co, ci, r), device=x.device, dtype=torch.float32)
        scratch = 0 if split == 1 else part.numel() * part.element_size()
        xx, gg = (dy, x) if trans else (x, dy)
        k1 = _backward_weight_flat[(triton.cdiv(co, bm) * triton.cdiv(ci * r, bn), split)](
            xx, gg, part, *normal, split, bm, bn, bk, **launch)
        kernels = [k1]
        if split > 1:
            kernels.append(_reduce_weight[(triton.cdiv(result.numel(), 128),)](
                part, result, result.numel(), split, 128, num_warps=4, num_stages=1))
    elif (direction == "fwd") != trans:
        # Normal forward, or transposed backward-input.
        src = x if direction == "fwd" else dy
        b_arg = bias if direction == "fwd" else None
        result = torch.empty(output_shape if direction == "fwd" else tuple(x.shape),
                             device=x.device, dtype=torch.float16)
        kernels = [_forward[(triton.cdiv(co, bm) * triton.cdiv(ho * q, bn), b)](
            src, w, b_arg if b_arg is not None else src, result, *normal,
            b_arg is not None, bm, bn, bk, **launch)]
    else:
        # Normal backward-input, or transposed forward. Dilation one is explicit.
        if d != 1:
            raise ValueError("Phase backward-input supports dilation one only")
        src = x if direction == "fwd" else dy
        b_arg = bias if direction == "fwd" else None
        result = torch.empty(output_shape if direction == "fwd" else tuple(x.shape),
                             device=x.device, dtype=torch.float16)
        kernels = [_backward_data_phase[(triton.cdiv(ci, bm) * triton.cdiv(triton.cdiv(h, s) * q, bn), b, s)](
            src, w, b_arg if b_arg is not None else src, result,
            b, ci, co, h, q, r, s, p, ho, b_arg is not None, bm, bn, bk, **launch)]
    _metadata(kernels, metadata, cfg)
    if metadata is not None:
        metadata.update(algorithm=algorithm, normal_geometry=list(normal),
                        scratch_bytes=scratch, im2col_bytes=0, atomics=False,
                        accumulation="FP32 dot2; fixed FP32 split reduction; FP16 output")
    return result


def native_forward(x, w, bias, case):
    kwargs = {k: tuple(case[k]) for k in ("stride", "padding", "dilation")}
    kwargs["groups"] = case.get("groups", 1)
    if "transpose" in case["kind"]:
        kwargs["output_padding"] = tuple(case["output_padding"])
    return getattr(F, case["kind"])(x, w, bias, **kwargs)


class _CandidateConv(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
    def forward(ctx, x, w, bias, case, directions):
        ctx.save_for_backward(x, w)
        ctx.case, ctx.directions, ctx.has_bias = case, directions, bias is not None
        if "fwd" in directions:
            return run(case, "fwd", x, w, bias=bias, config=directions["fwd"])
        return native_forward(x, w, bias, case)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        dy = grad_output.contiguous()
        c, configs = ctx.case, ctx.directions
        need_x, need_w, need_b = ctx.needs_input_grad[:3]
        native_x, native_w = need_x and "dx" not in configs, need_w and "dw" not in configs
        dx = dw = db = None
        if native_x or native_w:
            gx, gw, _ = torch.ops.aten.convolution_backward.default(
                dy, x, w, None, c["stride"], c["padding"], c["dilation"],
                "transpose" in c["kind"], c["output_padding"], c.get("groups", 1),
                [native_x, native_w, False])
            if native_x:
                dx = gx
            if native_w:
                dw = gw
        if need_x and "dx" in configs:
            dx = run(c, "dx", x, w, dy, config=configs["dx"])
        if need_w and "dw" in configs:
            dw = run(c, "dw", x, w, dy, config=configs["dw"])
        if need_b and ctx.has_bias:
            db = dy.sum((0, *range(2, dy.ndim)))
        return dx, dw, db, None, None


def apply(x, w, bias, case, directions):
    """Explicit AMP/autograd boundary; callers must select and validate directions."""
    return _CandidateConv.apply(x, w, bias, case, directions)


def case_key(case):
    from v08_solver_select import case_id
    return case_id(case)


@contextmanager
def measured_dispatch(manifest_path, *, hits=None):
    """Opt-in per-direction manifest; baseline v08 directions may be embedded.

    Intended to be entered after the operating v08 context. Conv1d cases with
    a new dW can thus preserve their earlier v08 forward/dX selection. All
    unmatched geometries keep the original module method.
    """
    if os.environ.get("APPLIO_RDNA2_V10_NATIVE", "0") != "1":
        yield
        return
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("schema") != "applio_v10_native_dispatch" or manifest.get("strict_gate_pass") is not True:
        raise ValueError("Validated v10 native-candidate manifest required")
    if manifest.get("source_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Kernel source changed after validation")
    if manifest.get("baseline_v08_source_sha256") != hashlib.sha256(Path(__file__).with_name("v08_direct_conv.py").read_bytes()).hexdigest():
        raise ValueError("Embedded v08 directions belong to another kernel source")
    entries = manifest.get("entries", {})
    by_shape = {}
    active_kinds = set()
    for key, row in entries.items():
        if row.get("strict_gate_pass") is not True or not row.get("contract_pass") or not row.get("directions"):
            raise ValueError("Every entry must pass operator and AMP/weight_norm contract gates")
        if key != case_key(row["case"]) or set(row["directions"]) - {"fwd", "dx", "dw"}:
            raise ValueError("Manifest case/direction mismatch")
        c = row["case"]
        lookup_key = (c["kind"], tuple(c["shape"]), tuple(c["weight"]), tuple(c["stride"]),
                      tuple(c["padding"]), tuple(c["dilation"]), c["groups"], tuple(c["output_padding"]))
        by_shape[lookup_key] = (key, row)
        active_kinds.add(c["kind"])
    observed = hits if hits is not None else {}
    old1, old2, oldt = torch.nn.Conv1d._conv_forward, torch.nn.Conv2d._conv_forward, torch.nn.ConvTranspose1d.forward

    def find(module, x, weight_shape, kind):
        lookup_key = (kind, tuple(x.shape), tuple(weight_shape), module.stride,
                      module.padding, module.dilation, module.groups,
                      getattr(module, "output_padding", (0,) * (x.ndim - 2)))
        return by_shape.get(lookup_key)

    def lookup(module, x, w, bias, kind):
        selected = find(module, x, w.shape, kind)
        if selected is None:
            return None
        if (x.device.type != "cuda" or not x.is_contiguous() or not w.is_contiguous() or
            module.padding_mode != "zeros" or not _is_gfx1030(x.device.index)):
            return None
        amp = torch.is_autocast_enabled("cuda")
        if amp and torch.get_autocast_dtype("cuda") != torch.float16:
            return None
        allowed = (torch.float16, torch.float32) if amp else (torch.float16,)
        if any(t is not None and t.dtype not in allowed for t in (x, w, bias)):
            return None
        key, row = selected
        observed[key] = observed.get(key, 0) + 1
        return apply(x, w, bias, row["case"], row["directions"])

    def hook1(module, x, w, bias):
        y = lookup(module, x, w, bias, "conv1d")
        return old1(module, x, w, bias) if y is None else y

    def hook2(module, x, w, bias):
        y = lookup(module, x, w, bias, "conv2d")
        return old2(module, x, w, bias) if y is None else y

    def hookt(module, x, output_size=None):
        if output_size is not None:
            return oldt(module, x, output_size)
        weight_shape = (module.in_channels, module.out_channels // module.groups, *module.kernel_size)
        if find(module, x, weight_shape, "conv_transpose1d") is None:
            return oldt(module, x, output_size)
        y = lookup(module, x, module.weight, module.bias, "conv_transpose1d")
        return oldt(module, x, output_size) if y is None else y

    if "conv1d" in active_kinds:
        torch.nn.Conv1d._conv_forward = hook1
    if "conv2d" in active_kinds:
        torch.nn.Conv2d._conv_forward = hook2
    if "conv_transpose1d" in active_kinds:
        torch.nn.ConvTranspose1d.forward = hookt
    try:
        yield
    finally:
        torch.nn.Conv1d._conv_forward, torch.nn.Conv2d._conv_forward = old1, old2
        torch.nn.ConvTranspose1d.forward = oldt
