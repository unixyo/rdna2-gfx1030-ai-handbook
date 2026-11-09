# SPDX-License-Identifier: Apache-2.0
"""Explicit gfx1030 INT8 GEMM with an optional FP32 dequantization epilogue.

The two scale products retain their order: float(acc) * (row_scale * col_scale).
rounding="separate" disables multiply/add contraction (MOSS eager semantics).
rounding="fused" permits the patientx Triton epilogue's FMA behavior.
No quantization, weight cache, model registration, or operating defaults change.
Only eager forward, INT8 inputs and FP32 dequantized outputs are implemented.
"""
from contextlib import contextmanager
from functools import lru_cache
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def _int8_dot_dequant(
    A, B, C, AS, BS, BIAS,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    AM: tl.constexpr, AK: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
    BM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr, GROUP_M: tl.constexpr,
    DEQUANT: tl.constexpr, SCALAR_B: tl.constexpr, HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, TN)
    group = pid // (GROUP_M * nn)
    first_m = group * GROUP_M
    group_m = tl.minimum(nm - first_m, GROUP_M)
    in_group = pid % (GROUP_M * nn)
    pm = first_m + in_group % group_m
    pn = in_group // group_m
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * TN + tl.arange(0, TN)
    rk = tl.arange(0, TK)
    ap = A + rm[:, None] * AM + rk[None, :] * AK
    bp = B + rk[:, None] * BK + rn[None, :] * BN
    acc = tl.zeros((BM, TN), tl.int32)
    for ki in range(tl.cdiv(K, TK)):
        if K % TK == 0:
            av = tl.load(ap, rm[:, None] < M, other=0)
            bv = tl.load(bp, rn[None, :] < N, other=0)
        else:
            kval = ki * TK + rk
            av = tl.load(ap, (rm[:, None] < M) & (kval[None, :] < K), other=0)
            bv = tl.load(bp, (kval[:, None] < K) & (rn[None, :] < N), other=0)
        acc = tl.dot(av, bv, acc)
        ap += TK * AK
        bp += TK * BK
    if DEQUANT:
        sa = tl.load(AS + rm, rm < M, other=0).to(tl.float32)
        if SCALAR_B:
            sb = tl.load(BS).to(tl.float32)
        else:
            sb = tl.load(BS + rn, rn < N, other=0).to(tl.float32)
        scale = sa[:, None] * sb if SCALAR_B else sa[:, None] * sb[None, :]
        value = acc.to(tl.float32) * scale
        if HAS_BIAS:
            bias = tl.load(BIAS + rn, rn < N, other=0).to(tl.float32)
            value = value + bias[None, :]
    else:
        value = acc
    tl.store(C + rm[:, None] * N + rn[None, :], value,
             (rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _int8_dot_dequant_wrapped(
    A, B, C, AS, BS, BIAS,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    AM: tl.constexpr, AK: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
    BM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr, GROUP_M: tl.constexpr,
    DEQUANT: tl.constexpr, SCALAR_B: tl.constexpr, HAS_BIAS: tl.constexpr,
    ADD_DOT: tl.constexpr,
):
    # Load every M/N lane from a valid row/column. The final partial tile wraps
    # its unused loads, but raw output indices remain masked, avoiding duplicate
    # stores. This isolates the effect of M/N load predicates from tile size.
    pid = tl.program_id(0)
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, TN)
    group = pid // (GROUP_M * nn)
    first_m = group * GROUP_M
    group_m = tl.minimum(nm - first_m, GROUP_M)
    in_group = pid % (GROUP_M * nn)
    rm = (first_m + in_group % group_m) * BM + tl.arange(0, BM)
    rn = (in_group // group_m) * TN + tl.arange(0, TN)
    load_m, load_n = rm % M, rn % N
    rk = tl.arange(0, TK)
    ap = A + load_m[:, None] * AM + rk[None, :] * AK
    bp = B + rk[:, None] * BK + load_n[None, :] * BN
    acc = tl.zeros((BM, TN), tl.int32)
    for ki in range(tl.cdiv(K, TK)):
        if K % TK == 0:
            av = tl.load(ap)
            bv = tl.load(bp)
        else:
            kval = ki * TK + rk
            av = tl.load(ap, kval[None, :] < K, other=0)
            bv = tl.load(bp, kval[:, None] < K, other=0)
        if ADD_DOT:
            acc += tl.dot(av, bv)
        else:
            acc = tl.dot(av, bv, acc)
        ap += TK * AK
        bp += TK * BK
    if DEQUANT:
        sa = tl.load(AS + load_m).to(tl.float32)
        if SCALAR_B:
            sb = tl.load(BS).to(tl.float32)
        else:
            sb = tl.load(BS + load_n).to(tl.float32)
        scale = sa[:, None] * sb if SCALAR_B else sa[:, None] * sb[None, :]
        value = acc.to(tl.float32) * scale
        if HAS_BIAS:
            bias = tl.load(BIAS + load_n).to(tl.float32)
            value = value + bias[None, :]
    else:
        value = acc
    tl.store(C + rm[:, None] * N + rn[None, :], value,
             (rm[:, None] < M) & (rn[None, :] < N))


@lru_cache(None)
def _arch(device_index):
    return torch.cuda.get_device_properties(device_index).gcnArchName.split(":")[0]


def _validate(a, b, tile, num_warps, num_stages, group_m):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Expected [M,K] @ [K,N]")
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise TypeError("Both GEMM operands must be INT8")
    if a.device.type != "cuda" or b.device != a.device:
        raise ValueError("Inputs must share a HIP device")
    if _arch(a.device.index) != "gfx1030":
        raise ValueError("This explicitly measured kernel targets gfx1030")
    if any(s < 0 for s in a.stride() + b.stride()):
        raise ValueError("Negative strides are unsupported")
    if a.shape[1] > 131071:
        raise ValueError("K could overflow signed INT32 for arbitrary signed INT8 inputs")
    if len(tile) != 3 or any(v < 16 or v > 256 or v & (v - 1) for v in tile):
        raise ValueError("Tile dimensions must be powers of two in [16,256]")
    if num_warps not in (4, 8) or num_stages not in (1, 2, 3) or group_m not in (1, 2, 4, 8):
        raise ValueError("Unsupported launch configuration")
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Only explicit eager calls are supported")


def _launch(a, b, a_scale, b_scale, bias, *, tile, num_warps,
            num_stages, group_m, rounding, return_kernel, wrapped=False, partial_dot=False):
    _validate(a, b, tile, num_warps, num_stages, group_m)
    if partial_dot and not wrapped:
        raise ValueError("partial_dot requires the explicitly wrapped-load variant")
    dequant = a_scale is not None
    m, k = a.shape
    n = b.shape[1]
    if rounding not in ("separate", "fused"):
        raise ValueError("rounding must be separate or fused")
    if dequant:
        for value, size, label in ((a_scale, m, "a_scale"), (b_scale, n, "b_scale")):
            if not isinstance(value, torch.Tensor):
                raise TypeError(label + " must be a device tensor")
            if value.device != a.device or value.dtype != torch.float32 or not value.is_contiguous():
                raise ValueError(label + " must be contiguous FP32 on the input device")
            if value.numel() != size and not (label == "b_scale" and value.numel() == 1):
                raise ValueError(label + " has the wrong number of elements")
            if torch.is_grad_enabled() and value.requires_grad:
                raise ValueError("Autograd is unsupported")
        if bias is not None:
            if (bias.device != a.device or bias.numel() != n or not bias.is_contiguous()
                    or bias.dtype not in (torch.float32, torch.float16, torch.bfloat16)):
                raise ValueError("Bias must be a contiguous floating point vector of length N")
            if torch.is_grad_enabled() and bias.requires_grad:
                raise ValueError("Autograd is unsupported")
    out = torch.empty((m, n), device=a.device, dtype=torch.float32 if dequant else torch.int32)
    if m == 0 or n == 0:
        return (out, None) if return_kernel else out
    bm, tn, tk = tile
    dummy = a
    kernel_fn = _int8_dot_dequant_wrapped if wrapped else _int8_dot_dequant
    extra_meta = {"ADD_DOT": partial_dot} if wrapped else {}
    kernel = kernel_fn[(triton.cdiv(m, bm) * triton.cdiv(n, tn),)](
        a, b, out, a_scale if dequant else dummy, b_scale if dequant else dummy,
        bias if bias is not None else dummy,
        m, n, k, *a.stride(), *b.stride(), bm, tn, tk, group_m,
        dequant, dequant and b_scale.numel() == 1, bias is not None,
        num_warps=num_warps, num_stages=num_stages,
        enable_fp_fusion=rounding == "fused", **extra_meta)
    return (out, kernel) if return_kernel else out


def int8_matmul(a, b, *, tile=(32, 64, 64), num_warps=4, num_stages=1,
                group_m=4, return_kernel=False, wrapped=False, partial_dot=False):
    """Exact INT8 -> INT32 matmul; accepts both [K,N] layouts without copying."""
    return _launch(a, b, None, None, None, tile=tile, num_warps=num_warps,
                   num_stages=num_stages, group_m=group_m, rounding="separate",
                   return_kernel=return_kernel, wrapped=wrapped, partial_dot=partial_dot)


def int8_gemm_dequant(a, b, a_scale, b_scale, bias=None, *, tile=(32, 64, 64),
                     num_warps=4, num_stages=1, group_m=4,
                     rounding="separate", return_kernel=False, wrapped=False, partial_dot=False):
    """INT32 GEMM and row/scalar or row/column FP32 dequantization in one kernel.

    Inputs are already quantized. A fresh contiguous output is allocated.
    Weight preparation and activation quantization remain the caller's responsibility.
    Finite scales/bias are a precondition; no GPU-wide finite scan is inserted.
    """
    return _launch(a, b, a_scale, b_scale, bias, tile=tile, num_warps=num_warps,
                   num_stages=num_stages, group_m=group_m, rounding=rounding,
                   return_kernel=return_kernel, wrapped=wrapped, partial_dot=partial_dot)


@contextmanager
def moss_int8_gemm_candidate(*, tile=(32, 64, 64), num_warps=4,
                             num_stages=1, group_m=4, wrapped=False, partial_dot=False):
    """Scoped MOSS eager export replacement in an exclusively owned process.

    Preserves Kitchen's input activation, ConvRot, row quantizer and residual helpers.
    FP32 input/output only; all other calls use the saved exported callable.
    No registry, code object, source file or model weight is changed.
    """
    from comfy_kitchen.backends import eager as backend
    from comfy_kitchen.backends.eager import quantization as q
    original = backend.int8_linear
    stats = {"hits": 0, "fallbacks": 0, "shapes": {}}

    def candidate(x, weight, weight_scale, bias=None, out_dtype=torch.bfloat16,
                  convrot=False, convrot_groupsize=256, input_act=None,
                  input_act_weight=None, input_act_eps=0.0,
                  residual=None, residual_scale=None):
        if x.dtype != torch.float32 or out_dtype != torch.float32:
            stats["fallbacks"] += 1
            return original(x, weight, weight_scale, bias=bias, out_dtype=out_dtype,
                            convrot=convrot, convrot_groupsize=convrot_groupsize,
                            input_act=input_act, input_act_weight=input_act_weight,
                            input_act_eps=input_act_eps, residual=residual,
                            residual_scale=residual_scale)
        x = q._apply_input_act(x, input_act, input_act_weight, input_act_eps)
        if x.shape[-1] != weight.shape[-1]:
            raise ValueError("Input and weight inner dimensions differ")
        weight = weight.to(device=x.device).contiguous()
        ws = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1)
        if ws.numel() not in (1, weight.shape[0]):
            raise ValueError("Weight scale must be scalar or per output channel")
        if convrot:
            if x.shape[-1] % convrot_groupsize:
                raise ValueError("ConvRot group size does not divide K")
            h = q._build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
            x = q._rotate_activation(x, h, convrot_groupsize)
        shape = x.shape
        a, sa = q.quantize_int8_rowwise(x.reshape(-1, shape[-1]))
        bias32 = None if bias is None else bias.to(device=x.device, dtype=torch.float32).reshape(-1)
        y = int8_gemm_dequant(a, weight.T, sa, ws, bias32, tile=tile,
                             num_warps=num_warps, num_stages=num_stages,
                             group_m=group_m, rounding="separate", wrapped=wrapped, partial_dot=partial_dot)
        stats["hits"] += 1
        key = str((a.shape[0], weight.shape[0], a.shape[1]))
        stats["shapes"][key] = stats["shapes"].get(key, 0) + 1
        return q._apply_residual(y.reshape(*shape[:-1], weight.shape[0]), residual, residual_scale)

    backend.int8_linear = candidate
    try:
        yield stats
    finally:
        if backend.int8_linear is not candidate:
            raise RuntimeError("MOSS eager export changed during the research context")
        backend.int8_linear = original


@contextmanager
def qwen_int8_gemm_candidate(patch_module=None, *, tile=(32, 64, 64),
                             num_warps=4, num_stages=1, group_m=4, wrapped=False, partial_dot=False):
    """Scoped replacement of the patientx bridge's two GEMM wrappers.

    The original patientx row quantizer stays in use. The patch module is discovered
    by its imported source path, or can be supplied explicitly. No import side effects
    are used to install patientx; the normal diagnostic server must load it first.
    """
    if patch_module is None:
        found = [m for m in tuple(sys.modules.values()) if m is not None
                 and str(getattr(m, "__file__", "")).endswith(
                     "RDNA2-Qwen21-INT8/rocm_int8_kitchen_patch.py")]
        if len(found) != 1:
            raise RuntimeError("Expected one already-loaded Qwen patientx bridge")
        patch_module = found[0]
    originals = {name: getattr(patch_module, name) for name in
                 ("triton_int8_linear", "triton_int8_linear_per_row")}
    stats = {"hits": 0, "fallbacks": 0, "shapes": {}}
    replacements = {}

    def make_wrapper(original, per_row):
        quantize = original.__globals__["triton_quantize_rowwise"]

        def candidate(x, weight, weight_scale, bias=None, compute_dtype=torch.float16):
            if x.dtype != torch.float32 or compute_dtype != torch.float32:
                stats["fallbacks"] += 1
                return original(x, weight, weight_scale, bias=bias, compute_dtype=compute_dtype)
            shape = x.shape
            x2d = x.reshape(-1, shape[-1])
            a, sa = quantize(x2d.float() if per_row else x2d)
            if per_row:
                ws = weight_scale.reshape(weight.shape[0]).contiguous()
            elif isinstance(weight_scale, torch.Tensor):
                ws = weight_scale.to(x.device, non_blocking=True).reshape(-1)
            else:
                ws = torch.tensor([weight_scale], device=x.device, dtype=torch.float32)
            y = int8_gemm_dequant(a, weight.T, sa, ws, bias, tile=tile,
                                 num_warps=num_warps, num_stages=num_stages,
                                 group_m=group_m, rounding="fused", wrapped=wrapped, partial_dot=partial_dot)
            stats["hits"] += 1
            key = str((a.shape[0], weight.shape[0], a.shape[1]))
            stats["shapes"][key] = stats["shapes"].get(key, 0) + 1
            return y.reshape(*shape[:-1], weight.shape[0])
        return candidate

    for name, original in originals.items():
        replacements[name] = make_wrapper(original, name.endswith("per_row"))
        setattr(patch_module, name, replacements[name])
    try:
        yield stats
    finally:
        for name in originals:
            if getattr(patch_module, name) is not replacements[name]:
                raise RuntimeError("Qwen patientx callable changed during the research context")
        for name, original in originals.items():
            setattr(patch_module, name, original)
