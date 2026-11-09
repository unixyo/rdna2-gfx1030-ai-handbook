"""Explicit BF16 GEMM alternatives for gfx1030; no automatic dispatch changes.

All public paths return BF16. ``fp16_f32`` uses ROCm half inputs with direct
FP32 output before rounding to BF16; ``fp16_f16`` also rounds through an FP16
output. ``dot2`` casts each loaded BF16 value to FP16 in-register and accumulates
with FP32 dot2. Its range precondition is exact FP16 representability of inputs.
``simt_fp32`` converts loads to FP32 and uses scalar-FMA dot lowering.
The default exact-range check for half paths synchronizes. Explicitly choosing
require_exact_fp16=False permits input rounding/underflow but still rejects
overflow/nonfinite values when check_fp16=True. Benchmark range checks are
reported separately; no accuracy guarantee follows from finite inputs alone.
No weight lifetime, LoRA, autograd, compile or graph support is implied.
"""
from functools import lru_cache
import torch
import triton
import triton.language as tl
from .gemm import _kernel_info


@triton.jit
def _bf16_mm(A, B, Bias, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             AB: tl.constexpr, AM: tl.constexpr, AK: tl.constexpr,
             BB: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
             HALF: tl.constexpr, HAS_BIAS: tl.constexpr,
             TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr):
    pid, batch = tl.program_id(0), tl.program_id(1)
    nm = tl.cdiv(M, TM)
    rows = pid % nm * TM + tl.arange(0, TM)
    cols = pid // nm * TN + tl.arange(0, TN)
    inner = tl.arange(0, TK)
    ap = A + batch * AB + rows[:, None] * AM + inner[None, :] * AK
    bp = B + batch * BB + inner[:, None] * BK + cols[None, :] * BN
    acc = tl.zeros((TM, TN), tl.float32)
    for step in range(tl.cdiv(K, TK)):
        kk = step * TK + inner
        av = tl.load(ap, (rows[:, None] < M) & (kk[None, :] < K), 0)
        bv = tl.load(bp, (kk[:, None] < K) & (cols[None, :] < N), 0)
        if HALF:
            av, bv = av.to(tl.float16), bv.to(tl.float16)
        else:
            av, bv = av.to(tl.float32), bv.to(tl.float32)
        acc = tl.dot(av, bv, acc, input_precision="ieee", out_dtype=tl.float32)
        ap += TK * AK
        bp += TK * BK
    if HAS_BIAS:
        acc += tl.load(Bias + cols, cols < N, 0).to(tl.float32)[None, :]
    tl.store(C + batch * M * N + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


@lru_cache(None)
def _arch(index):
    return torch.cuda.get_device_properties(index).gcnArchName.split(":")[0]


def fp16_exact_inputs(*tensors):
    """Check finite values and an exact BF16 -> FP16 -> BF16 round trip."""
    return all(bool(torch.isfinite(t).all()) and torch.equal(t, t.half().bfloat16())
               for t in tensors)


def fp16_finite_inputs(*tensors):
    """Finite FP16 conversion; unlike exact_inputs this permits underflow/rounding."""
    return all(bool(torch.isfinite(t.half()).all()) for t in tensors)


def _validate(a, b, bias):
    if a.ndim not in (2, 3) or b.ndim != a.ndim or a.shape[-1] != b.shape[-2]:
        raise ValueError("Expected same-rank [M,K]@[K,N] or [B,M,K]@[B,K,N]")
    if a.ndim == 3 and a.shape[0] != b.shape[0]:
        raise ValueError("Batches must match; broadcasting is not implemented")
    if bias is not None and a.ndim != 2:
        raise ValueError("Bias is supported only for 2D addmm")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError("BF16 operands required")
    if a.device.type != "cuda" or b.device != a.device or not torch.version.hip:
        raise ValueError("Inputs must share the existing HIP device")
    if _arch(a.device.index) != "gfx1030":
        raise ValueError("This explicitly measured implementation targets gfx1030")
    if (torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing()
            or torch.jit.is_tracing() or torch.jit.is_scripting()
            or torch.is_autocast_enabled("cuda")
            or torch._C._are_functorch_transforms_active()
            or getattr(torch.autograd.forward_ad, "_current_level", -1) >= 0
            or torch.is_grad_enabled() and (a.requires_grad or b.requires_grad)):
        raise ValueError("Only explicit eager inference is supported")
    for t in (a, b):
        if min(t.shape) <= 0 or any(s < 0 for s in t.stride()):
            raise ValueError("Nonempty, nonnegative-stride inputs required")
        if sum((n-1)*s for n, s in zip(t.shape, t.stride())) >= 2**31:
            raise ValueError("Supported address span is signed 32-bit elements")
    if bias is not None and (bias.shape != (b.shape[-1],) or bias.device != a.device
                             or bias.dtype not in (torch.bfloat16, torch.float32)
                             or not bias.is_contiguous() or bias.requires_grad):
        raise ValueError("Bias must be an inference BF16/FP32 vector of length N")


def bf16_gemm(a, b, *, bias=None, method="fp32", check_fp16=True, require_exact_fp16=True,
              tile=None, num_warps=4, metadata=None):
    """Explicit alternatives, including their per-call input/output conversions.

    Methods: native, fp32, fp16_f32, fp16_f16, dot2, simt_fp32. Half modes
    reject values which would overflow or change on FP16 re-expression by
    default. Explicit require_exact_fp16=False allows lossy input conversion;
    its finite check still rejects overflow/NaN/Inf. Disabling check_fp16 requires
    independently validating the current inputs under the chosen contract.
    All modes preserve the BF16 output dtype. Bias addition is in FP32 except
    for native addmm. Only alpha=beta=1 vector bias semantics are supported.
    """
    _validate(a, b, bias)
    methods = {"native", "fp32", "fp16_f32", "fp16_f16", "dot2", "simt_fp32"}
    if method not in methods:
        raise ValueError("Unknown method: " + method)
    if method in {"fp16_f32", "fp16_f16", "dot2"} and check_fp16:
        if require_exact_fp16 and not fp16_exact_inputs(a, b):
            raise ValueError("BF16 input is not exactly representable as finite FP16")
        if not require_exact_fp16 and not fp16_finite_inputs(a, b):
            raise ValueError("BF16 input becomes nonfinite in FP16")
    mm = torch.mm if a.ndim == 2 else torch.bmm
    if method == "native":
        if bias is not None and a.ndim == 2:
            return torch.addmm(bias.to(a.dtype), a, b)
        value = mm(a, b)
        return value if bias is None else value + bias.to(a.dtype)
    if method in {"fp32", "fp16_f32", "fp16_f16"}:
        if method == "fp32":
            value = mm(a.float(), b.float())
        elif method == "fp16_f32":
            value = mm(a.half(), b.half(), out_dtype=torch.float32)
        else:
            value = mm(a.half(), b.half()).float()
        if bias is not None:
            value = value + bias.float()
        return value.bfloat16()
    m, k = a.shape[-2:]
    n = b.shape[-1]
    batch = a.shape[0] if a.ndim == 3 else 1
    if m * n * batch >= 2**31:
        raise ValueError("Output exceeds supported indexing range")
    tile = tile or ((64, 64, 32) if method == "dot2" else (32, 32, 16))
    if (len(tile) != 3 or any(v not in (16, 32, 64, 128) for v in tile)
            or num_warps not in (4, 8)):
        raise ValueError("Unsupported tile/warps")
    tm, tn, tk = tile
    result = torch.empty((*a.shape[:-2], m, n), dtype=a.dtype, device=a.device)
    kernel = _bf16_mm[(triton.cdiv(m, tm) * triton.cdiv(n, tn), batch)](
        a, b, bias if bias is not None else a, result, m, n, k,
        a.stride(0) if a.ndim == 3 else 0, *a.stride()[-2:],
        b.stride(0) if b.ndim == 3 else 0, *b.stride()[-2:],
        method == "dot2", bias is not None, tm, tn, tk,
        num_warps=num_warps, num_stages=1, matrix_instr_nonkdim=0)
    if metadata is not None:
        metadata.update(_kernel_info(kernel), method=method, tile=list(tile),
                        num_warps=num_warps, input_dtype="bf16", output_dtype="bf16",
                        fp16_range_check_in_call=bool(check_fp16 and method == "dot2"),
                        require_exact_fp16=bool(require_exact_fp16))
    return result
