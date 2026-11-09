"""Explicit FP16, FP32-accumulating tiled GEMM for gfx1030 inference.

This is a measurement candidate, not an automatic replacement for rocBLAS.
It uses the existing Triton installation; it does not build/install a runtime.
The gfx1030 target lowers tl.dot through SIMD rather than MFMA/WMMA. Inspect
the optional metadata to see whether this compiler emitted dot2 or scalar FMA;
having an FP16 input does not by itself prove use of packed dot instructions.
"""
from collections import Counter
from functools import lru_cache
import hashlib
import re

import torch
import triton
import triton.language as tl


@triton.jit
def _fp16_gemm(
    A, B, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    AM: tl.constexpr, AK: tl.constexpr,
    BK: tl.constexpr, BN: tl.constexpr,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
):
    # Consecutive programs reuse a B tile across M tiles.
    m_tiles = tl.cdiv(M, TILE_M)
    pid = tl.program_id(0)
    rows = (pid % m_tiles) * TILE_M + tl.arange(0, TILE_M)
    cols = (pid // m_tiles) * TILE_N + tl.arange(0, TILE_N)
    inner = tl.arange(0, TILE_K)
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for step in range(tl.cdiv(K, TILE_K)):
        kk = step * TILE_K + inner
        av = tl.load(A + rows[:, None] * AM + kk[None, :] * AK,
                     mask=(rows[:, None] < M) & (kk[None, :] < K), other=0.0)
        bv = tl.load(B + kk[:, None] * BK + cols[None, :] * BN,
                     mask=(kk[:, None] < K) & (cols[None, :] < N), other=0.0)
        acc = tl.dot(av, bv, acc, input_precision="ieee", out_dtype=tl.float32)
    tl.store(C + rows[:, None] * N + cols[None, :], acc.to(tl.float16),
             mask=(rows[:, None] < M) & (cols[None, :] < N))


@lru_cache(maxsize=None)
def _gfx1030(device_index):
    return bool(torch.version.hip) and (
        torch.cuda.get_device_properties(device_index).gcnArchName.split(":")[0]
        == "gfx1030"
    )


def _kernel_info(kernel):
    asm = kernel.asm.get("amdgcn", "")
    instructions = Counter(re.findall(r"^\s*(v_[a-z0-9_]+)\s", asm, re.MULTILINE))
    selected = {name: count for name, count in instructions.items()
                if name.startswith(("v_dot", "v_fma", "v_mfma", "v_wmma"))}
    return {
        "target": "gfx1030",
        "assembly_available": bool(asm),
        "assembly_sha256": hashlib.sha256(asm.encode()).hexdigest() if asm else None,
        "instruction_counts_static": selected,
        "dot2_present": any(name.startswith("v_dot2") for name in instructions),
        "matrix_instructions_present": any(
            name.startswith(("v_mfma", "v_wmma")) for name in instructions),
        "registers": getattr(kernel, "n_regs", None),
        "spills": getattr(kernel, "n_spills", None),
        "shared_bytes": getattr(getattr(kernel, "metadata", None), "shared", None),
    }


def fp16_gemm(a, b, *, out=None, tile=(32, 64, 32), num_warps=4, metadata=None):
    """Return a @ b with FP16 inputs/output and an FP32 accumulator.

    Explicit eager inference API; unsupported devices/transforms raise rather
    than silently changing the benchmark's backend. Positive-stride 2-D input
    views, including a weight transpose, need no materialized transpose. An
    optional contiguous, non-aliasing FP16 ``out`` buffer avoids allocation.
    ``metadata`` is populated after launch; omit it from measured repetitions.
    Tile selection is fixed and explicit; there is no hidden autotuning.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Expected A[M,K] and B[K,N]")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("FP16 inputs are required")
    if a.device.type != "cuda" or b.device != a.device or not _gfx1030(a.device.index):
        raise ValueError("This kernel requires the existing ROCm gfx1030 device")
    if (torch.compiler.is_compiling() or torch.jit.is_tracing() or torch.jit.is_scripting()
            or torch._C._are_functorch_transforms_active()
            or getattr(torch.autograd.forward_ad, "_current_level", -1) >= 0
            or torch.is_autocast_enabled("cuda")
            or (torch.is_grad_enabled() and (a.requires_grad or b.requires_grad))):
        raise ValueError("fp16_gemm is an explicit eager inference operator")
    m, k = a.shape
    n = b.shape[1]
    if min(m, n, k) <= 0 or max(m * k, k * n, m * n) >= 2**31:
        raise ValueError("Use nonempty matrices with fewer than 2**31 elements")
    for tensor in (a, b):
        if min(tensor.stride()) <= 0 or sum(
                (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())) >= 2**31:
            raise ValueError("Input strides must be positive with 32-bit element offsets")
    if (len(tile) != 3 or any(v < 16 or v > 128 or v & (v - 1) for v in tile)
            or num_warps not in (1, 2, 4, 8)):
        raise ValueError("Tile dimensions must be powers of two in [16,128]; warps in {1,2,4,8}")
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=torch.float16)
    elif (out.shape != (m, n) or out.device != a.device or out.dtype != torch.float16
          or not out.is_contiguous() or out.requires_grad):
        raise ValueError("out must be a contiguous, inference FP16 [M,N] tensor on the input device")
    if any(out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr() for t in (a, b)):
        raise ValueError("out must not share input storage")
    bm, bn, bk = tile
    with torch.cuda.device(a.device):
        kernel = _fp16_gemm[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
            a, b, out, m, n, k, *a.stride(), *b.stride(), bm, bn, bk,
            num_warps=num_warps, num_stages=1, matrix_instr_nonkdim=0,
        )
    if metadata is not None:
        metadata.update(_kernel_info(kernel), tile=list(tile), num_warps=num_warps,
                        num_stages=1, matrix_instr_nonkdim=0)
    return out
