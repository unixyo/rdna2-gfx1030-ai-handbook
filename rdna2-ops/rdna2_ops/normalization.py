"""NCHW GroupNorm followed by SiLU, preserving intermediate dtype rounding."""
from __future__ import annotations

import ctypes
import functools
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

_DTYPES = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}


@functools.lru_cache(maxsize=1)
def _library():
    path = Path(os.environ.get("RDNA2_OPS_LIBRARY", Path(__file__).with_name("lib") / "librdna2_ops.so"))
    if not path.is_file():
        raise RuntimeError("Build the HIP extension first: python -m rdna2_ops.build --hipcc /path/to/hipcc")
    lib = ctypes.CDLL(str(path))
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    lib.rdna2_group_norm_silu.argtypes = [ptr, ptr, ptr, ptr, integer, integer, integer, integer, ctypes.c_float, integer, ptr, ptr]
    lib.rdna2_group_norm_silu.restype = integer
    lib.rdna2_group_norm_silu_workspace_bytes.argtypes = [integer] * 4
    lib.rdna2_group_norm_silu_workspace_bytes.restype = ctypes.c_size_t
    return lib


@functools.lru_cache(maxsize=8)
def _device_supported(index):
    if not torch.version.hip:
        return False
    return torch.cuda.get_device_properties(index).gcnArchName.split(":")[0] == "gfx1030"


def _supported(x, groups, weight, bias, eps):
    # These transformations must retain PyTorch's symbolic/derivative rules.
    if torch.compiler.is_compiling() or torch.jit.is_tracing() or torch.jit.is_scripting() or torch._C._are_functorch_transforms_active():
        return False
    if getattr(torch.autograd.forward_ad, "_current_level", -1) >= 0:
        return False
    # Autocast owns dtype selection (GroupNorm may run in FP32).
    if torch.is_autocast_enabled("cuda"):
        return False
    if x.device.type != "cuda" or x.dtype not in _DTYPES or x.ndim < 2:
        return False
    if not x.is_contiguous() or x.numel() == 0:
        return False
    if not isinstance(groups, int) or groups < 1 or x.shape[1] % groups:
        return False
    if not math.isfinite(eps) or eps < 0 or x.numel() > 2**31 - 1:
        return False
    if x.numel() // groups <= 1:
        return False
    for p in (weight, bias):
        if p is not None and (p.ndim != 1 or p.numel() != x.shape[1] or p.dtype != x.dtype or p.device != x.device or not p.is_contiguous()):
            return False
    if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (x, weight, bias)):
        return False
    return _device_supported(x.device.index)


def group_norm_silu(x, num_groups, weight=None, bias=None, eps=1e-5, *, backend="auto"):
    """Return ``silu(group_norm(x, num_groups, weight, bias, eps))``.

    The HIP path supports contiguous NCHW (or flattened spatial dimensions),
    FP32/FP16/BF16, same-dtype optional affine parameters, and gfx1030.
    Statistics accumulate in FP32. The intermediate normalized result is
    rounded to x.dtype before SiLU, just as in two ordinary PyTorch calls.
    Autograd, autocast, torch.compile, active GPU graph capture, noncontiguous
    tensors, CPU, and other GPUs use native PyTorch. Compiler calls can thus
    use native graph fusion. Graph infrastructure should select
    ``backend='torch'`` for warmup, capture, and replay; warmup is not capture
    and replay does not call this Python dispatcher.
    ``backend='hip'`` raises instead of silently falling back for validation.
    No model, global dispatch table, or environment is modified.
    """
    if backend not in ("auto", "hip", "torch"):
        raise ValueError("backend must be 'auto', 'hip', or 'torch'")
    supported = backend != "torch" and _supported(x, num_groups, weight, bias, eps)
    if not supported:
        if backend == "hip":
            raise ValueError("HIP kernel requires gfx1030, contiguous FP32/FP16/BF16, valid groups and same-dtype affine tensors, with autograd disabled")
        return F.silu(F.group_norm(x, num_groups, weight, bias, eps))
    n, c = x.shape[:2]
    spatial = x.numel() // (n * c)
    with torch.cuda.device(x.device):
        if torch.cuda.is_current_stream_capturing():
            if backend == "hip":
                raise ValueError("HIP backend is eager-only; use backend='torch' for graph warmup, capture and replay")
            return F.silu(F.group_norm(x, num_groups, weight, bias, eps))
        lib = _library()
        out = torch.empty_like(x)
        nbytes = lib.rdna2_group_norm_silu_workspace_bytes(n, c, spatial, num_groups)
        workspace = torch.empty(nbytes // 4, device=x.device, dtype=torch.float32) if nbytes else None
        stream = torch.cuda.current_stream(x.device).cuda_stream
        rc = lib.rdna2_group_norm_silu(x.data_ptr(), weight.data_ptr() if weight is not None else None, bias.data_ptr() if bias is not None else None, out.data_ptr(), n, c, spatial, num_groups, eps, _DTYPES[x.dtype], stream, workspace.data_ptr() if workspace is not None else None)
        if rc:
            raise RuntimeError(f"RDNA2 GroupNorm+SiLU HIP launch failed: {rc}")
    return out


class GroupNormSiLU(torch.nn.GroupNorm):
    """Drop-in combination of a GroupNorm module and its following SiLU.

    This class retains the ordinary GroupNorm parameter/state_dict names.
    It supports training through the native differentiable PyTorch path.
    """
    def forward(self, x):
        return group_norm_silu(x, self.num_groups, self.weight, self.bias, self.eps)
