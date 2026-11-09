"""Inference-only FP32 math attention for profiled gfx1030 workloads.

Keeps native Q/K pre-scaling, rocBLAS batch matmuls and native softmax.
A small kernel repairs only all-negative-infinity rows, avoiding the math
backend's full-size isneginf/where intermediate tensors. Explicit fallback retains PyTorch SDPA for every unsupported contract.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

_NATIVE_SDPA = F.scaled_dot_product_attention


def _eligible(q, k, v, mask, dropout_p, causal, enable_gqa):
    if torch.compiler.is_compiling() or torch.jit.is_tracing() or torch.jit.is_scripting() or torch._C._are_functorch_transforms_active():
        return False
    if getattr(torch.autograd.forward_ad, "_current_level", -1) >= 0:
        return False
    if mask is not None or dropout_p != 0.0 or causal or enable_gqa:
        return False
    if any(t.ndim != 4 or t.dtype != torch.float32 for t in (q, k, v)):
        return False
    if q.device.type != "cuda" or any(t.device != q.device for t in (k, v)):
        return False
    if not torch.version.hip or torch.is_autocast_enabled("cuda"):
        return False
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        return False
    if q.shape[0] != 1 or k.shape[:2] != q.shape[:2] or v.shape[:2] != q.shape[:2]:
        return False
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        return False
    if not (0 < q.shape[-2] and 0 < k.shape[-2] <= 4096):
        return False
    if q.shape[-1] == 0 or v.shape[-1] == 0:
        return False
    return torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0] == "gfx1030"


def _auto_shape_supported(q, k, v):
    return (q.ndim == 4 and q.shape[0] == 1 and q.shape[1] == 24
            and q.shape[-1] == 128 and 128 <= q.shape[-2] <= 4096
            and k.shape == q.shape and v.shape == q.shape
            and all(t.stride(-1) == 1 and t.stride(1) == 128 for t in (q, k, v))
            and q.stride(2) == k.stride(2) == 24 * 128
            and v.stride(2) in (24 * 128, 3 * 24 * 128))


def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0,
                                 is_causal=False, *, scale=None,
                                 enable_gqa=False, backend="auto"):
    """Same forward contract as torch SDPA; native path preserves gradients.

    auto selects measured FP32 [1,24,S,128], 128 <= S <= 4096,
    with interleaved Q/K and interleaved or packed-QKV V layouts.
    rdna2 requests the general eligible FP32 math path for benchmarking.
    Native SDPA handles compilation, masked/causal/dropout attention, other
    dtypes/devices, AD and every shape without measured automatic dispatch.
    This materializes the attention matrix, like PyTorch's math backend.
    """
    if backend not in ("auto", "rdna2", "torch"):
        raise ValueError("backend must be auto, rdna2, or torch")
    eligible = backend != "torch" and _eligible(q, k, v, attn_mask, dropout_p, is_causal, enable_gqa)
    measured = _auto_shape_supported(q, k, v)
    if not eligible or (backend == "auto" and not measured):
        if backend == "rdna2" and not eligible:
            raise ValueError("RDNA2 math attention requires inference, gfx1030, batch-one FP32, no mask/dropout/causal/GQA, K length <=4096")
        return _NATIVE_SDPA(q, k, v, attn_mask, dropout_p, is_causal,
                            scale=scale, enable_gqa=enable_gqa)
    from ._softmax import repair_all_negative_inf_rows_
    heads, length, dim = q.shape[1:]
    keys = k.shape[-2]
    actual_scale = 1.0 / math.sqrt(dim) if scale is None else scale
    root_scale = math.sqrt(abs(actual_scale))
    q3 = (q * (-root_scale if actual_scale < 0 else root_scale)).reshape(heads, length, dim)
    k3 = (k * root_scale).reshape(heads, keys, dim)
    v3 = v.reshape(heads, keys, v.shape[-1])
    scores = torch.bmm(q3, k3.transpose(1, 2))
    probabilities = torch.softmax(scores, dim=-1)
    repair_all_negative_inf_rows_(scores, probabilities)
    return torch.bmm(probabilities, v3).unsqueeze(0)

