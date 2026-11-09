# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# int8_linear is adapted from comfy_kitchen 0.2.40:
# comfy_kitchen/backends/eager/quantization.py, int8_linear.
# Inspected upstream file SHA256:
# 4412eb5bbd12cb2c9a09c5c9e076e744e16bf39632fe624e4df739b84681ef12
# Changes: call the existing gfx1030 GEMM with weight.T directly; replace only
# the eager module export and provide optional statistics. Quantization, scale order,
# chunking, dtype conversion, bias, and residual behavior follow the original.

"""Kitchen eager INT8 linear without a weight-transpose copy.

No upstream function bodies, bytecode, package files, or weight storage are changed.
Backend identities, priorities, and capabilities remain unchanged. The replacement
consumes the current weight on every call, so it
retains no stale weight or transpose cache across Comfy offload/LoRA updates.
"""

import os
from threading import Lock

import torch
from comfy_kitchen.backends import eager as eager_backend
from comfy_kitchen.backends.eager import quantization as eager
from comfy_kitchen.registry import registry


_matmul = None
_original_export = None
_installed = False
_installation_lock = Lock()
_stats_enabled = os.environ.get("MOSS_RDNA2_INT8_STATS", "0") == "1"
_stats_lock = Lock()
_calls = 0
_layouts = {}


def _record_layout(weight, rhs):
    global _calls
    key = (tuple(weight.shape), tuple(weight.stride()), tuple(rhs.stride()), str(weight.device))
    with _stats_lock:
        _calls += 1
        _layouts[key] = _layouts.get(key, 0) + 1


def get_stats():
    """Return successful GEMM counts/layouts when MOSS_RDNA2_INT8_STATS=1.

    Statistics are disabled by default and never retain tensors or log per call.
    """
    with _stats_lock:
        return {
            "enabled": _stats_enabled,
            "calls": _calls,
            "weight_layouts": [
                {"shape": shape, "weight_stride": stride, "rhs_stride": rhs_stride,
                 "device": device, "dtype": "torch.int8", "calls": count}
                for (shape, stride, rhs_stride, device), count in _layouts.items()
            ],
        }


def reset_stats():
    global _calls
    with _stats_lock:
        _calls = 0
        _layouts.clear()


def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    convrot: bool = False,
    convrot_groupsize: int = 256,
    input_act: str | None = None,
    input_act_weight: torch.Tensor | None = None,
    input_act_eps: float = 0.0,
    residual: torch.Tensor | None = None,
    residual_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pinned eager INT8 linear semantics with a stride-aware weight.T GEMM."""
    if _matmul is None:
        raise RuntimeError("Install the MOSS INT8 eager export before calling it")
    # The measured production path has FP32 activations/output. Keep the
    # external FP16 text encoder and other unmeasured precision paths original.
    if x.dtype != torch.float32 or out_dtype != torch.float32:
        return eager.int8_linear(
            x, weight, weight_scale, bias=bias, out_dtype=out_dtype,
            convrot=convrot, convrot_groupsize=convrot_groupsize,
            input_act=input_act, input_act_weight=input_act_weight,
            input_act_eps=input_act_eps, residual=residual,
            residual_scale=residual_scale,
        )
    x = eager._apply_input_act(x, input_act, input_act_weight, input_act_eps)
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"Input and weight inner dimensions must match, got {x.shape[-1]} and {weight.shape[-1]}"
        )

    weight = weight.to(device=x.device).contiguous()
    weight_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1)
    if weight_scale.numel() not in (1, weight.shape[0]):
        raise ValueError(
            f"INT8 weight scale must be scalar or per-output-channel, got {tuple(weight_scale.shape)} "
            f"for weight shape {tuple(weight.shape)}"
        )

    if convrot:
        if x.shape[-1] % convrot_groupsize != 0:
            raise ValueError(
                f"ConvRot group size {convrot_groupsize} does not divide input features {x.shape[-1]}"
            )
        h = eager._build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        x = eager._rotate_activation(x, h, convrot_groupsize)

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    x_8, x_scale = eager.quantize_int8_rowwise(x_2d)
    rhs = weight.T
    result = _matmul(x_8, rhs)
    if _stats_enabled:
        _record_layout(weight, rhs)

    # Preserve the original FP32 scaling order and bounded intermediate size.
    m, n = result.shape
    chunk_size = max(1, min(m, 256 * 1024 * 1024 // (n * 4)))
    weight_scale = weight_scale.reshape(1, -1)
    scaled_parts = []
    for i in range(0, m, chunk_size):
        end_i = min(i + chunk_size, m)
        chunk = result[i:end_i].float()
        chunk_scales = x_scale[i:end_i].to(device=chunk.device, dtype=torch.float32) * weight_scale
        chunk_scaled = chunk * chunk_scales
        chunk_scaled = chunk_scaled.to(out_dtype)
        scaled_parts.append(chunk_scaled)
    result = torch.cat(scaled_parts, dim=0)

    if bias is not None:
        result = result + bias.to(device=result.device, dtype=result.dtype).reshape(1, -1)
    result = result.reshape(*orig_shape[:-1], weight.shape[0])
    return eager._apply_residual(result, residual, residual_scale)


def install_backend(strided_int8_matmul):
    """Replace only the existing eager module's exported int8_linear callable.

    The custom-node entry point validates gfx1030 and installs the legacy GEMM
    first. The public Kitchen API and its Torch custom op resolve the selected
    backend's exported function through getattr on every call. This changes no
    registry identity, priority, constraint, or native-fusion capability signal.
    quantization.int8_linear and any direct aliases remain untouched.

    Call only before starting Comfy's worker or between completed workloads.
    """
    global _matmul, _original_export, _installed
    if not callable(strided_int8_matmul):
        raise TypeError("strided_int8_matmul must be callable")
    with _installation_lock:
        if _installed:
            if _matmul is not strided_int8_matmul:
                raise RuntimeError("MOSS INT8 export is already bound to a different GEMM")
            if eager_backend.int8_linear is not int8_linear:
                raise RuntimeError("Kitchen eager int8_linear changed while this patch was installed")
            return
        if eager._int8_matmul_accumulate is not strided_int8_matmul:
            raise RuntimeError("Install the existing gfx1030 GEMM workaround first")
        original = eager.int8_linear
        if eager_backend.int8_linear is not original:
            raise RuntimeError("Expected Kitchen's unmodified eager int8_linear export")
        if registry.get_implementation("int8_linear", backend="eager") is not original:
            raise RuntimeError("Kitchen eager dispatch does not resolve the expected export")

        _original_export = original
        _matmul = strided_int8_matmul
        try:
            eager_backend.int8_linear = int8_linear
            if registry.get_implementation("int8_linear", backend="eager") is not int8_linear:
                raise RuntimeError("Kitchen eager dispatch did not pick up its replacement export")
        except Exception:
            eager_backend.int8_linear = original
            _original_export = None
            _matmul = None
            raise
        _installed = True


def disable_backend():
    """Restore the original eager export, retaining the legacy GEMM workaround.

    For a persistent rollback, restart with MOSS_RDNA2_INT8_LAYOUT=legacy.
    Call only between workloads. Refuse to overwrite another patch's export.
    """
    global _installed
    with _installation_lock:
        if not _installed:
            return
        if eager_backend.int8_linear is not int8_linear:
            raise RuntimeError("Kitchen eager int8_linear changed; refusing to overwrite it")
        eager_backend.int8_linear = _original_export
        _installed = False
