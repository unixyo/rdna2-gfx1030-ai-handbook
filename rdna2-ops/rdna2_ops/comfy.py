"""Temporary GroupNorm+SiLU integration with ComfyUI's casting/offload rules."""
from contextlib import contextmanager
import torch
from .normalization import _supported, group_norm_silu

_MISSING = object()


def _native_transform(x):
    return (torch.compiler.is_compiling()
            or torch.jit.is_tracing() or torch.jit.is_scripting()
            or torch._C._are_functorch_transforms_active()
            or getattr(torch.autograd.forward_ad, "_current_level", -1) >= 0
            or torch.is_autocast_enabled("cuda")
            or (torch.is_grad_enabled() and x.requires_grad))


def _norm_forward(norm, original, activate, ops, backend, stats):
    def forward(x):
        if backend == "torch" or _native_transform(x):
            if stats is not None:
                stats["native_calls"] = stats.get("native_calls", 0) + 1
            return activate(original(x))
        if x.device.type == "cuda":
            with torch.cuda.device(x.device):
                if torch.cuda.is_current_stream_capturing():
                    return activate(original(x))

        def apply(weight, bias):
            if stats is not None:
                eligible = _supported(x, norm.num_groups, weight, bias, norm.eps)
                key = "hip_calls" if eligible else "native_calls"
                stats[key] = stats.get(key, 0) + 1
            return group_norm_silu(x, norm.num_groups, weight, bias, norm.eps, backend=backend)

        if ops is None:
            return apply(norm.weight, norm.bias)
        # Match Comfy's original GroupNorm forward including interrupt polling,
        # weight functions, dynamic prefetch, offload stream lifetime, and dtype.
        ops.run_every_op()
        needs_cast = (norm.comfy_cast_weights or len(norm.weight_function) > 0
                      or len(norm.bias_function) > 0)
        if needs_cast:
            if stats is not None:
                stats["cast_calls"] = stats.get("cast_calls", 0) + 1
            with ops.CastBiasWeightContext(norm, x, offloadable=True) as (weight, bias):
                return apply(weight, bias)
        return apply(norm.weight, norm.bias)
    return forward


@contextmanager
def fuse_comfy_vae_decoder(decoder, *, backend="auto", stats=None):
    """Temporarily fuse compatible 2-D VAE blocks without changing module paths.

    Module/parameter identities and state_dict names remain intact even while
    active. Comfy casting, weight functions, prefetch and release use its own
    CastBiasWeightContext. Unknown forward overrides and hooked blocks are
    skipped. Native transforms and graph capture call the original forwards.
    Use backend="torch" for the full warmup/capture/replay lifecycle.

    This instance must not be used concurrently or nested. Every original
    forward attribute is restored on exit, including exceptions. No global
    Comfy class, source file, model weight or dispatch registry is patched.
    The optional stats dict counts actual HIP/native/cast calls for validation.
    """
    if backend not in ("auto", "torch"):
        raise ValueError("Comfy integration backend must be auto or torch")
    if getattr(decoder, "_rdna2_fusion_active", False):
        raise RuntimeError("This decoder already has an active fusion context")
    try:
        import comfy.ops as ops
    except ImportError:
        ops = None
    from comfy.ldm.modules.diffusionmodules import model as comfy_model
    from torch.nn.modules import module as module_api
    if any(getattr(module_api, name, {}) for name in (
            "_global_forward_hooks", "_global_forward_pre_hooks",
            "_global_backward_hooks", "_global_backward_pre_hooks")):
        raise ValueError("Global module hooks require original norm/activation boundaries")
    comfy_forward = ops.disable_weight_init.GroupNorm.forward if ops is not None else None
    comfy_cast_forward = ops.disable_weight_init.GroupNorm.forward_comfy_cast_weights if ops is not None else None
    plan = []
    touched = set()
    for block in list(decoder.modules()):
        cls = type(block)
        if cls.__name__ != "ResnetBlock" or cls.__module__ != "comfy.ldm.modules.diffusionmodules.model":
            continue
        if getattr(block.forward, "__func__", None) is not comfy_model.ResnetBlock.forward:
            continue
        if getattr(block, "fused_norm", False):
            served = getattr(comfy_model, "fused_norm_served", None)
            # The installed gfx1030 runtime sets fused_norm=True but has no
            # served Kitchen FP16 conv backend; its ordinary path still runs.
            if served is None or any(served(torch.device("cuda"), dtype)
                                     for dtype in (torch.float16, torch.bfloat16, torch.float32)):
                continue
        if hasattr(block, "temb_proj") or type(block.swish) is not torch.nn.SiLU:
            continue
        norms = (block.norm1, block.norm2)
        if not all(isinstance(n, torch.nn.GroupNorm) for n in norms):
            continue
        modules = (*norms, block.swish)
        if any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks or m._backward_pre_hooks for m in modules):
            continue
        if any(id(m) in touched for m in modules):
            continue
        if getattr(block.swish.forward, "__func__", None) is not torch.nn.SiLU.forward:
            continue
        kinds = []
        for norm in norms:
            fn = getattr(norm.forward, "__func__", None)
            if type(norm) is torch.nn.GroupNorm and fn is torch.nn.GroupNorm.forward:
                kinds.append(None)
            elif (comfy_forward is not None and fn is comfy_forward
                  and getattr(norm.forward_comfy_cast_weights, "__func__", None) is comfy_cast_forward):
                kinds.append(ops)
            else:
                break
        if len(kinds) != 2:
            continue
        plan.append((block, norms, kinds))
        touched.update(id(m) for m in modules)
    if not plan:
        raise ValueError("No compatible unhooked ComfyUI VAE ResnetBlocks found")
    saved = []
    decoder._rdna2_fusion_active = True
    try:
        for block, norms, kinds in plan:
            activate = block.swish.forward
            for norm, norm_ops in zip(norms, kinds):
                original = norm.forward
                saved.append((norm, norm.__dict__.get("forward", _MISSING)))
                norm.forward = _norm_forward(norm, original, activate, norm_ops, backend, stats)
            saved.append((block.swish, block.swish.__dict__.get("forward", _MISSING)))
            block.swish.forward = lambda x: x
        if stats is not None:
            stats["blocks"] = len(plan)
        yield len(plan)
    finally:
        for module, old in reversed(saved):
            if old is _MISSING:
                del module.forward
            else:
                module.forward = old
        del decoder._rdna2_fusion_active
