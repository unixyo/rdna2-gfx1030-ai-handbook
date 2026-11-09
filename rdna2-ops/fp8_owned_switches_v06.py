"""Bounded immutable-request FP8 -> owned FP16 linear cache experiment.

Only the existing unscaled-FP8 Anima research request is supported. On first
use, borrow the same resident FP8 weight through Comfy's staging context, copy
weight AND bias into independently owned FP16 tensors, then release the context.
On cache hit, use only those owned tensors: no raw FP8 staging/context reentry.
No weight, geometry, operating setting or persistent module cache is changed.
LoRA/weight patches, training, graph capture and model mutation are unsupported.
"""
from collections import Counter
from contextlib import contextmanager
import hashlib
from pathlib import Path
import torch

from dtype_switches_v06 import activate as slow_activate, version


def signature(value):
    if value is None:
        return None
    return (id(value), value.data_ptr(), version(value), tuple(value.shape),
            tuple(value.stride()), value.dtype, value.device)


@contextmanager
def activate(mode, spec, state, bound, out):
    if mode == "fp8_control":
        with slow_activate(mode, spec, state, bound, out):
            yield
        return
    if mode != "fp8_owned_linear":
        raise ValueError("Only native FP8 control and immutable owned linear experiment supported")
    if spec.get("model") != "anima" or not spec.get("immutable_model_during_request", False):
        raise ValueError("This bounded experiment requires the explicit immutable existing Anima request")
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        raise ValueError("Eager inference only")
    import comfy.ops
    owner = comfy.ops.disable_weight_init.Linear
    original = owner.forward_comfy_cast_weights
    cache = {}
    budget = int(spec.get("fp8_prepared_cache_bytes", 5 * 1024**3))
    stats = {"mode": mode, "hits": 0, "candidate_calls": 0, "preparations": 0,
             "cache_reuses": 0, "cache_bytes": 0, "max_cache_bytes": budget,
             "fallbacks": Counter(), "shapes": Counter(),
             "source_weight_dtype": Counter(), "geometry_weight_dtype": Counter(),
             "prepared_from_weight_contracts": Counter(),
             "context_weight_is_resident_storage": Counter(),
             "bias_owned_copies": 0, "cache_hit_context_entries": 0,
             "preparation_in_stage_time": True,
             "contract": "Existing immutable Anima, raw unscaled FP8 resident, FP16 activation; owned weight+bias per request; no LoRA or model mutation",
             "memory_scope": "per-request PyTorch allocator, excludes external DynamicVRAM allocator; explicit owned cache bytes recorded"}
    root = Path(__file__).resolve().parent
    stats["source_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                              for name in ("fp8_owned_switches_v06.py", "dtype_switches_v06.py",
                                           "rdna2_ops/fp8_weight_gemm_v06.py")}
    stats["cuda_allocated_bytes_at_entry"] = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    state["candidate"] = stats

    def forward(module, input):
        weight = getattr(module, "weight", None)
        bias_source = getattr(module, "bias", None)
        if type(weight) not in (torch.Tensor, torch.nn.Parameter):
            stats["fallbacks"]["nonplain_weight"] += 1
            return original(module, input)
        staged_dtype = getattr(weight, "_model_dtype", weight.dtype) if hasattr(module, "_v") else weight.dtype
        stats["source_weight_dtype"][str(weight.dtype)] += 1
        stats["geometry_weight_dtype"][str(staged_dtype)] += 1
        if (staged_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
                or input.dtype != torch.float16 or input.device.type != "cuda"
                or input.requires_grad or weight.ndim != 2 or input.shape[-1] != weight.shape[1]
                or bias_source is not None and type(bias_source) not in (torch.Tensor, torch.nn.Parameter)):
            stats["fallbacks"]["contract"] += 1
            return original(module, input)
        if (len(getattr(module, "weight_function", ())) or len(getattr(module, "bias_function", ()))
                or getattr(module, "weight_lowvram_function", None)
                or getattr(module, "bias_lowvram_function", None)):
            stats["fallbacks"]["patch_or_lora"] += 1
            return original(module, input)
        key = (id(module), signature(weight), signature(bias_source), staged_dtype,
               getattr(bias_source, "_model_dtype", None), input.device, input.dtype)
        stats["hits"] += 1
        stats["shapes"][str((tuple(input.shape), tuple(weight.shape)))] += 1
        prepared = cache.get(key)
        if prepared is not None:
            # Both tensors have independent storage and immutable request
            # lifetime. No vbar tensor escapes its staging context here.
            stats["cache_reuses"] += 1
            stats["candidate_calls"] += 1
            return torch.nn.functional.linear(input, prepared[0], prepared[1])
        with comfy.ops.CastBiasWeightContext(module, input, dtype=staged_dtype,
                                             bias_dtype=input.dtype, offloadable=True) as (w, bias):
            if type(w) not in (torch.Tensor, torch.nn.Parameter) or w.dtype != staged_dtype:
                raise RuntimeError("Unexpected resident weight contract")
            stats["prepared_from_weight_contracts"][str((str(w.dtype), tuple(w.shape), tuple(w.stride()), str(w.device)))] += 1
            resident = getattr(module, "_v_weight", None)
            same = isinstance(resident, torch.Tensor) and w.dtype == resident.dtype and w.data_ptr() == resident.data_ptr() and w.shape == resident.shape and w.stride() == resident.stride()
            stats["context_weight_is_resident_storage"][str(same)] += 1
            need = w.numel() * input.element_size() + (bias.numel() * input.element_size() if bias is not None else 0)
            free, _ = torch.cuda.mem_get_info(input.device)
            if stats["cache_bytes"] + need > budget or free < need + 1024**3:
                stats["fallbacks"]["cache_budget_or_free_margin"] += 1
                return torch.nn.functional.linear(input, w.to(input.dtype), bias)
            owned_weight = w.to(input.dtype)  # FP8->FP16 necessarily allocates.
            owned_bias = None if bias is None else bias.clone()
            prepared = (owned_weight, owned_bias)
            cache[key] = prepared
            stats["preparations"] += 1
            stats["cache_bytes"] += need
            stats["bias_owned_copies"] += int(bias is not None)
            stats["candidate_calls"] += 1
            return torch.nn.functional.linear(input, *prepared)

    owner.forward_comfy_cast_weights = forward
    try:
        yield
    finally:
        owner.forward_comfy_cast_weights = original
        stats["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        stats["prepared_entries"] = len(cache)
        stats["request_cache_released"] = True
        cache.clear()
