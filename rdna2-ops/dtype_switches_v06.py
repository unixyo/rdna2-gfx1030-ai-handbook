"""Owned-request FP8 storage adapters for research_comfy_v05.py.

These modes require plain, unscaled FP8 weights, FP16 activations, no LoRA or
weight functions, and an immutable model during the context. The FP16 prepared
copy is scoped to one denoising request and released on exit. No module weight,
runtime source, operating defaults, or global pointer cache is modified.
"""
from collections import Counter
from contextlib import contextmanager
import hashlib
from pathlib import Path
import torch


def version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        return None


@contextmanager
def bf16_linear_mode(mode, state):
    """Replace the actual F.linear call after Comfy's existing cast/offload path."""
    from rdna2_ops.bf16_gemm_v06 import bf16_gemm
    methods = {"bf16_control": "native", "bf16_fp32": "fp32", "bf16_fp16_f16": "fp16_f16"}
    method = methods[mode]
    original = torch.nn.functional.linear
    stats = {"mode": mode, "method": method, "eligible_hits": 0, "candidate_calls": 0,
             "fallbacks": Counter(), "shapes": Counter(), "actual_dtypes": Counter(),
             "input_range_validation_in_timing": method == "fp16_f16",
             "half_output_overflow_not_excluded_by_input_finite_check": method == "fp16_f16"}
    root = Path(__file__).resolve().parent
    stats["source_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                              for name in ("dtype_switches_v06.py", "rdna2_ops/bf16_gemm_v06.py")}
    state["candidate"] = stats

    def linear(input, weight, bias=None):
        stats["actual_dtypes"][str((str(input.dtype), str(weight.dtype)))] += 1
        if (type(input) not in (torch.Tensor, torch.nn.Parameter)
                or type(weight) not in (torch.Tensor, torch.nn.Parameter)
                or input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
                or input.device.type != "cuda" or weight.device != input.device
                or input.ndim < 2 or weight.ndim != 2 or input.shape[-1] != weight.shape[-1]
                or bias is not None and (bias.dtype != torch.bfloat16 or bias.ndim != 1
                                        or not bias.is_contiguous())):
            stats["fallbacks"]["contract"] += 1
            return original(input, weight, bias)
        stats["eligible_hits"] += 1
        stats["shapes"][str((tuple(input.shape), tuple(weight.shape), tuple(input.stride()), tuple(weight.stride())))] += 1
        if method == "native":
            return original(input, weight, bias)
        a = input.reshape(-1, input.shape[-1])
        try:
            result = bf16_gemm(a, weight.T, bias=bias, method=method,
                               require_exact_fp16=False, check_fp16=True)
        except ValueError as exc:
            if "becomes nonfinite" in str(exc):
                stats["fallbacks"]["fp16_input_range"] += 1
                return original(input, weight, bias)
            raise
        stats["candidate_calls"] += 1
        return result.reshape(*input.shape[:-1], weight.shape[0])

    torch.nn.functional.linear = linear
    try:
        yield
    finally:
        torch.nn.functional.linear = original


@contextmanager
def activate(mode, spec, state, bound, out):
    if mode.startswith("bf16_"):
        model = bound["model"].model
        old_dtype = model.manual_cast_dtype
        force = bool(spec.get("bf16_request_compute_override", False))
        if force:
            model.manual_cast_dtype = torch.bfloat16
        try:
            with bf16_linear_mode(mode, state):
                state["candidate"]["previous_compute_dtype"] = str(old_dtype)
                state["candidate"]["request_compute_dtype"] = str(model.get_dtype_inference())
                state["candidate"]["request_compute_override"] = force
                yield
        finally:
            model.manual_cast_dtype = old_dtype
            state["candidate"]["restored_compute_dtype"] = str(model.manual_cast_dtype)
        return
    import comfy.ops
    from rdna2_ops.fp8_weight_gemm_v06 import decode_fp8_weight, fp8_weight_gemm, prepare_int8_weight
    choices = {"fp8_control", "fp8_prepared_fp16", "fp8_decode", "fp8_fused_dot2", "fp8_prepared_int8"}
    if mode not in choices:
        raise ValueError("Unknown FP8 experiment mode: " + mode)
    owner = comfy.ops.disable_weight_init.Linear
    original = owner.forward_comfy_cast_weights
    cache = {}
    shapes, fallbacks = Counter(), Counter()
    budget = int(spec.get("fp8_prepared_cache_bytes", 5 * 1024**3))
    int8_shapes = {tuple(map(int, shape)) for shape in spec.get("fp8_int8_allowed_shapes", [])}
    if mode == "fp8_prepared_int8" and not int8_shapes:
        raise ValueError("INT8 model research requires explicit operator-measured (M,N,K) shapes")
    stats = {"mode": mode, "hits": 0, "candidate_calls": 0, "cache_reuses": 0,
             "preparations": 0, "cache_bytes": 0,
             "max_cache_bytes": budget,
             "int8_allowed_shapes": sorted(int8_shapes),
             "contract": "plain unscaled FP8 weight, FP16 activation, immutable model, request-local prepared storage",
             "preparation_in_stage_time": True,
             "native_source_weight_dtype": Counter(), "geometry_weight_dtype": Counter(),
             "actual_resident_weight_dtype": Counter(), "actual_weight_contracts": Counter(),
             "context_weight_is_resident_storage": Counter(), "shapes": shapes, "fallbacks": fallbacks}
    root = Path(__file__).resolve().parent
    stats["source_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                              for name in ("dtype_switches_v06.py", "rdna2_ops/fp8_weight_gemm_v06.py",
                                           "rdna2_ops/int8_gemm.py")}
    state["candidate"] = stats

    # These are per-request PyTorch allocator peaks. DynamicVRAM allocations
    # outside the PyTorch allocator are not included; cache_bytes is explicit.
    stats["cuda_allocated_bytes_at_entry"] = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    stats["memory_scope"] = "per-request PyTorch allocator, excludes external DynamicVRAM allocator; explicit prepared cache bytes also recorded"

    def record_resident(w, module, *, context_return=False):
        stats["actual_resident_weight_dtype"][str(w.dtype)] += 1
        stats["actual_weight_contracts"][str((str(w.dtype), tuple(w.shape), tuple(w.stride()), str(w.device)))] += 1
        if context_return:
            resident = getattr(module, "_v_weight", None)
            key = "no_resident_tensor"
            if isinstance(resident, torch.Tensor):
                key = str(w.dtype == resident.dtype and w.data_ptr() == resident.data_ptr()
                          and w.shape == resident.shape and w.stride() == resident.stride())
            stats["context_weight_is_resident_storage"][key] += 1

    def forward(module, input):
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or type(weight) not in (torch.Tensor, torch.nn.Parameter):
            fallbacks["nonplain_weight"] += 1
            return original(module, input)
        stats["native_source_weight_dtype"][str(weight.dtype)] += 1
        # DynamicVRAM keeps the mmap parameter in its original dtype and uses
        # TensorGeometry._model_dtype for the resident representation. Reading
        # the raw parameter dtype alone would silently miss every FP8 call.
        staged_dtype = getattr(weight, "_model_dtype", weight.dtype) if hasattr(module, "_v") else weight.dtype
        stats["geometry_weight_dtype"][str(staged_dtype)] += 1
        if staged_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            fallbacks["not_fp8"] += 1
            return original(module, input)
        if input.dtype != torch.float16 or input.device.type != "cuda" or input.requires_grad:
            fallbacks["activation_contract"] += 1
            return original(module, input)
        if (len(getattr(module, "weight_function", ())) or len(getattr(module, "bias_function", ()))
                or getattr(module, "weight_lowvram_function", None)
                or getattr(module, "bias_lowvram_function", None)):
            fallbacks["weight_patch_or_lora"] += 1
            return original(module, input)
        if weight.ndim != 2 or input.shape[-1] != weight.shape[1]:
            fallbacks["shape"] += 1
            return original(module, input)
        mnk = (input.numel() // input.shape[-1], weight.shape[0], weight.shape[1])
        if mode == "fp8_prepared_int8" and mnk not in int8_shapes:
            fallbacks["unmeasured_or_slower_shape"] += 1
            return original(module, input)
        key = (id(module), id(weight), weight.data_ptr(), version(weight), tuple(weight.shape),
               tuple(weight.stride()), weight.dtype, staged_dtype, input.device)
        shape = str((tuple(input.shape), tuple(weight.shape), str(staged_dtype)))
        stats["hits"] += 1
        shapes[shape] += 1
        if mode == "fp8_control":
            value = original(module, input)
            record_resident(getattr(module, "_v_weight", weight), module)
            return value
        # Ask the existing Comfy staging/offload context for raw FP8 weights;
        # keep its pin/unpin and stream lifetime instead of bypassing offload.
        with comfy.ops.CastBiasWeightContext(module, input, dtype=staged_dtype,
                                             bias_dtype=input.dtype, offloadable=True) as (w, bias):
            record_resident(w, module, context_return=True)
            if type(w) not in (torch.Tensor, torch.nn.Parameter) or w.dtype != staged_dtype:
                raise RuntimeError("The observed Comfy cast path changed the raw FP8 contract")
            if mode == "fp8_decode":
                stats["candidate_calls"] += 1
                return torch.nn.functional.linear(input, decode_fp8_weight(w), bias)
            if mode == "fp8_fused_dot2":
                stats["candidate_calls"] += 1
                value = fp8_weight_gemm(input.reshape(-1, input.shape[-1]), w,
                                        bias=bias, method="fused_dot2")
                return value.reshape(*input.shape[:-1], w.shape[0])
            prepared = cache.get(key)
            if prepared is None:
                need = w.numel() * 2 if mode == "fp8_prepared_fp16" else w.numel() + w.shape[0] * 4
                free, _ = torch.cuda.mem_get_info(input.device)
                if stats["cache_bytes"] + need > budget or free < need + 1024**3:
                    fallbacks["prepared_cache_budget_or_free_margin"] += 1
                    return torch.nn.functional.linear(input, w.to(input.dtype), bias)
                if mode == "fp8_prepared_int8" and not bool(torch.isfinite(w.float()).all()):
                    fallbacks["nonfinite_weight"] += 1
                    return torch.nn.functional.linear(input, w.to(input.dtype), bias)
                prepared = w.to(input.dtype) if mode == "fp8_prepared_fp16" else prepare_int8_weight(w)
                cache[key] = prepared
                stats["preparations"] += 1
                stats["cache_bytes"] += need
            else:
                stats["cache_reuses"] += 1
            if mode == "fp8_prepared_fp16":
                stats["candidate_calls"] += 1
                return torch.nn.functional.linear(input, prepared, bias)
            if not bool(torch.isfinite(input).all()):
                fallbacks["nonfinite_activation"] += 1
                return torch.nn.functional.linear(input, w.to(input.dtype), bias)
            stats["candidate_calls"] += 1
            value = fp8_weight_gemm(input.reshape(-1, input.shape[-1]), w,
                                    bias=bias, method="prepared_int8", prepared=prepared)
            return value.reshape(*input.shape[:-1], w.shape[0])

    owner.forward_comfy_cast_weights = forward
    try:
        yield
    finally:
        owner.forward_comfy_cast_weights = original
        stats["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        stats["prepared_entries"] = len(cache)
        stats["request_cache_released"] = True
        cache.clear()
