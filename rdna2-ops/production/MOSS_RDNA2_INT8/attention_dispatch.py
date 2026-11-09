"""Comfy-specific SDPA dispatch; altered precision is measurement-only."""
import collections
import logging
import os
import torch

MODE = os.environ.get("MOSS_RDNA2_ATTENTION", "native")
MEASURE = bool(os.environ.get("MOSS_RDNA2_MEASURE_DIR"))
IN_DENOISE = False
COUNTS = collections.Counter()
LAYOUTS = {}


def install():
    import comfy.ops
    from . import math_attention
    from torch.nn.attention import SDPBackend, sdpa_kernel
    experimental = {"fp16_math", "fp16_math_reduced", "bf16_math", "fp16_flash", "fp16_efficient"}
    if MODE not in {"native", "auto"} | experimental:
        raise ValueError("Unknown MOSS_RDNA2_ATTENTION mode: " + MODE)
    if MODE in experimental and not MEASURE:
        raise ValueError("Precision experiments require MOSS_RDNA2_MEASURE_DIR")
    original = comfy.ops.scaled_dot_product_attention

    def dispatch(q, k, v, *args, **kwargs):
        if MEASURE and IN_DENOISE and tuple(q.shape) not in LAYOUTS:
            LAYOUTS[tuple(q.shape)] = {name: {"shape": list(t.shape), "stride": list(t.stride()),
                                             "offset": t.storage_offset(), "dtype": str(t.dtype)}
                                      for name, t in zip(("q", "k", "v"), (q, k, v))}
        mask = kwargs.get("attn_mask", args[0] if args else None)
        dropout = kwargs.get("dropout_p", args[1] if len(args) > 1 else 0.0)
        causal = kwargs.get("is_causal", args[2] if len(args) > 2 else False)
        eligible = MODE != "native" and math_attention._eligible(
            q, k, v, mask, dropout, causal, kwargs.get("enable_gqa", False))
        selected = eligible and math_attention._auto_shape_supported(q, k, v)
        if MODE == "auto" and selected:
            if MEASURE:
                COUNTS["exact_fp32"] += 1
            return math_attention.scaled_dot_product_attention(q, k, v, *args, **kwargs)
        if MODE in experimental and IN_DENOISE and eligible and q.shape[1] == 24 and q.shape[-1] == 128:
            dtype = torch.bfloat16 if MODE == "bf16_math" else torch.float16
            backend = (SDPBackend.FLASH_ATTENTION if MODE == "fp16_flash" else
                       SDPBackend.EFFICIENT_ATTENTION if MODE == "fp16_efficient" else SDPBackend.MATH)
            if MEASURE:
                COUNTS[MODE] += 1
            previous = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
            try:
                torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(MODE == "fp16_math_reduced")
                with sdpa_kernel(backend):
                    out = torch.nn.functional.scaled_dot_product_attention(
                        q.to(dtype), k.to(dtype), v.to(dtype), *args, **kwargs)
                return out.to(q.dtype)
            finally:
                torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous)
        if MEASURE and IN_DENOISE:
            COUNTS["native_denoise_" + str(q.dtype)] += 1
        return original(q, k, v, *args, **kwargs)

    comfy.ops.scaled_dot_product_attention = dispatch
    logging.getLogger("moss-rdna2").warning("MOSS_RDNA2_ATTENTION=%s", MODE)

