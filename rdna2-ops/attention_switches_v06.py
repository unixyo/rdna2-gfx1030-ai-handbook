"""Process-local Anima Attention experiment contexts; no operating edits."""
from collections import Counter
from contextlib import contextmanager
import importlib
import hashlib
from pathlib import Path
import torch


@contextmanager
def activate(mode, spec, state, bound, out):
    from rdna2_ops.fp16_attention_v06 import fp16_attention, METHODS
    method = mode.removeprefix("attn_")
    if method not in METHODS:
        raise ValueError("Unknown Attention experiment " + mode)
    ops = importlib.import_module("comfy.ops")
    original = ops.scaled_dot_product_attention
    stats = {"method": method, "hits": 0, "fallbacks": 0, "shapes": Counter()}
    stats["source_sha256"] = {
        name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
        for name in ("attention_switches_v06.py", "rdna2_ops/fp16_attention_v06.py")}
    state["candidate"] = stats

    def invoke(q, k, v, *args, **kwargs):
        mask = args[0] if args else kwargs.get("attn_mask")
        dropout = args[1] if len(args) > 1 else kwargs.get("dropout_p", 0.)
        causal = args[2] if len(args) > 2 else kwargs.get("is_causal", False)
        okay = (len(args) <= 3 and all(isinstance(x, torch.Tensor) for x in (q,k,v))
                and q.ndim == 4 and q.dtype == k.dtype == v.dtype == torch.float16
                and q.shape[-1] in (64,128) and q.shape[:2] == k.shape[:2]
                and k.shape == v.shape and q.shape[-1] == k.shape[-1]
                and mask is None and not dropout and not causal
                and (method != "prescale4d_native"
                     or torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
                and not kwargs.get("enable_gqa", False)
                and set(kwargs).issubset({"attn_mask","dropout_p","is_causal","scale","enable_gqa"}))
        if not okay:
            stats["fallbacks"] += 1
            return original(q,k,v,*args,**kwargs)
        stats["hits"] += 1
        stats["shapes"][str((tuple(q.shape),tuple(k.shape)))] += 1
        return fp16_attention(q,k,v,method=method,scale=kwargs.get("scale"))
    ops.scaled_dot_product_attention = invoke
    try:
        yield
    finally:
        ops.scaled_dot_product_attention = original
        stats["shapes"] = dict(stats["shapes"])
