"""Rehearsal candidates for Qwen-Image-2.1 split attention (scratch only)."""
import math
import torch

_state = {"orig": None, "hits": 0, "fallbacks": 0}


def split_view(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False,
               skip_output_reshape=False, **kw):
    """attention_split without the q/k/v .contiguous() copies and the r1 scratch.

    Same op sequence as ComfyUI attention_split for FP32, batch 1, no mask,
    one slice: einsum(q,k)*scale -> softmax -> einsum(s,v). Others fall back.
    """
    orig = _state["orig"]
    if (mask is not None or skip_reshape or skip_output_reshape or kw.get("enable_gqa")
            or q.dtype != torch.float32 or q.ndim != 3 or q.shape[0] != 1
            or k.dtype != torch.float32 or v.dtype != torch.float32):
        _state["fallbacks"] += 1
        return orig(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                    skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kw)
    from comfy import model_management
    inner = q.shape[-1]
    d = inner // heads
    scale = kw.get("scale", d ** -0.5)
    need = q.shape[1] * k.shape[1] * heads * 4 * 3
    if need > model_management.get_free_memory(q.device):
        _state["fallbacks"] += 1
        return orig(q, k, v, heads, mask=mask, attn_precision=attn_precision, **kw)
    # Contiguous per-head copies as in attention_split (strided rocBLAS was 2x slower).
    f = lambda t: t.view(1, -1, heads, d).permute(0, 2, 1, 3).reshape(heads, -1, d).contiguous()
    qh, kh, vh = f(q), f(k), f(v)
    with torch.autocast(enabled=False, device_type="cuda"):
        # Scale in the GEMM epilogue: FP32 acc rounded, then *scale rounded,
        # the same two roundings as einsum(...) * scale; saves one full pass.
        s1 = torch.baddbmm(torch.empty(heads, qh.shape[1], kh.shape[1], device=q.device),
                           qh, kh.transpose(1, 2), beta=0, alpha=scale)
    s2 = s1.softmax(dim=-1)
    del s1
    r = torch.bmm(s2, vh)
    del s2
    _state["hits"] += 1
    return r.unsqueeze(0).permute(0, 2, 1, 3).reshape(1, -1, inner)


import os as _os
_SAGE = {"fn": None, "hits": 0, "fallbacks": 0, "nonfinite": 0,
         "check": _os.environ.get("Q21_SAGE_FINITE_CHECK", "1") == "1"}


def sage_fp16(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False,
              skip_output_reshape=False, **kw):
    """Aceeton SageAttention (INT8 QK, FP16 PV) for FP32 [1,S,H*D] calls; output back to FP32."""
    prev = _state["prev"]
    if (mask is not None or skip_reshape or skip_output_reshape or kw.get("enable_gqa")
            or q.ndim != 3 or q.shape[0] != 1 or q.dtype != torch.float32):
        _SAGE["fallbacks"] += 1
        return prev(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                    skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kw)
    if _SAGE["fn"] is None:
        import sys
        vendor = "~/DevSpace/idol-rdna2-lab01/third_party/sageattention_rdna2"
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        from sageattention import sageattn
        _SAGE["fn"] = sageattn
    inner = q.shape[-1]
    d = inner // heads
    f = lambda t: t.view(1, -1, heads, d).permute(0, 2, 1, 3).to(torch.float16)
    o = _SAGE["fn"](f(q), f(k), f(v), tensor_layout="HND", sm_scale=kw.get("scale", d ** -0.5))
    o = o.float()
    if _SAGE.get("check", True) and not torch.isfinite(o).all():
        _SAGE["nonfinite"] += 1
    _SAGE["hits"] += 1
    return o.permute(0, 2, 1, 3).reshape(1, -1, inner)


def toggle(name, on):
    import comfy.ldm.qwen_image21.model as qm
    if name == "sage":
        if on:
            _state["prev"] = qm.optimized_attention
            qm.optimized_attention = sage_fp16
        else:
            qm.optimized_attention = _state["prev"]
        return
    if name not in ("split_view", "split_alpha"):
        raise ValueError(name)
    if on:
        _state["orig"] = _state["orig"] or _find_split()
        _state["prev"] = qm.optimized_attention
        inner_wrapper = qm.optimized_attention

        # Keep the harness's logging wrapper outermost: swap the function it wraps.
        qm.optimized_attention = _wrap(inner_wrapper)
    else:
        qm.optimized_attention = _state["prev"]


def _find_split():
    import comfy.ldm.modules.attention as A
    return A.attention_split


def _wrap(logger):
    def attn(q, k, v, heads, mask=None, **kw):
        return split_view(q, k, v, heads, mask=mask, **kw)
    return attn


def microbench():
    import sys, json, time
    sys.path.insert(0, "~/DevSpace/qwen-image-21-rdna2-lab01/runtime/ComfyUI")
    sys.argv = ["x", "--use-split-cross-attention"]
    import comfy.ldm.modules.attention as A
    _state["orig"] = A.attention_split
    torch.manual_seed(0)
    rows = []
    for sq, sk in ((2400, 4964), (2400, 2408), (156, 2564)):
        q = torch.randn(1, sq, 4096, device="cuda")
        k = torch.randn(1, sk, 4096, device="cuda")
        v = torch.randn(1, sk, 4096, device="cuda")
        a = A.attention_split(q, k, v, 32)
        b = split_view(q, k, v, 32)
        res = {"sq": sq, "sk": sk, "bitwise": bool(torch.equal(a, b)),
               "max_abs": (a - b).abs().max().item()}
        for name, fn in (("native", A.attention_split), ("view", split_view)):
            for _ in range(3): fn(q, k, v, 32)
            torch.cuda.synchronize()
            ts = []
            for _ in range(7):
                t = time.perf_counter(); fn(q, k, v, 32); torch.cuda.synchronize()
                ts.append((time.perf_counter() - t) * 1000)
            res[name + "_ms"] = sorted(ts)[3]
        res["speedup"] = res["native_ms"] / res["view_ms"]
        rows.append(res); print(json.dumps(res), flush=True)
    return rows


if __name__ == "__main__":
    microbench()
