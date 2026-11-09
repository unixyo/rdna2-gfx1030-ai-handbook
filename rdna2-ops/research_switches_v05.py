"""Process-local research adapters used ONLY by the owned v0.5 harness."""
from contextlib import contextmanager
import importlib
import importlib.util
from pathlib import Path
import torch


def resolve(symbol):
    module, attr = symbol.split(":", 1)
    return importlib.import_module(module), attr


@contextmanager
def activate(mode, spec, state, bound, out):
    model = spec["model"]
    if mode.startswith("int8"):
        from rdna2_ops.int8_gemm import moss_int8_gemm_candidate, qwen_int8_gemm_candidate
        config = dict(spec.get("int8_config", {}))
        if mode != "int8":
            config["tile"] = tuple(int(x) for x in mode.removeprefix("int8_").split("x"))
        context = moss_int8_gemm_candidate if model == "moss" else qwen_int8_gemm_candidate
        with context(**config) as stats:
            state["candidate"] = stats
            yield
        return
    if mode.startswith("mixed_"):
        if model != "moss":
            raise ValueError("This boundary experiment is pinned to the measured MOSS graph")
        from rdna2_ops.mixed_attention import protected_gemm_islands
        import comfy.ops
        choices = {"mixed_control": ("float", "float"),
                   "mixed_qk": ("half", "float"),
                   "mixed_pv": ("float", "half"),
                   "mixed_both": ("half", "half"),
                   "mixed_qk_split": ("split", "float"),
                   "mixed_pv_split": ("float", "split")}
        if mode not in choices:
            raise ValueError(mode)
        qk, pv = choices[mode]
        refs = state.setdefault("mixed_boundary_refs", {}) if spec.get("mixed_probes", True) else None
        with protected_gemm_islands(comfy.ops, qk=qk, pv=pv, scope="long",
                                    boundary_refs=refs, record_reference=mode == "mixed_control") as stats:
            state["candidate"] = stats
            yield
        return
    owner, name = resolve(spec["attention_target"])
    original = getattr(owner, name)
    stats = {"hits": 0, "fallbacks": 0, "mode": mode}
    if mode == "alpha":
        if model != "qwen": raise ValueError("alpha is the Qwen post-scale recipe")
        path = Path(__file__).resolve().parents[1] / "rehearsal_qwen_20261106/scripts/candidates.py"
        module_spec = importlib.util.spec_from_file_location("rdna2_qwen_alpha", path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        module._state.update(orig=original, hits=0, fallbacks=0)
        fn = module.split_view
        stats = module._state
    elif mode in ("fp32", "sage_qwen4964"):
        if mode == "sage_qwen4964":
            if model != "qwen":
                raise ValueError("The Sage RGB diagnosis is scoped to the measured Qwen capture")
            from rdna2_ops.sage_attention_v05 import sage_attention
            options = dict(spec.get("sage_config", {}))
            stats["scope"] = {"heads": 32, "query_length": 2400, "key_length": 4964, "head_dim": 128}
            stats["operator_gate_already_failed"] = True
        else:
            from rdna2_ops.fp32_attention import fp32_attention
            options = dict(spec.get("fp32_config", {}))
        stats["shapes"] = {}

        def fn(q, k, v, *args, **kw):
            mask = kw.get("mask", kw.get("attn_mask"))
            transformer_options = kw.get("transformer_options")
            override = (transformer_options is not None and
                        (not isinstance(transformer_options, dict) or
                         transformer_options.get("optimized_attention_override") is not None))
            # Comfy wrap_attn also handles containers and explicit override hooks.
            # Preserve those through the saved callable; this adapter only owns
            # the measured Tensor/head signature without positional extra options.
            signature_supported = (all(isinstance(t, torch.Tensor) for t in (q, k, v))
                                   and ((model == "qwen" and len(args) <= 1) or
                                        (model == "moss" and not args)))
            if (not signature_supported or override or mask is not None
                    or kw.get("skip_reshape") or kw.get("skip_output_reshape")
                    or kw.get("enable_gqa") or kw.get("dropout_p", 0) or kw.get("is_causal", False)
                    or q.dtype != torch.float32):
                stats["fallbacks"] += 1
                return original(q, k, v, *args, **kw)
            if model == "qwen":
                heads = args[0] if args else kw["heads"]
                b, length, channels = q.shape
                d = channels // heads
                convert = lambda t: t.reshape(b, -1, heads, d).transpose(1, 2)
                qh, kh, vh = map(convert, (q, k, v))
                if mode == "sage_qwen4964":
                    if b != 1 or heads != 32 or length != 2400 or kh.shape[-2] != 4964 or d != 128:
                        stats["fallbacks"] += 1
                        return original(q, k, v, *args, **kw)
                    result = sage_attention(qh, kh, vh, scale=kw.get("scale"), **options)
                else:
                    result = fp32_attention(qh, kh, vh, scale=kw.get("scale"), scale_mode="post", **options)
                result = result.transpose(1, 2).reshape(b, length, channels)
            else:
                if q.shape[-2] < 128:
                    stats["fallbacks"] += 1
                    return original(q, k, v, *args, **kw)
                result = fp32_attention(q, k, v, scale=kw.get("scale"), scale_mode="sdpa", **options)
            stats["hits"] += 1
            key = str((tuple(q.shape), tuple(k.shape), tuple(v.shape)))
            stats["shapes"][key] = stats["shapes"].get(key, 0) + 1
            return result
    else:
        raise ValueError(mode)
    setattr(owner, name, fn)
    try:
        state["candidate"] = stats
        yield
    finally:
        setattr(owner, name, original)
        # The candidate JSON must not retain a callable from the alpha adapter.
        state["candidate"] = {k: v for k, v in stats.items() if not callable(v)}
