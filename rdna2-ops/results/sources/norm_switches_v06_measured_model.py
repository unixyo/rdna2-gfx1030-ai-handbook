"""Temporary normalization adapters for the owned generic Comfy harness.

Only the sampling context changes. All callables are restored in finally;
neither package files, model weights, startup scripts nor settings are edited.
Anima block orchestration is expressed as three stages using existing modules.
Its source fingerprint is pinned so an upstream semantic change is declined.
"""
from collections import Counter
from contextlib import contextmanager, ExitStack
import functools
import hashlib
import importlib
import inspect
from pathlib import Path
import torch

from rdna2_ops.norm_fusions_v06 import row_norm, adaln, modulate

ANIMA_SOURCE_SHA256 = "8f3427c1199ea0f2845d5934ca3929bc151178d60a9373e6a363b9e63b2fd3ce"


def anima_block(self, x, emb, context, rope=None, lora=None, position=None,
                transformer_options=None, *, normalize):
    """Same module calls, residual casts and patch handling as the pinned block.

    ``normalize`` is the only substituted calculation. Keeping this function
    independent of a backend also permits CPU source-contract checks.
    """
    options = transformer_options if transformer_options is not None else {}
    residual_dtype, compute_dtype = x.dtype, emb.dtype
    if position is not None:
        x = x + position
    stages = ("self_attn", "cross_attn", "mlp")
    modulation = []
    for stage in stages:
        values = getattr(self, "adaln_modulation_" + stage)(emb)
        if self.use_adaln_lora:
            if lora is None:
                raise ValueError("Anima AdaLN LoRA input is required")
            values = values + lora
        modulation.append(tuple(t[:, :, None, None, :] for t in values.chunk(3, dim=-1)))
    for stage, (shift, scale, gate) in zip(stages, modulation):
        norm = getattr(self, "layer_norm_" + stage)
        z = normalize(x, norm, scale, shift, compute_dtype)
        if stage == "mlp":
            for patch in options.get("patches", {}).get("mlp_patch", []):
                data = patch({"x": z, "transformer_options": options})
                z = data["x"]
            value = self.mlp(z)
        else:
            b, t, h, w, d = z.shape
            value = getattr(self, stage)(z.reshape(b, t * h * w, d),
                                        None if stage == "self_attn" else context,
                                        rope_emb=rope, transformer_options=options)
            value = value.reshape(b, t, h, w, d)
        # Preserve the native single addcmul and its residual dtype. Inference
        # must not round the FP32 residual stream to the FP16 compute dtype.
        x = torch.addcmul(x, gate.to(residual_dtype), value.to(residual_dtype))
    return x


def eligible(x, tensors=()):
    return (isinstance(x, torch.Tensor) and x.device.type == "cuda"
            and x.dtype in (torch.float16, torch.float32, torch.bfloat16)
            and x.ndim >= 2 and x.is_contiguous() and 0 < x.numel() < 2**31
            and x.shape[-1] in (128, 768, 2048, 3072, 4096)
            and not torch.is_autocast_enabled("cuda")
            and not torch.compiler.is_compiling()
            and not (torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (x, *tensors))))


def measured_row_path(x, rms):
    """Bound the experimental adapter to useful measured dtype/width families.

    This is a model A/B filter, not a production performance guarantee for all
    row counts. Small FP32 D3072 RMS and FP16 D768 LN lost the operator screen.
    """
    if not isinstance(x, torch.Tensor) or x.ndim < 2:
        return False
    if rms:
        return ((x.dtype == torch.float16 and x.shape[-1] == 128)
                or (x.dtype == torch.float32 and x.shape[-1] == 4096))
    return x.dtype == torch.float32 and x.shape[-1] in (2048, 4096)


@contextmanager
def activate(mode, spec, state, bound, out):
    if mode.startswith("attn_"):
        attention = importlib.import_module("attention_switches_v06")
        with attention.activate(mode, spec, state, bound, out):
            yield
        return
    modes = {"norm_control", "norm_ln", "norm_rms", "norm_modulate", "norm_adaln"}
    if mode not in modes:
        raise ValueError("Unknown normalization experiment " + mode)
    ops = importlib.import_module("comfy.ops")
    management = importlib.import_module("comfy.model_management")
    stats = {"mode": mode, "hits": 0, "fallbacks": 0, "shapes": Counter(),
             "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                               for name in ("norm_switches_v06.py", "rdna2_ops/norm_fusions_v06.py")}}
    state["candidate"] = stats
    with ExitStack() as stack:
        # Optional independent attention control for both norm_control and
        # candidate modes; the generic harness records comparison references.
        attention_mode = spec.get("norm_attention_mode")
        if attention_mode:
            attention = importlib.import_module("attention_switches_v06")
            attention_state = {}
            stack.enter_context(attention.activate(attention_mode, spec, attention_state, bound, out))
            stats["attention_control"] = attention_state["candidate"]
        if mode in ("norm_modulate", "norm_adaln"):
            if spec["model"] != "anima":
                raise ValueError("This orchestration adapter owns only the pinned Anima block")
            cosmos = importlib.import_module("comfy.ldm.cosmos.predict2")
            src = Path(inspect.getsourcefile(cosmos.Block))
            actual = hashlib.sha256(src.read_bytes()).hexdigest()
            stats["baseline_source"] = {"path": str(src), "sha256": actual}
            if actual != ANIMA_SOURCE_SHA256:
                raise ValueError("Anima block changed; reread semantics before replacing its orchestration")
            original = cosmos.Block.forward

            def normalize(x, norm, scale, shift, compute_dtype):
                key = str((tuple(x.shape), str(x.dtype), str(scale.dtype), str(compute_dtype)))
                stats["shapes"][key] += 1
                if (norm.weight is not None or norm.bias is not None
                        or not eligible(x, (scale, shift)) or x.dtype != torch.float32
                        or x.shape[-1] != 2048 or compute_dtype != torch.float16
                        or scale.dtype != torch.float16 or shift.dtype != torch.float16):
                    stats["fallbacks"] += 1
                    return (norm(x) * (1 + scale) + shift).to(compute_dtype)
                stats["hits"] += 1
                if mode == "norm_modulate":
                    return modulate(norm(x), scale, shift, out_dtype=compute_dtype)
                ops.run_every_op()
                return adaln(x, scale, shift, norm.eps, rounding="eager", out_dtype=compute_dtype)

            @functools.wraps(original)
            def forward(self, x_B_T_H_W_D, emb_B_T_D, crossattn_emb,
                        rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=None,
                        extra_per_block_pos_emb=None, transformer_options=None):
                if management.in_training or torch.is_grad_enabled():
                    stats["fallbacks"] += 1
                    return original(self, x_B_T_H_W_D, emb_B_T_D, crossattn_emb,
                                    rope_emb_L_1_1_D, adaln_lora_B_T_3D,
                                    extra_per_block_pos_emb, transformer_options or {})
                return anima_block(self, x_B_T_H_W_D, emb_B_T_D, crossattn_emb,
                                   rope_emb_L_1_1_D, adaln_lora_B_T_3D,
                                   extra_per_block_pos_emb, transformer_options,
                                   normalize=normalize)

            cosmos.Block.forward = forward
            stack.callback(setattr, cosmos.Block, "forward", original)
        elif mode in ("norm_ln", "norm_rms"):
            rms = mode == "norm_rms"
            cls = ops.disable_weight_init.RMSNorm if rms else ops.disable_weight_init.LayerNorm
            original = cls.forward

            @functools.wraps(original)
            def forward(self, x):
                if (management.in_training or not eligible(x) or not measured_row_path(x, rms)
                        or tuple(self.normalized_shape) != (x.shape[-1],)):
                    stats["fallbacks"] += 1
                    return original(self, x)
                # Keep dynamic lowvram weight/bias casts and their lifetime.
                with ops.CastBiasWeightContext(self if self.weight is not None else None,
                                              x, offloadable=True) as (weight, bias):
                    if not eligible(x, (weight, bias)) or any(p is not None and
                            (p.dtype != x.dtype or not p.is_contiguous()) for p in (weight, bias)):
                        stats["fallbacks"] += 1
                        return original(self, x)
                    ops.run_every_op()
                    stats["hits"] += 1
                    stats["shapes"][str((tuple(x.shape), str(x.dtype), self.eps))] += 1
                    return row_norm(x, weight, bias if not rms else None, self.eps, rms=rms)

            cls.forward = forward
            stack.callback(setattr, cls, "forward", original)
        try:
            yield
        finally:
            stats["shapes"] = dict(stats["shapes"])
