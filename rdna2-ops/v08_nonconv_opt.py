"""Process-local, opt-in non-convolution candidates for Applio RDNA2 v0.8.

Import this file, then call apply(train_module) after importing the original
trainer and before train_module.run(). No operating file is edited.

Defaults preserve native behavior. Explicit options:
  APPLIO_RDNA2_ADAMW=native|single|foreach|fused       (default native)
  APPLIO_RDNA2_ADAMW_G=<mode>                         (default global mode)
  APPLIO_RDNA2_ADAMW_D=<mode>                         (default global mode)
  APPLIO_RDNA2_SLICE_GATHER=1                        (default 0)
  APPLIO_RDNA2_GRAD_NORM=native|device|foreach        (default native)
  APPLIO_RDNA2_D_WN_CACHE=1                          (default 0)
  APPLIO_RDNA2_SKIP_EMPTY_CACHE=1                    (default 0)

Diagnostic-only, outside timing:
  APPLIO_RDNA2_OPT_SNAPSHOT_DIR=<research directory>
  APPLIO_RDNA2_OPT_SNAPSHOT_AT=3  (optimizer step call; native skips are not calls)

All arithmetic parameters, FP32 weights/Adam moments, AMP policy, learning rate,
batch, data, losses and random-number consumption remain caller-controlled.
Only measured successful candidates should be enabled together. The returned
handle restores process-local overrides; unsetting options and starting a fresh
process is the normal rollback.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import functools
import hashlib
import os
from pathlib import Path
import sys
from typing import Any

import torch


def _choice(name, allowed, default):
    value = os.environ.get(name, default)
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")
    return value


def _flag(name):
    return _choice(name, {"0", "1"}, "0") == "1"


def slice_segments_gather(x, ids_str, segment_size=4, dim=2):
    """Exact valid-window slice with device indices and differentiable gather.

    Contract matches Applio's generated nonnegative, in-bounds window starts.
    Unsupported rank/dim or invalid window sizes fail instead of changing the
    sampling policy. No random values are generated here. For dim=3 each channel
    shares the same start; gather indices do not overlap inside a batch row.
    """
    if (dim == 2 and x.ndim != 2) or (dim == 3 and x.ndim != 3):
        raise ValueError("Applio slice expects rank 2/dim 2 or rank 3/dim 3")
    if dim not in (2, 3) or segment_size < 0 or segment_size > x.shape[-1]:
        raise ValueError("Unsupported or out-of-bounds Applio segment")
    if ids_str.ndim != 1 or ids_str.numel() != x.shape[0]:
        raise ValueError("One window start per batch row is required")
    starts = ids_str.to(device=x.device, dtype=torch.long)
    idx = starts[:, None] + torch.arange(segment_size, device=x.device)
    if dim == 3:
        idx = idx[:, None, :].expand(-1, x.shape[1], -1)
    return torch.gather(x, dim - 1, idx)


def grad_norm_device(parameters, norm_type=2.0):
    """Native reduction order, but leave this logging-only scalar on device."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return torch.linalg.vector_norm(
        torch.stack([g.norm(norm_type) for g in grads]), ord=norm_type
    )


def grad_norm_foreach(parameters, norm_type=2.0):
    """Fewer launches for logging-only norms; does not modify any gradient."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    # Applio has homogeneous FP32 optimizer gradients. Retain generic fallback.
    if len({(g.device, g.dtype) for g in grads}) != 1:
        return torch.linalg.vector_norm(
            torch.stack([g.norm(norm_type) for g in grads]), ord=norm_type
        )
    return torch.linalg.vector_norm(
        torch.stack(torch._foreach_norm(grads, norm_type)), ord=norm_type
    )


def optimizer_flags(mode):
    return {
        "native": {"foreach": None, "fused": None},
        "single": {"foreach": False, "fused": False},
        "foreach": {"foreach": True, "fused": False},
        "fused": {"foreach": False, "fused": True},
    }[mode].copy()


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(v) for v in value)
    return value


@dataclass
class PatchHandle:
    report: dict[str, Any]
    replacements: list = field(default_factory=list)
    optimizer_instances: list = field(default_factory=list)

    def set(self, owner, name, value):
        old = getattr(owner, name)
        self.replacements.append((owner, name, old, value))
        setattr(owner, name, value)

    def restore(self):
        for owner, name, old, installed in reversed(self.replacements):
            if getattr(owner, name) is installed:
                setattr(owner, name, old)
        self.replacements.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.restore()


def apply(train):
    """Apply explicitly requested overrides and return a reversible handle."""
    mode = _choice("APPLIO_RDNA2_ADAMW", {"native", "single", "foreach", "fused"}, "native")
    modes = {role: _choice("APPLIO_RDNA2_ADAMW_" + role,
                          {"native", "single", "foreach", "fused"}, mode)
             for role in ("G", "D")}
    norms = _choice("APPLIO_RDNA2_GRAD_NORM", {"native", "device", "foreach"}, "native")
    gather = _flag("APPLIO_RDNA2_SLICE_GATHER")
    wn_cache = _flag("APPLIO_RDNA2_D_WN_CACHE")
    skip_empty = _flag("APPLIO_RDNA2_SKIP_EMPTY_CACHE")
    snapshot_dir = os.environ.get("APPLIO_RDNA2_OPT_SNAPSHOT_DIR", "")
    snapshot_at = int(os.environ.get("APPLIO_RDNA2_OPT_SNAPSHOT_AT", "3"))
    if snapshot_at < 1:
        raise ValueError("Snapshot step must be positive")
    report = dict(
        schema="applio_nonconv_v08", adamw=mode, adamw_by_optimizer=modes,
        optimizer_order_contract="Native trainer constructs G index 0, D index 1; later optimizers use the global mode",
        slice_gather=gather,
        grad_norm=norms, discriminator_weight_norm_cache=wn_cache,
        skip_empty_cache=skip_empty, snapshot_dir=snapshot_dir,
        snapshot_at=snapshot_at, snapshots=[], optimizer_constructors=[],
        torch=str(torch.__version__), hip=torch.version.hip,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    handle = PatchHandle(report)
    if snapshot_dir:
        path = Path(snapshot_dir).resolve()
        research = Path(__file__).resolve().parent
        if not path.is_relative_to(research):
            raise ValueError("Optimizer snapshots must stay inside rdna2-ops")
        path.mkdir(parents=True, exist_ok=True)

    if mode != "native" or any(value != "native" for value in modes.values()) or snapshot_dir:
        base = torch.optim.AdamW

        class ResearchAdamW(base):
            def __init__(self, *args, **kwargs):
                index = len(handle.optimizer_instances)
                role = ("G", "D")[index] if index < 2 else "other"
                selected_mode = modes.get(role, mode)
                if selected_mode != "native":
                    kwargs.update(optimizer_flags(selected_mode))
                super().__init__(*args, **kwargs)
                self._rdna2_index = index
                self._rdna2_calls = 0
                self._rdna2_saved = False
                handle.optimizer_instances.append(self)
                report["optimizer_constructors"].append(
                    dict(index=self._rdna2_index, role=role, mode=selected_mode,
                         flags=[{key: group.get(key) for key in ("foreach", "fused")}
                                for group in self.param_groups], groups=[
                        {k: v for k, v in g.items() if k != "params"}
                        for g in self.param_groups
                    ])
                )

            def step(self, closure=None):
                self._rdna2_calls += 1
                if snapshot_dir and not self._rdna2_saved and self._rdna2_calls >= snapshot_at:
                    params = [p for g in self.param_groups for p in g["params"]]
                    grads = [None if p.grad is None else p.grad.detach().cpu().clone() for p in params]
                    finite = all(g is None or bool(torch.isfinite(g).all()) for g in grads)
                    if finite:
                        dst = Path(snapshot_dir) / f"optimizer_{self._rdna2_index}_before_step{self._rdna2_calls}.pt"
                        if dst.exists():
                            raise FileExistsError(f"Refusing to replace optimizer snapshot: {dst}")
                        torch.save(dict(
                            schema="applio_adamw_snapshot_v08",
                            optimizer_index=self._rdna2_index,
                            optimizer_step_call=self._rdna2_calls,
                            params=[p.detach().cpu().clone() for p in params],
                            grads=grads, optimizer_state=_cpu_tree(self.state_dict()),
                            torch=str(torch.__version__), hip=torch.version.hip,
                            parameter_shapes=[list(p.shape) for p in params],
                            note="Unscaled FP32 gradients immediately before a real optimizer update; diagnostic capture excluded from timing.",
                        ), dst)
                        self._rdna2_saved = True
                        report["snapshots"].append(str(dst))
                return super().step(closure=closure)

        handle.set(torch.optim, "AdamW", ResearchAdamW)

    if gather:
        # Resolve the import alias before patching so restore() is complete even
        # when apply() is called before train.run() first imports Synthesizer.
        from rvc.lib.algorithm import synthesizers as synth
        old = train.commons.slice_segments
        handle.set(train.commons, "slice_segments", slice_segments_gather)
        if getattr(synth, "slice_segments", None) is old:
            handle.set(synth, "slice_segments", slice_segments_gather)

    if norms != "native":
        handle.set(train.commons, "grad_norm",
                   grad_norm_device if norms == "device" else grad_norm_foreach)

    if wn_cache:
        if bool(getattr(train.config.model, "use_spectral_norm", False)):
            raise ValueError("Weight-norm cache candidate excludes stateful spectral norm")
        from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator
        original = MultiPeriodDiscriminator.forward

        @functools.wraps(original)
        def cached_forward(module, *args, **kwargs):
            if module.checkpointing:
                return original(module, *args, **kwargs)
            # This scope ends before optimizer.step. Never cache across an update.
            with torch.nn.utils.parametrize.cached():
                return original(module, *args, **kwargs)

        handle.set(MultiPeriodDiscriminator, "forward", cached_forward)

    if skip_empty:
        def keep_allocator_cache():
            report["empty_cache_calls_skipped"] = report.get("empty_cache_calls_skipped", 0) + 1
        handle.set(torch.cuda, "empty_cache", keep_allocator_cache)
    return handle
