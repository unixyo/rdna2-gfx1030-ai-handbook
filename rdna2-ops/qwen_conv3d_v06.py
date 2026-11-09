#!/usr/bin/env python3
"""Owned-process Qwen vision exact-tile Conv3D/Linear A/B, using the v05 harness.

Only TextEncodeQwenImage21's conditioning request is instrumented. Its original
CastBiasWeightContext, weight/bias transformations, and offload lifetime remain
in control; the borrowed tensors are used only inside proj._conv_forward.
The production Q21_RDNA2_CONV3D_PATCH stays disabled. This is FP32 inference only.

Run with the existing Qwen interpreter, AFTER the operator survey passes:
  qwen_conv3d_v06.py --out results/v06_qwen_conv3d --seed 20261106
The existing qwen_v05.json supplies the image/prompt/graph. Without --steps or
--seed the source workflow's 25 steps and seed are preserved. Defaults run
warm,native,candidate,native, with a fresh conditioning node every time.
The shared harness records final latent/RGB errors, byte digests, and GPU audit.
This file is not an operating custom node and does not modify installed files.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import functools
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import types


LINEAR_MODE = "qwen_conv3d_linear"
NATIVE_MODES = {"warm", "baseline", "capture", "profile"}
_MISSING = object()


def _backend_info(torch):
    """Observe effective process settings without setting a backend or flag."""
    info = {
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "environment": {name: os.environ.get(name) for name in (
            "MIOPEN_FIND_MODE", "MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR",
            "COMFYUI_ENABLE_MIOPEN")},
    }
    try:
        info["preferred_blas_library"] = str(torch.backends.cuda.preferred_blas_library())
    except (AttributeError, RuntimeError, TypeError) as exc:
        info["preferred_blas_library"] = None
        info["preferred_blas_library_error"] = type(exc).__name__ + ": " + str(exc)
    return info


def _tensor_info(t):
    if t is None:
        return None
    return {"shape": list(t.shape), "stride": list(t.stride()),
            "dtype": str(t.dtype), "device": str(t.device),
            "contiguous": t.is_contiguous()}


def _restore(obj, name, previous):
    if previous is _MISSING:
        delattr(obj, name)
    else:
        setattr(obj, name, previous)


@contextmanager
def activate(mode, spec, state, bound, out):
    """Sampler-side adapter is deliberately empty; work happened in node 3."""
    if mode != LINEAR_MODE:
        raise ValueError("Unknown Qwen Conv3D research mode: " + mode)
    row = state.get("candidate", {}).get("qwen_conv3d", {})
    if row.get("request_index") != state["request_index"] or not row.get("candidate_calls"):
        raise RuntimeError("No current-request vision Conv3D candidate hit; conditioning may be cached")
    yield


def install(*, nodes, spec, state, out, run_stage, digest, save):
    """Called by research_comfy_v05's optional instrument_module extension API."""
    import torch
    import torch.nn.functional as F

    if spec.get("env", {}).get("Q21_RDNA2_CONV3D_PATCH", "0") != "0":
        raise ValueError("Native control requires Q21_RDNA2_CONV3D_PATCH=0")
    allowed_rows = tuple(int(n) for n in spec.get("qwen_conv3d_allowed_rows", [128, 512, 2400]))
    installed = {}
    original_init = nodes.init_extra_nodes
    sampler = getattr(nodes, spec["sampler"][0])
    sample_name = spec["sampler"][1]
    original_sample_slot = sampler.__dict__.get(sample_name, _MISSING)
    original_sample = getattr(sampler, sample_name)

    def patch_registered_node():
        node = nodes.NODE_CLASS_MAPPINGS.get("TextEncodeQwenImage21")
        if node is None:
            raise RuntimeError("TextEncodeQwenImage21 was not registered by the existing runtime")
        if installed:
            if installed["node"] is not node:
                raise RuntimeError("Conditioning node was replaced after instrumentation")
            return
        execute_slot = inspect.getattr_static(node, "execute")
        if not isinstance(execute_slot, classmethod):
            raise TypeError("Expected the installed TextEncodeQwenImage21.execute classmethod")
        original_execute = execute_slot.__func__
        installed.update(node=node, execute=node.__dict__.get("execute", _MISSING),
                         fingerprint=node.__dict__.get("fingerprint_inputs", _MISSING))
        qwen35 = importlib.import_module("comfy.text_encoders.qwen35")
        patch_class = qwen35.Qwen35VisionPatchEmbed
        files = [Path(__file__), Path(inspect.getsourcefile(original_execute)),
                 Path(qwen35.__file__), Path(importlib.import_module("comfy.ops").__file__),
                 Path(importlib.import_module("comfy.text_encoders.qwen3vl").__file__),
                 Path(importlib.import_module("comfy.text_encoders.qwen_image21").__file__)]
        provenance = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

        @functools.wraps(original_execute)
        def execute(cls, *args, **kwargs):
            mode = state["mode"]
            if mode not in NATIVE_MODES | {LINEAR_MODE}:
                raise ValueError("Unsupported conditioning mode: " + str(mode))
            if state.get("qwen_conv3d_active"):
                raise RuntimeError("Concurrent/nested conditioning is outside this harness contract")
            state["qwen_conv3d_active"] = True
            row = {"mode": mode, "request_index": state["request_index"],
                   "node": "TextEncodeQwenImage21", "scope": "conditioning only",
                   "backend_state": _backend_info(torch),
                   "native_calls": 0, "candidate_calls": 0, "calls": [],
                   "allowed_rows": list(allowed_rows), "source_sha256": provenance,
                   "cache_invalidation": "V3 fingerprint_inputs returns a fresh NaN",
                   "scope_restored": False, "completed": False,
                   "bias_contract": "original cast bias passed to native Conv3D or F.linear; no bit-equivalence assumption",
                   "timing_contract": "conditioning synchronized wall includes VAE encode and both text encodes; per-call GPU events exclude outer weight staging"}
            state.setdefault("candidate", {})["qwen_conv3d"] = row
            events = []
            original_patch_forward = patch_class.forward

            @functools.wraps(original_patch_forward)
            def patch_forward(embed, x):
                proj = embed.proj
                previous = proj.__dict__.get("_conv_forward", _MISSING)
                original_conv = proj._conv_forward
                if getattr(original_conv, "_q21_exact_tile_patch", False):
                    raise RuntimeError("Production exact-tile patch is already active; native control is contaminated")

                def conv_forward(this, input, weight, bias, autopad=None, *extra, **kw):
                    arch = getattr(torch.cuda.get_device_properties(input.device), "gcnArchName", "") if input.is_cuda else "cpu"
                    eligible = (
                        input.is_cuda and bool(torch.version.hip) and arch.split(":")[0] == "gfx1030"
                        and input.dtype == weight.dtype == torch.float32
                        and bias is not None and bias.dtype == torch.float32
                        and input.ndim == 5 and weight.ndim == 5
                        and tuple(input.shape[1:]) == (3, 2, 16, 16)
                        and tuple(weight.shape) == (1152, 3, 2, 16, 16)
                        and tuple(bias.shape) == (1152,)
                        and input.shape[0] in allowed_rows
                        and tuple(this.stride) == (2, 16, 16)
                        and tuple(this.padding) == (0, 0, 0)
                        and tuple(this.dilation) == (1, 1, 1) and this.groups == 1
                        and this.padding_mode == "zeros" and autopad is None and not extra and not kw)
                    use_linear = mode == LINEAR_MODE and eligible
                    call = {"input": _tensor_info(input), "weight": _tensor_info(weight),
                            "bias": _tensor_info(bias), "arch": arch, "eligible": bool(eligible),
                            "backend_state": _backend_info(torch),
                            "implementation": "F.linear" if use_linear else "original Conv3d._conv_forward",
                            "original_function": getattr(original_conv, "__qualname__", str(original_conv)),
                            "autopad": autopad, "stride": list(this.stride),
                            "padding": list(this.padding), "dilation": list(this.dilation),
                            "groups": this.groups}
                    row["calls"].append(call)
                    start = torch.cuda.Event(enable_timing=True) if input.is_cuda else None
                    end = torch.cuda.Event(enable_timing=True) if input.is_cuda else None
                    if start is not None:
                        start.record()
                    if use_linear:
                        row["candidate_calls"] += 1
                        y = F.linear(input.reshape(input.shape[0], -1), weight.reshape(1152, -1), bias)
                        y = y.reshape(input.shape[0], 1152, 1, 1, 1)
                    else:
                        row["native_calls"] += 1
                        y = original_conv(input, weight, bias, autopad, *extra, **kw)
                    if end is not None:
                        end.record()
                        events.append((start, end, call))
                    call["output"] = _tensor_info(y)
                    return y

                proj._conv_forward = types.MethodType(conv_forward, proj)
                try:
                    return original_patch_forward(embed, x)
                finally:
                    _restore(proj, "_conv_forward", previous)

            patch_class.forward = patch_forward
            try:
                result = run_stage("conditioning", lambda: original_execute(cls, *args, **kwargs))
                row["completed"] = True
                for start, end, call in events:
                    call["gpu_event_ms"] = start.elapsed_time(end)
                row["sum_conv_gpu_event_ms"] = sum(call["gpu_event_ms"] for _, _, call in events)
                return result
            finally:
                patch_class.forward = original_patch_forward
                state["qwen_conv3d_active"] = False
                row["scope_restored"] = True
                save(f"conditioning_{state['request_index']:03d}_{mode}.json", row)

        node.execute = classmethod(execute)
        node.fingerprint_inputs = classmethod(lambda cls, **kwargs: float("nan"))
        save("qwen_conv3d_instrumentation.json", {
            "node_module": node.__module__, "node_class": node.__qualname__,
            "execute_descriptor": "classmethod", "source_sha256": provenance,
            "lifetime": "owned process; operation override restored after each vision call and conditioning request"})

    @functools.wraps(original_init)
    async def init_extra_nodes(*args, **kwargs):
        result = await original_init(*args, **kwargs)
        # Comfy's file loader does not use normal import hooks. Read its final
        # registry only after built-in/external nodes have all been registered.
        patch_registered_node()
        return result

    @functools.wraps(original_sample)
    def sample(self, *args, **kwargs):
        row = state.get("candidate", {}).get("qwen_conv3d", {})
        if (row.get("request_index") != state["request_index"] or not row.get("completed")
                or not row.get("scope_restored") or not row.get("calls")):
            raise RuntimeError("Current request did not execute and restore the vision Conv3D observer")
        if state["mode"] == LINEAR_MODE and not row.get("candidate_calls"):
            raise RuntimeError("Candidate had zero eligible Conv3D hits; see conditioning JSON")
        return original_sample(self, *args, **kwargs)

    nodes.init_extra_nodes = init_extra_nodes
    setattr(sampler, sample_name, sample)
    if "TextEncodeQwenImage21" in nodes.NODE_CLASS_MAPPINGS:
        patch_registered_node()

    def cleanup():
        nodes.init_extra_nodes = original_init
        _restore(sampler, sample_name, original_sample_slot)
        if installed:
            _restore(installed["node"], "execute", installed["execute"])
            _restore(installed["node"], "fingerprint_inputs", installed["fingerprint"])
    return cleanup


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default=str(Path(__file__).parent / "research_configs/qwen_v05.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="warm,baseline,qwen_conv3d_linear,baseline")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--port", type=int, default=8466)
    ap.add_argument("--allowed-rows", default="128,512,2400")
    cfg = ap.parse_args()
    spec_path = Path(cfg.spec).resolve()
    spec = json.loads(spec_path.read_text())
    graph = json.loads(Path(spec["workflow"]).read_text())
    out = Path(cfg.out).resolve()
    if (out / "runs.json").exists():
        raise FileExistsError("Use a fresh result directory: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    spec.update(instrument_module="qwen_conv3d_v06", switch_module="qwen_conv3d_v06",
                gpu_audit=True, qwen_conv3d_allowed_rows=[int(n) for n in cfg.allowed_rows.split(",")])
    spec["require_bitwise_modes"] = sorted(set(spec.get("require_bitwise_modes", [])) | {"baseline"})
    spec["qwen_conv3d_original_spec"] = {"path": str(spec_path),
                                         "sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest()}
    run_spec = out / "qwen_conv3d_spec.json"
    run_spec.write_text(json.dumps(spec, indent=2) + "\n")
    shutil.copy2(__file__, out / "qwen_conv3d_v06.py")
    seed = cfg.seed if cfg.seed is not None else graph[spec["sampler_node"]]["inputs"]["seed"]
    sys.argv = ["research_comfy_v05.py", "--spec", str(run_spec), "--out", str(out),
                "--modes", cfg.modes, "--seed", str(seed), "--port", str(cfg.port)]
    if cfg.steps is not None:
        sys.argv += ["--steps", str(cfg.steps)]
    from research_comfy_v05 import main as harness_main
    return harness_main()


if __name__ == "__main__":
    raise SystemExit(main())
