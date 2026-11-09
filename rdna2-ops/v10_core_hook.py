"""Observe the real core.py/train.py entry, with optional research-only B-prime.

Loaded only after the operating lab_hooks module imports naturally. Never used
as a replacement training entry point; the original spawn and worker path stays.
"""
from __future__ import annotations
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

def install(hooks):
    out = Path(os.environ["APPLIO_V10_OUTPUT"])
    torch = hooks.torch
    profile_at = int(os.environ.get("APPLIO_PROFILE_AT", "0"))
    state = {"schema": "applio_v10_operating_train", "pid": os.getpid(), "epochs": [],
             "steps": [], "conv_calls": {}, "direct_calls": {}, "profile_at": profile_at,
             "dataset": "voice_model_A", "batch_size": 8,
             "entrypoint": "operating core.py train -> operating train.py", "completed": False}
    pending, ranges, contexts, handles = [], {}, [], []
    current = {"epoch": None, "elapsed": None, "profile": False, "initial": False}

    def write():
        path = out / "training_state.json"
        tmp = out / "training_state.tmp"
        tmp.write_text(json.dumps(state, indent=2, allow_nan=False))
        tmp.replace(path)

    def digest_model(model):
        h = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            value = tensor.detach().cpu().contiguous()
            h.update(name.encode())
            h.update(str(value.dtype).encode())
            h.update(str(tuple(value.shape)).encode())
            h.update(value.numpy().tobytes())
        return h.hexdigest()

    def conv_pre(module, inputs):
        if not current["profile"] or not inputs:
            return
        if not isinstance(module, torch.nn.modules.conv._ConvNd):
            return
        x = inputs[0]
        trans = isinstance(module, torch.nn.modules.conv._ConvTransposeNd)
        shape = ([module.in_channels, module.out_channels // module.groups] if trans else
                 [module.out_channels, module.in_channels // module.groups]) + list(module.kernel_size)
        meta = {"kind": type(module).__name__, "input": list(x.shape), "weight": shape,
                "input_dtype": str(x.dtype), "weight_dtype": str(next(module.parameters()).dtype),
                "stride": list(module.stride), "padding": list(module.padding),
                "dilation": list(module.dilation), "groups": module.groups,
                "output_padding": list(getattr(module, "output_padding", []))}
        key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:12]
        row = state["conv_calls"].setdefault(key, {**meta, "count": 0})
        row["count"] += 1
        context = torch.profiler.record_function("V08_CONV_" + key)
        context.__enter__()
        ranges.setdefault(id(module), []).append(context)

    def conv_post(module, inputs, output):
        stack = ranges.get(id(module), [])
        if stack:
            stack.pop().__exit__(None, None, None)

    def wrap_direct(mod):
        for direction, name in [("fwd", "conv_fwd"), ("dx", "conv_dx"), ("dw", "conv_dw")]:
            original = getattr(mod, name)
            def wrapped(*args, _orig=original, _direction=direction, **kwargs):
                if not current["profile"]:
                    return _orig(*args, **kwargs)
                if _direction == "fwd":
                    xs, ws = list(args[0].shape), list(args[1].shape)
                elif _direction == "dx":
                    xs, ws = list(args[2]), list(args[1].shape)
                else:
                    xs, ws = list(args[0].shape), list(args[2])
                meta = {"direction": _direction, "input": xs, "weight": ws,
                        "dilation": kwargs.get("dilation", 1), "padding": kwargs.get("padding", 0),
                        "groups": kwargs.get("groups", 1), "dtype": "torch.float16",
                        "implementation": "direct_v08", "kind": "conv1d", "stride": [1],
                        "output_padding": [0], "config": kwargs.get("config") or {},
                        "split": (kwargs.get("config") or {}).get("split", 32) if _direction == "dw" else 0}
                key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:12]
                row = state["direct_calls"].setdefault(key, {**meta, "count": 0})
                row["count"] += 1
                with torch.profiler.record_function("V10_DIRECT_" + key):
                    return _orig(*args, **kwargs)
            setattr(mod, name, wrapped)

    def annotated(meta, original, args, kwargs):
        key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:12]
        state["direct_calls"].setdefault(key, {**meta, "count": 0})["count"] += 1
        with torch.profiler.record_function("V10_DIRECT_" + key):
            return original(*args, **kwargs)

    def wrap_fusion(mod):
        for direction, name in [("fwd", "conv_act_fwd"), ("dx", "conv_act_dx"), ("dw", "conv_act_dw")]:
            original = getattr(mod, name)
            def wrapped(*args, _orig=original, _direction=direction, **kwargs):
                if not current["profile"]:
                    return _orig(*args, **kwargs)
                x = args[2] if _direction == "dx" else args[0]
                ws = list(args[2]) if _direction == "dw" else list(args[1].shape)
                config = kwargs.get("config") or {}
                meta = {"direction": _direction, "input": list(x.shape), "weight": ws,
                        "kind": "conv1d", "stride": [1], "output_padding": [0],
                        "dilation": kwargs.get("dilation", 1), "padding": kwargs.get("padding", 0),
                        "groups": kwargs.get("groups", 1), "dtype": "torch.float16",
                        "source_dtype": str(x.dtype), "implementation": "fused_v10", "config": config,
                        "split": config.get("split", 32) if _direction == "dw" else 0,
                        "leaky_relu_on_load": _direction != "dx", "leaky_relu_gradient_mask": _direction == "dx",
                        "residual_add": kwargs.get("residual") is not None,
                        "residual_gradient_add": kwargs.get("residual_grad") is not None,
                        "residual_dtype": str(kwargs["residual"].dtype) if kwargs.get("residual") is not None else None,
                        "residual_gradient_dtype": str(kwargs["residual_grad"].dtype) if kwargs.get("residual_grad") is not None else None}
                return annotated(meta, _orig, args, kwargs)
            setattr(mod, name, wrapped)

    def wrap_native_candidate(mod):
        original = mod.run
        def wrapped(case, direction, x, w, dy=None, bias=None, **kwargs):
            if not current["profile"]:
                return original(case, direction, x, w, dy, bias, **kwargs)
            config = kwargs.get("config") or {}
            meta = {"direction": direction, "input": list(x.shape), "weight": list(w.shape),
                    "kind": case["kind"], "stride": case["stride"], "padding": case["padding"],
                    "dilation": case["dilation"], "groups": case["groups"],
                    "output_padding": case["output_padding"], "dtype": str(x.dtype),
                    "implementation": "candidate_v10", "config": config,
                    "split": config.get("split", 1) if direction == "dw" else 0}
            return annotated(meta, original, (case, direction, x, w, dy, bias), kwargs)
        mod.run = wrapped

    def export(prof):
        rows = []
        for event in prof.key_averages(group_by_input_shape=True):
            rows.append({"name": event.key, "count": event.count, "device_type": str(event.device_type),
                         "input_shapes": event.input_shapes, "cpu_us": event.cpu_time_total,
                         "self_cpu_us": event.self_cpu_time_total, "gpu_us": event.device_time_total,
                         "self_gpu_us": event.self_device_time_total})
        (out / "profile_rows.json").write_text(json.dumps(rows, indent=2))
        prof.export_chrome_trace(str(out / "profile_trace.json"))
        state["profile_saved"] = True
        print("[v10] operating profile exported", str(out), flush=True)

    original_install = hooks.install_rdna2
    def on_install(train):
        original_install(train)
        # multiprocessing.spawn/runpy copies the module dictionary, while the
        # function objects retain runpy's original globals. Patch the actual
        # run() globals as well as the module attribute used by the lab hook.
        train_globals = train.run.__globals__
        state["spawn_run_globals_same_as_module_dict"] = train_globals is vars(train)
        # A config/filelist-only clone lacks metadata. Read the source metadata
        # in place, preserving its speaker count and reference embedder without
        # copying a third training input or changing an operating file.
        source_info = Path("~/DevSpace/applio-lab/logs/voice_model_A/model_info.json")
        metadata = json.loads(source_info.read_text())
        train.model_info_path = str(source_info)
        train_globals["model_info_path"] = str(source_info)
        state["source_model_info"] = {"path": str(source_info),
            "sha256": hashlib.sha256(source_info.read_bytes()).hexdigest(),
            "speakers_id": metadata["speakers_id"], "embedder_model": metadata["embedder_model"]}
        # The operating exporter leaves two locals undefined when a fresh model
        # has no model_info.json. Use its exact function with only this missing
        # metadata branch repaired in the research process. Tensor conversion,
        # state keys, config construction and serialization stay native.
        original_export = train_globals["extract_model"]
        source = inspect.getsource(original_export)
        old_branch = "        else:\n            dataset_length = None"
        new_branch = ("        else:\n            dataset_length = _v10_metadata.get('total_dataset_duration')\n"
                      "            embedder_model = _v10_metadata['embedder_model']\n"
                      "            speakers_id = int(hps.model.spk_embed_dim)")
        if source.count(old_branch) != 1:
            raise RuntimeError("Operating export metadata branch changed; inspect before adapting")
        patched_source = source.replace(old_branch, new_branch)
        namespace = dict(original_export.__globals__)
        namespace["_v10_metadata"] = metadata
        runtime_export = out / "export_runtime_view.py"
        runtime_export.write_text(patched_source)
        exec(compile(patched_source, str(runtime_export), "exec"), namespace)
        train.extract_model = namespace["extract_model"]
        train_globals["extract_model"] = train.extract_model
        state["export_metadata_repair"] = {"scope": "missing metadata locals only, process-local function copy",
            "original_function_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "runtime_function_sha256": hashlib.sha256(patched_source.encode()).hexdigest()}
        state.update(torch=torch.__version__, hip=torch.version.hip,
                     train_dtype=str(train.train_dtype), effective_cudnn_benchmark=torch.backends.cudnn.benchmark,
                     seed=train.config.train.seed, model_name=train.model_name)
        from rvc.train.rdna2_v08 import v08_direct_conv
        if profile_at:
            wrap_direct(v08_direct_conv)
            handles.extend([torch.nn.modules.module.register_module_forward_pre_hook(conv_pre),
                            torch.nn.modules.module.register_module_forward_hook(conv_post)])
        selection = os.environ.get("APPLIO_V10_FUSION_SELECTION")
        if selection:
            import v10_fused_resblock
            if profile_at:
                wrap_fusion(v10_fused_resblock)
            state["fusion_hits"] = {}
            ctx = v10_fused_resblock.fused_resblocks(
                Path(__file__).parent / "results/applio_v08/direct_selected_final/dispatch.json",
                selection=selection, hits=state["fusion_hits"])
            ctx.__enter__()
            contexts.append(ctx)
        native_selection = os.environ.get("APPLIO_V10_NATIVE_SELECTION")
        if native_selection:
            import v10_native_conv
            if os.environ.get("APPLIO_RDNA2_V10_NATIVE") != "1":
                raise RuntimeError("Native selection requires the explicit research enable flag")
            if profile_at:
                wrap_native_candidate(v10_native_conv)
            state["native_hits"] = {}
            ctx = v10_native_conv.measured_dispatch(native_selection, hits=state["native_hits"])
            ctx.__enter__()
            contexts.append(ctx)
        original_epoch = train_globals["train_and_evaluate"]
        @functools.wraps(original_epoch)
        def epoch(*args, **kwargs):
            current["epoch"] = args[1]
            if not current["initial"]:
                nets, optims, loaders = args[3:6]
                effective_embedder = sys._getframe(1).f_locals.get("embedder_name")
                effective_speakers = int(args[2].model.spk_embed_dim)
                if effective_speakers != int(metadata["speakers_id"]) or effective_embedder != metadata["embedder_model"]:
                    raise RuntimeError("Fresh model metadata differs from source operating dataset")
                state["effective_model_metadata"] = {"speakers_id": effective_speakers,
                                                     "embedder_model": effective_embedder}
                state["initial_state_sha256"] = [digest_model(m) for m in nets]
                state["steps_per_epoch"] = len(loaders[0])
                state["dataset_rows"] = len(loaders[0].dataset)
                state["dataset_sampler_samples"] = loaders[0].batch_sampler.num_samples
                state["optimizer_groups"] = [{k: g.get(k) for k in ("foreach", "fused", "capturable", "eps", "weight_decay")} for opt in optims for g in opt.param_groups]
                current["initial"] = True
                write()
            return original_epoch(*args, **kwargs)
        train.train_and_evaluate = epoch
        train_globals["train_and_evaluate"] = epoch
        original_record = train.EpochRecorder.record
        def record(recorder):
            current["elapsed"] = train.ttime() - recorder.last_time
            return original_record(recorder)
        train.EpochRecorder.record = record
        write()
        print("[v10] operating path observer installed", os.getpid(), flush=True)
    hooks.install_rdna2 = on_install

    original_step = hooks.after_step
    def after_step(step, scaler, experiment_dir, loss_d=None, loss_g=None, grad_d=None, grad_g=None, scale_before=None):
        frame = sys._getframe(1)
        parts = [frame.f_locals[k].detach() for k in ("loss_mel", "loss_fm", "loss_kl", "loss_gen")]
        # Do not read scaler.get_scale() here: it synchronizes the GPU. The
        # next step's native scale_before gives this step's post-update scale.
        pending.append((step, [loss_d.detach(), loss_g.detach(), *parts], grad_d, grad_g,
                        scale_before, None))
        prof = hooks._prof
        original_step(step, scaler, experiment_dir, loss_d, loss_g, grad_d, grad_g, scale_before)
        if prof is not None and prof is not False and hooks._prof is False:
            current["profile"] = False
            export(prof)
        elif hooks._prof is not None and hooks._prof is not False:
            current["profile"] = True
    hooks.after_step = after_step

    original_after_epoch = hooks.after_epoch
    def after_epoch(number, scaler):
        original_after_epoch(number, scaler)
        # Tensor collection is after the original EpochRecorder sampled the clock.
        final_scale = scaler.get_scale()
        values = torch.stack([torch.stack(row[1]) for row in pending]).float().cpu().tolist()
        for index, (row, vals) in enumerate(zip(pending, values)):
            step, _, gd, gg, s0, s1 = row
            s1 = pending[index + 1][4] if index + 1 < len(pending) else final_scale
            gd, gg = float(gd), float(gg)
            metrics = dict(zip(("loss_d", "loss_g", "mel", "fm", "kl", "adv"), vals))
            metrics = {k: v if math.isfinite(v) else None for k, v in metrics.items()}
            state["steps"].append({"step": step, "epoch": number, **metrics,
                "grad_norm_d": gd if math.isfinite(gd) else None,
                "grad_norm_g": gg if math.isfinite(gg) else None,
                "scale_before": s0, "scale_after": s1, "scale_reduced": s0 is not None and s1 < s0})
        new_steps = state["steps"][-len(pending):]
        state["epochs"].append({"epoch": number, "seconds": current["elapsed"], "steps": len(pending),
            "nonfinite_losses": sum(row["loss_d"] is None or row["loss_g"] is None for row in new_steps),
            "scale_reductions": sum(row["scale_reduced"] for row in new_steps),
            "scale_after": final_scale, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "timing_scope": "same start/end as operating EpochRecorder, before research export/checkpoint"})
        state["direct_conv_hits"] = dict(hooks.rdna2_v08._state.get("hits", {}))
        pending.clear()
        write()
        print("[v10] epoch", json.dumps(state["epochs"][-1]), flush=True)
    hooks.after_epoch = after_epoch
