#!/usr/bin/env python3
"""Validate the actual installed ComfyUI VAE nodes on a captured denoiser latent."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import runpy
import statistics
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--project", type=Path, required=True)
ap.add_argument("--latent", type=Path, required=True)
ap.add_argument("--dtype", choices=["fp16", "bf16"], required=True)
ap.add_argument("--load-mode", choices=["normal", "lowvram"], default="normal")
ap.add_argument("--output", type=Path, required=True)
cfg = ap.parse_args()
project = cfg.project.resolve()
runtime = project / "runtime/ComfyUI"
os.environ.setdefault("OCL_SET_SVM_SIZE", "4194304")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
sys.path.insert(0, str(runtime))
sys.argv = [str(runtime / "main.py"), "--base-directory", str(project),
            "--models-directory", str(project / "models"),
            "--use-pytorch-cross-attention", "--disable-comfy-compiler",
            "--" + cfg.dtype + "-vae"]
if cfg.load_mode == "lowvram":
    # Real existing Comfy CLI policy, without allocating dummy VRAM or changing
    # the model manager's reported physical free memory.
    sys.argv += ["--lowvram", "--disable-dynamic-vram", "--reserve-vram", "15.4"]
runpy.run_path(str(runtime / "main.py"), run_name="rdna2_vae_runtime")

import torch
import nodes
from comfy import model_management
from rdna2_ops.comfy import fuse_comfy_vae_decoder

torch.set_num_threads(2)
cfg.output.parent.mkdir(parents=True, exist_ok=True)
latent = torch.load(cfg.latent, map_location="cpu", weights_only=True)
report = {"dtype": cfg.dtype, "load_mode": cfg.load_mode,
          "latent_source": str(cfg.latent.resolve()), "latent_shape": list(latent.shape),
          "comfy_source": str(runtime), "torch": torch.__version__,
          "rows": [], "status": "started"}
acceptance = {"max_abs": 0.01, "rmse": 0.001} if cfg.dtype == "fp16" else {"max_abs": 0.04, "rmse": 0.006}
report["pixel_acceptance"] = acceptance


def save():
    cfg.output.write_text(json.dumps(report, indent=2) + "\n")


def memory_state(vae):
    modules = list(vae.first_stage_model.modules())
    return {"dynamic": vae.patcher.is_dynamic(),
            "model_lowvram": bool(getattr(vae.first_stage_model, "model_lowvram", False)),
            "cast_modules": sum(bool(getattr(m, "comfy_cast_weights", False)) for m in modules),
            "cpu_weight_modules": sum(getattr(m, "weight", None) is not None and m.weight.device.type == "cpu" for m in modules),
            "loaded_bytes": vae.patcher.loaded_size(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()}


def time_call(fn, repeats):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        result = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


try:
    with torch.inference_mode():
        vae = nodes.VAELoader().load_vae("flux2-vae_small.safetensors")[0]
        report["actual_vae_dtype"] = str(vae.vae_dtype)
        expected = torch.float16 if cfg.dtype == "fp16" else torch.bfloat16
        assert vae.vae_dtype == expected
        decoder = vae.first_stage_model.decoder
        module_ids = {name: id(module) for name, module in decoder.named_modules()}
        parameter_names = tuple(decoder.state_dict().keys())
        samples = {"samples": latent}
        methods = {
            "regular": lambda: nodes.VAEDecode().decode(vae, samples)[0],
            "tiled_256_overlap64": lambda: nodes.VAEDecodeTiled().decode(
                vae, samples, tile_size=256, overlap=64, temporal_size=64, temporal_overlap=8)[0],
        }
        for path, decode in methods.items():
            torch.cuda.reset_peak_memory_stats()
            native = decode()
            stats = {}
            with fuse_comfy_vae_decoder(decoder, stats=stats) as count:
                assert module_ids == {name: id(module) for name, module in decoder.named_modules()}
                assert parameter_names == tuple(decoder.state_dict().keys())
                optimized = decode()
            torch.cuda.synchronize()
            diff = (optimized.float() - native.float()).abs()
            maximum = diff.max().item()
            rmse = diff.square().mean().sqrt().item()
            row = {"path": path, "image_shape": list(native.shape),
                   "fused_blocks": count, "dispatch": stats, "max_abs": maximum, "rmse": rmse,
                   "max_abs_8bit_levels": maximum * 255,
                   "finite": bool(torch.isfinite(native).all() and torch.isfinite(optimized).all()),
                   "memory": memory_state(vae)}
            row["accuracy_pass"] = row["finite"] and maximum <= acceptance["max_abs"] and rmse <= acceptance["rmse"]
            row["module_ids_restored"] = module_ids == {name: id(module) for name, module in decoder.named_modules()}
            row["state_dict_names_restored"] = parameter_names == tuple(decoder.state_dict().keys())
            if not row["accuracy_pass"] or not row["module_ids_restored"] or not row["state_dict_names_restored"]:
                report["rows"].append(row)
                save()
                raise AssertionError("Actual VAE integration accuracy/restoration failed: " + json.dumps(row))
            timings = {"native": [], "rdna2": []}
            repeats = 3 if path == "regular" else 1
            for trial in range(5):
                for mode in (["native", "rdna2"] if trial % 2 == 0 else ["rdna2", "native"]):
                    with fuse_comfy_vae_decoder(decoder) if mode == "rdna2" else nullcontext():
                        timings[mode].append(time_call(decode, repeats))
            row["samples_wall_ms"] = timings
            row["native_wall_ms"] = statistics.median(timings["native"])
            row["rdna2_wall_ms"] = statistics.median(timings["rdna2"])
            row["speedup"] = row["native_wall_ms"] / row["rdna2_wall_ms"]
            report["rows"].append(row)
            save()
            print(json.dumps(row), flush=True)
        if cfg.load_mode == "lowvram":
            assert any(r["memory"]["model_lowvram"] and not r["memory"]["dynamic"] for r in report["rows"]), "Low-VRAM partial-load branch was not exercised"
        report["status"] = "passed"
        model_management.unload_all_models()
except Exception:
    report["status"] = "failed"
    raise
finally:
    save()
