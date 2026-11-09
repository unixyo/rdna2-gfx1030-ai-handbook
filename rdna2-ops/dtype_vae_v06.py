#!/usr/bin/env python3
"""Actual captured MOSS latent -> existing Comfy BF16 VAE output A/B.

Only an owned inference process is affected. Actual dispatch coverage must be
measured: composite SDPA/MIOpen internal GEMMs can bypass TorchDispatchMode.
The SDPA boundary adapter covers the observed BF16 VAE attention path; no-hit is
not successful integration. Model/runtime/operating config remain untouched.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import runpy
import statistics
import sys
import time
import traceback

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from gpu_guard_v06 import GpuAudit
from rdna2_ops.bf16_gemm_v06 import bf16_gemm
from research_comfy_v05 import profile_tables


class BF16GemmMode(TorchDispatchMode):
    def __init__(self, method, *, metadata=False):
        super().__init__()
        self.method = method
        self.collect_metadata = metadata
        self.operations = Counter()
        self.hits = 0
        self.fallbacks = Counter()
        self.kernel_metadata = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kw = kwargs or {}
        supported = (torch.ops.aten.mm.default, torch.ops.aten.bmm.default,
                     torch.ops.aten.addmm.default)
        if func not in supported:
            return func(*args, **kw)
        add = func == torch.ops.aten.addmm.default
        a, b = args[1:3] if add else args[:2]
        key = str(func) + ":" + str(a.dtype) + ":" + str(list(a.shape)) + ":" + str(list(b.shape))
        self.operations[key] += 1
        if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
            return func(*args, **kw)
        if a.device.type != "cuda" or b.device != a.device:
            self.fallbacks["not_same_gpu"] += 1
            return func(*args, **kw)
        bias = args[0] if add else None
        if add and (kw.get("alpha", 1) != 1 or kw.get("beta", 1) != 1
                    or bias.ndim != 1 or not bias.is_contiguous()):
            self.fallbacks["unsupported_addmm_contract"] += 1
            return func(*args, **kw)
        self.hits += 1
        if self.method == "native":
            return func(*args, **kw)
        meta = {} if self.collect_metadata and key not in self.kernel_metadata else None
        try:
            result = bf16_gemm(a, b, bias=bias, method=self.method,
                               check_fp16=True, require_exact_fp16=False, metadata=meta)
        except ValueError as exc:
            if "becomes nonfinite" in str(exc) or "not exactly representable" in str(exc):
                self.fallbacks["half_range_guard"] += 1
                self.hits -= 1
                return func(*args, **kw)
            raise
        if meta is not None:
            self.kernel_metadata[key] = meta
        return result

    def report(self):
        return {"method": self.method, "eligible_hits": self.hits,
                "fallbacks": dict(self.fallbacks), "actual_aten_operations": dict(self.operations),
                "kernel_metadata": self.kernel_metadata,
                "half_input_rounding_underflow_permitted": self.method in ("dot2", "fp16_f16", "fp16_f32"),
                "half_range_check_in_timed_calls": self.method in ("dot2", "fp16_f16", "fp16_f32")}


def output_error(candidate, reference):
    c, r = candidate.float(), reference.float()
    delta = c - r
    maximum = float(delta.abs().max())
    rmse = float(delta.square().mean().sqrt())
    finite = bool(torch.isfinite(c).all() and torch.isfinite(r).all())
    value_equal = bool(torch.equal(candidate, reference))
    same_dtype_shape = candidate.dtype == reference.dtype and candidate.shape == reference.shape
    bit_equal = bool(same_dtype_shape and torch.equal(candidate.contiguous().view(torch.uint8),
                                                       reference.contiguous().view(torch.uint8)))
    return {"finite": finite, "bit_equal": bit_equal, "value_equal": value_equal,
            "candidate_dtype": str(candidate.dtype), "reference_dtype": str(reference.dtype),
            "equality_method": "bit_equal compares contiguous uint8 views with equal dtype/shape; value_equal uses torch.equal",
            "max_abs": maximum, "rmse": rmse, "max_levels_255": maximum * 255,
            "rmse_levels_255": rmse * 255,
            "strict_rgb_pass": finite and maximum <= 2 / 255 and rmse <= 0.25 / 255}


def save_image(value, path):
    from PIL import Image
    import numpy as np
    cpu = value.float().cpu().contiguous()
    torch.save(cpu, path.with_suffix(".pt"))
    raw = cpu.numpy()
    Image.fromarray(np.clip(raw[0] * 255, 0, 255).astype(np.uint8)).save(path.with_suffix(".png"))
    return {"float32_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
            "tensor_file": str(path.with_suffix(".pt")), "png_file": str(path.with_suffix(".png"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--latent", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--method", choices=("fp32", "fp16_f32", "fp16_f16", "dot2", "simt_fp32"), required=True)
    ap.add_argument("--paths", choices=("regular", "regular,tiled"), default="regular,tiled")
    ap.add_argument("--load-mode", choices=("normal", "lowvram"), default="normal")
    ap.add_argument("--boundary", choices=("dispatch", "sdpa"), default="dispatch")
    cfg = ap.parse_args()
    if cfg.boundary == "sdpa" and cfg.method != "fp32":
        ap.error("SDPA boundary experiment currently supports FP32 GEMM only")
    if cfg.boundary == "sdpa":
        from rdna2_ops.bf16_sdpa_gemm_v06 import BF16SdpaMode
        Mode = BF16SdpaMode
    else:
        Mode = BF16GemmMode
    cfg.output.mkdir(parents=True, exist_ok=True)
    report_path = cfg.output / "results.json"
    if report_path.exists():
        raise FileExistsError("Use a fresh output directory")
    report = {"status": "started", "method": cfg.method, "paths": cfg.paths,
              "boundary": cfg.boundary,
              "load_mode": cfg.load_mode, "latent": str(cfg.latent.resolve()),
              "final_rgb_gate_declared_before_run": {"max_abs": 2/255, "rmse": 0.25/255},
              "scope": "Existing real MOSS denoiser latent, existing BF16 VAE policy in this owned process; BF16 actual hit required",
              "timing_scope": "Native timings have no TorchDispatch observer; candidate includes its dispatch/guard overhead. Native observer is separate and must preserve output bits.",
              "rows": []}
    root = Path(__file__).resolve().parent
    report["source_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in ("dtype_vae_v06.py", "rdna2_ops/bf16_gemm_v06.py")}
    audit = GpuAudit()
    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    try:
        with audit, torch.inference_mode():
            audit.ensure_clear()
            project = cfg.project.resolve()
            runtime = project / "runtime/ComfyUI"
            os.environ.setdefault("OCL_SET_SVM_SIZE", "4194304")
            os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
            sys.path.insert(0, str(runtime))
            sys.argv = [str(runtime / "main.py"), "--base-directory", str(project),
                        "--models-directory", str(project / "models"),
                        "--use-pytorch-cross-attention", "--disable-comfy-compiler", "--bf16-vae"]
            if cfg.load_mode == "lowvram":
                sys.argv += ["--lowvram", "--disable-dynamic-vram", "--reserve-vram", "15.4"]
            runpy.run_path(str(runtime / "main.py"), run_name="rdna2_dtype_vae_runtime")
            import nodes
            from comfy import model_management
            torch.set_num_threads(2)
            latent = torch.load(cfg.latent, map_location="cpu", weights_only=True)
            report["latent_shape"] = list(latent.shape)
            report["latent_sha256"] = hashlib.sha256(cfg.latent.read_bytes()).hexdigest()
            vae = nodes.VAELoader().load_vae("flux2-vae_small.safetensors")[0]
            if vae.vae_dtype != torch.bfloat16:
                raise AssertionError("Actual VAE is not BF16")
            report["actual_vae_dtype"] = str(vae.vae_dtype)
            samples = {"samples": latent}
            paths = {"regular": lambda: nodes.VAEDecode().decode(vae, samples)[0]}
            if "tiled" in cfg.paths:
                paths["tiled_256_overlap64"] = lambda: nodes.VAEDecodeTiled().decode(
                    vae, samples, tile_size=256, overlap=64, temporal_size=64, temporal_overlap=8)[0]
            for path, decode in paths.items():
                audit.ensure_clear()
                row = {"path": path, "samples_wall_ms": {"native": [], cfg.method: []}}
                report["rows"].append(row)
                original = decode()
                with Mode("native", metadata=True) as mode:
                    native = decode()
                row["native_dispatch"] = mode.report()
                row["observer_control"] = output_error(native, original)
                if not row["observer_control"]["bit_equal"]:
                    raise AssertionError("TorchDispatch observer changed native final output")
                del original
                if cfg.boundary == "sdpa":
                    with Mode("reconstruct") as mode:
                        reconstructed = decode()
                    row["reconstructed_dispatch"] = mode.report()
                    row["reconstructed_control"] = output_error(reconstructed, native)
                    if not row["reconstructed_control"]["bit_equal"]:
                        raise AssertionError("Reconstructed BF16 SDPA changed native final output")
                    del reconstructed
                torch.cuda.reset_peak_memory_stats()
                with Mode(cfg.method, metadata=True) as mode:
                    candidate = decode()
                torch.cuda.synchronize()
                row["candidate_dispatch"] = mode.report()
                row["image_shape"] = list(native.shape)
                row["error"] = output_error(candidate, native)
                row["native_output"] = save_image(native, cfg.output / (path + "_native"))
                row["candidate_output"] = save_image(candidate, cfg.output / (path + "_" + cfg.method))
                row["actual_candidate_hit"] = row["candidate_dispatch"]["eligible_hits"] > 0
                for trial in range(3):
                    order = ("native", cfg.method) if trial % 2 == 0 else (cfg.method, "native")
                    for name in order:
                        audit.ensure_clear()
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        with Mode(name) if name != "native" else nullcontext():
                            value = decode()
                        torch.cuda.synchronize()
                        row["samples_wall_ms"][name].append((time.perf_counter() - t0) * 1000)
                row["native_ms"] = statistics.median(row["samples_wall_ms"]["native"])
                row["candidate_ms"] = statistics.median(row["samples_wall_ms"][cfg.method])
                row["speedup"] = row["native_ms"] / row["candidate_ms"]
                row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                row["model_lowvram"] = bool(getattr(vae.first_stage_model, "model_lowvram", False))
                row["dynamic_patcher"] = vae.patcher.is_dynamic()
                for name in ("native", cfg.method):
                    trace = cfg.output / (path + "_" + name + ".trace.json")
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA],
                                                record_shapes=True) as prof:
                        with Mode(name) if name != "native" else nullcontext():
                            value = decode()
                        torch.cuda.synchronize()
                    prof.export_chrome_trace(str(trace))
                    row[name + "_profile"] = profile_tables(prof, trace)
                row["integration_pass"] = row["actual_candidate_hit"] and row["error"]["strict_rgb_pass"]
                save()
                print(json.dumps({"path": path, "hits": row["candidate_dispatch"]["eligible_hits"],
                                  "error": row["error"], "speedup": row["speedup"]}), flush=True)
            model_management.unload_all_models()
            audit.ensure_clear()
            report["status"] = "measured"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        report["gpu_audit"] = audit.report()
        if not report["gpu_audit"]["valid_for_timing"]:
            report["status"] = "invalid_timing_discarded"
        save()


if __name__ == "__main__":
    main()
