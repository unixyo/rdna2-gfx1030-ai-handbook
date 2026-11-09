#!/usr/bin/env python3
"""Profile one existing ComfyUI API graph through its unmodified main.py.

Uses the installed runtime and original nodes, including dynamic VRAM and custom
nodes. No server source or global configuration is changed. The owned server
exits after its one prompt completes; denoising and VAE are separately marked.
"""
import argparse
import collections
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import signal
import socket
import sys
import threading
import time
import traceback
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--project", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=20261105)
ap.add_argument("--port", type=int, default=8299)
ap.add_argument("--summarize-only", action="store_true")
ap.add_argument("--compare-attention", action="store_true")
ap.add_argument("--compare-int8", action="store_true")
ap.add_argument("--compare-rounds", type=int, default=3)
ap.add_argument("--precision-modes", nargs="+", help="Diagnosis modes, native first; optional mode:short or mode:long")
cfg = ap.parse_args()
benchmark = cfg.compare_attention or cfg.compare_int8
diagnosing = bool(cfg.precision_modes)
if diagnosing and (benchmark or cfg.precision_modes[0] != "native"):
    ap.error("Precision diagnosis needs native first and no other compare flags")
project = Path(cfg.project).resolve()
out = Path(cfg.out).resolve()
out.mkdir(parents=True, exist_ok=True)
state = {"stage": None, "prof": None, "sdpa": collections.Counter(),
         "stage_wall_s": {}, "success": False, "mode": "torch", "runs": [],
         "reference_latent": None, "reference_image": None}
MARK = "rdna2_stage::"


def save_json(name, value):
    (out / name).write_text(json.dumps(value, indent=2) + "\n")


def summarize_trace(path):
    """Attribute each GPU activity once, using launch's external correlation."""
    trace = json.loads(path.read_text())
    events = trace["traceEvents"]
    spans = [e for e in events if e.get("ph") == "X"]
    stages = [e for e in spans if e.get("cat") == "user_annotation" and e.get("name", "").startswith(MARK)]
    cpu = {}
    for e in spans:
        if e.get("cat") in ("cpu_op", "user_annotation"):
            ext = e.get("args", {}).get("External id")
            if ext is not None:
                cpu[ext] = e
    sdpa_spans = [e for e in spans if e.get("cat") == "user_annotation" and e.get("name") == "rdna2::sdpa"]
    sums = collections.defaultdict(lambda: [0.0, 0])
    kernel_sums = collections.defaultdict(lambda: [0.0, 0])
    stage_sums = collections.Counter()
    attention_sums = collections.Counter()
    copy_sums = collections.Counter()
    unassigned = 0.0
    total = 0.0
    for e in spans:
        cat = e.get("cat", "").lower()
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        origin = cpu.get(e.get("args", {}).get("External id"))
        ts = origin["ts"] if origin else e["ts"]
        st = next((s["name"][len(MARK):] for s in stages
                   if s["ts"] <= ts <= s["ts"] + s["dur"]), None)
        if st is None:
            unassigned += e["dur"]
            continue
        duration = e["dur"]
        total += duration
        stage_sums[st] += duration
        key = origin["name"] if origin else "unattributed::" + e["name"]
        sums[key][0] += duration
        sums[key][1] += 1
        kernel_sums[e["name"]][0] += duration
        kernel_sums[e["name"]][1] += 1
        if cat != "kernel":
            copy_sums[st] += duration
        if any(s["ts"] <= ts <= s["ts"] + s["dur"] for s in sdpa_spans):
            attention_sums[st] += duration

    def rows(table):
        return [{"operator": name, "gpu_ms": values[0] / 1000,
                 "share_pct": 100 * values[0] / total if total else 0,
                 "gpu_activities": values[1]}
                for name, values in sorted(table.items(),
                                          key=lambda p: p[1][0], reverse=True)]
    return {"denominator": "sum of GPU kernel/memcpy/memset durations in denoising and VAE markers; overlapping streams may overlap in wall time",
            "gpu_total_ms": total / 1000,
            "stage_gpu_ms": {k: v / 1000 for k, v in stage_sums.items()},
            "sdpa_gpu_ms": {k: v / 1000 for k, v in attention_sums.items()},
            "transfer_memset_gpu_ms": {k: v / 1000 for k, v in copy_sums.items()},
            "excluded_outside_stages_ms": unassigned / 1000,
            "operators": rows(sums), "kernels": rows(kernel_sums),
            "sdpa_backend_cpu_ops": dict(collections.Counter(
                e["name"] for e in spans
                if "scaled_dot_product" in e.get("name", "") and e.get("cat") == "cpu_op"))}


def instrument(nodes):
    import torch
    import torch.nn.functional as F
    if diagnosing:
        import hashlib
        import inspect
        import comfy.ops
        from comfy_kitchen.backends.eager import quantization as quant
        from precision_diagnose import PrecisionDiagnosis
        diag = PrecisionDiagnosis(out / "precision_diagnosis.json", max_reference_bytes=256*1024**2)
        original_sample, original_decode = nodes.KSampler.sample, nodes.VAEDecode.decode
        signature = inspect.signature(original_sample)

        def sample_diagnosis(self, *args, **kwargs):
            state.pop("diagnosis_row", None)
            bound = signature.bind(self, *args, **kwargs).arguments
            hashes = {}
            for name in ("positive", "negative"):
                value = bound[name][0][0].detach().cpu().contiguous()
                hashes[name] = hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
            spec = state["precision_spec"].split(":")
            mode, scope = spec[0], spec[1] if len(spec) > 1 else "all"
            with diag.run(state["precision_spec"], mode=mode, scope=scope,
                          sdpa_owner=comfy.ops, denoiser=(bound["model"].model, "apply_model"),
                          quantizer_owner=quant,
                          metadata={"seed": int(bound["seed"]), "conditioning_sha256": hashes}):
                result = original_sample(self, *args, **kwargs)
                diag.checkpoint("latent", result[0]["samples"])
            return result

        def decode_diagnosis(self, vae, samples):
            result = original_decode(self, vae, samples)
            diag.checkpoint("image", result[0])
            diag.save()
            state["diagnosis_row"] = diag.report["runs"][-1]
            return result

        nodes.KSampler.sample = sample_diagnosis
        nodes.KSampler.IS_CHANGED = classmethod(lambda cls, **kwargs: float("nan"))
        nodes.VAEDecode.decode = decode_diagnosis
        return
    original_sdpa = F.scaled_dot_product_attention
    if benchmark:
        from rdna2_ops.attention import scaled_dot_product_attention as candidate_sdpa

    def sdpa(q, k, v, *args, **kwargs):
        if benchmark:
            if cfg.compare_attention and state["mode"] == "rdna2":
                return candidate_sdpa(q, k, v, *args, **kwargs)
            return original_sdpa(q, k, v, *args, **kwargs)
        if state["stage"]:
            mask = kwargs.get("attn_mask", args[0] if args else None)
            key = json.dumps({"stage": state["stage"], "q": list(q.shape),
                              "k": list(k.shape), "v": list(v.shape),
                              "dtype": str(q.dtype), "stride": list(q.stride()),
                              "mask": None if mask is None else {"shape": list(mask.shape), "dtype": str(mask.dtype)},
                              "dropout_p": kwargs.get("dropout_p", args[1] if len(args) > 1 else 0.0),
                              "is_causal": kwargs.get("is_causal", args[2] if len(args) > 2 else False)}, sort_keys=True)
            state["sdpa"][key] += 1
            with torch.profiler.record_function("rdna2::sdpa"):
                return original_sdpa(q, k, v, *args, **kwargs)
        return original_sdpa(q, k, v, *args, **kwargs)

    F.scaled_dot_product_attention = sdpa
    original_sample = nodes.KSampler.sample
    original_decode = nodes.VAEDecode.decode

    def sample(self, *args, **kwargs):
        if benchmark:
            state["stage"] = "denoise"
            torch.cuda.synchronize()
            begin = time.perf_counter()
            if cfg.compare_int8 and state["mode"] == "rdna2":
                import inspect
                import comfy_kitchen.backends.eager.quantization as quant
                from rdna2_ops.comfy_int8 import kitchen_int8_without_transpose_copy
                backend = quant._int8_matmul_accumulate
                if Path(inspect.getsourcefile(backend)).resolve() != project / "scripts/rdna2_int8_matmul.py":
                    raise RuntimeError("Expected the existing verified gfx1030 strided INT8 backend")
                with kitchen_int8_without_transpose_copy(backend):
                    result = original_sample(self, *args, **kwargs)
            else:
                result = original_sample(self, *args, **kwargs)
            torch.cuda.synchronize()
            state["stage_wall_s"]["denoise"] = time.perf_counter() - begin
            state["stage"] = None
            latent = result[0]["samples"].detach().float().cpu()
            if state["reference_latent"] is None:
                state["reference_latent"] = latent
            delta = latent - state["reference_latent"]
            state["latent_error"] = {"max_abs": delta.abs().max().item(),
                                     "rmse": delta.square().mean().sqrt().item(),
                                     "finite": bool(torch.isfinite(latent).all())}
            return result
        if state["prof"] is not None:
            raise RuntimeError("This harness profiles exactly one KSampler call")
        torch.cuda.synchronize()
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False)
        state["prof"] = prof
        prof.start()
        state["stage"] = "denoise"
        start = time.perf_counter()
        with torch.profiler.record_function(MARK + "denoise"):
            result = original_sample(self, *args, **kwargs)
            torch.cuda.synchronize()
        state["stage_wall_s"]["denoise"] = time.perf_counter() - start
        state["stage"] = None
        torch.save(result[0]["samples"].detach().cpu(), out / "actual_latent.pt")
        print("RDNA2_PROFILE_DENOISE_DONE", flush=True)
        return result

    def decode(self, vae, samples):
        if benchmark:
            state["stage"] = "vae"
            torch.cuda.synchronize()
            begin = time.perf_counter()
            result = original_decode(self, vae, samples)
            torch.cuda.synchronize()
            state["stage_wall_s"]["vae"] = time.perf_counter() - begin
            state["stage"] = None
            pixels = result[0].detach().float().cpu()
            if state["reference_image"] is None:
                state["reference_image"] = pixels
            delta = pixels - state["reference_image"]
            state["image_error"] = {"max_abs": delta.abs().max().item(),
                                    "rmse": delta.square().mean().sqrt().item(),
                                    "finite": bool(torch.isfinite(pixels).all())}
            torch.save(pixels, out / ("image_" + state["mode"] + ".pt"))
            return result
        if state["prof"] is None:
            return original_decode(self, vae, samples)
        state["stage"] = "vae"
        start = time.perf_counter()
        with torch.profiler.record_function(MARK + "vae"):
            result = original_decode(self, vae, samples)
            torch.cuda.synchronize()
        state["stage_wall_s"]["vae"] = time.perf_counter() - start
        state["stage"] = None
        prof = state["prof"]
        prof.stop()
        trace_path = out / "comfy_trace.json"
        prof.export_chrome_trace(str(trace_path))
        summary = summarize_trace(trace_path)
        props = torch.cuda.get_device_properties(0)
        summary.update({
            "seed": cfg.seed, "workflow": "sft_fast4_int8_512",
            "torch": torch.__version__, "hip": torch.version.hip,
            "gpu": torch.cuda.get_device_name(0), "gcn_arch": getattr(props, "gcnArchName", None),
            "stage_wall_s_profiled": state["stage_wall_s"],
            "sdpa_calls": [dict(json.loads(k), calls=n) for k, n in state["sdpa"].items()],
            "vae_dtype": str(vae.vae_dtype), "vae_device": str(vae.device),
            "image_shape": list(result[0].shape),
            "sdpa_flags": {"flash": torch.backends.cuda.flash_sdp_enabled(),
                           "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                           "math": torch.backends.cuda.math_sdp_enabled()},
            "note": "Real API graph through installed main.py; cold loads/profiler overhead make wall times diagnostic, not steady-state speed benchmarks."})
        save_json("comfy_profile.json", summary)
        torch.save(result[0].detach().cpu(), out / "actual_image_tensor.pt")
        print("RDNA2_PROFILE_TOP10 " + json.dumps(summary["operators"][:10]), flush=True)
        print("RDNA2_PROFILE_SDPA " + json.dumps(summary["sdpa_backend_cpu_ops"]), flush=True)
        return result

    nodes.KSampler.sample = sample
    nodes.VAEDecode.decode = decode
    if benchmark:
        # Force this actual node to execute for every same-seed A/B request.
        # Model and text-condition caches remain valid.
        nodes.KSampler.IS_CHANGED = classmethod(lambda cls, **kwargs: float("nan"))


class NodeLoader(importlib.abc.Loader):
    def __init__(self, base):
        self.base = base

    def create_module(self, spec):
        method = getattr(self.base, "create_module", None)
        return method(spec) if method else None

    def exec_module(self, module):
        self.base.exec_module(module)
        instrument(module)


class NodeFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "nodes":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.loader:
            spec.loader = NodeLoader(spec.loader)
        return spec


def client():
    endpoint = "http://127.0.0.1:" + str(cfg.port)

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(endpoint + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.load(response)

    try:
        for _ in range(180):
            try:
                request("/system_stats")
                break
            except (OSError, ValueError):
                time.sleep(1)
        else:
            raise RuntimeError("Owned ComfyUI server did not become ready")
        graph = json.loads((project / "workflows/sft_fast4_int8_512.json").read_text())
        graph["8"]["inputs"]["seed"] = cfg.seed
        modes = cfg.precision_modes if diagnosing else (["torch", "rdna2"] if benchmark else ["torch"])
        if benchmark:
            for n in range(cfg.compare_rounds):
                modes.extend(["torch", "rdna2"] if n % 2 == 0 else ["rdna2", "torch"])
        for run_index, mode in enumerate(modes):
            state["mode"] = mode
            state["precision_spec"] = mode
            state.pop("diagnosis_row", None)
            state["stage_wall_s"] = {}
            state["latent_error"] = None
            state["image_error"] = None
            graph["11"]["inputs"]["filename_prefix"] = "rdna2_" + mode + "_" + str(run_index)
            save_json("submitted_workflow.json", graph)
            accepted = request("/prompt", {"prompt": graph, "client_id": "rdna2-profiler"})
            prompt_id = accepted["prompt_id"]
            save_json("prompt_id.json", accepted)
            started = time.perf_counter()
            for _ in range(1500):
                history = request("/history/" + prompt_id)
                if prompt_id in history:
                    item = history[prompt_id]
                    save_json("comfy_history.json", item)
                    state["success"] = item["status"]["status_str"] == "success"
                    completion = {"success": state["success"],
                                  "request_wall_s": time.perf_counter() - started,
                                  "prompt_id": prompt_id, "outputs": item.get("outputs")}
                    save_json("completion.json", completion)
                    if diagnosing:
                        record = state.get("diagnosis_row", {})
                        print("RDNA2_PRECISION " + json.dumps({
                            "mode": mode, "success": state["success"],
                            "checkpoints": record.get("checkpoints", {}),
                            "first_attention_divergence": record.get("attention", {}).get("first_output_difference"),
                            "error_messages": item.get("status", {}).get("messages", []) if not state["success"] else []
                        }), flush=True)
                    if benchmark:
                        row = {"mode": mode, "warmup": run_index < 2,
                               "stage_wall_s": dict(state["stage_wall_s"]),
                               "latent_error": state.get("latent_error"),
                               "image_error": state.get("image_error"), **completion}
                        row["accuracy_pass"] = bool(
                            row["success"] and row["latent_error"] and row["image_error"]
                            and row["latent_error"]["finite"] and row["image_error"]["finite"]
                            and row["latent_error"]["max_abs"] == 0.0
                            and row["image_error"]["max_abs"] == 0.0)
                        state["runs"].append(row)
                        state["success"] = row["accuracy_pass"]
                        save_json("comfy_" + ("combined" if cfg.compare_attention and cfg.compare_int8 else "int8" if cfg.compare_int8 else "attention") + "_ab.json",
                                  {"seed": cfg.seed, "workflow": "sft_fast4_int8_512",
                                   "attention_optimized": cfg.compare_attention, "int8_copy_optimized": cfg.compare_int8,
                                   "same_seed_forced_rerun": True,
                                   "acceptance": "latent and image must be finite and bitwise identical",
                                   "status": "failed" if not state["success"] else "passed" if run_index + 1 == len(modes) else "running",
                                   "runs": state["runs"]})
                        print("RDNA2_COMFY_AB " + json.dumps(row), flush=True)
                    print("RDNA2_PROFILE_COMPLETE " + str(state["success"]), flush=True)
                    if not state["success"]:
                        return
                    break
                time.sleep(0.2)
            else:
                raise TimeoutError("ComfyUI graph exceeded bounded execution time")
    except Exception:
        save_json("harness_error.json", {"traceback": traceback.format_exc()})
        traceback.print_exc()
    finally:
        os.kill(os.getpid(), signal.SIGINT)


if cfg.summarize_only:
    summary = json.loads((out / "comfy_profile.json").read_text())
    summary.update(summarize_trace(out / "comfy_trace.json"))
    save_json("comfy_profile.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("operators", "kernels")}))
    print(json.dumps(summary["operators"][:10]))
    sys.exit(0)

# Refuse an occupied port before the client can submit a request to another
# ComfyUI process. The benchmark must own the server it talks to.
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_probe:
    port_probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        port_probe.bind(("127.0.0.1", cfg.port))
    except OSError as exc:
        raise RuntimeError(f"Port {cfg.port} is unavailable; choose a free --port") from exc

os.environ.setdefault("MOSS_RDNA2_INT8_PATCH", "1")
os.environ.setdefault("MOSS_RDNA2_INT8_LAYOUT", "legacy")
os.environ.setdefault("MOSS_RDNA2_ATTENTION", "native")
runtime = project / "runtime/ComfyUI"
sys.path.insert(0, str(runtime))
sys.meta_path.insert(0, NodeFinder())
sys.argv = [str(runtime / "main.py"),
            "--listen", "127.0.0.1", "--port", str(cfg.port),
            "--base-directory", str(project), "--models-directory", str(project / "models"),
            "--output-directory", str(out / "images"),
            "--input-directory", str(project / "input"),
            "--temp-directory", str(project / "temp"),
            "--user-directory", str(project / "user"),
            "--use-pytorch-cross-attention", "--disable-comfy-compiler"]
threading.Thread(target=client, name="rdna2-profile-client", daemon=True).start()
try:
    runpy.run_path(str(runtime / "main.py"), run_name="__main__")
except KeyboardInterrupt:
    pass
sys.exit(0 if state["success"] else 1)

