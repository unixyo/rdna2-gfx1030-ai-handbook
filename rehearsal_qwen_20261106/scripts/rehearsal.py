"""Qwen-Image-2.1 rehearsal harness: guide section 2.1 profile + optional A/B.

Owns a short-lived ComfyUI on a private port with the Studio production flags.
Inputs/outputs live in the scratch directory; the lab is not modified.
"""
import argparse, collections, importlib.abc, importlib.machinery, json, os, runpy
import signal, sys, threading, time, traceback, urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--modes", default="warm,profile")   # warm|profile|base|cand
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--seed", type=int, default=20261106)
ap.add_argument("--port", type=int, default=8461)
ap.add_argument("--candidate", default="none")
cfg = ap.parse_args()

LAB = Path("~/DevSpace/qwen-image-21-rdna2-lab01")
HERE = Path(__file__).resolve().parent
out = Path(cfg.out).resolve(); out.mkdir(parents=True, exist_ok=True)
state = {"mode": None, "stage": None, "attn": collections.Counter(), "runs": [],
         "ref_latent": None, "ref_image": None, "ok": True}

PROMPT = "[private prompt removed: two-character anime illustration edit with a reference image]"
NEGATIVE = "low quality, blurry, extra fingers, deformed hands, text, watermark"


def save(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2, default=str) + "\n")


def graph(mode, idx):
    g = json.loads((LAB / "workflows/request-gemini-edit-640x960-api.json").read_text())
    g["11"]["inputs"]["image"] = "ref_image.png"
    g["3"]["inputs"]["resolution"] = 0
    g["3"]["inputs"]["prompt"] = PROMPT
    g["3"]["inputs"]["negative_prompt"] = NEGATIVE
    g["8"]["inputs"]["seed"] = cfg.seed
    g["8"]["inputs"]["steps"] = cfg.steps
    g["10"]["inputs"]["filename_prefix"] = f"rehearsal_{idx}_{mode}"
    return g


def instrument(nodes):
    import torch
    import comfy.ldm.qwen_image21.model as qm
    base_attn = qm.optimized_attention

    def attn(q, k, v, heads, mask=None, **kw):
        if state["stage"] == "denoise":
            key = json.dumps({"q": list(q.shape), "k": list(k.shape), "heads": heads,
                              "dtype": str(q.dtype), "mask": None if mask is None else list(mask.shape),
                              "fn": base_attn.__name__})
            state["attn"][key] += 1
        return base_attn(q, k, v, heads, mask=mask, **kw)
    qm.optimized_attention = attn

    orig_sample, orig_decode = nodes.KSampler.sample, nodes.VAEDecodeTiled.decode

    def sample(self, *a, **kw):
        mode = state["mode"]
        torch.cuda.synchronize(); t = time.perf_counter()
        state["stage"] = "denoise"
        try:
            if mode == "profile":
                from torch.profiler import profile, ProfilerActivity
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                    r = orig_sample(self, *a, **kw); torch.cuda.synchronize()
                state["prof"] = prof
            else:
                r = orig_sample(self, *a, **kw); torch.cuda.synchronize()
        finally:
            state["stage"] = None
        state["wall"]["denoise"] = time.perf_counter() - t
        lat = r[0]["samples"].detach().float().cpu()
        state["latent"] = lat
        return r

    def decode(self, *a, **kw):
        torch.cuda.synchronize(); t = time.perf_counter()
        r = orig_decode(self, *a, **kw); torch.cuda.synchronize()
        state["wall"]["vae"] = time.perf_counter() - t
        state["image"] = r[0].detach().float().cpu()
        return r

    nodes.KSampler.sample = sample
    nodes.VAEDecodeTiled.decode = decode
    nodes.KSampler.IS_CHANGED = classmethod(lambda cls, **k: float("nan"))


def summarize_profile(prof):
    rows = []
    for e in sorted(prof.key_averages(), key=lambda e: e.self_device_time_total, reverse=True)[:25]:
        rows.append({"op": e.key, "self_gpu_ms": e.self_device_time_total / 1000, "count": e.count})
    shaped = []
    for e in sorted(prof.key_averages(group_by_input_shape=True),
                    key=lambda e: e.self_device_time_total, reverse=True)[:25]:
        shaped.append({"op": e.key, "shapes": str(e.input_shapes)[:300],
                       "self_gpu_ms": e.self_device_time_total / 1000, "count": e.count})
    total = sum(e.self_device_time_total for e in prof.key_averages()) / 1000
    return {"total_self_gpu_ms": total, "top_ops": rows, "top_ops_by_shape": shaped}


def err(a, b):
    d = a - b
    return {"max_abs": d.abs().max().item(), "rmse": d.square().mean().sqrt().item()}


class Loader(importlib.abc.Loader):
    def __init__(self, base): self.base = base
    def create_module(self, spec):
        m = getattr(self.base, "create_module", None); return m(spec) if m else None
    def exec_module(self, module):
        self.base.exec_module(module); instrument(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != "nodes": return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec and spec.loader: spec.loader = Loader(spec.loader)
        return spec


def client():
    ep = f"http://127.0.0.1:{cfg.port}"
    def req(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        r = urllib.request.Request(ep + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=10) as resp: return json.load(resp)
    try:
        for _ in range(240):
            try: req("/system_stats"); break
            except (OSError, ValueError): time.sleep(1)
        else: raise RuntimeError("server not ready")
        for idx, mode in enumerate(cfg.modes.split(",")):
            state.update(mode=mode, wall={}, latent=None, image=None)
            if mode == "cand": apply_candidate(True)
            started = time.perf_counter()
            pid = req("/prompt", {"prompt": graph(mode, idx), "client_id": "rehearsal"})["prompt_id"]
            for _ in range(6000):
                h = req("/history/" + pid)
                if pid in h: break
                time.sleep(0.2)
            else: raise TimeoutError
            if mode == "cand": apply_candidate(False)
            item = h[pid]; okay = item["status"]["status_str"] == "success"
            row = {"idx": idx, "mode": mode, "success": okay, "wall": dict(state["wall"]),
                   "request_wall_s": time.perf_counter() - started}
            if not okay: row["messages"] = item["status"].get("messages")
            if cfg.candidate != "none":
                import candidates
                row["cand_hits"], row["cand_fallbacks"] = candidates._state["hits"], candidates._state["fallbacks"]
            if okay and mode in ("base", "cand", "warm"):
                if state["ref_latent"] is None and mode != "cand":
                    state["ref_latent"], state["ref_image"] = state["latent"], state["image"]
                elif state["ref_latent"] is not None:
                    row["latent_err"] = err(state["latent"], state["ref_latent"])
                    row["image_err"] = err(state["image"], state["ref_image"])
            if mode == "profile" and okay:
                s = summarize_profile(state.pop("prof"))
                s["attention_calls"] = [dict(json.loads(k), calls=n) for k, n in state["attn"].items()]
                s["wall_profiled"] = dict(state["wall"])
                save("profile.json", s)
            state["runs"].append(row); save("runs.json", state["runs"])
            print("REHEARSAL " + json.dumps(row), flush=True)
            if not okay: state["ok"] = False; break
    except Exception:
        save("harness_error.json", {"tb": traceback.format_exc()}); traceback.print_exc(); state["ok"] = False
    finally:
        os.kill(os.getpid(), signal.SIGINT)


_cand = {}
def apply_candidate(on):
    """Candidate switch filled in after the profile names the bottleneck."""
    if cfg.candidate == "none":
        return
    import candidates
    candidates.toggle(cfg.candidate, on)


os.environ.update({"HF_HUB_OFFLINE": "1", "HIP_VISIBLE_DEVICES": "0", "Q21_RDNA2_INT8_PATCH": "1",
                   "Q21_RDNA2_ENCODER_CPU": "0", "Q21_RDNA2_CONV3D_PATCH": "0", "Q21_CFG1_PROBE": "0"})
runtime = LAB / "runtime/ComfyUI"
sys.path.insert(0, str(runtime)); sys.path.insert(0, str(HERE))
sys.meta_path.insert(0, Finder())
sys.argv = [str(runtime / "main.py"), "--listen", "127.0.0.1", "--port", str(cfg.port),
            "--extra-model-paths-config", str(LAB / "extra_model_paths.yaml"),
            "--disable-auto-launch", "--disable-triton-backend",
            "--use-split-cross-attention", "--reserve-vram", "1", "--vram-headroom", "3",
            "--input-directory", str(HERE / "input"), "--output-directory", str(out / "images"),
            "--temp-directory", str(HERE / "temp")]
os.chdir(runtime)
threading.Thread(target=client, daemon=True).start()
try:
    runpy.run_path(str(runtime / "main.py"), run_name="__main__")
except KeyboardInterrupt:
    pass
sys.exit(0 if state["ok"] else 1)
