"""Small research hooks for conditioning hashes and exact TunableOp CSV hits.

Imported only by the guarded model wrapper. No operating files are changed.
"""
from __future__ import annotations
import collections
import contextlib
import hashlib
import json
from pathlib import Path
import re
import time


HIT = re.compile(r"ResultEntry found for ([^,\r\n]+),([^\r\n]+)")
MISS = re.compile(r"missing (?:op|params)_signature, returning null ResultEntry for ([^,\r\n]+),([^\r\n]+)")


def install(nodes, runner_state, out, *, trace_csv_hits=False, profile_conditioning=False):
    import torch
    import comfy.sd
    import torch.cuda.tunable as tunable

    out = Path(out)
    log = out / "tunable_verbose.log"
    rows = {tuple(r[:2]): list(r) for r in tunable.get_results()}
    state = {"installed": True, "conditioning_calls": [], "sampler_calls": [],
             "hit_logging": trace_csv_hits, "profile_conditioning": profile_conditioning,
             "interpretation": "loaded rows are not hits; only ResultEntry found log lines count as exact CSV hits"}
    original_clip = comfy.sd.CLIP.encode_from_tokens_scheduled
    original_sample = nodes.KSampler.sample

    def snapshot(value):
        tensors, unknown = [], []
        def visit(v, path):
            if isinstance(v, torch.Tensor):
                cpu = v.detach().cpu().contiguous()
                raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
                item = {"path": path, "shape": list(cpu.shape), "dtype": str(cpu.dtype),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "finite": bool(torch.isfinite(cpu).all()) if cpu.is_floating_point() else True}
                tensors.append(item)
                return {"tensor": item}
            if isinstance(v, dict):
                return {str(k): visit(x, path + "." + str(k)) for k, x in sorted(v.items(), key=lambda z: str(z[0]))}
            if isinstance(v, (tuple, list)):
                return [visit(x, f"{path}[{i}]") for i, x in enumerate(v)]
            if v is None or isinstance(v, (str, int, float, bool)):
                return v
            name = type(v).__module__ + "." + type(v).__qualname__
            unknown.append({"path": path, "type": name})
            return {"unhashed_object_type": name}
        tree = visit(value, "$")
        digest = hashlib.sha256(json.dumps(tree, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {"sha256": digest, "tensors": tensors, "unhashed_objects": unknown,
                "all_tensor_leaves_finite": all(x["finite"] for x in tensors)}

    def offset():
        return log.stat().st_size if trace_csv_hits and log.exists() else 0

    def hits_since(start):
        if not trace_csv_hits or not log.exists():
            return {"enabled": False, "hits": [], "misses": []}
        with log.open("rb") as fp:
            fp.seek(start)
            text = fp.read().decode(errors="replace")
        counts = collections.Counter(HIT.findall(text))
        misses = collections.Counter(MISS.findall(text))
        return {"enabled": True, "start_byte": start, "end_byte": log.stat().st_size,
                "hits": [{"op": key[0], "signature": key[1], "calls": n,
                          "loaded_solution": rows.get(key, [None, None, None])[2]}
                         for key, n in counts.items()],
                "misses": [{"op": key[0], "signature": key[1], "calls": n}
                           for key, n in misses.items()]}

    def save():
        (out / "conditioning_capture.json").write_text(json.dumps(state, indent=2) + "\n")

    def clip(self, tokens, *args, **kwargs):
        idx = len(state["conditioning_calls"])
        row = {"index": idx, "mode": runner_state.get("mode"),
               "stage": "CLIP.encode_from_tokens_scheduled",
               "tunable_enabled": bool(tunable.is_enabled()), "tokens": snapshot(tokens)}
        start = offset()
        began = time.perf_counter()
        profiler = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA],
                                           record_shapes=True)
                    if profile_conditioning else contextlib.nullcontext())
        with profiler as prof:
            result = original_clip(self, tokens, *args, **kwargs)
            if profile_conditioning:
                torch.cuda.synchronize()
        row["wall_s"] = time.perf_counter() - began
        row["csv"] = hits_since(start)
        row["output"] = snapshot(result)
        if profile_conditioning:
            trace = out / f"conditioning_{idx:03d}.trace.json"
            prof.export_chrome_trace(str(trace))
            raw = json.loads(trace.read_text())
            names = collections.Counter(e.get("name", "") for e in raw.get("traceEvents", [])
                                        if e.get("cat") == "kernel" and "Cijk_" in e.get("name", ""))
            row["trace"] = str(trace)
            row["tensile_kernel_names"] = [{"name": name, "calls": n} for name, n in names.items()]
        state["conditioning_calls"].append(row)
        save()
        return result

    def sample(self, *args, **kwargs):
        names = ("model", "seed", "steps", "cfg", "sampler_name", "scheduler",
                 "positive", "negative", "latent_image", "denoise")
        values = dict(zip(names, args))
        values.update(kwargs)
        row = {"index": len(state["sampler_calls"]), "mode": runner_state.get("mode"),
               "seed": values.get("seed"), "steps": values.get("steps"),
               "clip_calls_seen": len(state["conditioning_calls"]),
               "tunable_enabled": bool(tunable.is_enabled()),
               "positive": snapshot(values.get("positive")),
               "negative": snapshot(values.get("negative")),
               "latent_input": snapshot(values.get("latent_image"))}
        # These hashes and log scans are outside the existing runner's denoise timer.
        start = offset()
        result = original_sample(self, *args, **kwargs)
        row["csv"] = hits_since(start)
        row["latent_output"] = snapshot(result)
        state["sampler_calls"].append(row)
        save()
        return result

    comfy.sd.CLIP.encode_from_tokens_scheduled = clip
    nodes.KSampler.sample = sample
    save()
    return state

