#!/usr/bin/env python3
"""Configurable owned-server ComfyUI research harness; no runtime source edits.

Run in the model project's EXISTING interpreter. A JSON spec names its runtime,
API workflow, node class methods, attention symbol, startup flags, and environment.
Only the owned process is instrumented. Capture/candidate hooks are restored after
samples; node timing wrappers live only in this process. The current node contract
is KSampler-style sample -> ({samples: Tensor},) and VAE decode -> (RGB Tensor,).
Modes baseline/warm/capture/profile are built in. Other modes are provided by
research_switches_v05 in this research folder. Profile mode measures denoise and
VAE separately. Stage clocks exclude report/trace export; profile timing is diagnostic.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import functools
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import inspect
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import signal
import socket
import sys
import tempfile
import threading
import time
import traceback


def resolve(name):
    module, attribute = name.split(":", 1)
    obj = importlib.import_module(module)
    parts = attribute.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    return obj, parts[-1]


def source(fn):
    fn = inspect.unwrap(fn)
    try:
        path = inspect.getsourcefile(fn)
    except TypeError:
        path = None
    return {"module": getattr(fn, "__module__", None),
            "name": getattr(fn, "__qualname__", str(fn)), "path": path}


def device_activity_table(trace_path):
    """Sum complete kernel/copy/memset records, excluding GPU range annotations.

    CUDA device_type also contains gpu_user_annotation records in PyTorch. A
    device-type-only key_averages filter therefore double-counts nested ranges.
    Chrome trace categories identify the actual activity records unambiguously.
    The denominator is summed device work, not elapsed wall or a stream union.
    """
    path = Path(trace_path)
    raw = path.read_bytes()
    trace = json.loads(raw)
    allowed = ("kernel", "gpu_memcpy", "gpu_memset")
    grouped, counts = Counter(), Counter()
    categories, category_counts = Counter(), Counter()
    excluded, excluded_counts = Counter(), Counter()
    for event in trace["traceEvents"]:
        if event.get("ph") != "X":
            continue
        duration = event.get("dur")
        if not isinstance(duration, (int, float)):
            continue
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid complete-event duration in " + str(path))
        category = event.get("cat", "")
        if category not in allowed:
            excluded[category] += duration
            excluded_counts[category] += 1
            continue
        device = event.get("args", {}).get("device")
        key = (category, event.get("name", ""), device)
        grouped[key] += duration
        counts[key] += 1
        categories[category] += duration
        category_counts[category] += 1
    rows = [
        {"name": name, "category": category, "device": device,
         "self_gpu_ms": duration / 1000, "count": counts[(category, name, device)],
         "device_type": "CUDA", "input_shapes": []}
        for (category, name, device), duration in grouped.items()
    ]
    rows.sort(key=lambda row: row["self_gpu_ms"], reverse=True)
    denominator = math.fsum(row["self_gpu_ms"] for row in rows)
    for row in rows:
        row["share_of_this_table"] = row["self_gpu_ms"] / denominator if denominator else 0.0
    return {
        "denominator_ms": denominator, "rows": rows, "top10": rows[:10],
        "aggregation": "sum Chrome trace complete events (ph=X), grouped by category/name/device",
        "included_categories": list(allowed),
        "category_totals": {category: {"count": category_counts[category],
                                     "duration_ms": categories[category] / 1000}
                            for category in allowed},
        "excluded_complete_event_categories": {
            category: {"count": excluded_counts[category], "duration_ms": duration / 1000}
            for category, duration in sorted(excluded.items())
        },
        "trace_file": path.name, "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "trace_retained": True, "trace_duration_unit": "microseconds",
        "denominator_semantics": "summed kernel/memcpy/memset work; never add CPU attribution or GPU annotations; may overlap across streams",
    }


def profile_tables(prof, trace_path=None):
    """Separate CPU attribution from real device activities.

    Pass an ALREADY EXPORTED Chrome trace path to retain it without exporting
    twice. Standalone callers may omit it: one temporary trace is exported inside
    this research folder, parsed, then removed; its hash remains in the report.
    """
    rows = []
    for event in prof.key_averages(group_by_input_shape=True):
        kind = str(event.device_type).rsplit(".", 1)[-1]
        if kind != "CPU" or event.self_device_time_total <= 0:
            continue
        rows.append({
            "name": event.key, "self_gpu_ms": event.self_device_time_total / 1000,
            "count": event.count, "device_type": str(event.device_type),
            "input_shapes": event.input_shapes,
        })
    rows.sort(key=lambda row: row["self_gpu_ms"], reverse=True)
    denominator = math.fsum(row["self_gpu_ms"] for row in rows)
    for row in rows:
        row["share_of_this_table"] = row["self_gpu_ms"] / denominator if denominator else 0.0
    if trace_path is None:
        with tempfile.TemporaryDirectory(prefix="rdna2_profile_", dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / "trace.json"
            prof.export_chrome_trace(str(path))
            device = device_activity_table(path)
        device.update(trace_file=None, trace_retained=False)
    else:
        device = device_activity_table(trace_path)
    return {"operator_attribution": {"denominator_ms": denominator, "rows": rows, "top10": rows[:10]},
            "device_activities": device}


def capture_tensor_view(value):
    """Borrow a Tensor without consuming Comfy's single-owner container.

    AttentionTensorContainer.peek() is a read-only borrow; take() must remain
    the original backend's responsibility. This helper never changes arguments
    passed to the observed callable.
    """
    import torch
    if isinstance(value, torch.Tensor):
        return value
    peek = getattr(value, "peek", None)
    if callable(peek):
        tensor = peek()
        if isinstance(tensor, torch.Tensor):
            return tensor
    raise TypeError("Unsupported Attention capture input " + type(value).__qualname__
                    + "; set attention_capture_target to comfy.ops:scaled_dot_product_attention")


def runtime_metadata(torch, runtime):
    """Collect the versions actually loaded by this owned server process."""
    try:
        triton_version = importlib.import_module("triton").__version__
    except ImportError:
        triton_version = None
    gpu = None
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        gpu = {"ordinal": index, "name": props.name,
               "gcn_arch_name": getattr(props, "gcnArchName", None),
               "total_memory_bytes": props.total_memory}
    math_flag = getattr(torch.backends.cuda, "fp16_bf16_reduction_math_sdp_allowed", None)
    return {
        "python": sys.version, "python_executable": sys.executable,
        "torch": str(torch.__version__), "hip": torch.version.hip,
        "triton": triton_version, "gpu": gpu, "comfy_root": str(runtime),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "research_source_files_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("research_switches_v05.py", "rdna2_ops/int8_gemm.py",
                         "rdna2_ops/fp32_attention.py", "rdna2_ops/mixed_attention.py",
                         "rdna2_ops/sage_attention_v05.py")
            if (Path(__file__).parent / name).is_file()
        },
        "startup_precision_flags": {
            "fp16_bf16_reduction_math_sdp_allowed": math_flag() if math_flag else None,
            "allow_fp16_reduced_precision_reduction":
                getattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", None),
        },
        "timing_contract": "synchronized stage wall; profile setup/collection included; report and trace export excluded",
        "profile_schema_version": 2,
        "profile_views": "CPU operator attribution is separate; device table uses only Chrome trace kernel/gpu_memcpy/gpu_memset complete events, excluding gpu_user_annotation",
        "node_contract": "KSampler-family synchronous sample -> ({samples: Tensor},); VAE decode -> (RGB Tensor,)",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="warm,baseline,capture,baseline")
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--port", type=int, default=8463)
    cfg = ap.parse_args()
    spec = json.loads(Path(cfg.spec).read_text())
    out = Path(cfg.out).resolve()
    if (out / "runs.json").exists():
        raise FileExistsError("Use a fresh result directory: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    for folder in ("input", "images", "temp", "user"):
        (out / folder).mkdir(exist_ok=True)
    for src in spec.get("input_files", []):
        shutil.copy2(src, out / "input" / Path(src).name)
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", cfg.port))
    runtime = Path(spec["runtime"])
    os.environ.update({"HF_HUB_OFFLINE": "1", **spec.get("env", {})})
    sys.path.insert(0, str(runtime))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    state = {"mode": None, "runs": [], "ok": True, "ref_latent": None,
             "ref_image": None, "captured": set(), "last_capture": {}}
    instrument_cleanups = []

    def save(name, obj):
        (out / name).write_text(json.dumps(obj, indent=2, default=str) + "\n")

    def digest(t):
        t = t.detach().cpu().contiguous()
        return hashlib.sha256(t.view(__import__("torch").uint8).numpy().tobytes()).hexdigest()

    def errors(t, ref):
        import torch
        t = t.detach().float().cpu()
        d = (t - ref).double()
        return {"finite": bool(torch.isfinite(t).all()), "max_abs": d.abs().max().item(),
                "rmse": d.square().mean().sqrt().item(), "byte_equal": digest(t) == digest(ref)}

    @contextmanager
    def capture():
        import torch
        from comfy_kitchen.registry import registry
        target = spec.get("attention_capture_target", spec["attention_target"])
        owner, attr = resolve(target)
        original = getattr(owner, attr)
        original_get = registry.get_implementation
        attn_calls, linear_calls = Counter(), Counter()
        wrapped_linear = {}

        @functools.wraps(original)
        def attention(q, k, v, *args, **kwargs):
            # Borrow the values for observation only. Pass the original
            # containers unchanged so their ownership/consumption is preserved.
            qt, kt, vt = [capture_tensor_view(t) for t in (q, k, v)]
            shapes = [list(t.shape) for t in (qt, kt, vt)]
            strides = [list(t.stride()) for t in (qt, kt, vt)]
            mask = kwargs.get("mask", kwargs.get("attn_mask"))
            heads = args[0] if args and type(args[0]) is int else kwargs.get("heads")
            if qt.ndim == 4 and heads is None:
                heads = qt.shape[1]
                if args:
                    mask = args[0]  # SDPA's positional attn_mask
            elif len(args) > 1 and mask is None:
                mask = args[1]  # Comfy attention's positional mask, after heads
            metadata = {"shapes": shapes, "strides": strides, "dtype": str(qt.dtype),
                        "heads": heads, "capture_target": target,
                        "input_wrappers": [type(t).__qualname__ for t in (q, k, v)],
                        "mask": None if mask is None else list(capture_tensor_view(mask).shape),
                        "source": source(original)}
            attn_calls[json.dumps(metadata, sort_keys=True)] += 1
            qlen = qt.shape[-2]
            klen = kt.shape[-2]
            key = (qlen, klen)
            if qlen >= 1000 and key not in state["captured"] and len(state["captured"]) < 2:
                state["captured"].add(key)
                torch.save({"q": qt.detach().cpu(), "k": kt.detach().cpu(), "v": vt.detach().cpu(),
                            "metadata": metadata, "scale": kwargs.get("scale")},
                           out / f"actual_qkv_{qlen}_{klen}.pt")
            return original(q, k, v, *args, **kwargs)

        def get(name, *args, **kwargs):
            fn = original_get(name, *args, **kwargs)
            if name != "int8_linear":
                return fn
            if fn not in wrapped_linear:
                @functools.wraps(fn)
                def linear(x, weight, weight_scale, *a, **kw):
                    w, ws = weight, weight_scale
                    md = {"x": list(x.shape), "weight": list(w.shape), "weight_stride": list(w.stride()),
                          "x_dtype": str(x.dtype), "scale": list(ws.shape),
                          "out_dtype": str(kw.get("out_dtype")), "convrot": kw.get("convrot"),
                          "input_act": kw.get("input_act"), "source": source(fn)}
                    linear_calls[json.dumps(md, sort_keys=True)] += 1
                    return fn(x, w, ws, *a, **kw)
                wrapped_linear[fn] = linear
            return wrapped_linear[fn]

        setattr(owner, attr, attention)
        registry.get_implementation = get
        try:
            yield
        finally:
            setattr(owner, attr, original)
            registry.get_implementation = original_get
            state["last_capture"] = {
                "attention": [dict(json.loads(k), calls=n) for k, n in attn_calls.items()],
                "int8_linear": [dict(json.loads(k), calls=n) for k, n in linear_calls.items()]}
            save("actual_call_paths.json", state["last_capture"])

    def instrument(nodes):
        import torch
        sampler_class = getattr(nodes, spec["sampler"][0])
        decoder_class = getattr(nodes, spec["decoder"][0])
        sample_name, decode_name = spec["sampler"][1], spec["decoder"][1]
        original_sample, original_decode = getattr(sampler_class, sample_name), getattr(decoder_class, decode_name)
        sig = inspect.signature(original_sample)
        metadata = runtime_metadata(torch, runtime)
        metadata["node_sources"] = {"sampler": source(original_sample),
                                    "decoder": source(original_decode)}
        save("runtime_metadata.json", metadata)

        def run_stage(stage, invoke):
            torch.cuda.synchronize()
            start = time.perf_counter()
            prof = None
            if state["mode"] == "profile":
                from torch.profiler import profile, ProfilerActivity, record_function
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                             record_shapes=True) as prof:
                    with record_function("rdna2_stage::" + stage):
                        result = invoke()
                        torch.cuda.synchronize()
            else:
                result = invoke()
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            # Stop the stage clock before summarizing, CPU output copies, or export.
            state["wall"][stage + "_s"] = state["wall"].get(stage + "_s", 0.0) + elapsed
            invocation = state["stage_calls"].get(stage, 0)
            state["stage_calls"][stage] = invocation + 1
            if prof is not None:
                prefix = f"profile_{state['request_index']:03d}_{stage}_{invocation:03d}"
                common = {"stage": stage, "request_index": state["request_index"],
                          "invocation": invocation, "profiled_stage_wall_s": elapsed,
                          "runtime_metadata": "runtime_metadata.json",
                          "wall_excludes_report_export": True, "profile_schema_version": 2}
                operators_file = prefix + "_operators.json"
                activities_file = prefix + "_kernels.json"
                trace_file = prefix + "_trace.json"
                # Export once, after stopping the stage clock. CUDA annotations
                # are nested ranges, so derive activity totals from trace cats.
                prof.export_chrome_trace(str(out / trace_file))
                reports = profile_tables(prof, trace_path=out / trace_file)
                save(operators_file, {
                    **common, **reports["operator_attribution"],
                    "denominator": "CPU operator self device attribution only; do not add device activity view",
                })
                save(activities_file, {
                    **common, **reports["device_activities"],
                    "denominator": "Chrome trace kernel/gpu_memcpy/gpu_memset complete events only; exclude GPU annotations and CPU attribution",
                })
                state["profile_artifacts"].append({
                    "stage": stage, "invocation": invocation,
                    "operators": operators_file, "kernels": activities_file, "trace": trace_file,
                })
            return result

        @functools.wraps(original_sample)
        def sample(self, *a, **kw):
            bound = sig.bind(self, *a, **kw).arguments
            hashes = {}
            for name in ("positive", "negative"):
                if name in bound:
                    hashes[name] = digest(bound[name][0][0])
            state["conditioning_first_embedding_sha256"] = hashes
            mode = state["mode"]
            context = capture() if mode == "capture" else nullcontext()
            if mode not in ("warm", "baseline", "capture", "profile"):
                switches = importlib.import_module(spec.get("switch_module", "research_switches_v05"))
                context = switches.activate(mode, spec, state, bound, out)
            with context:
                result = run_stage("denoise", lambda: original_sample(self, *a, **kw))
            state["latent"] = result[0]["samples"].detach().float().cpu()
            return result

        @functools.wraps(original_decode)
        def decode(self, *a, **kw):
            result = run_stage("vae", lambda: original_decode(self, *a, **kw))
            state["image"] = result[0].detach().float().cpu()
            return result

        setattr(sampler_class, sample_name, sample)
        setattr(decoder_class, decode_name, decode)
        sampler_class.IS_CHANGED = classmethod(lambda cls, **kw: float("nan"))
        # Optional owned-process instrumentation for operations outside the
        # sampler, such as a vision encoder in a conditioning node. The module
        # must preserve node descriptors and restore its hooks on cleanup.
        module_name = spec.get("instrument_module")
        if module_name:
            extension = importlib.import_module(module_name)
            cleanup = extension.install(nodes=nodes, spec=spec, state=state,
                                        out=out, run_stage=run_stage,
                                        digest=digest, save=save)
            if cleanup is not None:
                if not callable(cleanup):
                    raise TypeError("instrument_module.install must return a cleanup callable or None")
                instrument_cleanups.append(cleanup)

    class Loader(importlib.abc.Loader):
        def __init__(self, base): self.base = base
        def create_module(self, spec):
            fn = getattr(self.base, "create_module", None)
            return fn(spec) if fn else None
        def exec_module(self, module):
            self.base.exec_module(module)
            instrument(module)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != "nodes": return None
            result = importlib.machinery.PathFinder.find_spec(name, path)
            if result and result.loader: result.loader = Loader(result.loader)
            return result

    def client():
        import urllib.request
        endpoint = f"http://127.0.0.1:{cfg.port}"
        def request(path, body=None):
            payload = None if body is None else json.dumps(body).encode()
            req = urllib.request.Request(endpoint + path, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as response:
                return json.load(response)
        try:
            for _ in range(240):
                try: request("/system_stats"); break
                except (OSError, ValueError): time.sleep(0.5)
            else: raise RuntimeError("Owned server did not become ready")
            base_graph = json.loads(Path(spec["workflow"]).read_text())
            for node, values in spec.get("inputs", {}).items():
                base_graph[node]["inputs"].update(values)
            base_graph[spec["sampler_node"]]["inputs"]["seed"] = cfg.seed
            if cfg.steps is not None:
                base_graph[spec["sampler_node"]]["inputs"]["steps"] = cfg.steps
            save("submitted_workflow.json", base_graph)
            for index, mode in enumerate(cfg.modes.split(",")):
                state.update(mode=mode, request_index=index, wall={}, stage_calls={},
                             profile_artifacts=[], latent=None, image=None, candidate={})
                graph = json.loads(json.dumps(base_graph))
                graph[spec["save_node"]]["inputs"]["filename_prefix"] = f"v05_{index}_{mode}"
                started = time.perf_counter()
                pid = request("/prompt", {"prompt": graph, "client_id": "rdna2-v05-owned"})["prompt_id"]
                for _ in range(6000):
                    history = request("/history/" + pid)
                    if pid in history: break
                    time.sleep(0.2)
                else: raise TimeoutError("Workflow exceeded 20 minutes")
                item = history[pid]
                okay = item["status"]["status_str"] == "success"
                row = {"index": index, "mode": mode, "success": okay,
                       "wall": state["wall"], "request_wall_s": time.perf_counter() - started,
                       "conditioning_first_embedding_sha256": state.get("conditioning_first_embedding_sha256"),
                       "candidate": state.get("candidate", {}),
                       "stage_calls": dict(state["stage_calls"]),
                       "profile_artifacts": list(state["profile_artifacts"])}
                if not okay:
                    row["messages"] = item["status"].get("messages")
                else:
                    if state["latent"] is None or state["image"] is None:
                        raise RuntimeError("Sampler/decoder hook did not produce tensors; check node spec and cache invalidation")
                    if state["ref_latent"] is None:
                        if mode not in ("warm", "baseline", "capture"):
                            raise ValueError("First mode must produce a native reference")
                        state["ref_latent"], state["ref_image"] = state["latent"], state["image"]
                        if spec.get("save_reference_tensors", False):
                            import torch
                            torch.save({"latent": state["ref_latent"], "image": state["ref_image"],
                                        "seed": cfg.seed, "steps": cfg.steps,
                                        "conditioning_first_embedding_sha256": state.get("conditioning_first_embedding_sha256")},
                                       out / "reference_tensors.pt")
                    row["latent_error"] = errors(state["latent"], state["ref_latent"])
                    row["image_error"] = errors(state["image"], state["ref_image"])
                    # Optional same-process layout/candidate references. These
                    # retain CPU values only and never change the workflow.
                    refs = state.setdefault("comparison_refs", {})
                    row["comparison_errors"] = {
                        name: {"latent": errors(state["latent"], tensors[0]),
                               "image": errors(state["image"], tensors[1])}
                        for name, tensors in refs.items()}
                    if mode in spec.get("comparison_reference_modes", []) and mode not in refs:
                        refs[mode] = (state["latent"], state["image"])
                    e = row["image_error"]
                    row["image_gate"] = e["finite"] and e["max_abs"] <= 2/255 and e["rmse"] <= .25/255
                    row["latent_sha256"], row["image_sha256"] = digest(state["latent"]), digest(state["image"])
                if mode in spec.get("require_bitwise_modes", []):
                    row["required_bitwise_gate"] = (okay and row["latent_error"]["byte_equal"]
                                                    and row["image_error"]["byte_equal"])
                state["runs"].append(row)
                save("runs.json", {"spec": spec, "seed": cfg.seed,
                                   "rgb_gate": {"max_abs": 2/255, "rmse": .25/255}, "runs": state["runs"]})
                print("RDNA2_RESEARCH " + json.dumps(row), flush=True)
                if not okay or row.get("required_bitwise_gate") is False:
                    state["ok"] = False
                    break
        except Exception:
            state["ok"] = False
            save("harness_error.json", {"traceback": traceback.format_exc()})
            traceback.print_exc()
        finally:
            os.kill(os.getpid(), signal.SIGINT)

    save("run_spec.json", {"config": spec, "argv": vars(cfg)})
    sys.meta_path.insert(0, Finder())
    sys.argv = [str(runtime / "main.py"), "--listen", "127.0.0.1", "--port", str(cfg.port),
                *spec["server_args"], "--input-directory", str(out / "input"),
                "--output-directory", str(out / "images"), "--temp-directory", str(out / "temp"),
                "--user-directory", str(out / "user")]
    os.chdir(runtime)
    audit = None
    if spec.get("gpu_audit", False):
        from gpu_guard_v06 import GpuAudit
        audit = GpuAudit()
    try:
        with audit if audit is not None else nullcontext():
            if audit is not None:
                audit.ensure_clear()
            threading.Thread(target=client, daemon=True).start()
            try:
                runpy.run_path(str(runtime / "main.py"), run_name="__main__")
            except KeyboardInterrupt:
                pass
            finally:
                for cleanup in reversed(instrument_cleanups):
                    cleanup()
    finally:
        if audit is not None:
            report = audit.report()
            save("gpu_audit.json", report)
            if not report["valid_for_timing"]:
                state["ok"] = False
    return 0 if state["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
