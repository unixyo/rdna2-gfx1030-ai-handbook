"""Opt-in measurements in the real launcher server; absent from normal runs."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import torch


def install():
    import nodes
    from . import attention_dispatch, backend
    out = Path(os.environ["MOSS_RDNA2_MEASURE_DIR"]).resolve()
    reference = Path(os.environ.get("MOSS_RDNA2_REFERENCE_DIR", out / "reference")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    creating_reference = "MOSS_RDNA2_REFERENCE_DIR" not in os.environ
    if creating_reference and attention_dispatch.MODE != "native":
        raise ValueError("Reference creation requires native attention; provide MOSS_RDNA2_REFERENCE_DIR")
    if creating_reference:
        reference.mkdir(parents=True, exist_ok=True)
    state = {"index": 0}
    original_sample, original_decode = nodes.KSampler.sample, nodes.VAEDecode.decode
    signature = inspect.signature(original_sample)

    def check(kind, value):
        value = value.detach().cpu().contiguous()
        raw = value.view(torch.uint8).numpy().tobytes()
        name = f"{kind}_seed{state['seed']}_{state['latent_shape'][-2]}x{state['latent_shape'][-1]}.pt"
        dest = reference / name
        if creating_reference and not dest.exists():
            if attention_dispatch.MODE != "native":
                raise ValueError("A native reference must exist for this seed/shape")
            torch.save(value, dest)
        ref = torch.load(dest, map_location="cpu", weights_only=True)
        delta = value.float() - ref.float()
        exact = value.dtype == ref.dtype and value.shape == ref.shape and torch.equal(
            value.view(torch.uint8), ref.contiguous().view(torch.uint8))
        result = {"finite": bool(torch.isfinite(value).all()), "byte_equal": bool(exact),
                  "dtype": str(value.dtype), "shape": list(value.shape),
                  "sha256": hashlib.sha256(raw).hexdigest(),
                  "max_abs": delta.abs().max().item(), "rmse": delta.square().mean().sqrt().item()}
        if state["index"] == 1 or not exact:
            torch.save(value, out / f"{kind}_run{state['index']:03d}.pt")
        return result

    def sample(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs).arguments
        control_path = out / "control.json"
        selected_layout = os.environ.get("MOSS_RDNA2_INT8_LAYOUT", "view")
        if control_path.exists():
            control = json.loads(control_path.read_text())
            selected_layout = control["layout"]
            selected_attention = control["attention"]
            if selected_layout not in {"legacy", "view"} or selected_attention not in {
                    "native", "auto", "fp16_math", "fp16_math_reduced", "bf16_math", "fp16_flash", "fp16_efficient"}:
                raise ValueError("Invalid measurement-only mode selection")
            if selected_layout == "legacy":
                backend.disable_backend()
            else:
                from comfy_kitchen.backends.eager import quantization
                backend.install_backend(quantization._int8_matmul_accumulate)
            attention_dispatch.MODE = selected_attention
        state.clear()
        state.update(index=sample.count + 1, seed=int(bound["seed"]), stage_wall_s={}, layout=selected_layout)
        sample.count += 1
        attention_dispatch.COUNTS.clear()
        attention_dispatch.LAYOUTS.clear()
        backend.reset_stats()
        state["conditioning_sha256"] = {}
        for name in ("positive", "negative"):
            conditioning = bound[name][0][0].detach().cpu().contiguous()
            state["conditioning_sha256"][name] = hashlib.sha256(
                conditioning.view(torch.uint8).numpy().tobytes()).hexdigest()
            if state["index"] == 1:
                torch.save(conditioning, out / (name + ".pt"))
        model = bound["model"].model
        state["manual_cast_dtype"] = str(getattr(model, "manual_cast_dtype", None))
        state["model_dtype"] = str(model.get_dtype()) if hasattr(model, "get_dtype") else None
        attention_dispatch.IN_DENOISE = True
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            result = original_sample(self, *args, **kwargs)
            torch.cuda.synchronize()
        finally:
            attention_dispatch.IN_DENOISE = False
        state["stage_wall_s"]["denoise"] = time.perf_counter() - start
        state["latent_shape"] = list(result[0]["samples"].shape)
        state["latent"] = check("latent", result[0]["samples"])
        state["attention_counts"] = dict(attention_dispatch.COUNTS)
        state["attention_layouts"] = list(attention_dispatch.LAYOUTS.values())
        state["int8_stats"] = backend.get_stats()
        return result
    sample.count = 0

    def decode(self, vae, samples):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_decode(self, vae, samples)
        torch.cuda.synchronize()
        state["stage_wall_s"]["vae"] = time.perf_counter() - start
        state["image"] = check("image", result[0])
        exact_mode = attention_dispatch.MODE in {"native", "auto"}
        finite = state["image"]["finite"] and state["latent"]["finite"]
        state["accuracy_pass"] = finite and (
            state["image"]["byte_equal"] and state["latent"]["byte_equal"] if exact_mode else
            state["image"]["max_abs"] <= 2 / 255 and state["image"]["rmse"] <= 0.25 / 255)
        state["acceptance"] = {"exact": exact_mode, "rgb_max": 2 / 255, "rgb_rmse": 0.25 / 255}
        state["attention_mode"] = attention_dispatch.MODE
        state["math_reduced_precision_default"] = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        data = json.dumps(state, indent=2) + "\n"
        (out / f"run{state['index']:03d}.json").write_text(data)
        (out / "last.json").write_text(data)
        print("MOSS_RDNA2_MEASURE " + json.dumps(state), flush=True)
        return result

    nodes.KSampler.sample = sample
    nodes.KSampler.IS_CHANGED = classmethod(lambda cls, **kwargs: float("nan"))
    nodes.VAEDecode.decode = decode

