"""Bounded, process-local precision diagnosis for an owned Comfy workload.

This is an instrument, not a speed benchmark or a production dispatcher. It
does not save tensors, modify model weights, or edit installed source files.
Run native first, then candidates in the same process with the same seed and
conditioning. All temporary exported-callable changes are restored on exit.
Concurrent/nested use, including unrelated GPU work during flag changes, is
unsupported. The caller owns server startup and its normal sampler/decoder.

    diag = PrecisionDiagnosis(out / "precision_diagnosis.json")
    with diag.run("native", mode="native", sdpa_owner=comfy.ops,
                  denoiser=(model.model, "apply_model"),
                  quantizer_owner=quantization, metadata={"seed": seed}):
        result = original_sample(...)
        diag.checkpoint("latent", result[0]["samples"])
    # In the existing VAEDecode hook, after the original decoder returns:
    diag.checkpoint("image", pixels)
    diag.save()

Only the first attention of each sequence length per denoiser invocation gets
detailed paired local probes (eight total by default). Other attention outputs
retain only metadata and full-content SHA256. Full quantizer references are
limited to the first eight calls and a shared 256 MiB CPU-memory budget. The
first changed INT8 output receives a paired input/code/scale/dequantization
comparison. The explicit quantizer_rows contract (37, 1061 by default for the
512-square MOSS workload) separates activation matrices from cold-load weight
requantization through the same helper; excluded shapes are counted separately.
Use the actual activation lengths for another workflow or image geometry.
Code flips alone do not prove whole-model quantization causality.
The manual QK/softmax probes are explicitly shadow decompositions, not captured
internal kernel values, and their final outputs are checked against real SDPA.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import json
import math
from pathlib import Path
import threading

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


_EXCLUSIVE = threading.Lock()
_MATH_MODES = {"fp16_math", "fp16_math_reduced", "bf16_math"}
_ROUNDTRIP_MODES = {"fp16_q_roundtrip", "fp16_k_roundtrip", "fp16_v_roundtrip",
                    "fp16_qkv_roundtrip", "fp16_output_roundtrip",
                    "fp16_qkv_output_roundtrip"}


def _cpu(t):
    return t.detach().to(device="cpu", copy=True).contiguous()


def _digest(t):
    t = t.detach().contiguous()
    if t.device.type != "cpu":
        t = t.cpu()
    return hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _layout(t):
    return {"shape": list(t.shape), "stride": list(t.stride()),
            "dtype": str(t.dtype), "device": str(t.device),
            "storage_offset": int(t.storage_offset())}


def _stats(t):
    """Finite-only extrema are labelled; never hide nonfinite counts."""
    x = t.detach().float()
    finite = torch.isfinite(x)
    n_finite = int(finite.sum().item())
    result = {"shape": list(t.shape), "dtype": str(t.dtype),
              "numel": t.numel(), "finite": n_finite == t.numel(),
              "nan_count": int(torch.isnan(x).sum().item()),
              "positive_inf_count": int(torch.isposinf(x).sum().item()),
              "negative_inf_count": int(torch.isneginf(x).sum().item())}
    if n_finite:
        values = x if n_finite == x.numel() else x[finite]
        high = values.abs().max()
        rms = (values / high).square().mean().sqrt() * high if high.item() else high
        result.update(finite_min=values.min().item(), finite_max=values.max().item(),
                      finite_abs_max=high.item(), finite_rms=rms.item())
    else:
        result.update(finite_min=None, finite_max=None,
                      finite_abs_max=None, finite_rms=None)
    return result


def _error(actual, reference):
    a, b = _cpu(actual), _cpu(reference)
    same_shape = a.shape == b.shape
    exact = same_shape and a.dtype == b.dtype and torch.equal(
        a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))
    finite = bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
    result = {"byte_equal": bool(exact), "finite": finite,
              "same_shape": same_shape, "max_abs": None, "rmse": None,
              "relative_rms": None, "changed_fraction": None}
    if same_shape and finite:
        a, b = a.double(), b.double()
        delta = a - b
        rms = delta.square().mean().sqrt().item()
        ref_rms = b.square().mean().sqrt().item()
        result.update(max_abs=delta.abs().max().item(), rmse=rms,
                      relative_rms=rms / ref_rms if ref_rms else (0.0 if not rms else None),
                      changed_fraction=float(torch.count_nonzero(delta)) / delta.numel())
    return result


def _contract(args, kwargs):
    """Canonicalize the public SDPA contract without silently dropping options."""
    allowed = {"attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa"}
    if len(args) > 3 or set(kwargs) - allowed:
        return None
    names, defaults = ("attn_mask", "dropout_p", "is_causal"), (None, 0.0, False)
    result = {}
    for index, (name, default) in enumerate(zip(names, defaults)):
        if index < len(args) and name in kwargs:
            return None
        result[name] = args[index] if index < len(args) else kwargs.get(name, default)
    result.update(scale=kwargs.get("scale"), enable_gqa=kwargs.get("enable_gqa", False))
    if (result["attn_mask"] is not None or result["dropout_p"] != 0.0
            or result["is_causal"] or result["enable_gqa"]):
        return None
    scale = result["scale"]
    if scale is not None and (not isinstance(scale, (float, int)) or not math.isfinite(scale)):
        return None
    return result


def _math_call(native_sdpa, q, k, v, contract, reduced=False):
    previous = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    try:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(reduced)
        with sdpa_kernel(SDPBackend.MATH):
            return native_sdpa(q, k, v, **contract)
    finally:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous)


def _candidate(native_sdpa, q, k, v, contract, mode):
    if mode in _MATH_MODES:
        dtype = torch.bfloat16 if mode == "bf16_math" else torch.float16
        return _math_call(native_sdpa, q.to(dtype), k.to(dtype), v.to(dtype), contract,
                          reduced=mode == "fp16_math_reduced").to(q.dtype)
    values = [q, k, v]
    if mode in _ROUNDTRIP_MODES:
        what = mode.removeprefix("fp16_").removesuffix("_roundtrip")
        for i, name in enumerate(("q", "k", "v")):
            if what in (name, "qkv", "qkv_output"):
                values[i] = values[i].to(torch.float16).to(q.dtype)
        result = _math_call(native_sdpa, *values, contract)
        return result.to(torch.float16).to(q.dtype) if "output" in what else result
    raise ValueError("Unsupported diagnostic attention mode: " + mode)


def _shadow(q, k, v, contract, dtype, reduced, actual_sdpa):
    """Expose a shadow math pipeline and verify its endpoint against real SDPA."""
    values = [t.to(dtype) for t in (q, k, v)]
    if dtype in (torch.float16, torch.bfloat16) and not reduced:
        values = [t.float() for t in values]
    scale = contract["scale"]
    scale = 1 / math.sqrt(q.shape[-1]) if scale is None else scale
    root = math.sqrt(abs(scale))
    qs = values[0] * (-root if scale < 0 else root)
    ks = values[1] * root
    scores = torch.matmul(qs, ks.transpose(-1, -2))
    probabilities = torch.softmax(scores, dim=-1)
    all_negative_inf = torch.isneginf(scores).all(dim=-1, keepdim=True)
    probabilities = torch.where(all_negative_inf, 0.0, probabilities)
    result = torch.matmul(probabilities, values[2]).to(dtype).to(q.dtype)
    return {"kind": "shadow decomposition; endpoint checked, not internal capture",
            "scaled_q": _stats(qs), "scaled_k": _stats(ks),
            "logits": _stats(scores), "probabilities": _stats(probabilities),
            "all_negative_inf_rows": int(all_negative_inf.sum().item()),
            "output_vs_real_sdpa": _error(result, actual_sdpa)}


@torch.no_grad()
def inspect_attention(q, k, v, *, native_sdpa=F.scaled_dot_product_attention,
                      actual_native=None, scale=None):
    """Paired local probes on identical real FP32 QKV; returns JSON-only data.

    Input-only round trips retain FP32 arithmetic and a FP32 output. The
    additional QKV+output round trip is compared directly with FP16 math whose
    reductions are promoted to FP32, isolating conversion from backend math.
    """
    if any(t.dtype != torch.float32 or t.ndim != 4 for t in (q, k, v)):
        raise ValueError("inspect_attention expects four-dimensional FP32 QKV")
    contract = dict(attn_mask=None, dropout_p=0.0, is_causal=False,
                    scale=scale, enable_gqa=False)
    native = _math_call(native_sdpa, q, k, v, contract)
    report = {"inputs": {n: {**_layout(t), **_stats(t)}
                          for n, t in zip(("q", "k", "v"), (q, k, v))},
              "contract": contract,
              "original_vs_forced_math": _error(actual_native if actual_native is not None else native, native),
              "cast_inputs": {}, "outputs": {}, "shadow_stages": {}}
    # A float32 clone with preserve_format reproduces packing changes from
    # dtype conversion without rounding the values. A separate contiguous
    # control checks the logical BHSD path independently of packed QKV storage.
    for name, values in (
            ("fp32_clone_preserve_format", [t.clone(memory_format=torch.preserve_format) for t in (q, k, v)]),
            ("fp32_contiguous", [t.contiguous() for t in (q, k, v)])):
        output = _math_call(native_sdpa, *values, contract)
        report["outputs"][name] = {"error": _error(output, native),
                                    "layouts": [_layout(t) for t in values]}
    report["shadow_stages"]["fp32"] = _shadow(q, k, v, contract, torch.float32, False, native)
    outputs = {}
    for dtype_name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        report["cast_inputs"][dtype_name] = {}
        for name, original in zip(("q", "k", "v"), (q, k, v)):
            cast = original.to(dtype)
            report["cast_inputs"][dtype_name][name] = {
                **_layout(cast), **_stats(cast),
                "finite_to_nonfinite": int((torch.isfinite(original) & ~torch.isfinite(cast)).sum().item()),
                "nonzero_to_zero": int(((original != 0) & (cast == 0)).sum().item()),
                "roundtrip_error": _error(cast.float(), original)}
    for mode in ("fp16_math", "fp16_math_reduced", "bf16_math"):
        output = _candidate(native_sdpa, q, k, v, contract, mode)
        outputs[mode] = output
        report["outputs"][mode] = {"error": _error(output, native), "stats": _stats(output)}
        dtype = torch.bfloat16 if mode == "bf16_math" else torch.float16
        report["shadow_stages"][mode] = _shadow(
            q, k, v, contract, dtype, mode == "fp16_math_reduced", output)
    for mode in sorted(_ROUNDTRIP_MODES):
        output = _candidate(native_sdpa, q, k, v, contract, mode)
        report["outputs"][mode] = {"error": _error(output, native)}
        if mode == "fp16_qkv_output_roundtrip":
            report["fp16_math_vs_fp32_roundtrip_model"] = _error(outputs["fp16_math"], output)
    bf_values = [t.to(torch.bfloat16).float() for t in (q, k, v)]
    bf_roundtrip = _math_call(native_sdpa, *bf_values, contract).to(torch.bfloat16).float()
    report["bf16_math_vs_fp32_roundtrip_model"] = _error(outputs["bf16_math"], bf_roundtrip)
    return report


class PrecisionDiagnosis:
    """One native trajectory and bounded candidate traces in a shared process."""

    def __init__(self, path, *, max_reference_bytes=256 * 1024**2,
                 detailed_calls=8, quantizer_references=8,
                 quantizer_rows=(37, 1061),
                 num_heads=24, head_dim=128, native_sdpa=None):
        self.path = Path(path)
        self.max_reference_bytes = int(max_reference_bytes)
        self.detailed_calls = int(detailed_calls)
        self.quantizer_references = int(quantizer_references)
        self.quantizer_rows = tuple(sorted(set(quantizer_rows)))
        if not self.quantizer_rows or any(type(n) is not int or n <= 0 for n in self.quantizer_rows):
            raise ValueError("quantizer_rows must specify positive integer activation row counts")
        self.num_heads, self.head_dim = num_heads, head_dim
        self.native_sdpa = native_sdpa or F.scaled_dot_product_attention
        self.report = {"schema": 1, "purpose": "precision diagnosis; timings are not performance measurements",
                       "torch": torch.__version__, "hip": torch.version.hip,
                       "max_reference_bytes": self.max_reference_bytes,
                       "quantizer_rows": list(self.quantizer_rows),
                       "quantizer_filter": "Only 2D inputs with an explicitly listed row count enter paired activation traces; other calls are counted by shape.",
                       "notes": ["Same-process, same-input comparison required.",
                                 "Denoiser calls are not assumed to equal sampler steps.",
                                 "Shadow stage values are checked decompositions, not internal kernel captures.",
                                 "INT8 code flips and local amplification are evidence at that boundary, not proof of global causality."],
                       "runs": []}
        self._reference = None
        self._active = None
        self._cached_bytes = 0
        self._quant_refs, self._denoiser_refs, self._checkpoints = {}, {}, {}

    def save(self):
        self.report["reference_bytes_retained"] = self._cached_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n")

    def _remember(self, *tensors):
        size = sum(t.numel() * t.element_size() for t in tensors)
        if self._cached_bytes + size > self.max_reference_bytes:
            return None
        result = tuple(_cpu(t) for t in tensors)
        self._cached_bytes += size
        return result

    def checkpoint(self, name, value):
        """Attach latent/image data to the active or most recently finished run."""
        if not self.report["runs"]:
            raise RuntimeError("Start a diagnostic run before adding checkpoints")
        row = self._active or self.report["runs"][-1]
        cpu = _cpu(value)
        if row is self._reference and name not in self._checkpoints:
            cached = self._remember(cpu)
            if cached is None:
                raise RuntimeError("Reference memory cap exhausted before checkpoint: " + name)
            self._checkpoints[name] = cached[0]
        if name not in self._checkpoints:
            raise RuntimeError("Native checkpoint missing: " + name)
        row["checkpoints"][name] = {"stats": _stats(cpu), "sha256": _digest(cpu),
                                     "comparison_to_native": _error(cpu, self._checkpoints[name])}
        return row["checkpoints"][name]

    def _thread_check(self):
        if threading.get_ident() != self._thread:
            raise RuntimeError("Diagnostic hooks may only be used by their owning worker thread")

    def _inside_denoiser(self):
        return not self._has_denoiser or self._current_denoiser is not None

    def _sdpa(self, original):
        def hook(q, k, v, *args, **kwargs):
            self._thread_check()
            contract = _contract(args, kwargs)
            eligible = (self._inside_denoiser() and contract is not None
                        and all(t.ndim == 4 and t.dtype == torch.float32 for t in (q, k, v))
                        and q.shape == k.shape == v.shape and q.shape[0] == 1
                        and q.shape[1] == self.num_heads and q.shape[-1] == self.head_dim
                        and 0 < q.shape[-2] <= 4096)
            if not eligible:
                return original(q, k, v, *args, **kwargs)
            if torch.is_autocast_enabled(q.device.type) or (
                    torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v))):
                raise RuntimeError("Precision diagnosis requires inference without autocast")
            row = self._active
            index = len(row["attention"]["calls"])
            length = q.shape[-2]
            selected = row["scope"] == "all" or (row["scope"] == "short" and length == 37) or (
                row["scope"] == "long" and length >= 128)
            mode = row["mode"] if selected else "native"
            output = original(q, k, v, *args, **kwargs) if mode == "native" else _candidate(
                self.native_sdpa, q, k, v, contract, mode)
            cpu_output = _cpu(output)
            record = {"index": index, "denoiser_call": self._current_denoiser,
                      "mode_used": mode, "layouts": {n: _layout(t) for n, t in zip(("q", "k", "v"), (q, k, v))},
                      "contract": contract, "output_sha256": _digest(cpu_output)}
            if row is not self._reference:
                reference_calls = self._reference["attention"]["calls"]
                if index >= len(reference_calls):
                    raise RuntimeError("Candidate has more eligible SDPA calls than the native reference")
                ref = reference_calls[index]
                if (ref["denoiser_call"] != record["denoiser_call"] or ref["contract"] != contract
                        or any(ref["layouts"][n]["shape"] != record["layouts"][n]["shape"] for n in ("q", "k", "v"))):
                    raise RuntimeError("Attention reference alignment changed; refusing an unrelated comparison")
                record["output_hash_equal"] = ref["output_sha256"] == record["output_sha256"]
                if not record["output_hash_equal"] and row["attention"]["first_output_difference"] is None:
                    row["attention"]["first_output_difference"] = index
            detail_key = (self._current_denoiser, length)
            if detail_key not in self._detailed and len(self._detailed) < self.detailed_calls:
                self._detailed.add(detail_key)
                record["input_sha256"] = {n: _digest(t) for n, t in zip(("q", "k", "v"), (q, k, v))}
                record["output_stats"] = _stats(cpu_output)
                if row is self._reference:
                    record["paired_local"] = inspect_attention(
                        q, k, v, native_sdpa=self.native_sdpa,
                        actual_native=output, scale=contract["scale"])
                else:
                    record["input_stats"] = {n: _stats(t) for n, t in zip(("q", "k", "v"), (q, k, v))}
                    record["input_hash_equal"] = {n: record["input_sha256"][n] == ref.get("input_sha256", {}).get(n)
                                                    for n in ("q", "k", "v")}
                    local_native = _math_call(self.native_sdpa, q, k, v, contract)
                    record["same_input_local_error"] = _error(output, local_native)
            row["attention"]["calls"].append(record)
            return output
        return hook

    def _denoiser(self, original):
        signature = inspect.signature(original)

        def hook(*args, **kwargs):
            self._thread_check()
            if self._current_denoiser is not None:
                raise RuntimeError("Nested denoiser diagnosis is unsupported")
            bound = signature.bind(*args, **kwargs).arguments
            # Recent Comfy apply_model wrappers expose (*args, **kwargs);
            # their real _apply_model still receives x, timestep positionally.
            keyword_values = {**bound.get("kwargs", {}), **kwargs, **bound}
            x = keyword_values.get("x")
            timestep = keyword_values.get("timestep", keyword_values.get("t"))
            tensor_args = [a for a in args if isinstance(a, torch.Tensor)]
            if x is None and tensor_args:
                x = tensor_args[0]
            if timestep is None and len(tensor_args) >= 2:
                timestep = tensor_args[1]
            if not isinstance(x, torch.Tensor) or not isinstance(timestep, torch.Tensor):
                raise RuntimeError("Denoiser hook expects Tensor x and timestep arguments")
            row = self._active
            index = len(row["denoiser"])
            cpu_x, cpu_time = _cpu(x), _cpu(timestep)
            record = {"index": index, "timestep": cpu_time.flatten().tolist(),
                      "input": _stats(cpu_x), "input_sha256": _digest(cpu_x)}
            if row is not self._reference:
                if index not in self._denoiser_refs:
                    raise RuntimeError("Denoiser reference call missing")
                ref_x, ref_time, _ = self._denoiser_refs[index]
                if not _error(cpu_time, ref_time)["byte_equal"]:
                    raise RuntimeError("Denoiser timesteps changed; refusing unrelated trajectories")
                record["input_comparison_to_native"] = _error(cpu_x, ref_x)
            self._current_denoiser = index
            try:
                output = original(*args, **kwargs)
            finally:
                self._current_denoiser = None
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("Denoiser hook expects a Tensor output")
            cpu_out = _cpu(output)
            record.update(output=_stats(cpu_out), output_sha256=_digest(cpu_out))
            if row is self._reference:
                # Reserve a small margin for latent/RGB checkpoints: quantizer
                # snapshots stop before consuming this portion of the budget.
                reference = self._remember(cpu_x, cpu_time, cpu_out)
                if reference is None:
                    raise RuntimeError("Reference memory cap exhausted before denoiser output")
                self._denoiser_refs[index] = reference
            else:
                record["output_comparison_to_native"] = _error(cpu_out, self._denoiser_refs[index][2])
                if (not record["output_comparison_to_native"]["byte_equal"]
                        and row["first_denoiser_output_difference"] is None):
                    row["first_denoiser_output_difference"] = index
            row["denoiser"].append(record)
            return output
        return hook

    def _quantizer(self, original):
        def hook(x, *args, **kwargs):
            self._thread_check()
            result = original(x, *args, **kwargs)
            if not self._inside_denoiser():
                return result
            row = self._active
            table = row["quantizer"]
            table["raw_call_count"] += 1
            if x.ndim != 2 or x.shape[0] not in self.quantizer_rows:
                table["excluded_call_count"] += 1
                shape_key = json.dumps(list(x.shape), separators=(",", ":"))
                table["excluded_shape_counts"][shape_key] = table["excluded_shape_counts"].get(shape_key, 0) + 1
                return result
            index = table["call_count"]
            table["call_count"] += 1
            if row is not self._reference:
                refs = self._reference["quantizer"]["reference_hashes"]
                if index >= len(refs) or refs[index]["input_shape"] != list(x.shape) or (
                        refs[index]["denoiser_call"] != self._current_denoiser):
                    raise RuntimeError("INT8 activation quantizer reference alignment changed")
                # Keep shape/call alignment checks after the first difference;
                # only expensive tensor hashing/comparison stops at that point.
                if table["first_output_difference"] is not None:
                    return result
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("Expected quantize_int8_rowwise to return (codes, scales)")
            codes, scales = result
            c, s = _cpu(codes), _cpu(scales)
            record = {"index": index, "denoiser_call": self._current_denoiser,
                      "input_shape": list(x.shape), "codes_sha256": _digest(c), "scales_sha256": _digest(s)}
            if row is self._reference:
                table["reference_hashes"].append(record)
                size = x.numel() * x.element_size() + c.numel() * c.element_size() + s.numel() * s.element_size()
                if (index < self.quantizer_references
                        and self._cached_bytes + size <= max(0, self.max_reference_bytes - 32 * 1024**2)):
                    self._quant_refs[index] = self._remember(x, c, s)
            else:
                ref = refs[index]
                if ref["codes_sha256"] != record["codes_sha256"] or ref["scales_sha256"] != record["scales_sha256"]:
                    table["first_output_difference"] = record
                    if index in self._quant_refs:
                        rx, rc, rs = self._quant_refs[index]
                        record["input_stats"] = {"actual": _stats(x), "native": _stats(rx)}
                        record["input_comparison_to_native"] = _error(x, rx)
                        record["codes_comparison_to_native"] = _error(c, rc)
                        record["scales_comparison_to_native"] = _error(s, rs)
                        changed = c != rc
                        record["changed_code_count"] = int(changed.sum().item())
                        record["changed_row_fraction"] = changed.any(dim=-1).float().mean().item()
                        actual_dequant = c.float() * s.float().reshape(-1, 1)
                        reference_dequant = rc.float() * rs.float().reshape(-1, 1)
                        record["dequantized_comparison_to_native"] = _error(actual_dequant, reference_dequant)
                        before = record["input_comparison_to_native"]["rmse"]
                        after = record["dequantized_comparison_to_native"]["rmse"]
                        record["local_rmse_amplification"] = after / before if before and after is not None else None
                    else:
                        record["paired_input_unavailable"] = "outside bounded reference capture; rerun with a focused capture if needed"
            return result
        return hook

    @contextmanager
    def run(self, label, *, mode="native", scope="all", sdpa_owner,
            denoiser=None, quantizer_owner=None, metadata=None):
        """Temporarily instrument one sampler call; preserve its original API.

        scope='short' changes only S=37; scope='long' changes only S>=128.
        Unmasked inference FP32 [1, num_heads, S, head_dim] is targeted. Other
        inputs retain the original callable. Metadata must be identical to the
        first run, so include seed and conditioning hashes, not a trial label.
        """
        if mode not in {"native"} | _MATH_MODES | _ROUNDTRIP_MODES or scope not in {"all", "short", "long"}:
            raise ValueError("Unknown precision mode or scope")
        if self._reference is None and mode != "native":
            raise ValueError("The first diagnostic run must use native attention")
        if self._reference is not None and metadata != self._reference["metadata"]:
            raise ValueError("Seed/conditioning metadata changed; use a new diagnosis for unrelated inputs")
        if not _EXCLUSIVE.acquire(blocking=False):
            raise RuntimeError("Nested/concurrent precision diagnosis is unsupported")
        if any(r["label"] == label for r in self.report["runs"]):
            _EXCLUSIVE.release()
            raise ValueError("Diagnostic run labels must be unique")
        row = {"label": label, "mode": mode, "scope": scope, "metadata": metadata,
               "status": "running", "attention": {"calls": [], "first_output_difference": None},
               "denoiser": [], "first_denoiser_output_difference": None,
               "quantizer": {"call_count": 0, "raw_call_count": 0,
                             "excluded_call_count": 0, "excluded_shape_counts": {},
                             "first_output_difference": None, "reference_hashes": []},
               "checkpoints": {},
               "math_reduced_flag_before": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()}
        self.report["runs"].append(row)
        if self._reference is None:
            self._reference = row
        self._active, self._thread = row, threading.get_ident()
        self._current_denoiser, self._detailed = None, set()
        self._has_denoiser = denoiser is not None
        patches = []

        def patch(owner, name, factory):
            original = getattr(owner, name)
            # Instance methods may have been inherited. Deleting the temporary
            # instance attribute restores that descriptor binding exactly.
            own = name in vars(owner)
            replacement = factory(original)
            setattr(owner, name, replacement)
            patches.append((owner, name, original, replacement, own))

        try:
            patch(sdpa_owner, "scaled_dot_product_attention", self._sdpa)
            if denoiser is not None:
                patch(*denoiser, self._denoiser)
            if quantizer_owner is not None:
                patch(quantizer_owner, "quantize_int8_rowwise", self._quantizer)
            yield self
            if row is not self._reference:
                for key in ("denoiser",):
                    if len(row[key]) != len(self._reference[key]):
                        raise RuntimeError("Candidate/reference denoiser call counts differ")
                if len(row["attention"]["calls"]) != len(self._reference["attention"]["calls"]):
                    raise RuntimeError("Candidate/reference SDPA call counts differ")
                if row["quantizer"]["call_count"] != self._reference["quantizer"]["call_count"]:
                    raise RuntimeError("Candidate/reference quantizer call counts differ")
            row["status"] = "complete"
        except BaseException as exc:
            row.update(status="error", error=repr(exc))
            raise
        finally:
            conflicts = []
            for owner, name, original, replacement, own in reversed(patches):
                if getattr(owner, name) is not replacement:
                    conflicts.append(name)
                elif own:
                    setattr(owner, name, original)
                else:
                    delattr(owner, name)
            row["restore_conflicts"] = conflicts
            row["math_reduced_flag_after"] = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
            self._active = None
            _EXCLUSIVE.release()
            self.save()
            if conflicts:
                raise RuntimeError("Another patch replaced diagnostic exports: " + ", ".join(conflicts))
