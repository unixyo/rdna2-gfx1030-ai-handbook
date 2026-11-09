#!/usr/bin/env python3
"""Bounded LayerNorm/RMSNorm/AdaLN experiment; no model or environment writes.

Three shapes per basic operation plus the current kitchen-fused MOSS/Qwen
baselines. Values are synthetic, dimensions/origins are labelled. A passing
operator test is not a model-quality result. All timing uses GpuAudit.
"""
from __future__ import annotations
import argparse
import datetime as dt
import gc
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import traceback

import torch
import torch.nn.functional as F
import triton

from gpu_guard_v06 import GpuAudit
from bench_layout_blas_v06 import errors, tensor_info, measure, peak, call_profile
from rdna2_ops.norm_fusions_v06 import row_norm, adaln, modulate, _modulation_rows

CASES = {
    "ln_anima_residual": dict(op="ln", shape=(2, 4096, 2048), dtype="float32", origin="Anima FP32 residual LayerNorm dimensions"),
    "ln_qwen": dict(op="ln", shape=(1, 2400, 4096), dtype="float32", origin="Qwen target dimensions; bare LN is an operator comparison, inference uses kitchen AdaLN"),
    "ln_half_tail": dict(op="ln", shape=(8, 257, 768), dtype="float16", origin="Representative FP16 non-power-of-two row count; synthetic shape"),
    "rms_anima_cross": dict(op="rms", shape=(2, 512, 16, 128), dtype="float16", origin="Anima cross-attention K normalization dimensions; no replacement of fused RMSNorm+RoPE"),
    "rms_moss_text": dict(op="rms", shape=(1, 37, 3072), dtype="float32", origin="MOSS text RMSNorm dimensions; token count inferred from 1061-1024 capture"),
    "rms_qwen_text": dict(op="rms", shape=(1, 2564, 4096), dtype="float32", origin="Qwen text projection width and prefix length; ordinary affine RMSNorm after zero-centered weight preparation"),
    "adaln_anima": dict(op="adaln", shape=(2, 1, 64, 64, 2048), dtype="float32", scale_dtype="float16", out_dtype="float16", origin="Actual Anima residual/compute dtype contract"),
    "adaln_anima_small": dict(op="adaln", shape=(1, 1, 32, 32, 2048), dtype="float32", scale_dtype="float16", out_dtype="float16", origin="Representative shorter Anima-width sequence"),
    "adaln_anima_long": dict(op="adaln", shape=(1, 1, 64, 128, 2048), dtype="float32", scale_dtype="float16", out_dtype="float16", origin="Representative longer Anima-width sequence"),
    "kitchen_moss": dict(op="kitchen_rms", shape=(1, 1061, 3072), dtype="float32", origin="MOSS existing fused rms_adaln; weight-fold cost excluded equally from both paths"),
    "kitchen_qwen_prefix": dict(op="kitchen_ln", shape=(1, 4964, 4096), dtype="float32", prefix=2564, origin="Qwen first cache fill: target-all then prefix overwrite"),
    "kitchen_qwen_target": dict(op="kitchen_ln", shape=(1, 2400, 4096), dtype="float32", origin="Qwen cached target-only fused AdaLN"),
}
TOL = {"float16": dict(rtol=0.005, atol=0.005),
       "float32": dict(rtol=0.0001, atol=0.00003)}


def source(fn):
    path = inspect.getsourcefile(fn)
    return {"module": fn.__module__, "qualname": fn.__qualname__, "path": path,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None}


def sample_fp64(x, weight, scale, shift, prefix_scale, case):
    d = x.shape[-1]
    n = x.numel() // d
    rows = sorted({0, n // 3, n // 2, n - 1})
    xx = x.view(-1, d)[rows].cpu().double()
    rms = case["op"] in ("rms", "kitchen_rms")
    centered = xx if rms else xx - xx.mean(-1, keepdim=True)
    zz = centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + 1e-6)
    if weight is not None:
        zz *= weight.cpu().double()
    if scale is not None:
        ss, sg = _modulation_rows(scale, x)
        tt, tg = _modulation_rows(shift, x)
        ss = ss[[r // sg for r in rows]].cpu()
        tt = tt[[r // tg for r in rows]].cpu()
        if prefix_scale is not None:
            ps, pg = _modulation_rows(prefix_scale, x)
            ps = ps[[r // pg for r in rows]].cpu()
            choose = torch.tensor([r % x.shape[1] < case["prefix"] for r in rows])
            ss[choose] = ps[choose]
        if case["op"] == "adaln":
            # Eager dtype boundaries, replacing only the normalization statistic
            # with its independently computed double-precision value.
            zz = zz.to(x.dtype)
            zz = zz * (1 + ss) + tt
        else:
            zz = zz * (1 + ss.double()) + tt.double()
    out_dtype = getattr(torch, case.get("out_dtype", case["dtype"]))
    return rows, zz.to(out_dtype)


def build(case, kitchen):
    torch.manual_seed(20261108)
    dtype = getattr(torch, case["dtype"])
    x = torch.randn(case["shape"], device="cuda", dtype=dtype)
    d = x.shape[-1]
    weight = scale = shift = prefix_scale = None
    if case["op"] in ("ln", "rms"):
        rms = case["op"] == "rms"
        weight = (1 + torch.randn(d, device="cuda") * 0.05).to(dtype) if rms else None
        methods = {
            "native": lambda: row_norm(x, weight, eps=1e-6, rms=rms, backend="torch"),
            "triton_centered": lambda: row_norm(x, weight, eps=1e-6, rms=rms),
        }
        baseline_info = {"callable": "torch.nn.functional." + ("rms_norm" if rms else "layer_norm")}
    else:
        s_dtype = getattr(torch, case.get("scale_dtype", case["dtype"]))
        shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (d,)
        # Chunked non-contiguous modulation row strides exercise the real
        # linear(...).chunk(...).view(...) layout without materializing expands.
        pieces = 3 if case["op"] == "adaln" else 2
        packed = (torch.randn((*shape[:-1], d * pieces), device="cuda") * .05).to(s_dtype)
        chunks = packed.chunk(pieces, dim=-1)
        # Anima computes [shift, scale, gate], so its batch row stride is 3D.
        shift, scale = chunks[:2]
        out_dtype = getattr(torch, case.get("out_dtype", case["dtype"]))
        if case["op"] == "adaln":
            methods = {
                "native": lambda: adaln(x, scale, shift, out_dtype=out_dtype, backend="torch"),
                "native_norm_fused_mod": lambda: modulate(F.layer_norm(x, (d,), eps=1e-6), scale, shift, out_dtype=out_dtype),
                "triton_eager_rounding": lambda: adaln(x, scale, shift, out_dtype=out_dtype),
            }
            baseline_info = {"callable": "cosmos.predict2.Block._fn semantics + to(compute_dtype)",
                             "rounding": "norm FP32; 1+scale FP16; multiply/add FP32; output FP16"}
        else:
            rms = case["op"] == "kitchen_rms"
            entry = kitchen.rms_adaln if rms else kitchen.adaln
            opname = "rms_adaln" if rms else "adaln"
            if case.get("prefix"):
                prefix_scale = torch.randn_like(scale) * .05
                shift.zero_()
            def native():
                result = entry(x, scale, shift, 1e-6)
                if prefix_scale is not None:
                    prefix = case["prefix"]
                    result[:, :prefix] = entry(x[:, :prefix], prefix_scale, shift, 1e-6)
                return result
            methods = {
                "native": native,
                "triton_single_pass": lambda: adaln(x, scale, shift, rms=rms, rounding="fp32",
                    prefix_scale=prefix_scale, prefix_len=case.get("prefix", 0)),
            }
            impl = kitchen.registry.get_implementation(opname, kwargs={"x": x, "scale": scale, "shift": shift, "eps": 1e-6})
            baseline_info = {"public_callable": source(entry), "selected_implementation": source(impl),
                             "backend_setting": "Triton disabled like current Comfy server defaults; HIP/eager selection recorded"}
    inputs = {"x": tensor_info(x)}
    for name, t in (("weight", weight), ("scale", scale), ("shift", shift), ("prefix_scale", prefix_scale)):
        if t is not None:
            inputs[name] = tensor_info(t)
    oracle = sample_fp64(x, weight, scale, shift, prefix_scale, case)
    return methods, inputs, baseline_info, oracle


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--plan-only", action="store_true")
    cfg = ap.parse_args()
    if cfg.plan_only:
        print(json.dumps({"cases": {k: CASES[k] for k in cfg.cases}, "gpu_initialized": torch.cuda.is_initialized()}, indent=2))
        return
    if cfg.rounds < 3 or cfg.repeat < 1:
        ap.error("At least three alternating rounds and positive repeats required")
    if cfg.output.exists():
        ap.error("Evidence must use a new output path")
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parent
    result = {"status": "started", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "evidence": "operator A only, synthetic values; whole-model RGB is a separate gate",
              "torch": torch.__version__, "hip": torch.version.hip, "triton": triton.__version__,
              "gates_before_execution": {"operator_tolerance": TOL, "bitwise_required_for_native_norm_fused_mod": True,
                                         "model_rgb_max": 2 / 255, "model_rgb_rmse": .25 / 255},
              "timing": {"warmup": 3, "rounds": cfg.rounds, "repeat": cfg.repeat,
                         "includes": "allocation and every required input/output conversion; no GPU graphs",
                         "excludes": "JIT, correctness, independent oracle and profiling"},
              "sources": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in
                          ("bench_norm_fusions_v06.py", "rdna2_ops/norm_fusions_v06.py", "gpu_guard_v06.py",
                           "bench_layout_blas_v06.py", "research_comfy_v05.py")},
              "cases": []}
    def save():
        cfg.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    audit = GpuAudit()
    try:
        with audit, torch.inference_mode():
            audit.ensure_clear()
            prop = torch.cuda.get_device_properties(0)
            if prop.gcnArchName.split(":")[0] != "gfx1030":
                raise RuntimeError("This run must target gfx1030")
            result["device"] = {"name": prop.name, "arch": prop.gcnArchName}
            import comfy_kitchen as kitchen
            kitchen.registry.disable("cuda")
            kitchen.registry.disable("triton")
            result["kitchen_backends"] = str(kitchen.list_backends())
            for label in cfg.cases:
                audit.ensure_clear()
                case = dict(CASES[label], id=label, variants=[])
                result["cases"].append(case)
                methods, inputs, baseline_info, oracle = build(case, kitchen)
                case.update(inputs=inputs, baseline=baseline_info)
                ref = methods["native"]().cpu()
                active = {}
                for name, fn in methods.items():
                    row = {"name": name}
                    case["variants"].append(row)
                    try:
                        value = fn().cpu()
                        tol = TOL[str(value.dtype).removeprefix("torch.")]
                        err = errors(value, ref, tol)
                        err["byte_equal"] = torch.equal(value.contiguous().view(torch.uint8), ref.contiguous().view(torch.uint8))
                        row["error_vs_native"] = err
                        rows, fp64 = oracle
                        row["error_vs_fp64_sample"] = errors(value.reshape(-1, value.shape[-1])[rows], fp64, tol)
                        row["operator_pass"] = err["allclose"] and (name != "native_norm_fused_mod" or err["byte_equal"])
                        for _ in range(3):
                            fn()
                        torch.cuda.synchronize()
                        row["additional_peak_allocated_bytes"] = peak(fn)
                        row["timings"] = []
                        active[name] = row
                        del value
                    except Exception as exc:
                        row.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
                for trial in range(cfg.rounds):
                    order = list(active)
                    order = order[trial % len(order):] + order[:trial % len(order)] if order else []
                    if trial % 2:
                        order.reverse()
                    for name in order:
                        audit.ensure_clear()
                        active[name]["timings"].append(measure(methods[name], cfg.repeat))
                for name, row in active.items():
                    row["wall_ms"] = statistics.median(v["wall_ms"] for v in row["timings"])
                    row["event_ms"] = statistics.median(v["event_ms"] for v in row["timings"])
                    row["profile"] = call_profile(methods[name])
                    row["status"] = "measured"
                if "native" in active:
                    for row in active.values():
                        row["speedup_wall"] = active["native"]["wall_ms"] / row["wall_ms"]
                save()
                print(json.dumps({"case": label, "variants": [{k: r.get(k) for k in
                      ("name", "status", "wall_ms", "speedup_wall", "operator_pass")} for r in case["variants"]]}), flush=True)
                del methods, active, ref, oracle
                gc.collect()
                torch.cuda.empty_cache()
            audit.ensure_clear()
            result["status"] = "measured"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        result["gpu_audit"] = audit.report()
        if not result["gpu_audit"]["valid_for_timing"]:
            result["status"] = "discarded_overlap_or_unobservable"
        result["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
