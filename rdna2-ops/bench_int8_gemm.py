#!/usr/bin/env python3
"""Measure explicit gfx1030 INT8 GEMM candidates against both installed backends.

No model files or operating configuration are changed. Baseline kernels are loaded
from their existing source paths. GPU work must be run alone. Default cases are
the two models' common shapes; pass --cases for the wider shape sweep.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys
import time
import traceback

import torch
import triton

from rdna2_ops.int8_gemm import int8_matmul, int8_gemm_dequant


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moss", type=Path, default=Path("~/DevSpace/moss-image-rdna2-lab01"))
    ap.add_argument("--qwen", type=Path, default=Path("~/DevSpace/qwen-image-21-rdna2-lab01"))
    ap.add_argument("--cases", default="moss:1061,3072,3072;qwen:2400,4096,4096")
    ap.add_argument("--configs", default="16,64,64,4,1,4;32,32,64,4,1,4;32,64,32,4,1,4;32,64,64,4,1,4;32,64,128,4,1,4;64,32,64,4,1,4;64,64,32,4,1,4;64,64,64,4,1,4")
    ap.add_argument("--layouts", default="weight_nk")
    ap.add_argument("--bias-mode", choices=("none", "float32"), default="none")
    ap.add_argument("--wrapped", action="store_true", help="wrap M/N load indices; mask stores")
    ap.add_argument("--partial-dot", action="store_true", help="use zero-initialized dot then INT32 add")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--no-pipeline", action="store_true")
    ap.add_argument("--no-small-checks", action="store_true")
    cfg = ap.parse_args()
    if cfg.partial_dot and not cfg.wrapped:
        raise ValueError("--partial-dot requires --wrapped")
    if cfg.output.exists():
        raise FileExistsError(cfg.output)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(20261106)
    moss_file = cfg.moss / "scripts/rdna2_int8_matmul.py"
    qwen_file = cfg.qwen / "custom_nodes/RDNA2-Qwen21-INT8/int8_fused_kernel.py"
    moss = load_file("rdna2_moss_native_v05", moss_file)
    qwen = load_file("rdna2_qwen_native_v05", qwen_file)
    configs = [tuple(map(int, s.split(","))) for s in cfg.configs.split(";")]
    if any(len(c) != 6 for c in configs):
        raise ValueError("Config is BM,BN,BK,warps,stages,group_m")
    cases = []
    for item in cfg.cases.split(";"):
        backend, shape = item.split(":")
        if backend not in ("moss", "qwen"):
            raise ValueError(backend)
        cases.append((backend, tuple(map(int, shape.split(",")))))
    if cfg.rounds < 3 or cfg.iters < 1:
        raise ValueError("At least three alternating rounds are required")
    report = {
        "purpose": "Operator microbenchmark, synthetic data at observed model GEMM shapes",
        "gpu": torch.cuda.get_device_name(0),
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
        "torch": torch.__version__, "hip": torch.version.hip, "triton": triton.__version__,
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    (moss_file, qwen_file, Path(__file__), Path(__file__).parent / "rdna2_ops/int8_gemm.py")},
        "contract": "FP32 output; exact integer dot. MOSS separate FP32 scale multiply then bias. Qwen permits FMA as in native patientx.",
        "timing": {"unit": "milliseconds", "method": "synchronized ordinary wall, alternating rounds",
                   "rounds": cfg.rounds, "iters": cfg.iters, "warmups": 2,
                   "excluded": ["JIT/autotune", "input generation", "correctness checks"],
                   "included": ["dispatch", "output allocation", "MOSS separate dequant/bias if native"],
                   "quantization": "excluded from GEMM rows; included separately in pipeline rows",
                   "weight_layout": "Prepared once before timing; preparation cost and resident bytes reported"},
        "gate": {"integer": "bitwise CPU INT64 oracle", "backend_fp32": "bitwise native output",
                 "nonfinite": 0},
        "small_checks": [], "rows": [], "failures": [],
    }

    def save():
        cfg.output.write_text(json.dumps(report, indent=2, default=str) + "\n")

    def error(out, ref):
        finite = bool(torch.isfinite(out).all())
        d = out.float() - ref.float()
        return {"bitwise": bool(torch.equal(out.view(torch.int32), ref.view(torch.int32))),
                "finite": finite, "max_abs": d.abs().max().item(),
                "rmse": d.square().mean().sqrt().item()}

    def metadata(kernel):
        if kernel is None:
            return None
        asm = kernel.asm.get("amdgcn", "")
        meta = getattr(kernel, "metadata", None)
        return {"n_regs": getattr(kernel, "n_regs", None),
                "n_spills": getattr(kernel, "n_spills", None),
                "shared": getattr(meta, "shared", None),
                "v_dot4": len(re.findall(r"\bv_dot4(?:c)?_i32_i8\b", asm)),
                "v_dot2": len(re.findall(r"\bv_dot2(?:c)?_f32_f16\b", asm)),
                "mfma_wmma": len(re.findall(r"\bv_(?:mfma|wmma)\w*", asm)),
                "assembly_sha256": hashlib.sha256(asm.encode()).hexdigest()}

    def timed(fn):
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(cfg.iters):
            y = fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1000 / cfg.iters

    def peak(fn):
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        y = fn()
        torch.cuda.synchronize()
        value = torch.cuda.max_memory_allocated() - base
        del y
        return value

    def native(backend, a, b, sa, sb, bias, return_kernel=False):
        if backend == "moss":
            acc = moss.int8_matmul(a, b)
            m, n = acc.shape
            chunk_size = max(1, min(m, 256 * 1024 * 1024 // (n * 4)))
            parts = []
            for off in range(0, m, chunk_size):
                chunk = acc[off:off + chunk_size].float()
                scales = sa[off:off + chunk_size].reshape(-1, 1) * sb.reshape(1, -1)
                parts.append(chunk * scales)
            out = torch.cat(parts, dim=0)
            if bias is not None:
                out = out + bias.float().reshape(1, -1)
            return (out, None) if return_kernel else out
        m, k = a.shape
        n = b.shape[1]
        out = torch.empty((m, n), device=a.device, dtype=torch.float32)
        fn = (qwen._int8_matmul_dequant_kernel if sb.numel() == 1
              else qwen._int8_matmul_dequant_per_row_kernel)
        grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(n, META["BLOCK_N"]),)
        kernel = fn[grid](a, b, out, sa, sb, bias if bias is not None else a,
                          m, n, k, *a.stride(), *b.stride(), *out.stride(),
                          HAS_BIAS=bias is not None)
        return (out, kernel) if return_kernel else out

    def candidate(backend, conf, a, b, sa, sb, bias, return_kernel=False):
        bm, bn, bk, nw, ns, gm = conf
        return int8_gemm_dequant(a, b, sa, sb, bias, tile=(bm, bn, bk),
                                num_warps=nw, num_stages=ns, group_m=gm,
                                rounding="separate" if backend == "moss" else "fused",
                                return_kernel=return_kernel, wrapped=cfg.wrapped, partial_dot=cfg.partial_dot)

    with torch.inference_mode():
        half_a = torch.randn(1, 16, 32, device="cuda", dtype=torch.float16)
        half_b = torch.randn(1, 32, 16, device="cuda", dtype=torch.float16)
        try:
            out32 = torch.bmm(half_a, half_b, out_dtype=torch.float32)
            oracle32 = torch.bmm(half_a.float(), half_b.float())
            report["fp16_bmm_fp32_output_probe"] = {
                "supported": True, "dtype": str(out32.dtype),
                "finite": bool(torch.isfinite(out32).all()),
                "max_abs_vs_fp32_input_roundtrip": (out32 - oracle32).abs().max().item()}
        except Exception as exc:
            report["fp16_bmm_fp32_output_probe"] = {"supported": False, "error": repr(exc)}
        print("FP16_BMM " + json.dumps(report["fp16_bmm_fp32_output_probe"]), flush=True)
        del half_a, half_b
        if not cfg.no_small_checks:
            for conf in configs:
                for layout in ("weight_nk", "rhs_kn"):
                    for scalar in (False, True):
                        m, n, k = 37, 53, 79
                        a_store = torch.randint(-128, 128, (m + 2, k + 3), device="cuda", dtype=torch.int8)
                        a = a_store[1:1 + m, 1:1 + k]
                        w = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int8)
                        b = w.T if layout == "weight_nk" else w.T.contiguous()
                        sa = torch.rand(m, 1, device="cuda") * 0.02 + 0.001
                        sb = torch.rand(1 if scalar else n, device="cuda") * 0.02 + 0.001
                        bias = torch.randn(n, device="cuda")
                        oracle_i64 = a.cpu().to(torch.int64) @ b.cpu().to(torch.int64)
                        bm, bn, bk, nw, ns, gm = conf
                        got_i32 = int8_matmul(a, b, tile=(bm, bn, bk), num_warps=nw,
                                             num_stages=ns, group_m=gm, wrapped=cfg.wrapped, partial_dot=cfg.partial_dot)
                        exact_integer = torch.equal(got_i32.cpu().to(torch.int64), oracle_i64)
                        # Independent integer oracle, with explicit separate FP32 GPU epilogue.
                        ref = oracle_i64.to(device="cuda", dtype=torch.float32) * (sa * sb.reshape(1, -1)) + bias
                        out = candidate("moss", conf, a, b, sa, sb, bias)
                        e = error(out, ref)
                        entry = {"config": conf, "layout": layout, "scalar_scale": scalar,
                                 "offset_and_row_padding": True, "exact_integer": exact_integer, **e}
                        report["small_checks"].append(entry)
                        if not exact_integer or not e["bitwise"] or not e["finite"]:
                            save()
                            raise AssertionError(entry)
            save()
            print("SMALL " + json.dumps({"checks": len(report["small_checks"]), "passed": True}), flush=True)

        for backend, (m, n, k) in cases:
            for layout in cfg.layouts.split(","):
                if layout not in ("weight_nk", "rhs_kn"):
                    raise ValueError(layout)
                print("CASE " + json.dumps({"backend": backend, "shape": [m, n, k], "layout": layout}), flush=True)
                a = torch.randint(-128, 128, (m, k), device="cuda", dtype=torch.int8)
                w = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int8)
                sa = torch.rand(m, 1, device="cuda") * 0.02 + 0.001
                sb = torch.rand(n, device="cuda") * 0.02 + 0.001
                bias = None if cfg.bias_mode == "none" else torch.randn(n, device="cuda")
                torch.cuda.synchronize()
                t = time.perf_counter()
                b = w.T if layout == "weight_nk" else w.T.contiguous()
                torch.cuda.synchronize()
                prep_ms = (time.perf_counter() - t) * 1000
                cold_start = time.perf_counter()
                ref, native_kernel = native(backend, a, b, sa, sb, bias, True)
                torch.cuda.synchronize()
                native_prepare_s = time.perf_counter() - cold_start
                row = {"backend": backend, "m": m, "n": n, "k": k, "layout": layout,
                       "bias_mode": cfg.bias_mode, "wrapped_loads": cfg.wrapped,
                       "partial_dot": cfg.partial_dot,
                       "a_stride": list(a.stride()), "b_stride": list(b.stride()),
                       "weight_prepare_ms": prep_ms,
                       "extra_weight_resident_bytes": 0 if layout == "weight_nk" else b.numel(),
                       "native_cold_autotune_s": native_prepare_s,
                       "native_kernel": metadata(native_kernel),
                       "native_config": str(getattr(qwen._int8_matmul_dequant_per_row_kernel,
                                                     "best_config", None)) if backend == "qwen" else [32,64,32,4,1],
                       "native": {}, "candidates": []}
                fns = {"native": lambda: native(backend, a, b, sa, sb, bias)}
                for conf in configs:
                    label = ",".join(map(str, conf))
                    try:
                        out, kernel = candidate(backend, conf, a, b, sa, sb, bias, True)
                        e = error(out, ref)
                        bm, bn, bk, nw, ns, gm = conf
                        oracle = a[:13].cpu().to(torch.int64) @ b[:, :11].cpu().to(torch.int64)
                        got_int = int8_matmul(a, b, tile=(bm, bn, bk), num_warps=nw,
                                             num_stages=ns, group_m=gm, wrapped=cfg.wrapped, partial_dot=cfg.partial_dot)
                        int_exact = torch.equal(got_int[:13, :11].cpu().to(torch.int64), oracle)
                        del got_int
                        item = {"config": conf, "key": label, "error": e,
                                "integer_sample_exact": int_exact, "kernel": metadata(kernel)}
                        item["accepted"] = e["bitwise"] and e["finite"] and int_exact
                        row["candidates"].append(item)
                        del out
                        if item["accepted"]:
                            fns[label] = lambda c=conf: candidate(backend, c, a, b, sa, sb, bias)
                    except Exception:
                        row["candidates"].append({"config": conf, "key": label, "accepted": False,
                                                  "error_text": traceback.format_exc()})
                samples = {label: [] for label in fns}
                for fn in fns.values():
                    for _ in range(2):
                        fn()
                for trial in range(cfg.rounds):
                    labels = list(fns)
                    if trial % 2:
                        labels.reverse()
                    for label in labels:
                        samples[label].append(timed(fns[label]))
                row["native"] = {"samples_wall_ms": samples["native"],
                                 "median_wall_ms": statistics.median(samples["native"]),
                                 "additional_peak_bytes": peak(fns["native"])}
                for item in row["candidates"]:
                    if not item["accepted"]:
                        continue
                    label = item["key"]
                    item["samples_wall_ms"] = samples[label]
                    item["median_wall_ms"] = statistics.median(samples[label])
                    item["speedup"] = row["native"]["median_wall_ms"] / item["median_wall_ms"]
                    item["additional_peak_bytes"] = peak(fns[label])
                passed = [c for c in row["candidates"] if c["accepted"]]
                if passed:
                    best = min(passed, key=lambda c: c["median_wall_ms"])
                    row["best_config"] = best["config"]
                    if not cfg.no_pipeline:
                        from comfy_kitchen.backends.eager import quantization as kitchen
                        x = torch.randn((m, k), device="cuda", dtype=torch.float32)
                        quantize = kitchen.quantize_int8_rowwise if backend == "moss" else qwen.triton_quantize_rowwise
                        def pipe_native():
                            qa, qs = quantize(x)
                            return native(backend, qa, b, qs, sb, bias)
                        def pipe_candidate():
                            qa, qs = quantize(x)
                            return candidate(backend, tuple(best["config"]), qa, b, qs, sb, bias)
                        e_pipe = error(pipe_candidate(), pipe_native())
                        pipe_samples = {"native": [], "candidate": []}
                        for _ in range(2):
                            pipe_native(); pipe_candidate()
                        for trial in range(cfg.rounds):
                            items = [("native", pipe_native), ("candidate", pipe_candidate)]
                            if trial % 2:
                                items.reverse()
                            for label, fn in items:
                                pipe_samples[label].append(timed(fn))
                        med = {name: statistics.median(values) for name, values in pipe_samples.items()}
                        row["pipeline"] = {"scope": "row quantization + GEMM/dequant; no ConvRot/input-act/residual",
                                           "error": e_pipe, "samples_wall_ms": pipe_samples,
                                           "median_wall_ms": med, "speedup": med["native"] / med["candidate"]}
                report["rows"].append(row)
                save()
                print("RESULT " + json.dumps(row), flush=True)
                del a, w, b, sa, sb, bias, ref
    save()
    print("DONE " + str(cfg.output), flush=True)


if __name__ == "__main__":
    main()
