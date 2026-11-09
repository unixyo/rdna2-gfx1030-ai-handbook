#!/usr/bin/env python3
"""Measured register-dot2 and streaming-copy ceilings for v0.8 roofline bounds.

The dot loop deliberately has eight independent FP32 accumulators and FP16
packed input pairs. Its throughput is an optimistic register arithmetic ceiling,
NOT a claim that a convolution can reach it. Streaming copy touches 512 MiB
(read+write) by default so the working set exceeds gfx1030 Infinity Cache.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import time

for _name in ("MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1",
              "MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53"):
    os.environ[_name] = "0"

import torch
import triton
import triton.language as tl
from gpu_guard_v06 import GpuAudit


@triton.jit
def _dot_eight(a, b, c0, c1, c2, c3, c4, c5, c6, c7):
    return tl.inline_asm_elementwise(
        """
        v_dot2c_f32_f16 $0, $8, $9;
        v_dot2c_f32_f16 $1, $8, $9;
        v_dot2c_f32_f16 $2, $8, $9;
        v_dot2c_f32_f16 $3, $8, $9;
        v_dot2c_f32_f16 $4, $8, $9;
        v_dot2c_f32_f16 $5, $8, $9;
        v_dot2c_f32_f16 $6, $8, $9;
        v_dot2c_f32_f16 $7, $8, $9;
        """,
        constraints="=v,=v,=v,=v,=v,=v,=v,=v,v,v,0,1,2,3,4,5,6,7",
        args=[a, b, c0, c1, c2, c3, c4, c5, c6, c7],
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32,
               tl.float32, tl.float32, tl.float32, tl.float32),
        is_pure=True, pack=1)


@triton.jit
def _dot_loop(A, B, O, N: tl.constexpr, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + idx, idx < N, 0)
    b = tl.load(B + idx, idx < N, 0)
    c0 = tl.full((BLOCK,), 0., tl.float32)
    c1 = tl.full((BLOCK,), 1., tl.float32)
    c2 = tl.full((BLOCK,), 2., tl.float32)
    c3 = tl.full((BLOCK,), 3., tl.float32)
    c4 = tl.full((BLOCK,), 4., tl.float32)
    c5 = tl.full((BLOCK,), 5., tl.float32)
    c6 = tl.full((BLOCK,), 6., tl.float32)
    c7 = tl.full((BLOCK,), 7., tl.float32)
    for i in range(ITERS):
        c0, c1, c2, c3, c4, c5, c6, c7 = _dot_eight(
            a, b, c0, c1, c2, c3, c4, c5, c6, c7)
    tl.store(O + idx, c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7, idx < N)


@triton.jit
def _stream_copy(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + idx, idx < N, 0)
    tl.store(Y + idx, value, idx < N)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--copy-mib", type=int, default=256, help="MiB in each copy buffer")
    p.add_argument("--dot-elements", type=int, default=262144)
    p.add_argument("--dot-iters", type=int, default=1024)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    out = args.out.resolve()
    if not out.is_relative_to(root) or out.exists():
        p.error("Use a fresh output path inside the research folder")
    out.mkdir(parents=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(root / "results/applio_v08/triton_cache"))
    result = dict(schema="applio_v08_hardware_calibration",
                  caution="Optimistic measured arithmetic/copy ceilings, not predicted achieved Conv performance",
                  torch=torch.__version__, triton=triton.__version__, hip=torch.version.hip,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  repetitions=args.reps, rounds=args.rounds)

    def timed(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        rows = []
        for _ in range(args.rounds):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            wall = time.perf_counter()
            start.record()
            for _ in range(args.reps):
                fn()
            end.record()
            end.synchronize()
            rows.append(dict(event_ms=start.elapsed_time(end) / args.reps,
                             wall_ms=(time.perf_counter() - wall) * 1000 / args.reps))
        return dict(rounds=rows, event_median_ms=statistics.median(x["event_ms"] for x in rows),
                    wall_median_ms=statistics.median(x["wall_ms"] for x in rows))

    with GpuAudit() as audit:
        try:
            audit.ensure_clear()
            prop = torch.cuda.get_device_properties(0)
            if prop.gcnArchName.split(":")[0] != "gfx1030":
                raise RuntimeError("This instruction calibration requires gfx1030")
            result.update(device=prop.name, arch=prop.gcnArchName)
            n = args.dot_elements
            gen = torch.Generator(device="cpu").manual_seed(803117)
            ah = (.1 + .1 * torch.rand((n, 2), generator=gen)).half().cuda()
            bh = (.01 + .01 * torch.rand((n, 2), generator=gen)).half().cuda()
            a, b = ah.view(torch.int32).flatten(), bh.view(torch.int32).flatten()
            y = torch.empty((n,), device="cuda", dtype=torch.float32)
            def dot():
                return _dot_loop[(triton.cdiv(n, 256),)](
                    a, b, y, n, args.dot_iters, 256, num_warps=4, num_stages=1)
            first = time.perf_counter()
            kernel = dot()
            torch.cuda.synchronize()
            first_s = time.perf_counter() - first
            asm = kernel.asm.get("amdgcn", "")
            (out / "dot2_calibration.s").write_text(asm)
            counts = {name: len(re.findall(r"(?m)^\s*" + name + r"\s", asm))
                      for name in ("v_dot2c_f32_f16", "v_dot2_f32_f16", "v_fma_f32",
                                   "v_fmac_f32")}
            if counts["v_dot2c_f32_f16"] < 8:
                raise RuntimeError("Expected eight retained dot2 accumulators; inspect saved ISA")
            reference = 28 + 8 * args.dot_iters * (ah.float() * bh.float()).sum(1)
            error = (y - reference).abs()
            result["dot2"] = dict(elements=n, iterations=args.dot_iters,
                                  independent_accumulators=8, flop_count=n * args.dot_iters * 8 * 4,
                                  first_compile_and_call_s=first_s,
                                  finite=bool(torch.isfinite(y).all()),
                                  max_abs=error.max().item(),
                                  max_relative=(error / reference.abs().clamp_min(1e-12)).max().item(),
                                  instruction_counts_static=counts,
                                  assembly_sha256=hashlib.sha256(asm.encode()).hexdigest(),
                                  registers=getattr(kernel, "n_regs", None),
                                  spills=getattr(kernel, "n_spills", None))
            result["dot2"].update(timed(dot))
            result["dot2"]["measured_tflop_s"] = (
                result["dot2"]["flop_count"] / (result["dot2"]["event_median_ms"] * 1e9))
            audit.ensure_clear()
            for label, mib in (("copy", args.copy_mib), ("copy_cached", 8)):
                numel = mib * (1 << 20) // 4
                x = torch.arange(numel, device="cuda", dtype=torch.float32)
                target = torch.empty_like(x)
                def copy():
                    return _stream_copy[(triton.cdiv(numel, 4096),)](
                        x, target, numel, 4096, num_warps=8, num_stages=1)
                copy()
                torch.cuda.synchronize()
                correct = torch.equal(x, target)
                result[label] = dict(buffer_bytes=numel * 4, read_write_bytes=numel * 8,
                                      bitwise_correct=correct, implementation="Triton contiguous load/store",
                                      cache_scope="larger-than-Infinity-Cache" if label == "copy" else
                                                  "warm small working set; may be cache resident")
                result[label].update(timed(copy))
                seconds = result[label]["event_median_ms"] * 1e-3
                result[label]["measured_GB_s"] = numel * 8 / seconds / 1e9
                result[label]["measured_GiB_s"] = numel * 8 / seconds / (1 << 30)
                del x, target
        except Exception as exc:
            result["error"] = dict(type=type(exc).__name__, message=str(exc))
            raise
        finally:
            result["gpu_audit"] = audit.report()
            result["valid_for_timing"] = audit.report()["valid_for_timing"] and "error" not in result
            (out / "result.json").write_text(json.dumps(result, indent=2))
    result["gpu_audit"] = audit.report()
    result["valid_for_timing"] = audit.report()["valid_for_timing"] and "error" not in result
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
