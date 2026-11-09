#!/usr/bin/env python3
"""Actual profiled SDPA shapes, preserving FP32 and the interleaved QKV layout."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
import torch.nn.functional as F
from rdna2_ops.attention import scaled_dot_product_attention as fast


def timed(fn, count=10):
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin = time.perf_counter()
    a.record()
    for _ in range(count):
        y = fn()
    b.record()
    torch.cuda.synchronize()
    return {"gpu_ms": a.elapsed_time(b) / count,
            "wall_ms": (time.perf_counter() - begin) * 1000 / count}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("results/attention_math.json"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20261105)
    report = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "dtype": "float32", "rtol": 2e-4, "atol": 2e-5, "rows": []}
    with torch.inference_mode():
        for heads, length, dim in [(24, 37, 128), (24, 1061, 128), (4, 2048, 128)]:
            q, k, v = [torch.randn(1, length, heads, dim, device="cuda").transpose(1, 2) for _ in range(3)]
            native = lambda: F.scaled_dot_product_attention(q, k, v)
            rdna2 = lambda: fast(q, k, v, backend="rdna2")
            ref, result = native(), rdna2()
            torch.testing.assert_close(result, ref, rtol=2e-4, atol=2e-5)
            error = (result - ref).abs()
            row = {"shape": list(q.shape), "stride": list(q.stride()),
                   "max_abs": error.max().item(), "rmse": error.square().mean().sqrt().item(),
                   "accuracy_pass": True, "samples": {"native": [], "rdna2": []}}
            for _ in range(3):
                native(); rdna2()
            for i in range(5):
                for key, fn in ([("native", native), ("rdna2", rdna2)] if i % 2 == 0 else [("rdna2", rdna2), ("native", native)]):
                    row["samples"][key].append(timed(fn))
            for key in row["samples"]:
                for metric in ("gpu_ms", "wall_ms"):
                    row[key + "_" + metric] = statistics.median(s[metric] for s in row["samples"][key])
            row["speedup"] = row["native_wall_ms"] / row["rdna2_wall_ms"]
            report["rows"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
