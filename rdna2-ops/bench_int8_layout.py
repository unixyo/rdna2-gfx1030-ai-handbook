#!/usr/bin/env python3
"""Check the installed stride-aware INT8 GEMM before removing a transpose copy."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--project", type=Path, required=True)
ap.add_argument("--output", type=Path, default=Path("results/int8_layout.json"))
cfg = ap.parse_args()
sys.path.insert(0, str(cfg.project / "scripts"))
from rdna2_int8_matmul import int8_matmul
torch.set_num_threads(2)
torch.manual_seed(20261105)


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(5):
        result = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / 5


report = {"kernel": str(cfg.project / "scripts/rdna2_int8_matmul.py"),
          "gpu": torch.cuda.get_device_name(0), "rows": []}
with torch.inference_mode():
    for m, n, k in [(37, 64, 256), (1061, 9216, 3072), (1061, 3072, 3072), (1061, 16384, 3072), (1061, 3072, 8192)]:
        a = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
        w = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
        baseline = lambda: int8_matmul(a, w.T.contiguous())
        strided = lambda: int8_matmul(a, w.T)
        torch.cuda.synchronize()
        prepare_start = time.perf_counter()
        wt = w.T.contiguous()
        torch.cuda.synchronize()
        prepare_ms = (time.perf_counter() - prepare_start) * 1000
        prepared = lambda: int8_matmul(a, wt)
        old, new = baseline(), strided()
        assert torch.equal(old, new)
        assert torch.equal(old, prepared())
        if m == 37:
            oracle = a.cpu().to(torch.int32) @ w.cpu().to(torch.int32).T
            assert torch.equal(new.cpu(), oracle)
        for _ in range(2):
            baseline(); strided(); prepared()
        samples = {"copy": [], "view": [], "pretransposed": []}
        for trial in range(3):
            order = [("copy", baseline), ("view", strided), ("pretransposed", prepared)]
            for name, fn in (order if trial % 2 == 0 else order[::-1]):
                samples[name].append(timed(fn))
        row = {"m": m, "n": n, "k": k, "rhs_stride": list(w.T.stride()),
               "int32_bitwise_equal": True, "samples_wall_ms": samples,
               "copy_wall_ms": statistics.median(samples["copy"]),
               "view_wall_ms": statistics.median(samples["view"])}
        row.update(pretransposed_wall_ms=statistics.median(samples["pretransposed"]),
                   one_time_prepare_ms=prepare_ms, extra_resident_bytes=wt.numel()*wt.element_size())
        row["speedup"] = row["copy_wall_ms"] / row["view_wall_ms"]
        report["rows"].append(row)
        print(json.dumps(row), flush=True)
cfg.output.parent.mkdir(parents=True, exist_ok=True)
cfg.output.write_text(json.dumps(report, indent=2) + "\n")

