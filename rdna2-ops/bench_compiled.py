#!/usr/bin/env python3
"""Compare HIP fusion against native eager and the existing torch.compile.

HIP GroupNorm is eager-only and participates only in ordinary-call timings.
Graph warmup, capture, and replay measure native PyTorch and compiled callables
only; HIP graph timing is policy_disabled.
"""
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F
from rdna2_ops import group_norm_silu
from validate_bench import timed_call, error_metrics, fp64_reference, capture_callable


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261105)
    report = {
        "torch": torch.__version__, "shape": [1,320,64,64], "rows": [],
        "graph_policy": {
            "hip": "policy_disabled", "measured_backends": ["native", "compiled"],
            "reason": "HIP GroupNorm is eager-only; excluded from graph warmup, capture, and replay.",
        },
    }
    path = Path("results/compiled_comparison.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for dtype in (torch.float32, torch.float16):
            x = torch.randn(report["shape"], device="cuda", dtype=dtype)
            w = torch.randn(320, device="cuda", dtype=dtype) * .2 + 1
            b = torch.randn_like(w) * .1
            def native(x, w, b):
                return F.silu(F.group_norm(x, 32, w, b, 1e-5))
            row = {"dtype": str(dtype)}
            try:
                start = time.perf_counter()
                compiled = torch.compile(native, fullgraph=True)
                y = compiled(x,w,b)
                torch.cuda.synchronize()
                row["compile_and_first_call_seconds"] = time.perf_counter() - start
                row["compiled_vs_fp64"] = error_metrics(y, fp64_reference(x,32,w,b,1e-5), dtype)
                if not row["compiled_vs_fp64"]["pass"]:
                    raise AssertionError("compiled reference accuracy failure")
                funcs = {"native": lambda:native(x,w,b), "compiled": lambda:compiled(x,w,b), "hip": lambda:group_norm_silu(x,32,w,b,1e-5,backend="hip")}
                for fn in funcs.values():
                    for _ in range(6): fn()
                samples={k:[] for k in funcs}
                for i in range(5):
                    for k in (list(funcs) if i%2==0 else list(reversed(funcs))):
                        samples[k].append(timed_call(funcs[k], 40))
                row["samples"]=samples
                row["median_wall_ms"]={k:statistics.median(s["wall_ms"] for s in v) for k,v in samples.items()}
                graph_funcs={k:funcs[k] for k in ("native", "compiled")}
                captured={k:capture_callable(fn,16) for k,fn in graph_funcs.items()}
                graph={k:[] for k in graph_funcs}
                for i in range(5):
                    for k in (list(graph_funcs) if i%2==0 else list(reversed(graph_funcs))):
                        graph[k].append(timed_call(captured[k][0].replay, 5)["event_ms"]/16)
                row["graph_ms"]=graph
                row["median_graph_ms"]={k:statistics.median(v) for k,v in graph.items()}
                row["status"]="ok"
            except Exception as exc:
                row["status"]="failed"
                row["error"]=f"{type(exc).__name__}: {exc}"
            report["rows"].append(row)
            path.write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps({k:v for k,v in row.items() if k not in ("samples","graph_ms")}),flush=True)


if __name__=="__main__":
    main()
