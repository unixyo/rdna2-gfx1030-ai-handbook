#!/usr/bin/env python3
"""Compare the installed ComfyUI VAE ResnetBlock with explicit norm fusion.

Uses seeded inputs and weights, not a pretrained checkpoint or generated image.
The original installed module is imported read-only; its source is not patched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch
from rdna2_ops import group_norm_silu
from validate_bench import timed_call, error_metrics, capture_callable


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comfy", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("results/comfy_blocks.json"))
    args = ap.parse_args()
    sys.path.insert(0, str(args.comfy))
    saved_argv = sys.argv
    sys.argv = ["rdna2_block_benchmark"]
    from comfy.ldm.modules.diffusionmodules import model
    sys.argv = saved_argv
    torch.set_num_threads(2)
    torch.manual_seed(20261105)
    source = Path(model.__file__)
    report = {"workload": "Installed ComfyUI ResnetBlock forward; seeded random weights/input, same parameters in both paths", "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "full_model_or_image_generation": False, "rows": []}
    with torch.inference_mode():
        for dtype in (torch.float32, torch.float16):
            for c, h in ((64, 128), (128, 64), (256, 32)):
                block = model.ResnetBlock(in_channels=c, temb_channels=0, dropout=0.0, conv_op=torch.nn.Conv2d, norm_op=lambda channels: torch.nn.GroupNorm(32, channels, eps=1e-6)).eval().to(device="cuda", dtype=dtype)
                x = torch.randn(1, c, h, h, device="cuda", dtype=dtype)
                def native():
                    return block(x)
                def fused():
                    a = block.norm1
                    y = group_norm_silu(x, a.num_groups, a.weight, a.bias, a.eps, backend="hip")
                    y = block.conv1(y)
                    a = block.norm2
                    y = group_norm_silu(y, a.num_groups, a.weight, a.bias, a.eps, backend="hip")
                    return x + block.conv2(block.dropout(y))
                baseline, candidate = native(), fused()
                check = error_metrics(candidate, baseline, dtype)
                if not check["pass"]:
                    raise AssertionError(f"ResnetBlock output differs: {check}")
                functions = {"native": native, "fused": fused}
                for _ in range(8):
                    native(); fused()
                timing = {name: [] for name in functions}
                for i in range(5):
                    for name in (("native", "fused") if i % 2 == 0 else ("fused", "native")):
                        timing[name].append(timed_call(functions[name], 20))
                graph_timing = {name: [] for name in functions}
                captured = {name: capture_callable(fn, 8) for name, fn in functions.items()}
                for i in range(5):
                    for name in (("native", "fused") if i % 2 == 0 else ("fused", "native")):
                        graph_timing[name].append(timed_call(captured[name][0].replay, 5)["event_ms"] / 8)
                n = statistics.median(t["wall_ms"] for t in timing["native"])
                f = statistics.median(t["wall_ms"] for t in timing["fused"])
                row = {"dtype": str(dtype), "shape": list(x.shape), "output_check": check, "samples": timing, "native_wall_ms": n, "fused_wall_ms": f, "wall_speedup": n/f, "graph_ms": graph_timing, "graph_speedup": statistics.median(graph_timing["native"]) / statistics.median(graph_timing["fused"])}
                report["rows"].append(row)
                print(json.dumps({k:v for k,v in row.items() if k not in ("samples","graph_ms")}), flush=True)
    report["source_unchanged"] = hashlib.sha256(source.read_bytes()).hexdigest() == report["source_sha256"]
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(str(args.output.resolve()), flush=True)


if __name__ == "__main__":
    main()
