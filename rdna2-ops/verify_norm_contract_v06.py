#!/usr/bin/env python3
"""CPU-only orchestration/broadcast checks against the installed Anima source.

This does not compile a GPU kernel or claim operator/model numerical evidence.
The pinned forward's AST is evaluated only as an independent CPU reference;
the running Comfy module and its bytecode are never replaced by this test.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Callable, Optional

import torch
from einops import rearrange
from norm_switches_v06 import ANIMA_SOURCE_SHA256, anima_block
from rdna2_ops.norm_fusions_v06 import _modulation_rows
from bench_norm_fusions_v06 import CASES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior evidence")
    source = args.source.read_bytes()
    sha = hashlib.sha256(source).hexdigest()
    assert sha == ANIMA_SOURCE_SHA256
    tree = ast.parse(source)
    block = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Block")
    forward = next(n for n in block.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    namespace = {"torch": torch, "Optional": Optional, "Callable": Callable, "rearrange": rearrange}
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(args.source), "exec"), namespace)
    reference = namespace["forward"]
    torch.set_num_threads(1)
    torch.manual_seed(42)

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8, dtype=torch.float16)

        def forward(self, x, context=None, **kwargs):
            y = self.proj(x).tanh()
            return y if context is None else y + context.mean() * 0.01

    class Block(torch.nn.Module):
        def __init__(self, lora):
            super().__init__()
            self.use_adaln_lora = lora
            for name in ("self_attn", "cross_attn", "mlp"):
                setattr(self, "layer_norm_" + name, torch.nn.LayerNorm(8, eps=1e-6, elementwise_affine=False))
                setattr(self, "adaln_modulation_" + name, torch.nn.Linear(8, 24, dtype=torch.float16))
            self.self_attn, self.cross_attn = Attention(), Attention()
            self.mlp = torch.nn.Sequential(torch.nn.Linear(8, 16, dtype=torch.float16),
                                          torch.nn.SiLU(), torch.nn.Linear(16, 8, dtype=torch.float16))

    rows = []
    with torch.inference_mode():
        for lora in (False, True):
            for position in (False, True):
                for patches in (False, True):
                    block = Block(lora)
                    x = torch.randn(2, 1, 3, 5, 8)
                    emb = torch.randn(2, 1, 8).half()
                    context = torch.randn(2, 4, 8).half()
                    residual = torch.randn(2, 1, 24).half() if lora else None
                    pos = torch.randn_like(x) if position else None
                    options = {"patches": {"mlp_patch": [lambda data: {**data, "x": data["x"] + .001}]}} if patches else {}
                    ref = reference(block, x, emb, context, None, residual, pos, options)
                    got = anima_block(block, x, emb, context, None, residual, pos, options,
                                      normalize=lambda xx, n, s, sh, dt: (n(xx) * (1 + s) + sh).to(dt))
                    equal = torch.equal(ref, got)
                    rows.append(dict(lora=lora, position=position, patches=patches, bit_equal=equal,
                                     max_abs=float((ref - got).abs().max())))
                    assert equal, rows[-1]
        broadcast = []
        for shape, modulation in (((2, 1, 3, 5, 8), (2, 1, 1, 1, 8)),
                                  ((2, 7, 8), (2, 1, 8)), ((2, 7, 8), (1, 1, 8))):
            x = torch.empty(shape)
            packed = torch.randn(*modulation[:-1], 16)
            t = packed[..., :8]
            flat, group = _modulation_rows(t, x)
            out = flat[torch.arange(x.numel() // 8) // group]
            equal = torch.equal(out, t.expand(shape).reshape(-1, 8))
            broadcast.append(dict(shape=shape, modulation=modulation, row_stride=flat.stride(0),
                                  group=group, bit_equal=equal))
            assert equal
        rejected = False
        try:
            _modulation_rows(torch.ones(1, 3, 8), torch.empty(2, 3, 8))
        except ValueError:
            rejected = True
        assert rejected
    result = {"status": "cpu_contract_pass", "gpu_initialized": torch.cuda.is_initialized(),
              "scope": "CPU orchestration and compact broadcast only; GPU kernel/model A/B pending",
              "source": {"path": str(args.source), "sha256": sha}, "orchestration": rows,
              "broadcast": broadcast, "alternating_broadcast_rejected": rejected,
              "operator_plan_cases": list(CASES)}
    assert result["gpu_initialized"] is False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
