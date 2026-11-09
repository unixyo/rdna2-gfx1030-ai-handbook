#!/usr/bin/env python3
"""Focused runtime contracts: safe-softmax specials and opaque-HIP bypass."""
import argparse
import json
from pathlib import Path
import statistics
import torch
import torch.nn.functional as F
import rdna2_ops.normalization as norm
from rdna2_ops._softmax import repair_all_negative_inf_rows_
from rdna2_ops.attention import scaled_dot_product_attention
from validate_bench import capture_callable, timed_call

ap = argparse.ArgumentParser()
ap.add_argument("--output", type=Path, default=Path("results/runtime_contracts.json"))
cfg = ap.parse_args()
torch.set_num_threads(2)
torch.manual_seed(20261105)
report = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
          "safe_softmax": [], "attention_specials": []}
with torch.inference_mode():
    for length in (1, 37, 1061, 4096):
        scores = torch.randn(7, length, device="cuda")
        scores[1].fill_(-float("inf"))
        scores[2, 0] = -float("inf")
        scores[3, 0] = float("inf")
        scores[4, 0] = float("nan")
        scores[5, -1] = float("nan")
        scores[6].fill_(0)
        expected = torch.ops.aten._safe_softmax.default(scores, -1)
        actual = torch.softmax(scores, dim=-1)
        repair_all_negative_inf_rows_(scores, actual)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        finite = torch.isfinite(expected)
        assert torch.equal(actual[finite].view(torch.int32), expected[finite].view(torch.int32))
        report["safe_softmax"].append({"length": length, "equal_including_nonfinite": True,
                                       "finite_bitwise_equal": True, "all_negative_inf_zero": bool((actual[1] == 0).all())})
    for kind in ("overflow_negative", "positive_inf", "nan", "negative_scale", "zero_scale"):
        q = torch.randn(1, 2, 3, 128, device="cuda")
        k = torch.randn(1, 2, 5, 128, device="cuda")
        v = torch.randn(1, 2, 5, 128, device="cuda")
        scale = None
        if kind == "overflow_negative":
            q.fill_(1e30); k.fill_(-1e30)
        elif kind == "positive_inf":
            q[0, 0, 0, 0] = float("inf")
        elif kind == "nan":
            k[0, 1, -1, -1] = float("nan")
        elif kind == "negative_scale":
            scale = -0.1
        else:
            scale = 0.0
        expected = F.scaled_dot_product_attention(q, k, v, scale=scale)
        actual = scaled_dot_product_attention(q, k, v, scale=scale, backend="rdna2")
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6, equal_nan=True)
        report["attention_specials"].append({"case": kind, "pass": True})

    x = torch.randn(1, 320, 64, 64, dtype=torch.float16, device="cuda")
    w = torch.randn(320, dtype=x.dtype, device=x.device)
    b = torch.randn_like(w)
    def auto(x, w, b):
        return norm.group_norm_silu(x, 32, w, b)
    def native(x, w, b):
        return F.silu(F.group_norm(x, 32, w, b))
    # Eager still uses HIP; then forbid the C ABI loader to prove capture,
    # compilation and tracing do not accidentally enter an opaque kernel.
    auto(x, w, b)
    report["gn_eager_wall_ms"] = statistics.median(timed_call(lambda: auto(x,w,b), 30)["wall_ms"] for _ in range(3))
    original_library = norm._library
    def forbidden():
        raise AssertionError("Opaque HIP kernel was used during a native transform")
    norm._library = forbidden
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                native(x,w,b)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = auto(x,w,b)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, native(x,w,b))
        report["gn_capture_native_bitwise_equal"] = True
        traced = torch.jit.trace(auto, (x,w,b), check_trace=False)
        x2 = torch.randn_like(x)
        assert torch.equal(traced(x2,w,b), native(x2,w,b))
        report["gn_trace_new_input_native_bitwise_equal"] = True
        compiled = torch.compile(auto, fullgraph=True)
        y = compiled(x,w,b)
        torch.testing.assert_close(y, native(x,w,b), rtol=0.003, atol=0.003)
        report["gn_compile_accuracy_pass"] = True
        graphs = {"native": capture_callable(lambda: native(x,w,b), 16),
                  "compiled_auto": capture_callable(lambda: compiled(x,w,b), 16)}
        for captured_graph, _ in graphs.values():
            captured_graph.replay()
        torch.cuda.synchronize()
        report["gn_graph_us"] = {
            key: 1000 * statistics.median(timed_call(item[0].replay, 5)["event_ms"] / 16 for _ in range(3))
            for key, item in graphs.items()}
    finally:
        norm._library = original_library
report["status"] = "passed"
cfg.output.parent.mkdir(parents=True, exist_ok=True)
cfg.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)

