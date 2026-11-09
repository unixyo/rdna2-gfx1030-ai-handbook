"""Same-process check of the production MOSS node's LAYOUT=fused (Claude, 2026-11-07).

Server starts with MOSS_RDNA2_INT8_LAYOUT=fused, so "baseline" is the node's fused export.
"viewonly" temporarily restores the node's view export; "int8_fused" is the research
candidate (rdna2_ops.int8_gemm) via int4_model_switches_v07.
"""
from contextlib import contextmanager
import int4_model_switches_v07 as inner


@contextmanager
def activate(mode, spec, state, bound, out):
    if mode == "viewonly":
        from comfy_kitchen.backends import eager as eager_backend
        import sys
        found = [m for n, m in list(sys.modules.items())
                 if n.endswith("fused_gemm") and hasattr(m, "_view_export") and hasattr(m, "install_fused")]
        if len(found) != 1:
            raise RuntimeError(f"Expected one loaded node fused_gemm module, found {len(found)}")
        fused = found[0]
        current = eager_backend.int8_linear
        if current is not fused.int8_linear:
            raise RuntimeError("Expected the node fused export to be active")
        eager_backend.int8_linear = fused._view_export
        state["candidate"] = {"mode": mode}
        try:
            yield
        finally:
            eager_backend.int8_linear = current
        return
    with inner.activate(mode, spec, state, bound, out):
        yield
