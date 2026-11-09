"""Project-local gfx1030 INT8 support and optional transpose-copy removal.

MOSS_RDNA2_INT8_PATCH=1 enables the existing Triton INT32 GEMM replacement.
MOSS_RDNA2_INT8_LAYOUT=view replaces the existing eager linear export.
MOSS_RDNA2_INT8_LAYOUT=legacy restores the previous working GEMM-only behavior.
MOSS_RDNA2_INT8_STATS=1 enables optional backend counters; no per-call logging.

No installed package or model files are edited. Disabling the entire PATCH also
removes the required gfx1030 GEMM workaround; use LAYOUT=legacy for rollback.
"""

import logging
import os
from pathlib import Path
import sys


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


if os.environ.get("MOSS_RDNA2_INT8_PATCH") == "1":
    layout = os.environ.get("MOSS_RDNA2_INT8_LAYOUT", "view")
    if layout not in {"legacy", "view"}:
        raise ValueError("MOSS_RDNA2_INT8_LAYOUT must be 'legacy' or 'view'")

    import torch

    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("MOSS RDNA2 patch requires a ROCm GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx1030":
        raise RuntimeError("This selective INT8 patch is only validated on gfx1030, not " + arch)

    import comfy_kitchen.backends.eager.quantization as ck_quant

    root = Path(__file__).resolve().parents[2]
    scripts_path = str(root / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    from rdna2_int8_matmul import int8_matmul

    previous_accumulator = ck_quant._int8_matmul_accumulate
    try:
        ck_quant._int8_matmul_accumulate = int8_matmul
        if layout == "view":
            from . import backend

            backend.install_backend(int8_matmul)
    except Exception:
        ck_quant._int8_matmul_accumulate = previous_accumulator
        raise

    logging.getLogger("moss-rdna2").warning(
        "MOSS_RDNA2_INT8_PATCH active on %s: gfx1030 Triton INT32 GEMM, layout=%s%s",
        arch,
        layout,
        " (FP32 input/output only, eager export, weight.T view)" if layout == "view" else " (original eager linear)",
    )
    from . import attention_dispatch
    attention_dispatch.install()
    if os.environ.get("MOSS_RDNA2_MEASURE_DIR"):
        from . import measurement
        measurement.install()
