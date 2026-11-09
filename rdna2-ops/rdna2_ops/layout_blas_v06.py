"""Explicit physical-storage variants for ordinary mm/bmm.

N/T name ROW/COLUMN-major storage of the SAME logical operands, not a request
to change A@B. PyTorch may swap operands when calling column-major BLAS, so
these labels are not the rocBLAS transA/transB flag values. No automatic policy.
"""
import torch

LAYOUTS = ("NN", "NT", "TN", "STRIDED")


def prepare_operands(a, b, layout):
    if layout not in LAYOUTS:
        raise ValueError("Unknown physical layout: " + str(layout))
    if a.ndim not in (2, 3) or b.ndim != a.ndim:
        raise ValueError("Explicit 2-D mm or 3-D bmm operands required")
    if a.shape[-1] != b.shape[-2] or a.shape[:-2] != b.shape[:-2]:
        raise ValueError("Matching contraction and batch dimensions required")
    if a.dtype != b.dtype or a.device != b.device:
        raise ValueError("Matching dtype and device required")
    aa = (a.transpose(-1, -2).contiguous().transpose(-1, -2)
          if layout == "TN" else a.contiguous())
    if layout == "NT":
        bb = b.transpose(-1, -2).contiguous().transpose(-1, -2)
    elif layout == "STRIDED" and b.ndim == 3:
        # Interleave batch/head storage: logical [B,K,N], stride [N,B*N,1].
        # This matches the B=1, heads-as-batch family, not a claim about the
        # flattened strides of Anima's B=2,H=16 tensor before its contiguous().
        bb = b.transpose(0, 1).contiguous().transpose(0, 1)
    elif layout == "STRIDED":
        # Non-unit column stride, with holes. PyTorch may insert a pack/copy.
        backing = torch.empty((b.shape[0], 2 * b.shape[1]), device=b.device, dtype=b.dtype)
        bb = backing[:, ::2]
        bb.copy_(b)
    else:
        bb = b.contiguous()
    return aa, bb


def multiply_prepared(a, b):
    return torch.mm(a, b) if a.ndim == 2 else torch.bmm(a, b)


def matmul_layout(a, b, layout="NT"):
    """Copy-inclusive explicit experiment; caller chooses a measured layout."""
    return multiply_prepared(*prepare_operands(a, b, layout))
