"""BF16 reduced-precision SDPA with explicit FP32 GEMM islands.

Preserves native math's four-dimensional Q/K sqrt pre-scaling, BF16 scores,
BF16 softmax output and BF16 final output. It replaces only the two GEMMs.
The reconstructed-native control must match the current installation before
using the FP32 candidate. No runtime source/default/backend flag is modified.
"""
from collections import Counter
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import torch

from ._softmax import repair_all_negative_inf_rows_


def bf16_sdpa(q, k, v, *, method, scale=None):
    if method not in ("reconstruct", "fp32"):
        raise ValueError("Expected reconstructed native or FP32 GEMM candidate")
    if any(type(x) not in (torch.Tensor, torch.nn.Parameter) or x.ndim != 4
           or x.dtype != torch.bfloat16 or x.device != q.device for x in (q,k,v)):
        raise ValueError("Plain same-device BF16 rank-four Q/K/V required")
    if q.device.type != "cuda" or not torch.version.hip:
        raise ValueError("HIP input required")
    if torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0] != "gfx1030":
        raise ValueError("Only gfx1030 is measured")
    if q.shape[:2] != k.shape[:2] or k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("Matching B/H/D and equal K/V required")
    if (not torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
            or torch.is_grad_enabled() and any(x.requires_grad for x in (q,k,v))
            or torch.is_autocast_enabled("cuda") or torch.compiler.is_compiling()
            or torch.cuda.is_current_stream_capturing()):
        raise ValueError("Requires eager reduced-precision math SDPA inference")
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("Finite scale required")
    root = math.sqrt(abs(scale))
    # Retain the multiply-before-transpose layout chosen by native 4D matmul.
    qs = q * (-root if scale < 0 else root)
    ks = k.transpose(-2, -1) * root
    scores = (torch.matmul(qs, ks) if method == "reconstruct" else
              torch.matmul(qs.float(), ks.float()).bfloat16())
    probability = torch.softmax(scores, dim=-1)
    repair_all_negative_inf_rows_(scores, probability)
    return (torch.matmul(probability, v) if method == "reconstruct" else
            torch.matmul(probability.float(), v.float()).bfloat16())


class BF16SdpaMode:
    def __init__(self, method, *, metadata=False):
        self.method = method
        self.hits = 0
        self.operations = Counter()
        self.fallbacks = Counter()
        self.original = None
        self.shapes = Counter()

    def __enter__(self):
        import comfy.ops
        self.original = comfy.ops.scaled_dot_product_attention
        original = self.original

        def invoke(q,k,v,*args,**kw):
            mask = args[0] if args else kw.get("attn_mask")
            dropout = args[1] if len(args)>1 else kw.get("dropout_p",0.)
            causal = args[2] if len(args)>2 else kw.get("is_causal",False)
            tensor_ok = all(isinstance(x,torch.Tensor) for x in (q,k,v))
            if tensor_ok:
                self.operations[str((str(q.dtype),tuple(q.shape),tuple(k.shape),tuple(v.shape)))] += 1
            eligible = (tensor_ok and len(args)<=3 and q.ndim==k.ndim==v.ndim==4
                        and q.dtype==k.dtype==v.dtype==torch.bfloat16
                        and q.shape[:2]==k.shape[:2] and k.shape==v.shape
                        and q.shape[-1]==k.shape[-1] and mask is None and not dropout
                        and not causal and not kw.get("enable_gqa",False)
                        and set(kw).issubset({"attn_mask","dropout_p","is_causal","scale","enable_gqa"})
                        and torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
            if not eligible:
                self.fallbacks["contract_or_fp32_sdpa_policy"] += 1
                return original(q,k,v,*args,**kw)
            self.hits += 1
            self.shapes[str((tuple(q.shape),tuple(k.shape),tuple(q.stride()),tuple(k.stride()),tuple(v.stride())))] += 1
            if self.method == "native":
                return original(q,k,v,*args,**kw)
            return bf16_sdpa(q,k,v,method=self.method,scale=kw.get("scale"))
        comfy.ops.scaled_dot_product_attention = invoke
        return self

    def __exit__(self, exc_type, exc, tb):
        import comfy.ops
        comfy.ops.scaled_dot_product_attention = self.original
        return False

    def report(self):
        return {"method":self.method,"eligible_hits":self.hits,
                "actual_replaced_gemms":2*self.hits if self.method=="fp32" else 0,
                "fallbacks":dict(self.fallbacks),"actual_sdpa_inputs":dict(self.operations),
                "shapes_strides":dict(self.shapes),
                "math_allow_fp16_bf16_reduction":torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
                "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
