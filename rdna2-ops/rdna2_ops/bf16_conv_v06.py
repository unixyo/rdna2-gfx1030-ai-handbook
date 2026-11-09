"""Inference-only BF16 Conv2d through FP32, preserving the caller's dtype.

Hook the functional convolution after ComfyUI's weight casting/offload boundary.
No cached weight copies, model mutation, or persistent framework patch. Default
bias ordering matches PyTorch MIOpen: round convolution to BF16, then add BF16 bias.
This is an experimental opt-in policy until shape and final-output checks pass.
"""
from collections import Counter
import hashlib
import os
from pathlib import Path
import torch
import torch.nn.functional as F


def bf16_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1,
                groups=1, *, bias_rounding="bf16", native=None):
    native = F.conv2d if native is None else native
    if input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("BF16 input and weight required")
    if bias is not None and bias.dtype != input.dtype:
        raise ValueError("Bias must use the input dtype")
    if bias_rounding not in ("bf16", "fp32"):
        raise ValueError("bias_rounding must be bf16 or fp32")
    # Disable a surrounding autocast so explicit FP32 operands stay FP32.
    with torch.autocast(device_type=input.device.type, enabled=False):
        bf = bias.float() if bias is not None and bias_rounding == "fp32" else None
        y = native(input.float(), weight.float(), bf, stride, padding, dilation, groups).bfloat16()
        if bias is not None and bias_rounding == "bf16":
            y.add_(bias.reshape(1,-1,1,1))
        return y


class BF16Conv2dMode:
    def __init__(self, method, *, metadata=False, bias_rounding="bf16"):
        if method not in ("native", "fp32"):
            raise ValueError("Only native observation or FP32 convolution supported")
        self.method = method
        self.bias_rounding = bias_rounding
        self.metadata = metadata
        self.hits = 0
        self.operations = Counter()
        self.fallbacks = Counter()
        self.original = None

    def __enter__(self):
        self.original = F.conv2d
        original = self.original
        def invoke(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
            tensor_ok = isinstance(input,torch.Tensor) and isinstance(weight,torch.Tensor)
            if tensor_ok:
                key = str((str(input.dtype),tuple(input.shape),tuple(weight.shape),
                           tuple(input.stride()),tuple(weight.stride()),stride,padding,dilation,groups))
                self.operations[key] += 1
            eligible = (tensor_ok and input.ndim == weight.ndim == 4
                and input.dtype == weight.dtype == torch.bfloat16
                and input.device.type == "cuda" and weight.device == input.device
                and (bias is None or (bias.dtype == input.dtype and bias.device == input.device))
                and not torch.is_grad_enabled() and groups == 1)
            if not eligible:
                self.fallbacks["dtype_device_grad_or_group_contract"] += 1
                return original(input,weight,bias,stride,padding,dilation,groups)
            self.hits += 1
            if self.method == "native":
                return original(input,weight,bias,stride,padding,dilation,groups)
            return bf16_conv2d(input,weight,bias,stride,padding,dilation,groups,
                               bias_rounding=self.bias_rounding,native=original)
        F.conv2d = invoke
        return self

    def __exit__(self, exc_type, exc, tb):
        F.conv2d = self.original
        return False

    def report(self):
        return {"method":self.method,"eligible_hits":self.hits,
                "actual_replaced_convolutions":self.hits if self.method=="fp32" else 0,
                "boundary":"torch.nn.functional.conv2d after caller weight cast",
                "bias_rounding":self.bias_rounding,"fallbacks":dict(self.fallbacks),
                "actual_convolution_inputs":dict(self.operations),"weight_cache":False,
                "miopen_find_mode":os.environ.get("MIOPEN_FIND_MODE"),
                "cudnn_benchmark":torch.backends.cudnn.benchmark,
                "inference_only":True,"group_scope":1,
                "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
