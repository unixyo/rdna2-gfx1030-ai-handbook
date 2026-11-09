"""Unscaled e4m3fn/e5m2 weight GEMM experiments for gfx1030.

FP8 is a storage representation here. The software decoder consumes raw bytes;
no gfx1030 FP8 arithmetic instruction is assumed. Activation/output are FP16.
``upcast`` is the native full-weight FP16 cast followed by F.linear baseline.
``decode`` replaces that cast with a Triton software decoder; ``fused_dot2``
decodes weight tiles in-register. ``int8`` quantizes activation rows and decoded
weight rows to signed INT8, then uses this project's dot4/dequant GEMM.
Prepared weights are explicit caller-owned tensors, not an automatic pointer
cache. Discard/reprepare after weight, scale, LoRA, device or lifetime changes.
Finite inputs and finite unscaled FP8 values are required for GEMM/quantization.
"""
from functools import lru_cache
import torch
import triton
import triton.language as tl
from .gemm import _kernel_info
from .int8_gemm import int8_gemm_dequant


@triton.jit
def _decode_bits(raw, E5: tl.constexpr):
    u = raw.to(tl.uint32)
    sign = (u & 128) << 24
    if E5:
        exponent, mantissa = (u >> 2) & 31, u & 3
        normal = ((exponent + 112) << 23) | (mantissa << 21)
        small = (mantissa.to(tl.float32) * 0.0000152587890625).to(tl.uint32, bitcast=True)
        bits = tl.where(exponent == 0, small, normal)
        bits = tl.where(exponent == 31, 0x7f800000 | (mantissa << 21), bits)
    else:
        exponent, mantissa = (u >> 3) & 15, u & 7
        normal = ((exponent + 120) << 23) | (mantissa << 20)
        small = (mantissa.to(tl.float32) * 0.001953125).to(tl.uint32, bitcast=True)
        bits = tl.where(exponent == 0, small, normal)
        bits = tl.where((exponent == 15) & (mantissa == 7), 0x7fc00000, bits)
    return (bits | sign).to(tl.float32, bitcast=True)


@triton.jit
def _decode_weight(W, Out, N: tl.constexpr, K: tl.constexpr,
                   WN: tl.constexpr, WK: tl.constexpr,
                   E5: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    raw = tl.load(W + (i // K) * WN + (i % K) * WK, i < N*K, 0)
    tl.store(Out + i, _decode_bits(raw, E5), i < N*K)


@triton.jit
def _fp8_linear(A, W, Bias, Out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                AM: tl.constexpr, AK: tl.constexpr, WN: tl.constexpr, WK: tl.constexpr,
                E5: tl.constexpr, BIAS: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    rows = (pid % nm) * BM + tl.arange(0, BM)
    cols = (pid // nm) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    ap = A + rows[:, None] * AM + inner[None, :] * AK
    wp = W + cols[None, :] * WN + inner[:, None] * WK
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + inner
        a = tl.load(ap, (rows[:, None] < M) & (kk[None, :] < K), 0)
        raw = tl.load(wp, (kk[:, None] < K) & (cols[None, :] < N), 0)
        w = _decode_bits(raw, E5).to(tl.float16)
        acc = tl.dot(a, w, acc, out_dtype=tl.float32)
        ap += BK * AK
        wp += BK * WK
    if BIAS:
        acc += tl.load(Bias + cols, cols < N, 0).to(tl.float32)[None, :]
    tl.store(Out + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _quant_rows(X, Q, S, ROWS: tl.constexpr, K: tl.constexpr,
                XM: tl.constexpr, XK: tl.constexpr,
                FP8: tl.constexpr, E5: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * XM + col * XK, col < K, 0)
    if FP8:
        x = _decode_bits(x, E5)
    else:
        x = x.to(tl.float32)
    step = tl.maximum(tl.max(tl.abs(x), 0) / 127.0, 1.0e-20)
    z = x / step
    q = tl.where(z >= 0, tl.floor(z + 0.5), tl.ceil(z - 0.5))
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
    tl.store(Q + row * K + col, q, col < K)
    tl.store(S + row, step)


@lru_cache(None)
def _gfx1030(index):
    return bool(torch.version.hip) and torch.cuda.get_device_properties(index).gcnArchName.split(":")[0] == "gfx1030"


def _weight(w):
    if w.ndim != 2 or w.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise TypeError("Weight must be a plain unscaled FP8 [N,K] tensor")
    if w.device.type != "cuda" or not _gfx1030(w.device.index):
        raise ValueError("This explicit implementation targets HIP gfx1030")
    if min(w.shape) <= 0 or min(w.stride()) <= 0 or sum((n-1)*s for n,s in zip(w.shape,w.stride())) >= 2**31:
        raise ValueError("Nonempty positive strides and signed-32-bit address span required")


def decode_fp8_weight(w, *, out_dtype=torch.float16, metadata=None):
    """Decode each storage value; all finite FP8 values are exact in FP16.

    Decoder tests include all 256 byte patterns and check NaN/Inf classes.
    Finite-GEMM callers must still reject nonfinite input values.
    """
    _weight(w)
    if out_dtype not in (torch.float16, torch.float32):
        raise TypeError("Decode output must be FP16 or FP32")
    n,k=w.shape
    out=torch.empty((n,k),dtype=out_dtype,device=w.device)
    kernel=_decode_weight[(triton.cdiv(n*k,1024),)](w.view(torch.uint8),out,n,k,*w.stride(),
                                                  w.dtype==torch.float8_e5m2,1024,num_warps=4)
    if metadata is not None:
        metadata.update(_kernel_info(kernel), method="software_decode", output_dtype=str(out_dtype))
    return out


def prepare_int8_weight(w):
    """Return caller-owned (INT8 [N,K], FP32 per-output scale), no hidden cache."""
    _weight(w)
    n,k=w.shape
    if k>32768:
        raise ValueError("This bounded row quantizer supports K<=32768")
    packed=torch.empty((n,k),dtype=torch.int8,device=w.device)
    scales=torch.empty(n,dtype=torch.float32,device=w.device)
    _quant_rows[(n,)](w.view(torch.uint8),packed,scales,n,k,*w.stride(),True,
                      w.dtype==torch.float8_e5m2,triton.next_power_of_2(k),num_warps=4)
    return packed,scales


def fp8_weight_gemm(a,w,*,bias=None,method="upcast",prepared=None,
                    tile=(64,64,32),num_warps=4,metadata=None):
    """FP16 A[M,K] times unscaled FP8 weight[N,K], returning FP16 [M,N].

    Methods: upcast, decode, prepared_fp16, fused_dot2, int8, prepared_int8.
    For prepared modes the caller must provide a current converted weight or
    (INT8 weight, scale). Preparation is separate and must be counted when not
    amortized. Original weights are never changed. Only explicit eager inference.
    """
    _weight(w)
    if a.ndim!=2 or a.shape[1]!=w.shape[1] or a.dtype!=torch.float16 or a.device!=w.device:
        raise ValueError("FP16 [M,K] activation and matching FP8 [N,K] weight required")
    if (torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing()
            or torch.is_autocast_enabled("cuda") or torch.jit.is_tracing() or torch.jit.is_scripting()
            or torch._C._are_functorch_transforms_active()
            or getattr(torch.autograd.forward_ad,"_current_level",-1)>=0
            or torch.is_grad_enabled() and (a.requires_grad or w.requires_grad)):
        raise ValueError("Only explicit eager inference is supported")
    if min(a.shape)<=0 or min(a.stride())<=0 or sum((n-1)*s for n,s in zip(a.shape,a.stride()))>=2**31:
        raise ValueError("Nonempty positive-stride activation required")
    m,k=a.shape; n=w.shape[0]
    if m*n>=2**31:
        raise ValueError("Output exceeds supported indexing span")
    if bias is not None and (bias.shape!=(n,) or bias.device!=a.device or bias.dtype!=a.dtype or not bias.is_contiguous()
                             or torch.is_grad_enabled() and bias.requires_grad):
        raise ValueError("Bias must be a contiguous FP16 vector of length N")
    if method=="upcast":
        return torch.nn.functional.linear(a,w.to(a.dtype),bias)
    if method=="decode":
        return torch.nn.functional.linear(a,decode_fp8_weight(w),bias)
    if method=="prepared_fp16":
        if prepared is None or prepared.shape!=w.shape or prepared.device!=w.device or prepared.dtype!=a.dtype:
            raise ValueError("Provide the current FP16 prepared weight")
        return torch.nn.functional.linear(a,prepared,bias)
    if method in ("int8","prepared_int8"):
        if method=="prepared_int8" and (not isinstance(prepared,tuple) or len(prepared)!=2):
            raise ValueError("Provide the current (INT8 weight, FP32 row scale)")
        qw,sw=prepare_int8_weight(w) if method=="int8" else prepared
        if (qw.shape!=w.shape or qw.device!=w.device or qw.dtype!=torch.int8
                or sw.shape!=(n,) or sw.device!=w.device or sw.dtype!=torch.float32):
            raise ValueError("Invalid prepared INT8 weight/scales")
        qa=torch.empty((m,k),dtype=torch.int8,device=a.device)
        sa=torch.empty(m,dtype=torch.float32,device=a.device)
        if k>32768:
            raise ValueError("This bounded row quantizer supports K<=32768")
        _quant_rows[(m,)](a,qa,sa,m,k,*a.stride(),False,False,
                          triton.next_power_of_2(k),num_warps=4)
        result=int8_gemm_dequant(qa,qw.T,sa,sw,bias,
                                tile=(64,128,64),num_warps=4,num_stages=2,group_m=8,
                                return_kernel=metadata is not None)
        if metadata is not None:
            result,kernel=result
            metadata.update(_kernel_info(kernel),method=method)
        return result.to(a.dtype)
    if method!="fused_dot2":
        raise ValueError("Unknown method: "+method)
    if len(tile)!=3 or any(x not in (16,32,64,128) for x in tile) or num_warps not in (4,8):
        raise ValueError("Unsupported tile or warp count")
    bm,bn,bk=tile
    result=torch.empty((m,n),dtype=a.dtype,device=a.device)
    kernel=_fp8_linear[(triton.cdiv(m,bm)*triton.cdiv(n,bn),)](
        a,w.view(torch.uint8),bias if bias is not None else a,result,m,n,k,*a.stride(),*w.stride(),
        w.dtype==torch.float8_e5m2,bias is not None,bm,bn,bk,
        num_warps=num_warps,num_stages=1,matrix_instr_nonkdim=0)
    if metadata is not None:
        metadata.update(_kernel_info(kernel),method=method,tile=list(tile),num_warps=num_warps)
    return result
