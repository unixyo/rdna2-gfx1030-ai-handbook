"""Explicit grouped W4A8/W4A16 inference kernels for gfx1030 / Triton 3.8.

Storage is signed 4-bit q and FP32 scale/zero point per output-row K group.
W=(q-zero)*scale.  This is linear INT4, never an IQ/codebook format.
The caller owns packed weights and invalidates them after weight/LoRA changes.
No model registration, operating setting, or global dispatch is changed.
dot8 preserves every A8 value using two signed dot8 operations and correction;
it is not W4A4.  W4A16 uses FP16 decode+rocBLAS or an explicit dot2 candidate.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import re

import torch
import triton
import triton.language as tl


@dataclass
class PackedInt4:
    data: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    sums: torch.Tensor
    n: int
    k: int
    group_size: int
    symmetric: bool
    layout: str = "nk"
    order: str = "low"

    def to(self, device):
        return PackedInt4(*(t.to(device) for t in
                            (self.data, self.scales, self.zeros, self.sums)),
                          self.n, self.k, self.group_size, self.symmetric,
                          self.layout, self.order)

    @property
    def bytes(self):
        return sum(t.numel()*t.element_size() for t in
                   (self.data, self.scales, self.zeros, self.sums))

    @property
    def logical_stride(self):
        return self.data.stride() if self.layout == "nk" else self.data.T.stride()


def pack_int4_cpu(w, *, group_size=64, symmetric=True, layout="nk", order="low"):
    """Independent CPU packing/reference, ties-to-even and explicit range."""
    if w.device.type != "cpu" or w.ndim != 2 or not w.is_floating_point():
        raise ValueError("CPU floating-point [N,K] weight required")
    n,k=w.shape
    if group_size not in (32,64,128) or k % group_size or layout not in ("nk","kn") or order not in ("low","high"):
        raise ValueError("K must be divisible by group size 32/64/128; valid layout/order required")
    if not torch.isfinite(w).all():
        raise ValueError("Finite weight required")
    x=w.float().reshape(n,k//group_size,group_size)
    if symmetric:
        scales=x.abs().amax(-1)/7
        scales=torch.where(scales==0,torch.ones_like(scales),scales)
        zeros=torch.zeros_like(scales,dtype=torch.int32)
        q=torch.round(x/scales[...,None]).clamp(-7,7).to(torch.int32)
    else:
        lo=torch.minimum(x.amin(-1),torch.zeros((),dtype=torch.float32))
        hi=torch.maximum(x.amax(-1),torch.zeros((),dtype=torch.float32))
        scales=(hi-lo)/15
        empty=scales==0
        scales=torch.where(empty,torch.ones_like(scales),scales)
        zu=torch.round(-lo/scales).clamp(0,15).to(torch.int32)
        # zero groups use q=zero=0 in signed-nibble coordinates.
        zeros=torch.where(empty,torch.zeros_like(zu),zu-8)
        q=(torch.round(x/scales[...,None])+zu[...,None]).clamp(0,15).to(torch.int32)-8
        q=torch.where(empty[...,None],torch.zeros_like(q),q)
    sums=q.sum(-1,dtype=torch.int32)
    q8=q.reshape(n,k//8,8).to(torch.int64)&15
    shifts=torch.arange(8,dtype=torch.int64)*4
    if order=="high": shifts=28-shifts
    words=(q8<<shifts).sum(-1).to(torch.int32)
    if layout=="kn": words=words.T.contiguous()
    return PackedInt4(words,scales.contiguous(),zeros.contiguous(),sums.contiguous(),
                      n,k,group_size,bool(symmetric),layout,order)


def unpack_int4_cpu(p, *, floating=True):
    words=p.data.cpu() if p.layout=="nk" else p.data.cpu().T
    shifts=torch.arange(8,dtype=torch.int64)*4
    if p.order=="high": shifts=28-shifts
    q=((words.to(torch.int64)[...,None]>>shifts)&15).to(torch.int32)
    q=torch.where(q>=8,q-16,q).reshape(p.n,p.k//p.group_size,p.group_size)
    centered=q-p.zeros.cpu()[...,None]
    return (centered.float()*p.scales.cpu()[...,None]).reshape(p.n,p.k) if floating else centered.reshape(p.n,p.k)


@triton.jit
def _round_even(x):
    flo=tl.floor(x)
    fraction=x-flo
    up=(fraction>0.5)|((fraction==0.5)&((flo.to(tl.int32)&1)!=0))
    return flo+up.to(tl.float32)


@triton.jit
def _pack_weight(W,P,S,Z,Sum, N:tl.constexpr,K:tl.constexpr,WN:tl.constexpr,WK:tl.constexpr,
                 PN:tl.constexpr,PK:tl.constexpr,G:tl.constexpr,SYM:tl.constexpr,REV:tl.constexpr):
    group=tl.program_id(0)
    row=group//(K//G)
    gi=group%(K//G)
    r=tl.arange(0,G)
    x=tl.load(W+row*WN+(gi*G+r)*WK).to(tl.float32)
    if SYM:
        scale=tl.div_rn(tl.max(tl.abs(x),0),7.0)
        scale=tl.where(scale==0,1.0,scale)
        zero=tl.full((),0,tl.int32)
        qi=tl.minimum(tl.maximum(_round_even(tl.div_rn(x,scale)),-7.0),7.0).to(tl.int32)
    else:
        lo=tl.minimum(tl.min(x,0),0.0)
        hi=tl.maximum(tl.max(x,0),0.0)
        step=tl.div_rn(hi-lo,15.0)
        empty=step==0
        scale=tl.where(empty,1.0,step)
        zu=tl.minimum(tl.maximum(_round_even(tl.div_rn(-lo,scale)),0.0),15.0).to(tl.int32)
        zero=tl.where(empty,0,zu-8)
        qi=tl.minimum(tl.maximum(_round_even(tl.div_rn(x,scale))+zu,0.0),15.0).to(tl.int32)-8
        qi=tl.where(empty,0,qi)
    qwords=tl.reshape(qi,(G//8,8)).to(tl.uint32)&15
    shifts=tl.arange(0,8)*4
    if REV: shifts=28-shifts
    words=tl.sum(qwords<<shifts[None,:],1)
    tl.store(P+row*PN+(gi*(G//8)+tl.arange(0,G//8))*PK,words)
    tl.store(S+group,scale)
    tl.store(Z+group,zero)
    tl.store(Sum+group,tl.sum(qi,0))


@triton.jit
def _unpack_weight(P,S,Z,O,N:tl.constexpr,K:tl.constexpr,G:tl.constexpr,
                   PN:tl.constexpr,PK:tl.constexpr,REV:tl.constexpr,BLOCK:tl.constexpr,
                   INTEGER:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=i//K; kk=i%K
    word=tl.load(P+row*PN+(kk//8)*PK,i<N*K,0).to(tl.uint32)
    shift=(kk%8)*4
    if REV: shift=28-shift
    q=((word>>shift)&15).to(tl.int32)
    q=tl.where(q>=8,q-16,q)
    z=tl.load(Z+row*(K//G)+kk//G,i<N*K,0)
    val=q-z
    if not INTEGER:
        s=tl.load(S+row*(K//G)+kk//G,i<N*K,0)
        val=val.to(tl.float32)*s
    tl.store(O+i,val,i<N*K)


@triton.jit
def _pack_a8(A,AL,AH,ASum,M:tl.constexpr,K:tl.constexpr,AM:tl.constexpr,AK:tl.constexpr,
             G:tl.constexpr,REV:tl.constexpr):
    p=tl.program_id(0); row=p//(K//G); gi=p%(K//G)
    r=tl.arange(0,G)
    a=tl.load(A+row*AM+(gi*G+r)*AK).to(tl.int32)
    lo=((a&15)^8).to(tl.uint32)
    hi=((a>>4)&15).to(tl.uint32)
    shifts=tl.arange(0,8)*4
    if REV: shifts=28-shifts
    low=tl.sum(tl.reshape(lo,(G//8,8))<<shifts[None,:],1)
    high=tl.sum(tl.reshape(hi,(G//8,8))<<shifts[None,:],1)
    index=row*(K//8)+gi*(G//8)+tl.arange(0,G//8)
    tl.store(AL+index,low); tl.store(AH+index,high)
    tl.store(ASum+p,tl.sum(a,0))


@triton.jit
def _int8_to_half(A,S,O,M:tl.constexpr,K:tl.constexpr,AM:tl.constexpr,AK:tl.constexpr,
                  BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=i//K; k=i%K
    a=tl.load(A+m*AM+k*AK,i<M*K,0).to(tl.float32)
    s=tl.load(S+m,i<M*K,0)
    tl.store(O+i,a*s,i<M*K)


@triton.jit
def _linear_dot4(A,P,AS,WS,WZ,Bias,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                 AM:tl.constexpr,AK:tl.constexpr,PN:tl.constexpr,PK:tl.constexpr,
                 G:tl.constexpr,REV:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,
                 GROUP_M:tl.constexpr,HAS_BIAS:tl.constexpr,INTEGER:tl.constexpr,
                 PREUNPACK:tl.constexpr):
    pid=tl.program_id(0); nm=tl.cdiv(M,BM); nn=tl.cdiv(N,BN)
    gid=pid//(GROUP_M*nn); first=gid*GROUP_M; gm=tl.minimum(nm-first,GROUP_M)
    mi=first+(pid%(GROUP_M*nn))%gm; ni=(pid%(GROUP_M*nn))//gm
    m=mi*BM+tl.arange(0,BM); n=ni*BN+tl.arange(0,BN)
    ml=m%M; nl=n%N; r=tl.arange(0,G)
    total=tl.zeros((BM,BN),tl.float32)
    exact=tl.zeros((BM,BN),tl.int32)
    for gi in range(K//G):
        kk=gi*G+r
        av=tl.load(A+ml[:,None]*AM+kk[None,:]*AK)
        if PREUNPACK:
            w=tl.load(P+nl[None,:]*PN+kk[:,None]*PK)
        else:
            word=tl.load(P+nl[None,:]*PN+(kk[:,None]//8)*PK).to(tl.uint32)
            shift=(kk%8)*4
            if REV: shift=28-shift
            q=((word>>shift[:,None])&15).to(tl.int32)
            q=tl.where(q>=8,q-16,q)
            z=tl.load(WZ+nl*(K//G)+gi)
            w=(q-z[None,:]).to(tl.int8)
        acc=tl.dot(av,w,out_dtype=tl.int32)
        if INTEGER: exact+=acc
        else:
            ws=tl.load(WS+nl*(K//G)+gi)
            total+=acc.to(tl.float32)*ws[None,:]
    if INTEGER: result=exact
    else:
        sa=tl.load(AS+ml)
        result=total*sa[:,None]
        if HAS_BIAS: result+=tl.load(Bias+nl).to(tl.float32)[None,:]
    tl.store(C+m[:,None]*N+n[None,:],result,(m[:,None]<M)&(n[None,:]<N))


@triton.jit
def _linear_dot4_split(A,P,AS,WS,WZ,Bias,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                       AM:tl.constexpr,AK:tl.constexpr,PN:tl.constexpr,PK:tl.constexpr,
                       G:tl.constexpr,BK:tl.constexpr,REV:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,
                       GROUP_M:tl.constexpr,HAS_BIAS:tl.constexpr,INTEGER:tl.constexpr):
    # Quantization G and the dot/load reduction width are independent. Preserve
    # the exact integer partial sum across sub-K chunks before one group scale.
    pid=tl.program_id(0); nm=tl.cdiv(M,BM); nn=tl.cdiv(N,BN)
    gid=pid//(GROUP_M*nn); first=gid*GROUP_M; gm=tl.minimum(nm-first,GROUP_M)
    mi=first+(pid%(GROUP_M*nn))%gm; ni=(pid%(GROUP_M*nn))//gm
    m=mi*BM+tl.arange(0,BM); n=ni*BN+tl.arange(0,BN)
    ml=m%M; nl=n%N; r=tl.arange(0,BK)
    total=tl.zeros((BM,BN),tl.float32)
    exact=tl.zeros((BM,BN),tl.int32)
    for gi in range(K//G):
        z=tl.load(WZ+nl*(K//G)+gi)
        acc=tl.zeros((BM,BN),tl.int32)
        for sub in tl.range(0,G,BK,loop_unroll_factor=1):
            kk=gi*G+sub+r
            av=tl.load(A+ml[:,None]*AM+kk[None,:]*AK)
            word=tl.load(P+nl[None,:]*PN+(kk[:,None]//8)*PK).to(tl.uint32)
            shift=(kk%8)*4
            if REV: shift=28-shift
            q=((word>>shift[:,None])&15).to(tl.int32)
            q=tl.where(q>=8,q-16,q)
            w=(q-z[None,:]).to(tl.int8)
            acc=tl.dot(av,w,acc,out_dtype=tl.int32)
        if INTEGER: exact+=acc
        else:
            ws=tl.load(WS+nl*(K//G)+gi)
            total+=acc.to(tl.float32)*ws[None,:]
    if INTEGER: result=exact
    else:
        sa=tl.load(AS+ml)
        result=total*sa[:,None]
        if HAS_BIAS: result+=tl.load(Bias+nl).to(tl.float32)[None,:]
    tl.store(C+m[:,None]*N+n[None,:],result,(m[:,None]<M)&(n[None,:]<N))


@triton.jit
def _sdot8(a,b,c):
    return tl.inline_asm_elementwise("v_dot8_i32_i4 $0, $1, $2, $3;",
                                    constraints="=v,v,v,v",args=[a,b,c],
                                    dtype=tl.int32,is_pure=True,pack=1)


@triton.jit
def _sdot4c(a,b,c):
    # The accumulator is tied to the output register: the compact VOP2 form.
    return tl.inline_asm_elementwise("v_dot4c_i32_i8 $0, $1, $2;",
                                    constraints="=v,v,v,0",args=[a,b,c],
                                    dtype=tl.int32,is_pure=True,pack=1)


@triton.jit
def _four_nibbles_to_bytes(word,REV:tl.constexpr,SECOND:tl.constexpr):
    # Only unpack one N-vector of words, before broadcasting to output rows.
    if REV:
        u=word if SECOND else word>>16
        unsigned=((u>>12)&15)|(u&0x00000f00)|((u<<12)&0x000f0000)|((u<<24)&0x0f000000)
    else:
        u=word>>16 if SECOND else word
        unsigned=(u&15)|((u<<4)&0x00000f00)|((u<<8)&0x000f0000)|((u<<12)&0x0f000000)
    # No byte-wise subtract with cross-byte borrow. Each bit-3 expands to 0xf8.
    return (unsigned&0x07070707)|((unsigned&0x08080808)*31)


@triton.jit
def _linear_dot4_words(A,P,AS,WS,WZ,Bias,C,
                       M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,AM:tl.constexpr,
                       PN:tl.constexpr,PK:tl.constexpr,G:tl.constexpr,REV:tl.constexpr,
                       BM:tl.constexpr,BN:tl.constexpr,SYM:tl.constexpr,
                       HAS_BIAS:tl.constexpr,INTEGER:tl.constexpr):
    pid=tl.program_id(0);nm=tl.cdiv(M,BM);nn=tl.cdiv(N,BN)
    group=pid//(8*nn);first=group*8;gm=tl.minimum(nm-first,8)
    mi=first+(pid%(8*nn))%gm;ni=(pid%(8*nn))//gm
    m=mi*BM+tl.arange(0,BM);n=ni*BN+tl.arange(0,BN)
    ml=m%M;nl=n%N
    total=tl.zeros((BM,BN),tl.float32);exact=tl.zeros((BM,BN),tl.int32)
    for gi in range(K//G):
        acc=tl.zeros((BM,BN),tl.int32)
        asum=tl.zeros((BM,),tl.int32)
        for j in tl.static_range(G//8):
            kp=gi*(G//8)+j
            a0=tl.load(A+ml*AM+2*kp)
            a1=tl.load(A+ml*AM+2*kp+1)
            word=tl.load(P+nl*PN+kp*PK).to(tl.uint32)
            b0=_four_nibbles_to_bytes(word,REV,False)
            b1=_four_nibbles_to_bytes(word,REV,True)
            acc=_sdot4c(a0[:,None],b0[None,:],acc)
            acc=_sdot4c(a1[:,None],b1[None,:],acc)
            if not SYM:
                asum=_sdot4c(a0,tl.full((),0x01010101,tl.int32),asum)
                asum=_sdot4c(a1,tl.full((),0x01010101,tl.int32),asum)
        if not SYM:
            wz=tl.load(WZ+nl*(K//G)+gi)
            acc-=asum[:,None]*wz[None,:]
        if INTEGER:exact+=acc
        else:
            ws=tl.load(WS+nl*(K//G)+gi)
            total+=acc.to(tl.float32)*ws[None,:]
    if INTEGER:result=exact
    else:
        sa=tl.load(AS+ml)
        result=total*sa[:,None]
        if HAS_BIAS:result+=tl.load(Bias+nl).to(tl.float32)[None,:]
    tl.store(C+m[:,None]*N+n[None,:],result,(m[:,None]<M)&(n[None,:]<N))


@triton.jit
def _linear_dot8(AL,AH,ASum,P,AS,WS,WZ,WSum,Bias,C,
                 M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,PN:tl.constexpr,PK:tl.constexpr,
                 G:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,
                 HAS_BIAS:tl.constexpr,INTEGER:tl.constexpr,SYM:tl.constexpr):
    mi=tl.program_id(0); ni=tl.program_id(1)
    m=mi*BM+tl.arange(0,BM); n=ni*BN+tl.arange(0,BN)
    ml=m%M; nl=n%N
    total=tl.zeros((BM,BN),tl.float32); exact=tl.zeros((BM,BN),tl.int32)
    for gi in range(K//G):
        low=tl.zeros((BM,BN),tl.int32); high=tl.zeros((BM,BN),tl.int32)
        for j in tl.static_range(G//8):
            kp=gi*(G//8)+j
            al=tl.load(AL+ml*(K//8)+kp)
            ah=tl.load(AH+ml*(K//8)+kp)
            w=tl.load(P+nl*PN+kp*PK)
            low=_sdot8(al[:,None],w[None,:],low)
            high=_sdot8(ah[:,None],w[None,:],high)
        sw=tl.load(WSum+nl*(K//G)+gi)
        acc=low+16*high+8*sw[None,:]
        if not SYM:
            az=tl.load(ASum+ml*(K//G)+gi)
            wz=tl.load(WZ+nl*(K//G)+gi)
            acc-=az[:,None]*wz[None,:]
        if INTEGER: exact+=acc
        else:
            ws=tl.load(WS+nl*(K//G)+gi)
            total+=acc.to(tl.float32)*ws[None,:]
    if INTEGER: result=exact
    else:
        sa=tl.load(AS+ml)
        result=total*sa[:,None]
        if HAS_BIAS: result+=tl.load(Bias+nl).to(tl.float32)[None,:]
    tl.store(C+m[:,None]*N+n[None,:],result,(m[:,None]<M)&(n[None,:]<N))


@triton.jit
def _linear_dot2(A,P,WS,WZ,Bias,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                 AM:tl.constexpr,AK:tl.constexpr,PN:tl.constexpr,PK:tl.constexpr,
                 G:tl.constexpr,REV:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
                 HAS_BIAS:tl.constexpr):
    pid=tl.program_id(0); nm=tl.cdiv(M,BM)
    m=(pid%nm)*BM+tl.arange(0,BM); n=(pid//nm)*BN+tl.arange(0,BN)
    ml=m%M; nl=n%N; r=tl.arange(0,BK)
    total=tl.zeros((BM,BN),tl.float32)
    for bi in range(K//BK):
        kk=bi*BK+r
        av=tl.load(A+ml[:,None]*AM+kk[None,:]*AK)
        word=tl.load(P+nl[None,:]*PN+(kk[:,None]//8)*PK).to(tl.uint32)
        shift=(kk%8)*4
        if REV: shift=28-shift
        q=((word>>shift[:,None])&15).to(tl.int32)
        q=tl.where(q>=8,q-16,q)
        z=tl.load(WZ+nl[None,:]*(K//G)+kk[:,None]//G)
        s=tl.load(WS+nl[None,:]*(K//G)+kk[:,None]//G)
        w=((q-z).to(tl.float32)*s).to(tl.float16)
        total=tl.dot(av,w,total,out_dtype=tl.float32)
    if HAS_BIAS: total+=tl.load(Bias+nl).to(tl.float32)[None,:]
    tl.store(C+m[:,None]*N+n[None,:],total,(m[:,None]<M)&(n[None,:]<N))


@lru_cache(None)
def _arch(index):
    return bool(torch.version.hip) and torch.cuda.get_device_properties(index).gcnArchName.split(":")[0]=="gfx1030"


def _eager(*tensors):
    if (torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing()
            or torch.is_autocast_enabled("cuda") or torch.jit.is_tracing()
            or torch._C._are_functorch_transforms_active()
            or torch.is_grad_enabled() and any(t.requires_grad for t in tensors)):
        raise ValueError("Explicit eager inference only")


def kernel_metadata(kernel):
    asm=kernel.asm.get("amdgcn","")
    return {"n_regs":getattr(kernel,"n_regs",None),"n_spills":getattr(kernel,"n_spills",None),
            "shared_bytes":getattr(getattr(kernel,"metadata",None),"shared",None),
            "instruction_counts":{name:len(re.findall(r"\b"+name+r"\b",asm)) for name in
                                  ("v_dot8_i32_i4","v_dot4c_i32_i8","v_dot4_i32_i8","v_dot2c_f32_f16","v_dot2_f32_f16")},
            "asm_sha256":hashlib.sha256(asm.encode()).hexdigest(),
            "kernel_name":getattr(kernel,"name",None)}


def pack_int4(w,*,group_size=64,symmetric=True,layout="nk",order="low"):
    if w.device.type!="cuda" or not _arch(w.device.index) or w.ndim!=2 or not w.is_floating_point():
        raise ValueError("Floating [N,K] weight on gfx1030 required")
    _eager(w)
    n,k=w.shape
    if group_size not in (32,64,128) or k%group_size or min(n,k)<=0 or n*k>=2**31:
        raise ValueError("Nonempty bounded shape and K divisible by 32/64/128 required")
    if layout not in ("nk","kn") or order not in ("low","high"):
        raise ValueError("Unknown packing layout/order")
    if min(w.stride())<=0: raise ValueError("Positive strides required")
    data=torch.empty((n,k//8) if layout=="nk" else (k//8,n),device=w.device,dtype=torch.int32)
    scales=torch.empty((n,k//group_size),device=w.device,dtype=torch.float32)
    zeros=torch.empty_like(scales,dtype=torch.int32); sums=torch.empty_like(zeros)
    p=PackedInt4(data,scales,zeros,sums,n,k,group_size,bool(symmetric),layout,order)
    _pack_weight[(n*(k//group_size),)](w,data,scales,zeros,sums,n,k,*w.stride(),
                                    *p.logical_stride,group_size,symmetric,order=="high",
                                    num_warps=4,enable_fp_fusion=False)
    return p


def _validate(p):
    if not isinstance(p,PackedInt4) or p.data.device.type!="cuda" or not _arch(p.data.device.index):
        raise ValueError("PackedInt4 on gfx1030 required")
    if (p.group_size not in (32,64,128) or min(p.n,p.k)<=0 or p.k%p.group_size or p.n*p.k>=2**31
            or p.layout not in ("nk","kn") or p.order not in ("low","high")
            or p.data.dtype!=torch.int32 or not p.data.is_contiguous()
            or tuple(p.data.shape)!=((p.n,p.k//8) if p.layout=="nk" else (p.k//8,p.n))):
        raise ValueError("Invalid packed shape/storage")
    for t,dtype in ((p.scales,torch.float32),(p.zeros,torch.int32),(p.sums,torch.int32)):
        if t.device!=p.data.device or t.dtype!=dtype or tuple(t.shape)!=(p.n,p.k//p.group_size) or not t.is_contiguous():
            raise ValueError("Invalid scale/zero/sum storage")
    _eager(p.data,p.scales)


def unpack_int4(p,*,dtype=torch.float16):
    _validate(p)
    if dtype not in (torch.float16,torch.float32,torch.int8): raise TypeError(dtype)
    out=torch.empty((p.n,p.k),device=p.data.device,dtype=dtype)
    _unpack_weight[(triton.cdiv(p.n*p.k,1024),)](p.data,p.scales,p.zeros,out,p.n,p.k,p.group_size,
                                              *p.logical_stride,p.order=="high",1024,dtype==torch.int8,
                                              num_warps=4,enable_fp_fusion=False)
    return out


def prepare_dot8_activation(a,p):
    """Exact A8 recoding. Return caller-owned (lo4 words, hi4 words, group sums)."""
    _validate(p)
    if a.dtype!=torch.int8 or a.ndim!=2 or a.shape[1]!=p.k or a.device!=p.data.device or min(a.stride())<=0:
        raise ValueError("Matching positive-stride A8[M,K] required")
    m,k=a.shape
    lo=torch.empty((m,k//8),device=a.device,dtype=torch.int32); hi=torch.empty_like(lo)
    sums=torch.empty((m,k//p.group_size),device=a.device,dtype=torch.int32)
    _pack_a8[(m*(k//p.group_size),)](a,lo,hi,sums,m,k,*a.stride(),p.group_size,p.order=="high",num_warps=4)
    return lo,hi,sums


def int8_activation_to_fp16(a,scale):
    m,k=a.shape
    out=torch.empty((m,k),device=a.device,dtype=torch.float16)
    _int8_to_half[(triton.cdiv(m*k,1024),)](a,scale,out,m,k,*a.stride(),1024,num_warps=4)
    return out


def int4_linear_a8(a,p,a_scale,*,bias=None,method="dot4",tile=(64,64),num_warps=4,
                   prepared=None,integer=False,return_kernel=False):
    """Compute (A8 * row_scale) @ reconstructed W4.T, FP32 output.

    dot4 unpacks in each GEMM tile. dot4_prepared accepts centered INT8 weights
    but keeps group scale accumulation. dot8 optionally accepts prepacked exact
    A8; otherwise its recoding is included in this call. Weight-only INT4 never
    means silently requantizing activations to four bits.
    """
    _validate(p); _eager(a)
    if (a.ndim!=2 or a.dtype!=torch.int8 or a.device!=p.data.device or a.shape[1]!=p.k
            or min(a.stride())<=0 or a.shape[0]*p.n>=2**31): raise ValueError("Matching A8[M,K] required")
    m,k=a.shape
    if a_scale.shape not in ((m,),(m,1)) or a_scale.device!=a.device or a_scale.dtype!=torch.float32 or not a_scale.is_contiguous():
        raise ValueError("Contiguous FP32 per-activation-row scale required")
    if bias is not None and (bias.shape!=(p.n,) or bias.device!=a.device or not bias.is_contiguous() or not bias.is_floating_point()):
        raise ValueError("Floating bias [N] required")
    if len(tile)!=2 or any(x not in (8,16,32,64,128) for x in tile) or num_warps not in (4,8):
        raise ValueError("Unsupported tile/warps")
    if k*128*15>=2**31: raise ValueError("Integer accumulator bound exceeded")
    out=torch.empty((m,p.n),device=a.device,dtype=torch.int32 if integer else torch.float32)
    bm,bn=tile
    if method=="dot4_words":
        # The ordinary rowwise quantizer produces contiguous A8. Copy a rare
        # padded/nonaligned view explicitly; its cost belongs to this call.
        av=a if a.stride(1)==1 and a.stride(0)%4==0 and a.storage_offset()%4==0 else a.contiguous()
        aw=av.view(torch.int32)
        kernel=_linear_dot4_words[(triton.cdiv(m,bm)*triton.cdiv(p.n,bn),)](
            aw,p.data,a_scale,p.scales,p.zeros,bias if bias is not None else a,out,
            m,p.n,k,aw.stride(0),*p.logical_stride,p.group_size,p.order=="high",bm,bn,p.symmetric,
            bias is not None,integer,num_warps=num_warps,num_stages=1,enable_fp_fusion=False)
    elif method=="dot8":
        packed=prepare_dot8_activation(a,p) if prepared is None else prepared
        if not isinstance(packed,tuple) or len(packed)!=3: raise ValueError("Dot8 prepared tuple required")
        al,ah,asum=packed
        if any(t.device!=a.device or t.dtype!=torch.int32 or not t.is_contiguous() for t in packed) or al.shape!=(m,k//8) or ah.shape!=al.shape or asum.shape!=(m,k//p.group_size):
            raise ValueError("Invalid prepared activation shape/dtype")
        kernel=_linear_dot8[(triton.cdiv(m,bm),triton.cdiv(p.n,bn))](
            al,ah,asum,p.data,a_scale,p.scales,p.zeros,p.sums,bias if bias is not None else a,out,
            m,p.n,k,*p.logical_stride,p.group_size,bm,bn,bias is not None,integer,p.symmetric,
            num_warps=num_warps,num_stages=1,enable_fp_fusion=False)
    elif method in ("dot4","dot4_prepared"):
        w=p.data
        stride=p.logical_stride
        if method=="dot4_prepared":
            if prepared is None or prepared.shape!=(p.n,k) or prepared.dtype!=torch.int8 or prepared.device!=a.device:
                raise ValueError("Centered INT8 prepared weight required")
            w=prepared; stride=w.stride()
        if bm<16 or bn<16: raise ValueError("tl.dot tile dimensions must be >=16")
        kernel=_linear_dot4[(triton.cdiv(m,bm)*triton.cdiv(p.n,bn),)](
            a,w,a_scale,p.scales,p.zeros,bias if bias is not None else a,out,m,p.n,k,*a.stride(),*stride,
            p.group_size,p.order=="high",bm,bn,8,bias is not None,integer,method=="dot4_prepared",
            num_warps=num_warps,num_stages=1,enable_fp_fusion=False)
    elif method=="dot4_split":
        if bm<16 or bn<16: raise ValueError("tl.dot tile dimensions must be >=16")
        kernel=_linear_dot4_split[(triton.cdiv(m,bm)*triton.cdiv(p.n,bn),)](
            a,p.data,a_scale,p.scales,p.zeros,bias if bias is not None else a,out,
            m,p.n,k,*a.stride(),*p.logical_stride,p.group_size,min(p.group_size,64),
            p.order=="high",bm,bn,8,bias is not None,integer,
            num_warps=num_warps,num_stages=1,enable_fp_fusion=False)
    else: raise ValueError(method)
    return (out,kernel) if return_kernel else out


def int4_linear_a16(a,p,*,bias=None,method="dequant",prepared=None,tile=(64,64,32),
                    num_warps=4,return_kernel=False):
    """FP16 activation/output. Decode+rocBLAS, prepared rocBLAS, fused dot2."""
    _validate(p); _eager(a)
    if a.dtype!=torch.float16 or a.ndim!=2 or a.shape[1]!=p.k or a.device!=p.data.device:
        raise ValueError("Matching FP16 activation [M,K] required")
    if min(a.stride())<=0 or a.shape[0]*p.n>=2**31: raise ValueError("Unsupported shape/stride")
    if bias is not None and (bias.dtype!=a.dtype or bias.shape!=(p.n,) or bias.device!=a.device):
        raise ValueError("FP16 bias [N] required")
    if method in ("dequant","prepared"):
        w=unpack_int4(p,dtype=torch.float16) if method=="dequant" else prepared
        if w is None or w.shape!=(p.n,p.k) or w.dtype!=torch.float16 or w.device!=a.device:
            raise ValueError("FP16 prepared weight required")
        result=torch.nn.functional.linear(a,w,bias)
        return (result,None) if return_kernel else result
    if method!="dot2": raise ValueError(method)
    if len(tile)!=3 or any(x not in (16,32,64,128) for x in tile) or p.k%tile[2]:
        raise ValueError("Supported dot2 tile required")
    bm,bn,bk=tile; m,k=a.shape
    out=torch.empty((m,p.n),device=a.device,dtype=torch.float16)
    kernel=_linear_dot2[(triton.cdiv(m,bm)*triton.cdiv(p.n,bn),)](
        a,p.data,p.scales,p.zeros,bias if bias is not None else a,out,m,p.n,k,*a.stride(),
        *p.logical_stride,p.group_size,p.order=="high",bm,bn,bk,bias is not None,
        num_warps=num_warps,num_stages=1,enable_fp_fusion=False)
    return (out,kernel) if return_kernel else out
