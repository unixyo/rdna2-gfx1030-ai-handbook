"""gfx1030 FP32 block-sparse attention, explicit forward-only research API.

The Sparge-inspired selector implements block mean, pairwise cosine safety gate
and top-CDF selection. It does not port NVIDIA Sage kernels, quantize Q/K, skip
PV with a warp threshold, or claim upstream performance. Spatial masks below
are an image-only Radial-inspired experiment, not the original temporal mask.

The streaming kernel allocates no token-by-token score tensor. A Boolean block
map is quadratic in the number of blocks; do not call total storage O(S) when
the block size is fixed. Outputs are renormalized over retained keys.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _sparse_fp32_stream(
    Q, K, V, Mask, O,
    L: tl.constexpr, S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    QB: tl.constexpr, QH: tl.constexpr, QS: tl.constexpr,
    KB: tl.constexpr, KH: tl.constexpr, KS: tl.constexpr,
    VB: tl.constexpr, VH: tl.constexpr, VS: tl.constexpr,
    NQB: tl.constexpr, NKB: tl.constexpr, MQ: tl.constexpr, MK: tl.constexpr,
    SCALE: tl.constexpr, CAUSAL_OFFSET: tl.constexpr, PREFIX_BIAS_OFFSET: tl.constexpr,
    BQ: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr, BP: tl.constexpr,
):
    qt, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // H, bh % H
    rows = qt * BQ + tl.arange(0, BQ)
    dk = tl.arange(0, BD)
    dv = tl.arange(0, D)
    kc = tl.arange(0, BK)
    mx = tl.full((BQ,), -float("inf"), tl.float32)
    den = tl.zeros((BQ,), tl.float32)
    num = tl.zeros((BQ, D), tl.float32)
    mask_q = (qt * BQ) // MQ
    for block in range(NKB):
        selected = tl.load(Mask + (bh * NQB + mask_q) * NKB + block)
        if selected:
            for subtile in range(MK // BK):
                start = block * MK + subtile * BK
                keys = start + kc
                logits = tl.zeros((BQ, BK), tl.float32)
                for dc in tl.range(0, D, BD, loop_unroll_factor=1):
                    dims = dc + dk
                    q = tl.load(Q + batch * QB + head * QH + rows[:, None] * QS + dims[None, :],
                                mask=rows[:, None] < L, other=0.)
                    k = tl.load(K + batch * KB + head * KH + keys[None, :] * KS + dims[:, None],
                                mask=keys[None, :] < S, other=0.)
                    logits = tl.dot(q, k, logits, input_precision="ieee")
                logits = logits * SCALE
                if PREFIX_BIAS_OFFSET >= 0:
                    logits += (keys[None, :] <= rows[:, None] + PREFIX_BIAS_OFFSET).to(tl.float32)
                valid = keys[None, :] < S
                if CAUSAL_OFFSET >= 0:
                    valid = valid & (keys[None, :] <= rows[:, None] + CAUSAL_OFFSET)
                logits = tl.where(valid, logits, -float("inf"))
                next_mx = tl.maximum(mx, tl.max(logits, 1))
                safe_mx = tl.where(next_mx == -float("inf"), 0., next_mx)
                old_weight = tl.exp(mx - safe_mx)
                p = tl.exp(logits - safe_mx[:, None])
                den = den * old_weight + tl.sum(p, 1)
                num = num * old_weight[:, None]
                for pv in tl.range(0, BK, BP, loop_unroll_factor=1):
                    pi = pv + tl.arange(0, BP)
                    pp = tl.gather(p, tl.broadcast_to(pi[None, :], (BQ, BP)), 1)
                    values = tl.load(V + batch * VB + head * VH + (start + pi[:, None]) * VS + dv[None, :],
                                     mask=(start + pi[:, None]) < S, other=0.)
                    num = tl.dot(pp, values, num, input_precision="ieee")
                mx = next_mx
    value = tl.where(den[:, None] > 0, num / den[:, None], 0.)
    tl.store(O + (bh * L + rows[:, None]) * D + dv[None, :], value, mask=rows[:, None] < L)


@dataclass(frozen=True)
class BlockMap:
    keep: torch.Tensor
    query_block: int = 128
    key_block: int = 64


@triton.jit
def _pool_mean_cosine(X, Mean, Cosine,
                     L: tl.constexpr,H: tl.constexpr,D: tl.constexpr,
                     XB: tl.constexpr,XH: tl.constexpr,XS: tl.constexpr,
                     N: tl.constexpr,BLOCK: tl.constexpr):
    block,bh=tl.program_id(0),tl.program_id(1)
    b,h=bh//H,bh%H
    rows=block*BLOCK+tl.arange(0,BLOCK)
    dims=tl.arange(0,D)
    x=tl.load(X+b*XB+h*XH+rows[:,None]*XS+dims[None,:],mask=rows[:,None]<L,other=0.)
    n=tl.minimum(BLOCK,L-block*BLOCK).to(tl.float32)
    mean=tl.sum(x,0)/n
    norm=tl.sqrt(tl.sum(x*x,1))
    normalized=x/tl.maximum(norm[:,None],1.e-12)
    total=tl.sum(normalized,0)
    diag=tl.sum(tl.sum(normalized*normalized,1),0)
    cosine=(tl.sum(total*total,0)-diag)/tl.maximum(n*(n-1.),1.)
    tl.store(Mean+(bh*N+block)*D+dims,mean)
    tl.store(Cosine+bh*N+block,cosine)


def sparse_attention(q, k, v, block_map, *, scale=None, causal_offset=None, prefix_bias_offset=None,
                     block_q=32, block_k=64, block_d=16, block_pv=16,
                     num_warps=8, return_kernel=False):
    """FP32 BHLD -> FP32 BHLD, explicit blocks; no dropout/GQA/backward/dispatch.

    causal_offset optionally keeps key_index <= query_index + offset for a
    logical prefix mask. Every valid query row
    must have at least one retained visible key. All-false rows return zero.
    Input finite/score finite is an explicit caller contract; timing excludes a
    global finite scan. Mask contents must be trusted Boolean metadata.

    prefix_bias_offset instead adds +1 to key_index <= query_index + offset.
    This explicit option preserves the observed Comfy attention_split behavior
    for a Boolean mask (scores += mask). It is not logical masking and cannot
    be combined with causal_offset. The model adapter leaves masked calls native.
    """
    ts = (q, k, v)
    if any(not isinstance(t, torch.Tensor) or t.ndim != 4 or t.dtype != torch.float32 for t in ts):
        raise ValueError("Expected FP32 BHLD Q/K/V")
    if q.device.type != "cuda" or not torch.version.hip or any(t.device != q.device for t in ts):
        raise ValueError("Expected one ROCm device")
    if k.shape != v.shape or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1] or q.shape[-1] not in (64, 128):
        raise ValueError("B/H/D must match; K/V shapes must match; D64/128")
    if min(*q.shape, *k.shape) <= 0 or any(t.stride(-1) != 1 or min(t.stride()) < 0 for t in ts):
        raise ValueError("Positive dimensions, unit channel stride and nonnegative strides required")
    if any(sum((n - 1) * s for n, s in zip(t.shape, t.stride())) >= 2**31 for t in ts):
        raise ValueError("Input address span exceeds 32-bit indexing")
    if torch.is_grad_enabled() and any(t.requires_grad for t in ts):
        raise ValueError("Inference only; use no_grad")
    if torch.compiler.is_compiling() or torch.jit.is_scripting() or torch.jit.is_tracing() or torch.is_autocast_enabled("cuda"):
        raise ValueError("Explicit eager FP32 inference only")
    if torch._C._are_functorch_transforms_active() or getattr(torch.autograd.forward_ad, "_current_level", -1) >= 0:
        raise ValueError("Automatic differentiation is not implemented")
    if (block_q not in (16, 32) or block_k not in (32, 64) or block_d not in (16, 32)
            or block_pv != 16 or num_warps not in (4, 8)):
        raise ValueError("Unsupported bounded tile")
    b, h, length, dim = q.shape
    keys = k.shape[-2]
    mq, mk, keep = block_map.query_block, block_map.key_block, block_map.keep
    if mq <= 0 or mk <= 0 or mq % block_q or mk % block_k:
        raise ValueError("Mask block sizes must be positive multiples of compute tile sizes")
    if keep.device != q.device or keep.dtype != torch.bool or not keep.is_contiguous():
        raise ValueError("Contiguous Boolean block map on the same device required")
    if tuple(keep.shape) != (b, h, triton.cdiv(length, mq), triton.cdiv(keys, mk)):
        raise ValueError("Block map geometry mismatch")
    if q.numel() >= 2**31 or keep.numel() >= 2**31 or b*h > 65535:
        raise ValueError("Output/map/grid bounds exceeded")
    if causal_offset is not None and (type(causal_offset) is not int or causal_offset < 0):
        raise ValueError("causal_offset must be a nonnegative integer")
    if prefix_bias_offset is not None and (type(prefix_bias_offset) is not int or prefix_bias_offset < 0):
        raise ValueError("prefix_bias_offset must be a nonnegative integer")
    if causal_offset is not None and prefix_bias_offset is not None:
        raise ValueError("Choose logical exclusion or additive prefix bias explicitly")
    sm = dim**-.5 if scale is None else float(scale)
    if not math.isfinite(sm):
        raise ValueError("Finite scale required")
    with torch.cuda.device(q.device):
        if torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0] != "gfx1030":
            raise ValueError("This measured experiment is restricted to gfx1030")
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("Graph capture is not supported")
        result = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        kernel = _sparse_fp32_stream[(triton.cdiv(length, block_q), b*h)](
            q, k, v, keep, result, length, keys, h, dim,
            *q.stride()[:3], *k.stride()[:3], *v.stride()[:3],
            keep.shape[-2], keep.shape[-1], mq, mk, sm,
            -1 if causal_offset is None else causal_offset,
            -1 if prefix_bias_offset is None else prefix_bias_offset,
            block_q, block_k, block_d, block_pv, num_warps=num_warps, num_stages=1)
    return (result, kernel) if return_kernel else result


def block_mean_cosine_reference(x, block):
    """Exact average pairwise cosine (off-diagonal), without a BxB matrix."""
    length = x.shape[-2]
    n = triton.cdiv(length, block)
    padded = F.pad(x, (0, 0, 0, n*block-length)).reshape(*x.shape[:2], n, block, x.shape[-1])
    counts = (length - torch.arange(n, device=x.device)*block).clamp(1, block).float()
    mean = padded.sum(-2) / counts[:, None]
    normalized = F.normalize(padded, p=2, dim=-1, eps=1e-12)
    norm_sum = normalized.sum(-2)
    diag = normalized.square().sum((-2, -1))
    cosine = (norm_sum.square().sum(-1) - diag) / (counts * (counts-1)).clamp_min(1)
    return mean, cosine, counts


def _block_mean_cosine(x,block):
    if x.device.type!="cuda":return block_mean_cosine_reference(x,block)
    b,h,length,d=x.shape;n=triton.cdiv(length,block)
    if x.dtype!=torch.float32 or d not in (64,128) or block not in (64,128) or x.stride(-1)!=1:
        raise ValueError("Fused selector pool requires FP32, D64/128, block64/128, channel stride1")
    mean=torch.empty((b,h,n,d),device=x.device,dtype=torch.float32)
    cosine=torch.empty((b,h,n),device=x.device,dtype=torch.float32)
    _pool_mean_cosine[(n,b*h)](x,mean,cosine,length,h,d,*x.stride()[:3],n,block,num_warps=8,num_stages=1)
    counts=(length-torch.arange(n,device=x.device)*block).clamp(1,block).float()
    return mean,cosine,counts


def sparge_map(q, k, *, mass=.95, similarity=.1, query_block=128, key_block=64, return_details=False):
    """Mean-pooled selective compression; top-CDF includes the crossing block.

    Low-cosine Q rows/K columns remain dense. A partial final key block is
    weighted by its token count. No offline tuning/training/model downloads.
    """
    if not 0 < mass <= 1 or not -1 <= similarity <= 1:
        raise ValueError("mass in (0,1], similarity in [-1,1] required")
    qm, qc, _ = _block_mean_cosine(q, query_block)
    km, kc, counts = _block_mean_cosine(k, key_block)
    logits = torch.matmul(qm, km.transpose(-1,-2)) * (q.shape[-1]**-.5)
    logits = logits + counts.log()[None, None, None, :]
    fixed_k = kc < similarity
    masked = logits.masked_fill(fixed_k.unsqueeze(-2), -torch.inf)
    # All-key-fixed is handled explicitly; no NaN probability enters selection.
    all_fixed = fixed_k.all(-1, keepdim=True).unsqueeze(-2)
    probs = torch.softmax(torch.where(all_fixed, torch.zeros_like(masked), masked), -1)
    vals, indices = probs.sort(dim=-1, descending=True, stable=True)
    chosen = (vals.cumsum(-1) - vals) < mass
    keep = torch.zeros_like(chosen).scatter_(-1, indices, chosen)
    keep |= fixed_k.unsqueeze(-2) | (qc < similarity).unsqueeze(-1)
    block_map = BlockMap(keep.contiguous(), query_block, key_block)
    if return_details:
        return block_map, {"query_cosine": qc, "key_cosine": kc, "predicted_probability": probs}
    return block_map


def fixed_map(q, k, *, fraction=.5, query_block=128, key_block=64):
    """Deterministic distributed key blocks plus local block and first/last."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    nq, nk = triton.cdiv(q.shape[-2],query_block), triton.cdiv(k.shape[-2],key_block)
    qi = torch.arange(nq,device=q.device)[:,None]
    ki = torch.arange(nk,device=q.device)[None,:]
    stride = max(1, int(round(1/fraction)))
    local = ((qi*query_block + query_block//2) * nk // q.shape[-2]).clamp(0,nk-1)
    keep = ((ki % stride)==0) | ((ki-local).abs()<=1) | (ki==nk-1)
    return BlockMap(keep.expand(q.shape[0],q.shape[1],nq,nk).contiguous(),query_block,key_block)


def radial_map(q, k, geometry, *, radius=8., query_block=128, key_block=64):
    """Image-only distance mask; all text retained, both image regions tested.

    A block is kept if ANY query/key token pair in the two blocks is within the
    configured Euclidean pixel-token radius. Boxes conservatively bound the
    flattened row spans. This may retain extra pixels; it never drops a nearby
    token due to a row-major block crossing. No temporal Radial claim is made.
    Geometry must come from the actual model's build_sequence, never filename.
    """
    if radius < 0 or not math.isfinite(radius):
        raise ValueError("Finite nonnegative radius required")
    tq = geometry["target"]
    length, keys = q.shape[-2], k.shape[-2]
    if tq["end"]-tq["start"] != length or geometry["key_length"] != keys:
        raise ValueError("Only captured target-query geometry is supported")
    nq,nk=triton.cdiv(length,query_block),triton.cdiv(keys,key_block)
    # Geometry construction is small CPU work; static maps may be cached by
    # callers. End-to-end timings include the first construction explicitly.
    keep = torch.zeros((nq,nk),dtype=torch.bool)
    def bbox(start,end,width):
        if start//width == (end-1)//width:
            return start//width,(end-1)//width,start%width,(end-1)%width
        return start//width,(end-1)//width,0,width-1
    for i in range(nq):
        qa,qb=i*query_block,min(length,(i+1)*query_block)
        y0,y1,x0,x1=bbox(qa,qb,tq["width"])
        for j in range(nk):
            ka,kb=j*key_block,min(keys,(j+1)*key_block)
            for text in geometry["text_segments"]:
                if max(ka,text["start"]) < min(kb,text["end"]):
                    keep[i,j]=True
            for image in geometry["image_segments"]:
                lo,hi=max(ka,image["start"]),min(kb,image["end"])
                if lo>=hi:
                    continue
                if image["height"] != tq["height"] or image["width"] != tq["width"]:
                    # No unvalidated resize/coordinate correspondence.
                    keep[i,j]=True
                    continue
                a,b,c,d=bbox(lo-image["start"],hi-image["start"],image["width"])
                dy=max(0,y0-b,a-y1);dx=max(0,x0-d,c-x1)
                if dx*dx+dy*dy <= radius*radius:
                    keep[i,j]=True
    return BlockMap(keep.to(q.device).expand(q.shape[0],q.shape[1],nq,nk).contiguous(),query_block,key_block)


def dense_alpha(q,k,v,*,scale=None,causal_offset=None,prefix_bias_offset=None,block_map=None):
    """Reference: original Qwen FP32 head-contiguous baddbmm alpha + softmax + PV."""
    b,h,length,d=q.shape;keys=k.shape[-2]
    qh,kh,vh=[t.contiguous().reshape(b*h,t.shape[-2],d) for t in (q,k,v)]
    scores=torch.baddbmm(torch.empty((b*h,length,keys),device=q.device,dtype=q.dtype),qh,kh.transpose(1,2),
                       beta=0,alpha=d**-.5 if scale is None else scale)
    if prefix_bias_offset is not None:
        bias=torch.arange(keys,device=q.device)[None,:] <= (torch.arange(length,device=q.device)[:,None]+prefix_bias_offset)
        scores.add_(bias)
    if causal_offset is not None:
        visible=torch.arange(keys,device=q.device)[None,:] <= (torch.arange(length,device=q.device)[:,None]+causal_offset)
        scores.masked_fill_(~visible,-torch.inf)
    if block_map is not None:
        qi=torch.arange(length,device=q.device)//block_map.query_block
        ki=torch.arange(keys,device=q.device)//block_map.key_block
        full=block_map.keep[:,:,qi[:,None],ki[None,:]].reshape(b*h,length,keys)
        scores.masked_fill_(~full,-torch.inf)
    probs=scores.softmax(-1)
    del scores
    return torch.bmm(probs,vh).reshape(b,h,length,d)


def map_statistics(q,k,maps,*,scale=None,causal_offset=None,prefix_bias_offset=None,query_block=128,key_block=64):
    """Exact FP32 post-softmax mass for every query, with bounded row chunks.

    Oracle top-CDF uses the mean probability per query block. Full block means
    and retained-mass quantiles expose lost rows rather than only a grand mean.
    Only a query_block x all-key score chunk is allocated at a time.
    """
    b,h,length,d=q.shape;keys=k.shape[-2];nk=triton.cdiv(keys,key_block)
    kh=k.contiguous().reshape(b*h,keys,d)
    masses=[];retained={name:[] for name in maps};entropy=[]
    key_idx=torch.arange(keys,device=q.device)
    for first in range(0,length,query_block):
        stop=min(first+query_block,length)
        qq=q[:,:,first:stop].contiguous().reshape(b*h,stop-first,d)
        scores=torch.baddbmm(torch.empty((b*h,stop-first,keys),device=q.device),qq,kh.transpose(1,2),
                           beta=0,alpha=d**-.5 if scale is None else scale)
        if prefix_bias_offset is not None:
            bias=key_idx[None,:] <= (torch.arange(first,stop,device=q.device)[:,None]+prefix_bias_offset)
            scores.add_(bias)
        if causal_offset is not None:
            visible=key_idx[None,:] <= (torch.arange(first,stop,device=q.device)[:,None]+causal_offset)
            scores.masked_fill_(~visible,-torch.inf)
        probs=scores.softmax(-1).reshape(b,h,stop-first,keys)
        entropy.append(-(probs*probs.clamp_min(1e-30).log()).sum(-1).cpu())
        blocks=F.pad(probs,(0,nk*key_block-keys)).reshape(b,h,stop-first,nk,key_block).sum(-1)
        masses.append(blocks.mean(-2).cpu())
        for name,bm in maps.items():
            row_ids=torch.arange(first,stop,device=q.device)//bm.query_block
            key_ids=key_idx//bm.key_block
            mk=bm.keep[:,:,row_ids[:,None],key_ids[None,:]]
            retained[name].append((probs*mk).sum(-1).cpu())
    block_mass=torch.stack(masses,dim=-2)
    order=block_mass.sort(-1,descending=True).values
    cdf=order.cumsum(-1)
    def quantiles(x):
        x=x.double().flatten()
        return {"min":x.min().item(),"p01":torch.quantile(x,.01).item(),"median":x.median().item(),
                "p99":torch.quantile(x,.99).item(),"max":x.max().item(),"mean":x.mean().item()}
    summary={"block_shape":[query_block,key_block],"full_probability_finite":bool(torch.isfinite(block_mass).all()),
             "oracle_density_for_block_mean_mass":{},"masks":{},"entropy_nats":quantiles(torch.cat(entropy,-1))}
    for mass in (.9,.95,.99,.999):
        required=((cdf-order)<mass).sum(-1).float()/nk
        summary["oracle_density_for_block_mean_mass"][str(mass)]=quantiles(required)
    for name,parts in retained.items():
        bm=maps[name]
        summary["masks"][name]={"block_density":bm.keep.float().mean().item(),
                                 "retained_probability_mass":quantiles(torch.cat(parts,-1))}
    return summary,block_mass
