"""Owned-process MOSS W4 diagnostic, never a production/default patch.

Native W8A8 -> W4 grouped integer reference -> W4 custom dot4 comparison.
MOSS's existing input activation, Hadamard, A8 row quantization, bias/residual
contracts are reused through its v0.5 scoped adapter. The original checkpoint
is untouched. Packed weights are created per call and not retained beyond the
call: no borrowed staging tensor or mutable pointer-key cache escapes its owner.
This diagnostic measures quality; per-call packing is included in stage time.
"""
from collections import Counter
from contextlib import contextmanager
import hashlib
from pathlib import Path
import torch

from rdna2_ops.int4_weight_gemm_v07 import pack_int4,unpack_int4,int4_linear_a8


def grouped_integer_reference(a,p,sa,bias=None):
    """Independent torch FP32 GEMM computes each integer dot exactly.

    Within one group, |sum A8*(q-z)| <= 128*15*128 = 245760 < 2**24.
    All operands/products/partial integer sums are exactly representable FP32.
    Then explicit torch mul/add reproduce the declared grouped scale order.
    No custom int4 GEMM code is used in this reference.
    """
    q=unpack_int4(p,dtype=torch.int8)
    total=torch.zeros((a.shape[0],p.n),device=a.device,dtype=torch.float32)
    for gi in range(p.k//p.group_size):
        k0=gi*p.group_size;k1=k0+p.group_size
        exact=torch.mm(a[:,k0:k1].float(),q[:,k0:k1].float().T)
        total=total+exact*p.scales[:,gi][None,:]
    y=total*sa.reshape(-1,1)
    return y if bias is None else y+bias.float().reshape(1,-1)


@contextmanager
def activate(mode,spec,state,bound,out):
    if spec.get('model')!='moss':raise ValueError('This quality diagnostic owns only the existing MOSS request')
    import rdna2_ops.int8_gemm as backend
    if mode=='int8_fused':
        with backend.moss_int8_gemm_candidate(**dict(spec.get('int8_config',{}))) as stats:
            stats['scope']='Actual requested MOSS v0.5 INT8 fused baseline'
            state['candidate']=stats
            yield
        return
    if mode not in ('int4_reference','int4_dot4','int4_dot8','int4_dot4_words'):
        raise ValueError(mode)
    opts=dict(spec.get('int4_config',{}))
    group=int(opts.get('group_size',32));sym=bool(opts.get('symmetric',False))
    layout=opts.get('layout','kn');order=opts.get('order','low')
    tile=tuple(opts.get('tile',[32,64]))
    old=backend.int8_gemm_dequant
    stats={'mode':mode,'hits':0,'shapes':Counter(),'bias_dtype':Counter(),
           'max_transient_packed_bytes':0,'group_size':group,'symmetric':sym,
           'layout':layout,'order':order,'tile':list(tile),'preparations':0,
           'source':'original checkpoint W8 reconstructed in FP32 within call; no new checkpoint',
           'activation':'existing A8 and row scale preserved',
           'weight_lifetime':'fresh per-call owned W4; no persistent pointer cache',
           'preparation_in_stage_time':True,'performance_adoption':False,
           'same_stored_reference':'sequential group FP32 torch.mm on exact integers, explicit scale mul/add; integer sum bound 245760',
           'protected_original_export_restored':False}
    root=Path(__file__).resolve().parent
    stats['source_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__),root/'rdna2_ops/int4_weight_gemm_v07.py',root/'rdna2_ops/int8_gemm.py')}

    def call(a,b,sa,bs,bias=None,**kwargs):
        if a.dtype!=torch.int8 or b.dtype!=torch.int8 or b.shape[0]!=a.shape[1]:
            raise ValueError('Existing MOSS quantized operand contract changed')
        w=b.T.float()*bs.reshape(1 if bs.numel()==1 else -1,1)
        p=pack_int4(w,group_size=group,symmetric=sym,layout=layout,order=order)
        stats['hits']+=1;stats['preparations']+=1
        stats['shapes'][str((a.shape[0],b.shape[1],a.shape[1]))]+=1
        stats['bias_dtype']['none' if bias is None else str(bias.dtype)]+=1
        stats['max_transient_packed_bytes']=max(stats['max_transient_packed_bytes'],p.bytes)
        if mode=='int4_reference':return grouped_integer_reference(a,p,sa,bias)
        return int4_linear_a8(a,p,sa,bias=bias,method=mode.removeprefix('int4_'),tile=tile)

    with backend.moss_int8_gemm_candidate(**dict(spec.get('int8_config',{}))) as outer_stats:
        backend.int8_gemm_dequant=call
        state['candidate']=stats
        try:yield
        finally:
            if backend.int8_gemm_dequant is not call:
                raise RuntimeError('Research export changed concurrently')
            backend.int8_gemm_dequant=old
            stats['outer_adapter_hits']=outer_stats['hits']
            stats['outer_adapter_fallbacks']=outer_stats['fallbacks']
            stats['protected_original_export_restored']=True
            if stats['hits']==0:
                raise RuntimeError('W4 diagnostic replaced no calls; no model success claim is possible')
