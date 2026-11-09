#!/usr/bin/env python3
"""Bounded gfx1030 INT8-QK experiment with end-to-end quantization costs."""
import argparse
import gc
import hashlib
import json
import re
import statistics
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from rdna2_ops.quantized_attention import quantized_attention, prepare_qk


def error(y, ref):
    a,b=y.double(),ref.double()
    diff=a-b
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return dict(finite=False,max_abs=None,rmse=None,relative_rmse=None,
                    cosine=None,operator_gate=False)
    rms=diff.square().mean().sqrt().item()
    rel=rms/max(b.square().mean().sqrt().item(),1.e-15)
    cos=(a.flatten()@b.flatten()/(a.norm()*b.norm()).clamp_min(1.e-30)).item()
    mx=diff.abs().max().item()
    return dict(finite=finite,max_abs=mx,rmse=rms,relative_rmse=rel,cosine=cos,
                operator_gate=finite and mx<=.02 and rel<=.03 and cos>=.999)


def timed_once(fn):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize();t=time.perf_counter();a.record();out=fn();b.record();b.synchronize()
    sample=dict(wall_ms=(time.perf_counter()-t)*1000,event_ms=a.elapsed_time(b))
    del out
    return sample


def measure_peak(fn):
    gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
    base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    out=fn();torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated()-base
    del out
    return peak


def measure_interleaved(functions, repeat=3):
    # Accuracy/ISA preparation already compiled every candidate. Warm up the
    # complete callables once more before recording any native/candidate time.
    for fn in functions.values():
        fn()
    torch.cuda.synchronize()
    samples={name:[] for name in functions}
    orders=[]
    for trial in range(repeat):
        order=list(functions) if trial % 2 == 0 else list(reversed(functions))
        orders.append(order)
        for position,name in enumerate(order):
            sample=timed_once(functions[name])
            sample.update(round=trial,position=position)
            samples[name].append(sample)
    measured={}
    for name,fn in functions.items():
        values=samples[name]
        measured[name]=dict(
            median_wall_ms=statistics.median(x['wall_ms'] for x in values),
            median_event_ms=statistics.median(x['event_ms'] for x in values),
            samples=values,
            # Preserve the allocation measurement as a separate phase. Its
            # allocator-cache flush occurs after every timing sample is saved.
            peak_additional_allocated_bytes=measure_peak(fn))
    return measured,orders


def native(q,k,v):
    with sdpa_kernel(SDPBackend.MATH): return F.scaled_dot_product_attention(q,k,v)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--lengths',type=int,nargs='+',default=[4097,8192])
    ap.add_argument('--dims',type=int,nargs='+',default=[64,128])
    ap.add_argument('--heads',type=int,default=4)
    ap.add_argument('--repeat',type=int,default=3)
    ap.add_argument('--compare-tiled',action='store_true',
                    help='Compare FP16 tiled attention on the same Q/K/V (D64: 32x64, D128: 32x32)')
    cfg=ap.parse_args()
    if cfg.output.exists(): raise FileExistsError(cfg.output)
    if cfg.repeat < 1: ap.error('--repeat must be positive')
    if cfg.compare_tiled:
        from rdna2_ops.tiled_attention import tiled_attention
    cfg.output.parent.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(20261106)
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    report=dict(device=torch.cuda.get_device_name(),arch=torch.cuda.get_device_properties(0).gcnArchName,
                torch=torch.__version__,seed=20261106,evidence_grade='A',
                prototype='Per-token symmetric INT8 QK, optional K centering, streaming softmax, FP32 accumulator PV',
                operator_thresholds=dict(max_abs=.02,relative_rmse=.03,cosine=.999,finite=True),
                image_thresholds_not_measured=dict(max_abs=2/255,rmse=.25/255),
                timing='Equal repeat count per method, round order alternates forward/reverse; raw samples and medians. Includes fresh K mean, Q/K quantization, allocations and dispatch; preparation/JIT/warmup excluded. Peak-memory phase follows all timing samples. Seeded synthetic inputs, no model quality claim.',
                repeat=cfg.repeat,
                peak_memory='PyTorch peak allocated increment, including output and quantization temporaries; excludes input/reference residency.',
                cases=[])
    if cfg.compare_tiled:
        report['tiled_comparison']=dict(input='same Q/K/V as native and INT8 candidates',
                                        compute_dtype='torch.float16',
                                        tiles_by_head_dim={'64':[32,64],'128':[32,32]})
    def save(): cfg.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for s in cfg.lengths:
      for d in cfg.dims:
        q,k,v=[torch.randn(1,cfg.heads,s,d,device='cuda',dtype=torch.float16) for _ in range(3)]
        row=dict(shape=list(q.shape),status='started',candidates=[]);report['cases'].append(row);save()
        try:
            estimate=13*cfg.heads*s*s
            free=torch.cuda.mem_get_info()[0]
            ref=None
            functions={}
            if estimate <= min(10*1024**3,.8*free):
                try:
                    ref=native(q,k,v)
                    row['native']=dict(status='available')
                    functions['native_math']=lambda:native(q,k,v)
                except torch.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    row['native']=dict(status='out_of_memory',error=str(exc))
            else:
                row['native']=dict(status='skipped_memory_budget',estimated_bytes=estimate,free_bytes=free)
            if ref is None:
                ref=native(q[:,:,:16],k,v)
                row['reference_scope']='first 16 query rows against native math; no full native speedup claim'
            else:
                row['reference_scope']='full output against native math'
            if cfg.compare_tiled:
                block_q,block_k=(32,64) if d==64 else (32,32)
                tiled=dict(status='started',block_q=block_q,block_k=block_k,
                           compute_dtype='torch.float16',timing_key='tiled_fp16',
                           reference_scope=row['reference_scope'])
                row['tiled_baseline']=tiled
                y=tiled_attention(q,k,v,block_q=block_q,block_k=block_k,
                                  compute_dtype=torch.float16)
                tiled['accuracy']=error(y if ref.shape==y.shape else y[:,:,:16],ref)
                tiled['full_output_finite']=bool(torch.isfinite(y).all())
                tiled['operator_gate']=tiled['accuracy']['operator_gate'] and tiled['full_output_finite']
                tiled['status']='available'
                # Insert after native so every forward/reverse round pairs
                # both baselines with the same four complete INT8 callables.
                functions['tiled_fp16']=lambda bq=block_q,bk=block_k:tiled_attention(
                    q,k,v,block_q=bq,block_k=bk,compute_dtype=torch.float16)
                del y;save()
            for smooth in (True,False):
              for pv in ('fp16','fp32'):
                candidate=dict(smooth_k=smooth,pv=pv);row['candidates'].append(candidate)
                y,compiled=quantized_attention(q,k,v,smooth_k=smooth,pv_dtype=pv,return_kernel=True)
                candidate['accuracy']=error(y if ref.shape==y.shape else y[:,:,:16],ref)
                candidate['full_output_finite']=bool(torch.isfinite(y).all())
                candidate['operator_gate']=candidate['accuracy']['operator_gate'] and candidate['full_output_finite']
                asm=compiled.asm.get('amdgcn','')
                patterns=('v_dot4_i32_i8','v_dot4c_i32_i8','v_dot2_f32_f16','v_dot2c_f32_f16',
                          'v_fma_f32','v_fmac_f32','v_mfma','v_wmma')
                candidate['isa_counts']={p:len(re.findall(p,asm)) for p in patterns}
                candidate['asm_sha256']=hashlib.sha256(asm.encode('utf-8')).hexdigest() if asm else None
                candidate['kernel_registers']=getattr(compiled,'n_regs',None)
                candidate['kernel_spills']=getattr(compiled,'n_spills',None)
                metadata=getattr(compiled,'metadata',None)
                candidate['kernel_shared_bytes']=(metadata.get('shared') if isinstance(metadata,dict)
                                                  else getattr(metadata,'shared',None))
                key=f'smooth_{int(smooth)}_pv_{pv}'
                candidate['timing_key']=key
                # Bind the current mode: every method is called later, after
                # all four candidates have completed compilation/validation.
                functions[key]=lambda smooth=smooth,pv=pv:quantized_attention(
                    q,k,v,smooth_k=smooth,pv_dtype=pv)
                del y;save()
            measurements,orders=measure_interleaved(functions,cfg.repeat)
            row['timing_round_order']=orders
            if 'native_math' in measurements:
                row['native'].update(status='measured',**measurements['native_math'])
            if cfg.compare_tiled:
                row['tiled_baseline'].update(status='measured',timing=measurements['tiled_fp16'])
            for candidate in row['candidates']:
                candidate['timing']=measurements[candidate['timing_key']]
                have_native=row['native']['status']=='measured'
                candidate['speedup']=(row['native']['median_wall_ms']/candidate['timing']['median_wall_ms']
                                      if have_native else None)
                candidate['speedup_event']=(row['native']['median_event_ms']/candidate['timing']['median_event_ms']
                                            if have_native else None)
                if cfg.compare_tiled:
                    baseline=measurements['tiled_fp16']
                    candidate['speedup_over_tiled']=baseline['median_wall_ms']/candidate['timing']['median_wall_ms']
                    candidate['speedup_event_over_tiled']=baseline['median_event_ms']/candidate['timing']['median_event_ms']
            save()
            # Isolate kernel arithmetic from quantization using a small query slice.
            q8,k8,qs,ks=prepare_qk(q,k,smooth_k=True)
            deq_q=q8.float()*qs[...,None];deq_k=k8.float()*ks[...,None]
            with sdpa_kernel(SDPBackend.MATH):
                deq_ref=F.scaled_dot_product_attention(deq_q[:,:,:16],deq_k,v.float())
            full=quantized_attention(q,k,v,smooth_k=True,pv_dtype='fp32')
            row['kernel_vs_dequantized_math_first16']=error(full[:,:,:16].float(),deq_ref)
            row['status']='measured'
            row['all_operator_gates_passed']=all(c['operator_gate'] for c in row['candidates'])
            del q8,k8,qs,ks,deq_q,deq_k,deq_ref,full,ref
        except Exception as exc:
            row.update(status='failed',error=repr(exc))
        save();print(json.dumps(row),flush=True)
        del q,k,v;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    with torch.inference_mode(): main()
