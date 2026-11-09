#!/usr/bin/env python3
"""Serial gfx1030 W4 kernels: predeclared gates, actual weights, real Linear shapes.

Run only with the root's exclusive GPU lease. No operating model/source is edited.
Output JSON is append-progress and never silently overwrites a prior experiment.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch
import triton
from gpu_guard_v06 import GpuAudit
from rdna2_ops.int4_weight_gemm_v07 import (
    PackedInt4,pack_int4_cpu,unpack_int4_cpu,pack_int4,unpack_int4,
    prepare_dot8_activation,int4_linear_a8,int4_linear_a16,
    int8_activation_to_fp16,kernel_metadata)
from rdna2_ops.int8_gemm import int8_gemm_dequant

ROOT=Path(__file__).resolve().parent
DEV=Path('~/DevSpace')
SHAPES={
 'moss_proj':('moss',1061,3072,3072,'layers.0.proj.o.weight'),
 'moss_mlp_up':('moss',1061,16384,3072,'layers.0.proj.ffn_up_gate.weight'),
 'moss_mlp_down':('moss',1061,3072,8192,'layers.0.proj.ffn_down.weight'),
 'qwen_proj':('qwen',2400,4096,4096,'transformer_blocks.0.attn.to_q.weight'),
 'qwen_mlp_up':('qwen',2400,24576,4096,'transformer_blocks.0.img_mlp.gate_up.weight'),
 'qwen_mlp_down':('qwen',2400,4096,12288,'transformer_blocks.0.img_mlp.out.weight'),
 'anima_proj':('anima',8192,2048,2048,'net.blocks.0.self_attn.q_proj.weight'),
 'anima_mlp_up':('anima',8192,8192,2048,'net.blocks.0.mlp.layer1.weight'),
 'anima_mlp_down':('anima',8192,2048,8192,'net.blocks.0.mlp.layer2.weight'),
}
MODELS={
 'moss':DEV/'moss-image-rdna2-lab01/models/diffusion_models/moss_image_sft_int8_convrot.safetensors',
 'qwen':DEV/'qwen-image-21-rdna2-lab01/models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors',
 'anima':DEV/'anima-rdna2-lab01/models/diffusion_models/anima-base-v1.0.safetensors',
}


def digest(x):
    return hashlib.sha256(x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def metric(y,ref):
    a=y.detach().double().cpu(); b=ref.detach().double().cpu()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return {'finite':False,'max_abs':None,'rmse':None,'reference_max_abs':None,'relative_l2':None,
                'nonfinite_output_count':int((~torch.isfinite(a)).sum()),
                'nonfinite_reference_count':int((~torch.isfinite(b)).sum())}
    d=a-b
    return {'finite':True,'max_abs':d.abs().max().item(),
            'rmse':d.square().mean().sqrt().item(),
            'reference_max_abs':b.abs().max().item(),
            'relative_l2':(d.square().sum()/b.square().sum().clamp_min(1e-300)).sqrt().item()}


def gate(e,half):
    rtol=2e-3 if half else 1e-4; atol=2e-3 if half else 1e-4
    return e['finite'] and e['relative_l2']<=(1e-3 if half else 2e-5) and e['max_abs']<=atol+rtol*e['reference_max_abs']


def word_info(p):
    return dict(group_size=p.group_size,symmetric=p.symmetric,layout=p.layout,order=p.order,
                packed_bytes=p.data.numel()*4,scale_bytes=p.scales.numel()*4,
                zero_bytes=p.zeros.numel()*4,sum_bytes=p.sums.numel()*4,total_resident_bytes=p.bytes,
                logical_stride=list(p.logical_stride))


def record_kernel(kernel,outdir,label):
    if kernel is None:return None
    md=kernel_metadata(kernel)
    path=outdir/'isa'/f"{label}_{md['asm_sha256'][:12]}.amdgcn"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():path.write_text(kernel.asm.get('amdgcn',''))
    md['asm_file']=str(path.relative_to(outdir))
    return md


def smoke(report,outdir,save,*,groups=(32,64,128),integer_methods=('dot4','dot8','dot4_words'),
          float_methods=('dot4','dot8','dot4_words'),half_methods=('dequant','dot2')):
    """All packing combinations; small integer oracle and representative floating cases."""
    gen=torch.Generator().manual_seed(20261107)
    m,n,k=17,19,384
    a=torch.randint(-128,128,(m,k),dtype=torch.int8,generator=gen)
    a[0]=torch.arange(k).remainder(256).to(torch.int8)
    w=torch.randn((n,k),generator=gen)*.2
    w[0]=0
    w[1]=torch.arange(k).remainder(16).float()-8
    w[2]=torch.arange(k).remainder(15).float()-7
    ag=a.cuda(); sa=(torch.arange(1,m+1).float()*.007).cuda()
    report['smoke']=[]
    for g in groups:
      for sym in (True,False):
       for layout in ('nk','kn'):
        for order in ('low','high'):
            label=f'g{g}_{"sym" if sym else "aff"}_{layout}_{order}'
            row={'key':label,'shape':[m,n,k],'failures':[]}
            report['smoke'].append(row)
            try:
                cpu=pack_int4_cpu(w,group_size=g,symmetric=sym,layout=layout,order=order)
                p=pack_int4(w.cuda(),group_size=g,symmetric=sym,layout=layout,order=order)
                row['packing']=word_info(p)
                row['cpu_gpu_pack_equal']={name:torch.equal(getattr(cpu,name),getattr(p,name).cpu()) for name in ('data','scales','zeros','sums')}
                qi=unpack_int4_cpu(cpu,floating=False)
                qgpu=unpack_int4(p,dtype=torch.int8)
                row['decoder_exact']=torch.equal(qgpu.cpu().to(torch.int32),qi)
                oracle=a.to(torch.int64)@qi.T.to(torch.int64)
                row['integer']={}
                for method in integer_methods:
                    y,kern=int4_linear_a8(ag,p,sa,method=method,tile=(16,32),integer=True,return_kernel=True)
                    exact=torch.equal(y.cpu().to(torch.int64),oracle)
                    row['integer'][method]={'byte_value_exact':exact,'max_abs':(y.cpu().to(torch.int64)-oracle).abs().max().item(),
                                             'kernel':record_kernel(kern,outdir,f'{method}_i32_{label}')}
                    if not exact:row['failures'].append(method+'_integer')
                # No giant model tensor is needed for the independent FP64 sample.
                if layout=='nk' and order=='low':
                    w64=qi.double()*cpu.scales.double().repeat_interleave(g,dim=1)
                    a64=a.double()*sa.cpu().double()[:,None]
                    ref=a64@w64.T
                    row['floating']={}
                    for method in float_methods:
                        y,kern=int4_linear_a8(ag,p,sa,method=method,tile=(16,32),return_kernel=True)
                        e=metric(y,ref)
                        row['floating'][method]={'error':e,'gate':gate(e,False),'kernel':record_kernel(kern,outdir,f'{method}_f32_{label}')}
                        if not gate(e,False):row['failures'].append(method+'_float')
                    ah=(a64*.1).half().cuda(); refh=ah.cpu().double()@w64.T
                    for method in half_methods:
                        y,kern=int4_linear_a16(ah,p,method=method,tile=(16,32,32),return_kernel=True)
                        e=metric(y,refh)
                        row['floating'][method]={'error':e,'gate':gate(e,True),'kernel':record_kernel(kern,outdir,f'{method}_{label}')}
                        if not gate(e,True):row['failures'].append(method+'_float')
                if not all(row['cpu_gpu_pack_equal'].values()):row['failures'].append('packing')
                if not row['decoder_exact']:row['failures'].append('decode')
            except Exception:
                row['failures'].append(traceback.format_exc())
            save()
            print(json.dumps({'smoke':label,'failures':row['failures']}),flush=True)
            if row['failures']:
                raise RuntimeError('Smoke failed; inspect raw result, do not benchmark yet')
    report['smoke_passed']=all(not r['failures'] for r in report['smoke'])


def reference_check(report,outdir,save):
    from int4_model_switches_v07 import grouped_integer_reference
    report['reference_check']=[]
    for m,n,k in ((37,53,384),(19,3072,3072),(1061,3072,3072)):
        torch.manual_seed(20261107)
        a=torch.randint(-128,128,(m,k),device='cuda',dtype=torch.int8)
        sa=torch.rand(m,device='cuda')*.02+.001
        w=torch.randn((n,k),device='cuda')*.01
        bias=torch.randn(n,device='cuda')*.01
        p=pack_int4(w,group_size=32,symmetric=False,layout='kn',order='low')
        reference=grouped_integer_reference(a,p,sa,bias)
        row={'shape':[m,n,k],'group_size':32,'symmetric':False,'bias':True,'paths':[]}
        for method in ('dot4','dot8','dot4_words'):
            got,kernel=int4_linear_a8(a,p,sa,bias=bias,method=method,tile=(32,64),return_kernel=True)
            same=torch.equal(got.contiguous().view(torch.int32),reference.contiguous().view(torch.int32))
            row['paths'].append({'method':method,'byte_equal':same,'error':metric(got,reference),
                                 'output_sha256':digest(got),'reference_sha256':digest(reference),
                                 'kernel':record_kernel(kernel,outdir,f'ref_{method}_{m}_{n}_{k}')})
            if not same:
                report['reference_check'].append(row);save()
                raise RuntimeError('Independent group reference differs; model diagnostic must not assume equivalence')
        report['reference_check'].append(row);save()
        print(json.dumps({'reference_check':[m,n,k],'paths':[{x['method']:x['byte_equal']} for x in row['paths']]}),flush=True)


def load_qwen():
    p=DEV/'qwen-image-21-rdna2-lab01/custom_nodes/RDNA2-Qwen21-INT8/int8_fused_kernel.py'
    spec=importlib.util.spec_from_file_location('v07_patientx_original',p)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod,p


def native_a8(model,a,w,sa,sw,qwen,return_kernel=False):
    if model=='moss':
        return int8_gemm_dequant(a,w.T,sa,sw,tile=(64,128,64),num_warps=4,
                                num_stages=2,group_m=8,rounding='separate',return_kernel=return_kernel)
    m,k=a.shape;n=w.shape[0]
    y=torch.empty((m,n),dtype=torch.float32,device=a.device)
    kernel=qwen._int8_matmul_dequant_per_row_kernel[
        lambda meta:(triton.cdiv(m,meta['BLOCK_M'])*triton.cdiv(n,meta['BLOCK_N']),)](
            a,w,y,sa,sw,a,m,n,k,*a.stride(),w.stride(1),w.stride(0),*y.stride(),HAS_BIAS=False)
    return (y,kernel) if return_kernel else y


def read_weight(model,name):
    from safetensors import safe_open
    path=MODELS[model]
    with safe_open(path,framework='pt',device='cpu') as f:
        w=f.get_tensor(name)
        scale=f.get_tensor(name.replace('.weight','.weight_scale')).reshape(-1) if model!='anima' else None
    return w,scale,{'file':str(path),'name':name,'dtype':str(w.dtype),'shape':list(w.shape),
                    'tensor_sha256':digest(w),'scale_sha256':None if scale is None else digest(scale)}


def timing(fn,iters):
    torch.cuda.synchronize()
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    start=time.perf_counter();a.record()
    for _ in range(iters):y=fn()
    b.record();torch.cuda.synchronize()
    wall=(time.perf_counter()-start)*1000/iters
    event=a.elapsed_time(b)/iters
    return {'event_ms':event,'wall_ms':wall}


def peak(fn):
    torch.cuda.synchronize();start=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    y=fn();torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()-start


def profiler(fn,outdir,label):
    from torch.profiler import profile,ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as p:
        fn();torch.cuda.synchronize()
    path=outdir/f'profile_{label}.json'
    p.export_chrome_trace(str(path))
    trace=json.loads(path.read_text())
    kernels=[{'name':e.get('name'),'duration_us':e.get('dur')} for e in trace['traceEvents'] if e.get('cat')=='kernel' and e.get('ph')=='X']
    return {'trace':path.name,'kernels':kernels}


def sweep(report,outdir,save,cfg):
    qwen,qwen_path=load_qwen()
    report['sources'][str(qwen_path)]=hashlib.sha256(qwen_path.read_bytes()).hexdigest()
    report['rows']=[]
    keys=list(SHAPES) if cfg.cases=='all' else cfg.cases.split(',')
    for key in keys:
        model,m,n,k,weight_name=SHAPES[key]
        wc,sc,weight_md=read_weight(model,weight_name)
        assert tuple(wc.shape)==(n,k)
        torch.manual_seed(20261107)
        if model=='anima':
            a=torch.randn((m,k),device='cuda',dtype=torch.float16)
            w=wc.to(device='cuda',dtype=torch.float16)
            qa,sa=qwen.triton_quantize_rowwise(a.float())
            sa=sa.reshape(-1)
            native=lambda:torch.nn.functional.linear(a,w)
            source_a=a
            original_w=w.float()
        else:
            qa=torch.randint(-128,128,(m,k),device='cuda',dtype=torch.int8)
            sa=torch.rand(m,device='cuda')*.02+.001
            w=wc.cuda();sw=sc.cuda()
            native=lambda:native_a8(model,qa,w,sa,sw,qwen)
            source_a=qa.float()*sa[:,None]
            original_w=w.float()*sw[:,None]
        first=time.perf_counter();native();torch.cuda.synchronize()
        native_cold_s=time.perf_counter()-first
        baseline_profile=profiler(native,outdir,key+'_native')
        native_y=native()
        # Fixed scattered independent FP64 output sample, not a crop chosen after errors.
        rows=torch.linspace(0,m-1,17).long();cols=torch.linspace(0,n-1,19).long()
        aa=source_a[rows.cuda()].cpu().double() if model=='anima' else qa[rows.cuda()].cpu().double()*sa[rows.cuda()].cpu().double()[:,None]
        wold=original_w[cols.cuda()].cpu().double()
        original_ref=aa@wold.T
        for g in map(int,cfg.groups.split(',')):
         for quant in cfg.quant.split(','):
          for layout in cfg.layouts.split(','):
           for order in cfg.orders.split(','):
            row={'case':key,'model':model,'shape':[m,n,k],'weight':weight_md,
                 'input_values':'synthetic seeded activations at actual observed shape',
                 'unit_scope':'Bias-free GEMM at actual Linear weight/shape; not full Linear pipeline',
                 'bias_in_unit':False,
                 'baseline':'MOSS v0.5 fused INT8' if model=='moss' else 'patientx per-row' if model=='qwen' else 'FP16 rocBLAS F.linear',
                 'native_first_compile_autotune_s':native_cold_s,'native_profile':baseline_profile,
                 'candidates':[]}
            report['rows'].append(row)
            sym=quant=='sym'
            prep_start=time.perf_counter()
            p=pack_int4(original_w,group_size=g,symmetric=sym,layout=layout,order=order)
            torch.cuda.synchronize()
            row['weight_packing_s']=time.perf_counter()-prep_start
            row['weight_packing_scope']='Single first preparation; may include JIT; excluded from warm call comparison'
            row['storage']=word_info(p)
            ph=unpack_int4(p,dtype=torch.float16)
            pi=unpack_int4(p,dtype=torch.int8)
            ap=prepare_dot8_activation(qa,p)
            ah=int8_activation_to_fp16(qa,sa) if model!='anima' else a
            ps=p.to('cpu')
            ps_sample=PackedInt4(ps.data[cols].contiguous() if layout=='nk' else ps.data[:,cols].contiguous(),
                                 ps.scales[cols].contiguous(),ps.zeros[cols].contiguous(),ps.sums[cols].contiguous(),
                                 len(cols),k,g,sym,layout,order)
            centered=unpack_int4_cpu(ps_sample,floating=False)
            w4=centered.double()*ps.scales[cols].double().repeat_interleave(g,1)
            ref_a16=aa@w4.T
            # The A8 operand itself is an exact stored integer. Its scale is FP32.
            a8=qa[rows.cuda()].cpu().double()*sa[rows.cuda()].cpu().double()[:,None]
            ref_a8=a8@w4.T
            row['weight_representation_error']=metric(ref_a16,original_ref)
            row['activation_representation_error']=metric(ref_a8,ref_a16) if model=='anima' else {'relative_l2':0.0,'max_abs':0.0,'reason':'Existing A8 and row scales retained'}
            row['native_compute_error']=metric(native_y[rows.cuda()][:,cols.cuda()],original_ref)
            row['weight_dequant_fp16_error']=metric(ph[cols.cuda()],w4)
            fns={'native':native}
            defs=[('dot4',None),('dot4_split',None),('dot4_prepared',pi),('dot4_words',None),('dot8',None),('dot8_prepared_a',ap),('fp16_dequant',None),('fp16_prepared_w',ph),('dot2',None)]
            if cfg.methods!='all':defs=[x for x in defs if x[0] in cfg.methods.split(',')]
            for method,prepared in defs:
                item={'method':method};row['candidates'].append(item)
                try:
                    if method in ('dot4','dot4_split','dot4_prepared','dot4_words','dot8','dot8_prepared_a'):
                        meth='dot8' if method.startswith('dot8') else method
                        tile=tuple(map(int,cfg.dot8_tile.split(','))) if meth=='dot8' else tuple(map(int,cfg.dot4_tile.split(',')))
                        # Core time keeps activation row quantization out, as the existing INT8 GEMM baseline does.
                        def core(ret=False,meth=meth,prepared=prepared,tile=tile):
                            return int4_linear_a8(qa,p,sa,method=meth,tile=tile,prepared=prepared,return_kernel=ret)
                        y,kern=core(True)
                        if method=='dot4_split':
                            original_dot4=int4_linear_a8(qa,p,sa,method='dot4',tile=tile)
                            item['original_dot4_byte_equal']=torch.equal(y.view(torch.int32),original_dot4.view(torch.int32))
                            del original_dot4
                            if not item['original_dot4_byte_equal']:
                                raise RuntimeError('Sub-K split changed original W4 group calculation bytes')
                        reference=ref_a8
                        half=False
                        if model=='anima':
                            def fn(meth=meth,prepared=prepared,tile=tile,is_prepared_w=method=='dot4_prepared'):
                                xq,xs=qwen.triton_quantize_rowwise(a.float())
                                # dot8 prepared-A is a core-only diagnostic; genuine invocation cannot reuse dynamic activations.
                                return int4_linear_a8(xq,p,xs.reshape(-1),method=meth,tile=tile,
                                                      prepared=prepared if is_prepared_w else None).half()
                        else:fn=lambda core=core:core(False)
                    else:
                        meth={'fp16_dequant':'dequant','fp16_prepared_w':'prepared','dot2':'dot2'}[method]
                        def core(ret=False,meth=meth,prepared=prepared):
                            return int4_linear_a16(ah,p,method=meth,prepared=prepared,
                                                   tile=tuple(map(int,cfg.dot2_tile.split(','))),return_kernel=ret)
                        y,kern=core(True)
                        reference=ref_a16 if model=='anima' else ref_a8
                        half=True
                        if model=='anima':fn=lambda core=core:core(False)
                        else:
                            def fn(meth=meth,prepared=prepared):
                                xh=int8_activation_to_fp16(qa,sa)
                                return int4_linear_a16(xh,p,method=meth,prepared=prepared,
                                                       tile=tuple(map(int,cfg.dot2_tile.split(',')))).float()
                    if model=='anima' and not half:
                        item['core_fp32_error']=metric(y[rows.cuda()][:,cols.cuda()],reference)
                        y=y.half()
                        half=True
                    e=metric(y[rows.cuda()][:,cols.cuda()],reference)
                    item.update(error_same_stored_values=e,compute_gate=gate(e,half),kernel=record_kernel(kern,outdir,key+'_'+method+f'_g{g}_{quant}_{layout}_{order}'),
                                weight_prepare_excluded='prepared' in method,
                                activation_prepare_excluded_from_core=model=='anima' or method=='dot8_prepared_a',
                                candidate_core_dtype=str(y.dtype))
                    if item['compute_gate']:
                        fns[method]=fn
                        item['prepared_core_sample']=timing(lambda core=core:core(False),cfg.iters)
                        item['additional_peak_bytes']=peak(fn)
                    del y
                except Exception:item['error_text']=traceback.format_exc();item['compute_gate']=False
                save()
            for fn in fns.values():
                fn();fn()
            samples={name:[] for name in fns}
            for trial in range(cfg.rounds):
                labels=list(fns)
                if trial%2:labels.reverse()
                for name in labels:samples[name].append(timing(fns[name],cfg.iters))
            row['native_samples']=samples['native']
            row['native_median_wall_ms']=statistics.median(x['wall_ms'] for x in samples['native'])
            row['native_median_event_ms']=statistics.median(x['event_ms'] for x in samples['native'])
            for item in row['candidates']:
                name=item['method']
                if name not in samples:continue
                item['samples']=samples[name]
                item['median_wall_ms']=statistics.median(x['wall_ms'] for x in samples[name])
                item['median_event_ms']=statistics.median(x['event_ms'] for x in samples[name])
                item['speedup_wall']=row['native_median_wall_ms']/item['median_wall_ms']
                item['speedup_event']=row['native_median_event_ms']/item['median_event_ms']
                item['model_candidate']=item['speedup_wall']>=1.05 and name!='dot8_prepared_a'
            save()
            print(json.dumps({'case':key,'group':g,'quant':quant,'layout':layout,'order':order,'native_ms':row['native_median_wall_ms'],
                              'candidate':[{x['method']:x.get('speedup_wall',x.get('error_text','gate failed'))} for x in row['candidates']]}),flush=True)
            del p,ps,ph,pi,ap,ah,w4,centered,ref_a16,ref_a8
        del wc,sc,w,source_a,original_w,native_y,qa,sa
        torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['smoke','split_smoke','sweep','reference'],required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--cases',default='all')
    ap.add_argument('--groups',default='32,64,128')
    ap.add_argument('--quant',default='sym,aff')
    ap.add_argument('--layouts',default='nk')
    ap.add_argument('--orders',default='low')
    ap.add_argument('--methods',default='all')
    ap.add_argument('--dot4-tile',default='64,64')
    ap.add_argument('--dot8-tile',default='16,64')
    ap.add_argument('--dot2-tile',default='64,64,32')
    ap.add_argument('--rounds',type=int,default=3)
    ap.add_argument('--iters',type=int,default=3)
    cfg=ap.parse_args()
    if cfg.out.exists():raise FileExistsError(cfg.out)
    if cfg.rounds<3 or cfg.iters<3:raise ValueError('At least 3 alternating rounds and 3 repeats')
    cfg.out.mkdir(parents=True)
    snapshot=cfg.out/'source';snapshot.mkdir()
    for p in (Path(__file__),ROOT/'rdna2_ops/int4_weight_gemm_v07.py',ROOT/'int4_model_switches_v07.py'):
        if p.exists():(snapshot/p.name).write_bytes(p.read_bytes())
    criteria=ROOT/'results/int4_v07/INT4_PLAN_V07_KO.md'
    (snapshot/criteria.name).write_bytes(criteria.read_bytes())
    torch.set_num_threads(2)
    report={'schema':1,'purpose':'W4 real model Linear shapes; actual checkpoint weight, synthetic activation; no model quality claim',
            'command':sys.argv,'mode':cfg.mode,'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  (Path(__file__),ROOT/'rdna2_ops/int4_weight_gemm_v07.py',ROOT/'rdna2_ops/int8_gemm.py',ROOT/'gpu_guard_v06.py')},
            'criteria_file':'source/INT4_PLAN_V07_KO.md',
            'criteria_sha256':hashlib.sha256(criteria.read_bytes()).hexdigest(),'started_unix_s':time.time(),
            'criteria':{'fp32_relative_l2':2e-5,'fp32_max_abs':'1e-4 + 1e-4*reference_max_abs',
                        'fp16_relative_l2':1e-3,'fp16_max_abs':'2e-3 + 2e-3*reference_max_abs',
                        'model_rgb_max_abs':2/255,'model_rgb_rmse':.25/255,'performance_promotion':1.05},
            'timing':{'rounds':cfg.rounds,'iterations':cfg.iters,'warmups':2,'interleaved_forward_reverse':True,
                       'wall_scope':'synchronized ordinary call with output allocation and documented activation/weight conversion',
                       'event_scope':'HIP events around repeated ordinary calls; may contain host launch gaps',
                       'weight_cache':'explicit caller owned; resident bytes include sums and zeros',
                       'cold':'first compile/autotune/preparation separately; never warm throughput'},
            'success':False}
    def save():
        (cfg.out/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    audit=GpuAudit()
    try:
        with audit,torch.inference_mode():
            audit.ensure_clear()
            report['runtime']={'torch':torch.__version__,'triton':triton.__version__,'hip':torch.version.hip,
                               'gpu':torch.cuda.get_device_name(0),'arch':torch.cuda.get_device_properties(0).gcnArchName}
            if not triton.__version__.startswith('3.8'):raise RuntimeError('Triton 3.8 required; no environment changes')
            save()
            if cfg.mode=='smoke':smoke(report,cfg.out,save)
            elif cfg.mode=='split_smoke':smoke(report,cfg.out,save,groups=(128,),integer_methods=('dot4_split',),
                                               float_methods=('dot4_split',),half_methods=())
            elif cfg.mode=='reference':reference_check(report,cfg.out,save)
            else:sweep(report,cfg.out,save,cfg)
            audit.ensure_clear()
            report['success']=True
    except Exception:
        report['failure']=traceback.format_exc();raise
    finally:
        report['gpu_audit']=audit.report();report['ended_unix_s']=time.time();save()


if __name__=='__main__':main()
