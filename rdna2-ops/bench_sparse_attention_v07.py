#!/usr/bin/env python3
"""Actual Qwen captures, FP32 sparse mask/kernels, guarded serial GPU evidence."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback
import torch
import triton

from gpu_guard_v06 import GpuAudit
from rdna2_ops.sparse_attention_v07 import (BlockMap,sparse_attention,sparge_map,radial_map,
                                          fixed_map,dense_alpha,map_statistics,_block_mean_cosine,
                                          block_mean_cosine_reference)
from bench_fp32_attention import kernel_info,qwen_baselines


KERNEL_TOL={"atol":3e-5,"rtol":3e-4}


def err(y,ref):
    a=y.detach().double().cpu();b=ref.detach().double().cpu();delta=a-b
    rmse=delta.square().mean().sqrt().item()
    return {"finite":bool(torch.isfinite(a).all()),"reference_finite":bool(torch.isfinite(b).all()),
            "output_nonfinite_count":int((~torch.isfinite(a)).sum()),
            "reference_nonfinite_count":int((~torch.isfinite(b)).sum()),
            "max_abs":delta.abs().max().item(),"rmse":rmse,
            "relative_rmse":rmse/max(b.square().mean().sqrt().item(),1e-30),
            "relative_l1":delta.abs().sum().item()/max(b.abs().sum().item(),1e-30),
            "same_values":bool(torch.equal(a,b)),
            "gate":bool(torch.isfinite(a).all() and torch.isfinite(b).all() and torch.allclose(a,b,**KERNEL_TOL))}


def standard_json_report(report):
    """Keep nonfinite metric paths and values without invalid JSON constants."""
    nonfinite=[]
    def convert(value,path):
        if isinstance(value,float) and not math.isfinite(value):
            label="NaN" if math.isnan(value) else ("Infinity" if value>0 else "-Infinity")
            nonfinite.append({"path":path,"value":label,"failure":"nonfinite_numeric_metric"})
            return label
        if isinstance(value,dict):return {str(k):convert(v,path+"."+str(k)) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [convert(v,path+"["+str(i)+"]") for i,v in enumerate(value)]
        return value
    clean=convert(report,"report")
    clean["nonfinite_metric_failure"]=bool(nonfinite)
    clean["nonfinite_metrics"]=nonfinite
    return clean


def fp64_worst_rows(q,k,v,got,reference,block_map,offset,prefix_bias_offset=None,count=3):
    """Diagnose failed replacement tolerance with independent CPU FP64 rows.

    This does not loosen the preregistered gate or turn selected-row evidence
    into full-tensor correctness. It separates native and streaming rounding.
    """
    gap=(got-reference).abs().amax(-1).detach().cpu()
    rows=[];h=gap.shape[1];length=gap.shape[2]
    for idx in gap.flatten().topk(min(count,gap.numel())).indices.tolist():
        batch,rem=divmod(idx,h*length);head,query=divmod(rem,length)
        qr=q[batch,head,query].detach().cpu().double()
        kr=k[batch,head].detach().cpu().double()
        vr=v[batch,head].detach().cpu().double()
        score=(kr@qr)*(q.shape[-1]**-.5)
        if prefix_bias_offset is not None:
            score += torch.arange(k.shape[-2]) <= query+prefix_bias_offset
        mask=block_map.keep[batch,head,query//block_map.query_block].detach().cpu()
        visible=mask.repeat_interleave(block_map.key_block)[:k.shape[-2]]
        if offset is not None:visible &= torch.arange(k.shape[-2])<=query+offset
        exact=(score.masked_fill(~visible,-torch.inf).softmax(-1))@vr
        rows.append({"batch":batch,"head":head,"query":query,
            "retained_visible_keys":int(visible.sum()),"all_keys_masked":not bool(visible.any()),
            "max_logit":score.max().item(),"min_logit":score.min().item(),
            "streaming_vs_fp64":err(got[batch,head,query],exact),
            "rocblas_reference_vs_fp64":err(reference[batch,head,query],exact)})
    return rows


def measure(functions,warmup=2,rounds=3,iterations=2):
    report={n:{"samples":[]} for n in functions};order=[]
    for name,fn in functions.items():
        start=time.perf_counter();y=fn();torch.cuda.synchronize()
        report[name]["first_timed_call_s_after_correctness"] = time.perf_counter()-start
        del y
        for _ in range(warmup):
            y=fn();torch.cuda.synchronize();del y
    names=list(functions)
    for round_idx in range(rounds):
        rotated=names[round_idx:]+names[:round_idx]
        if round_idx%2:rotated=rotated[::-1]
        for name in rotated:
            for rep in range(iterations):
                fn=functions[name];torch.cuda.synchronize()
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();t=time.perf_counter();y=fn();b.record();b.synchronize()
                ms=(time.perf_counter()-t)*1000
                report[name]["samples"].append({"round":round_idx,"repeat":rep,"wall_ms":ms,"event_ms":a.elapsed_time(b)})
                order.append(name);del y
    for name,fn in functions.items():
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated()
        y=fn();torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()-before;del y
        report[name]["median_wall_ms"]=statistics.median(x["wall_ms"] for x in report[name]["samples"])
        report[name]["median_event_ms"]=statistics.median(x["event_ms"] for x in report[name]["samples"])
        report[name]["incremental_peak_bytes"]=peak
    return {"methods":report,"order":order,"warmup":warmup,"rounds":rounds,"iterations":iterations}


def small_checks():
    rows=[]
    for dim,ql,kl,offset,bias in [(64,35,139,None,None),(128,17,77,None,None),
                                (128,35,139,104,None),(128,35,139,None,104)]:
        q=torch.randn(1,ql,2,dim,device="cuda")*.125
        k=torch.randn(1,kl,2,dim,device="cuda")*.125
        v=torch.randn(1,kl,2,dim,device="cuda")*.125
        q,k,v=[t.transpose(1,2) for t in (q,k,v)]
        bm=fixed_map(q,k)
        got,kernel=sparse_attention(q,k,v,bm,causal_offset=offset,prefix_bias_offset=bias,return_kernel=True)
        qr,kr,vr=[t.cpu().double() for t in (q,k,v)]
        logits=qr@kr.transpose(-1,-2)*(dim**-.5)
        if bias is not None:logits += torch.arange(kl)[None,:]<=torch.arange(ql)[:,None]+bias
        qi=torch.arange(ql)//bm.query_block;ki=torch.arange(kl)//bm.key_block
        visible=bm.keep.cpu()[:,:,qi[:,None],ki[None,:]]
        if offset is not None:visible=visible & (torch.arange(kl)[None,:]<=torch.arange(ql)[:,None]+offset)
        ref=logits.masked_fill(~visible,-torch.inf).softmax(-1)@vr
        pooled=_block_mean_cosine(k,64);pool_ref=block_mean_cosine_reference(k.cpu(),64)
        pool_error={"mean_max_abs":(pooled[0].cpu()-pool_ref[0]).abs().max().item(),
                    "cosine_max_abs":(pooled[1].cpu()-pool_ref[1]).abs().max().item()}
        pool_error["gate"]=pool_error["mean_max_abs"]<=1e-5 and pool_error["cosine_max_abs"]<=1e-5
        rows.append({"shape":[1,2,ql,kl,dim],"causal_offset":offset,"prefix_bias_offset":bias,"error_vs_fp64":err(got,ref),
                     "tile":{"block_q":32,"block_k":64,"num_warps":8},
                     "fused_pool_vs_cpu":pool_error,"kernel":kernel_info(kernel)})
        got,kernel=sparse_attention(q,k,v,bm,causal_offset=offset,prefix_bias_offset=bias,block_q=16,num_warps=4,return_kernel=True)
        rows.append({"shape":[1,2,ql,kl,dim],"causal_offset":offset,"prefix_bias_offset":bias,"error_vs_fp64":err(got,ref),
                     "tile":{"block_q":16,"block_k":64,"num_warps":4},
                     "fused_pool_vs_cpu":pool_error,"kernel":kernel_info(kernel)})
    return rows


def load_capture(path):
    data=torch.load(path,map_location="cpu",weights_only=True)
    heads=data.get("heads",data.get("metadata",{}).get("heads",32))
    ts=[]
    for name in ("q","k","v"):
        x=data[name].to("cuda")
        ts.append(x.reshape(x.shape[0],x.shape[1],heads,-1).transpose(1,2) if x.ndim==3 else x)
    q,k,v=ts
    mask=data.get("mask")
    offset=None
    if mask is not None:
        if mask.dtype!=torch.bool or mask.ndim!=2:
            raise ValueError("Only the observed rectangular Boolean prefix mask is supported")
        offset=int(mask[0].sum().item())-1
        expected=torch.arange(k.shape[-2])[None,:] <= torch.arange(q.shape[-2])[:,None]+offset
        if not torch.equal(mask,expected):raise ValueError("Mask is not the declared prefix-causal geometry")
    geometry=data.get("geometry")
    if geometry and (q.shape[-2]!=geometry["target"]["end"]-geometry["target"]["start"] or k.shape[-2]!=geometry["key_length"]):
        # A captured image-prefix query uses the earlier image segment as its grid.
        images=[s for s in geometry["image_segments"] if s["end"]==k.shape[-2] and s["end"]-s["start"]==q.shape[-2]]
        if images:
            geometry={**geometry,"target":images[0],"key_length":k.shape[-2],
                      "image_segments":[s for s in geometry["image_segments"] if s["end"]<=k.shape[-2]],
                      "text_segments":[s for s in geometry["text_segments"] if s["end"]<=k.shape[-2]]}
        else:geometry=None
    if geometry and geometry["key_length"]!=k.shape[-2]:geometry=None
    return data,ts,offset,geometry


def run_case(path,out,functions,cfg):
    data,(q,k,v),prefix_bias_offset,geometry=load_capture(path)
    # The captured Boolean prefix tensor is ADDITIVE in this actual backend:
    # attention_split executes scores += mask, rather than masked_fill(~mask).
    # Keep this contract separate from a logical prefix/causal exclusion mask.
    offset=None
    b,h,length,d=q.shape;keys=k.shape[-2]
    q3,k3,v3=[t.transpose(1,2).reshape(b,t.shape[-2],h*d) for t in (q,k,v)]
    native,alpha=functions
    mask=data.get("mask");mask=mask.to(q.device) if mask is not None else None
    restore=lambda y:y.reshape(b,length,h,d).transpose(1,2)
    fns={"attention_split":lambda:restore(native(q3,k3,v3,h,mask=mask)),
         "split_alpha":lambda:restore(alpha(q3,k3,v3,h,mask=mask)),
         "reference_alpha":lambda:dense_alpha(q,k,v,prefix_bias_offset=prefix_bias_offset)}
    torch.cuda.synchronize();first_native_start=time.perf_counter()
    reference=fns["attention_split"]().cpu()
    first_native_s=time.perf_counter()-first_native_start
    nq,nk=triton.cdiv(length,128),triton.cdiv(keys,64)
    maps={"dense_stream":BlockMap(torch.ones((b,h,nq,nk),device=q.device,dtype=torch.bool))}
    selectors={}
    for mass in (.9,.95,.99):
        name=f"sparge_p{mass:g}_c0.1";selectors[name]=lambda mass=mass:sparge_map(q,k,mass=mass,similarity=.1)
    selectors["sparge_p0.95_c0.5"]=lambda:sparge_map(q,k,mass=.95,similarity=.5)
    selectors["fixed_50"]=lambda:fixed_map(q,k,fraction=.5)
    if geometry and mask is None:
        for radius in (4.,8.,16.):
            selectors[f"radial_r{radius:g}"]=lambda radius=radius:radial_map(q,k,geometry,radius=radius)
    prepare={}
    for name,fn in selectors.items():
        torch.cuda.synchronize();t=time.perf_counter();maps[name]=fn();torch.cuda.synchronize()
        prepare[name]=(time.perf_counter()-t)*1000
    row={"capture":str(path),"capture_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
         "q_shape":list(q.shape),"k_shape":list(k.shape),"q_stride":list(q.stride()),"k_stride":list(k.stride()),
         "causal_offset":offset,"prefix_bias_offset":prefix_bias_offset,"geometry":geometry,"map_prepare_ms":prepare,"accuracy":{},"isa":{},
         "first_native_call_s_before_correctness":first_native_s,"first_candidate_call_s_before_correctness":{}}
    native_path=Path(inspect.getsourcefile(inspect.unwrap(native)))
    row["native_callable"]={"module":native.__module__,"qualname":native.__qualname__,
                             "source":str(native_path),"source_sha256":hashlib.sha256(native_path.read_bytes()).hexdigest()}
    if data.get("mask") is not None:
        raw=data["mask"].contiguous()
        values,counts=torch.unique(raw,return_counts=True)
        row["raw_captured_mask"]={"dtype":str(raw.dtype),"shape":list(raw.shape),
            "values_and_counts":[[bool(a),int(b)] for a,b in zip(values,counts)],
            "raw_bytes_sha256":hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest(),
            "matches_prefix_geometry":True,"prefix_offset":prefix_bias_offset,
            "capture_metadata":data.get("metadata"),
            "observed_backend_contract":"scores += bool mask, hence +1 for true and +0 for false; no key exclusion"}
    summary,raw_mass=map_statistics(q,k,maps,prefix_bias_offset=prefix_bias_offset)
    row["mass_statistics"]=summary
    mass_path=out/(path.stem+"_mass.pt")
    torch.save({"block_mass":raw_mass,"maps":{n:m.keep.cpu() for n,m in maps.items()}},mass_path)
    row["mass_file"]={"path":str(mass_path),"sha256":hashlib.sha256(mass_path.read_bytes()).hexdigest()}
    for name,fn in fns.items():row["accuracy"][name]={"approximation_vs_native":err(fn(),reference)}
    for name,bm in maps.items():
        same_mask=dense_alpha(q,k,v,prefix_bias_offset=prefix_bias_offset,block_map=bm)
        for suffix,tile in [("",{"block_q":32,"num_warps":8}),("_q16",{"block_q":16,"num_warps":4})]:
            variant=name+suffix
            try:
                torch.cuda.synchronize();first_start=time.perf_counter()
                got,kernel=sparse_attention(q,k,v,bm,prefix_bias_offset=prefix_bias_offset,return_kernel=True,**tile)
                torch.cuda.synchronize()
                row["first_candidate_call_s_before_correctness"][variant]=time.perf_counter()-first_start
                row["accuracy"][variant]={"tile":tile,"same_mask_compute_error":err(got,same_mask),
                                          "approximation_vs_native":err(got,reference)}
                row["isa"][variant]=kernel_info(kernel)
                if not row["accuracy"][variant]["same_mask_compute_error"]["gate"]:
                    row["accuracy"][variant]["fp64_diagnostic_selected_rows"]=fp64_worst_rows(q,k,v,got,same_mask,bm,offset,prefix_bias_offset)
                    del got
                    continue
                fns[variant+"_prepared"]=lambda bm=bm,tile=tile:sparse_attention(q,k,v,bm,prefix_bias_offset=prefix_bias_offset,**tile)
                if name.startswith("sparge"):
                    fns[variant+"_inclusive"]=lambda name=name,tile=tile:sparse_attention(q,k,v,selectors[name](),prefix_bias_offset=prefix_bias_offset,**tile)
                del got
            except Exception:
                row["accuracy"][variant]={"tile":tile,"exception":traceback.format_exc()}
        del same_mask
    row.update(measure(fns,cfg.warmup,cfg.rounds,cfg.iterations))
    alpha_ms=row["methods"]["split_alpha"]["median_wall_ms"]
    for x in row["methods"].values():x["speedup_vs_actual_split_alpha"]=alpha_ms/x["median_wall_ms"]
    # Profiling is a separate diagnostic after all clocks above have stopped.
    from torch.profiler import profile,ProfilerActivity
    from research_comfy_v05 import profile_tables
    eligible=[n for n in fns if n.endswith("_inclusive") or (n.endswith("_prepared") and not n.startswith("sparge"))]
    best=min(eligible,key=lambda n:row["methods"][n]["median_wall_ms"]) if eligible else None
    row["kernel_profiles"]={}
    for name in ["attention_split","split_alpha"]+([best] if best else []):
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True) as prof:
            y=fns[name]();torch.cuda.synchronize();del y
        trace=out/(path.stem+"_"+name+"_trace.json")
        prof.export_chrome_trace(str(trace))
        tables=profile_tables(prof,trace_path=trace)
        row["kernel_profiles"][name]=tables["device_activities"]
    return row


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--captures",type=Path,nargs="*",default=[])
    ap.add_argument("--warmup",type=int,default=2)
    ap.add_argument("--rounds",type=int,default=3)
    ap.add_argument("--iterations",type=int,default=2)
    cfg=ap.parse_args()
    if (cfg.out/"results.json").exists():ap.error("Use a fresh evidence directory")
    cfg.out.mkdir(parents=True,exist_ok=True)
    report={"started_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"argv":vars(cfg),
            "compute_gate_fixed_before_execution":KERNEL_TOL,
            "rgb_gate_fixed_before_execution":{"max_abs":2/255,"rmse":.25/255},
            "approximation_gate":"RGB is independent; no probability-mass threshold substitutes for it",
            "timing":"ordinary calls; two warmups then rotating/reversed 3x2 samples; first native/candidate calls before correctness recorded separately (persistent Triton cache may be warm); first timed call after correctness is not cold JIT; static map prep separately",
            "memory":"incremental torch allocated peak; prepared masks and input/reference excluded; driver scratch not measured",
            "small_checks":[],"benchmarks":[],"source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (Path(__file__),Path(__file__).parent/"rdna2_ops/sparse_attention_v07.py")},"status":"started"}
    def save():
        (cfg.out/"results.json").write_text(json.dumps(standard_json_report(report),indent=2,default=str,allow_nan=False)+"\n")
    audit=GpuAudit();save()
    try:
        with audit:
            audit.ensure_clear()
            torch.set_num_threads(2);torch.manual_seed(20261107)
            props=torch.cuda.get_device_properties(0)
            report["environment"]={"torch":str(torch.__version__),"hip":torch.version.hip,"triton":triton.__version__,"gpu":props.name,"arch":props.gcnArchName}
            with torch.inference_mode():
                report["small_checks"]=small_checks();save()
                if not all(x["error_vs_fp64"]["gate"] and x["fused_pool_vs_cpu"]["gate"] for x in report["small_checks"]):
                    raise RuntimeError("Small independent FP64 correctness failed")
                functions=qwen_baselines(Path("~/DevSpace/qwen-image-21-rdna2-lab01/runtime/ComfyUI"),Path(__file__).resolve().parents[1]/"rehearsal_qwen_20261106")
                for path in cfg.captures:
                    row=run_case(path,cfg.out,functions,cfg)
                    report["benchmarks"].append(row)
                    report["gpu_audit_progress"]=audit.report()
                    save()
                    print(json.dumps({"shape":row["q_shape"],"keys":row["k_shape"],"methods":{n:{"ms":x["median_wall_ms"],"speedup":x["speedup_vs_actual_split_alpha"]} for n,x in row["methods"].items()}}),flush=True)
            report["status"]="completed"
    except Exception:
        report["status"]="failed";report["traceback"]=traceback.format_exc();print(report["traceback"],flush=True)
    finally:
        report["gpu_audit"]=audit.report()
        (cfg.out/"gpu_audit.json").write_text(json.dumps(report["gpu_audit"],indent=2,allow_nan=False)+"\n")
        if not report["gpu_audit"]["valid_for_timing"]:report["status"]="invalid_overlap"
        save()
    return 0 if report["status"]=="completed" else 1


if __name__=="__main__":raise SystemExit(main())
