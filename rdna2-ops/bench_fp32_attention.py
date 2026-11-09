#!/usr/bin/env python3
"""FP32 gfx1030 attention sweep against actual Comfy split and math paths.

Run in the model's EXISTING environment. This file changes no production source
or settings. Inputs are synthetic real-shape layouts unless --captured is used.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import torch
import torch.nn.functional as F
import triton
from torch.nn.attention import SDPBackend, sdpa_kernel

from rdna2_ops.fp32_attention import fp32_attention
from rdna2_ops.attention import scaled_dot_product_attention as exact_math
from rdna2_ops.tiled_attention import tiled_attention, _tiled_forward
from bench_tiled_attention import make_inputs, layouts, reference64, errors, measure


TOL = {"rtol": 3e-4, "atol": 3e-5}
# BQ, BK, QK reduction block, PV channel block, warps, key splits.
TILES = {
    "q16k16v128w4": (16,16,128,128,4,1),
    "q16k32v128w8": (16,32,128,128,8,1),
    "q32k32v128w8": (32,32,128,128,8,1),
    "q32k64v128w8": (32,64,128,128,8,1),
    "q32k32v128w16": (32,32,128,128,16,1),
    "q16k32d64v128w4": (16,32,64,128,4,1),
    "q32k32d64v128w8": (32,32,64,128,8,1),
    "q32k32d64v64w4": (32,32,64,64,4,1),
    "q16k32d64v64w4": (16,32,64,64,4,1),
    "q32k64d64v64w8": (32,64,64,64,8,1),
    "q16k32d64v32w4": (16,32,64,32,4,1),
    "q32k32d64v128w8s4": (32,32,64,128,8,4),
    "q32k32d64v64w4s4": (32,32,64,64,4,4),
    "q64k32d64v128w8": (64,32,64,128,8,1),
    "q64k32d64v128w16": (64,32,64,128,16,1),
    "q16k16d16v128w4r": (16,16,16,128,4,1,False),
    "q32k16d16v128w8r": (32,16,16,128,8,1,False),
    "q64k16d16v128w16r": (64,16,16,128,16,1,False),
    "q16k16d32v128w4r": (16,16,32,128,4,1,False),
    "q32k16d32v128w8r": (32,16,32,128,8,1,False),
    "q32k32d16v64w8r": (32,32,16,64,8,1,False),
    "q32k16d16v64w4r": (32,16,16,64,4,1,False),
    "q32k32d16v128w8p16": (32,32,16,128,8,1,False,16),
    "q32k64d16v128w8p16": (32,64,16,128,8,1,False,16),
    "q64k32d16v128w16p16": (64,32,16,128,16,1,False,16),
    "q64k64d16v128w16p16": (64,64,16,128,16,1,False,16),
}


def params(name):
    return dict(zip(("block_q","block_k","block_d","block_v","num_warps","split_k","unroll_d","block_pv"), TILES[name]))


def kernel_info(kernel):
    asm = kernel.asm.get("amdgcn", "")
    meta = getattr(kernel, "metadata", None)
    return {"n_regs": getattr(kernel,"n_regs",None),
            "n_spills": getattr(kernel,"n_spills",None),
            "shared_bytes": getattr(meta,"shared",None),
            "asm_sha256": hashlib.sha256(asm.encode()).hexdigest(),
            "instructions": {p: len(re.findall(p,asm)) for p in
                (r"\bv_fmac_f32\b",r"\bv_fma_f32\b",r"\bv_dot2c_f32_f16\b",
                 r"\bv_dot4c_i32_i8\b",r"\bv_mfma",r"\bv_wmma",r"\bscratch_load",r"\bscratch_store")},
            "private_segment": re.findall(r"amdhsa_private_segment_fixed_size\s+(\d+)",asm)}


def old_kernel_info(q,k,v,bq,bk):
    out = torch.empty(q.shape,device=q.device,dtype=q.dtype)
    compiled = _tiled_forward[(triton.cdiv(q.shape[-2],bq),q.shape[0]*q.shape[1])](
        q,k,v,out,q.shape[-2],k.shape[-2],q.shape[1],
        *q.stride()[:3],*k.stride()[:3],*v.stride()[:3],
        (q.shape[-1]**-0.5)*math.log2(math.e),
        HEAD_DIM=q.shape[-1],BLOCK_Q=bq,BLOCK_K=bk,CAUSAL=False,HALF_COMPUTE=False,
        num_warps=4,num_stages=1)
    return kernel_info(compiled)


def qwen_baselines(runtime, rehearsal):
    sys.path.insert(0,str(runtime))
    sys.argv=["fp32-benchmark","--use-split-cross-attention","--disable-triton-backend"]
    import comfy.ldm.modules.attention as A
    import importlib.util
    spec=importlib.util.spec_from_file_location("qwen_rehearsal_alpha",rehearsal/"scripts/candidates.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module._state["orig"]=A.attention_split
    return A.attention_split,module.split_view


def small_correctness(tile_names):
    result=[]
    for name in tile_names:
        q,k,v=make_inputs(1,2,17,37,128,torch.float32,"packed_v")
        ref=reference64((q,k,v),False)
        row={"tile":name,"case":"tail_rectangular_packed_v"}
        try:
            value,kernels=fp32_attention(q,k,v,return_kernels=True,**params(name))
            row.update(errors(value,ref,TOL),kernels=[kernel_info(x) for x in kernels])
        except Exception as exc:
            row.update(passed=False,error=f"{type(exc).__name__}: {exc}")
        result.append(row)
    # Independent edge checks of each distinct algorithm (single/split + causal),
    # rather than duplicating every possible tuning choice.
    for name in ("q32k32d64v64w4","q32k32d64v64w4s4"):
        for causal,scale in ((True,None),(False,-0.125),(False,0.0)):
            q,k,v=make_inputs(1,2,33,17,128,torch.float32,"interleaved")
            ref=reference64((q,k,v),causal,scale)
            value=fp32_attention(q,k,v,is_causal=causal,scale=scale,**params(name))
            result.append({"tile":name,"case":"causal_or_scale","causal":causal,
                           "scale":scale,**errors(value,ref,TOL)})
    for split in (1,4):
        q,k,v=make_inputs(1,2,19,35,64,torch.float32,"packed_v")
        ref=reference64((q,k,v),True)
        value=fp32_attention(q,k,v,is_causal=True,block_d=64,block_v=64,split_k=split)
        result.append({"tile":f"d64_split{split}","case":"d64_causal_tail",**errors(value,ref,TOL)})
    return result


def load_capture(path):
    data=torch.load(path,map_location="cpu",weights_only=True)
    tensors=[]
    stride_info=data.get("strides",data.get("metadata",{}).get("strides",{}))
    for index,name in enumerate(("q","k","v")):
        t=data[name]
        target_stride=stride_info.get(name) if isinstance(stride_info,dict) else stride_info[index]
        if target_stride is not None:
            gpu=torch.empty_strided(t.shape,target_stride,dtype=t.dtype,device="cuda")
            gpu.copy_(t)
        else:
            gpu=t.to("cuda")
        tensors.append(gpu)
    return data,tensors


def rocblas_query_tiled(q,k,v,query_block,scale_mode):
    """Research hybrid: one fixed Q-block x all-K score tile, full-K softmax.

    QKV head copies/scaling, output and one score tile are included in the call.
    Its global scratch is linear in K for a fixed query_block. This deliberately
    uses rocBLAS/native softmax rather than the Triton streaming kernel above.
    """
    b,h,length,d=q.shape; keys=k.shape[-2]
    scale=d**-0.5
    if scale_mode=="sdpa":
        qh=torch.empty(q.shape,device=q.device,dtype=torch.float32)
        kh=torch.empty(k.shape,device=k.device,dtype=torch.float32)
        torch.mul(q,math.sqrt(scale),out=qh)
        torch.mul(k,math.sqrt(scale),out=kh)
        alpha=1.0
    else:
        qh,kh=q.contiguous(),k.contiguous()
        alpha=scale
    qh,kh=qh.reshape(b*h,length,d),kh.reshape(b*h,keys,d)
    vh=v.contiguous().reshape(b*h,keys,d)
    out=torch.empty(q.shape,device=q.device,dtype=torch.float32)
    out3=out.reshape(b*h,length,d)
    for first in range(0,length,query_block):
        stop=min(first+query_block,length)
        scores=torch.empty((b*h,stop-first,keys),device=q.device,dtype=torch.float32)
        torch.baddbmm(scores,qh[:,first:stop],kh.transpose(1,2),beta=0,alpha=alpha,out=scores)
        torch.softmax(scores,dim=-1,out=scores)
        torch.bmm(scores,vh,out=out3[:,first:stop])
        del scores
    return out


def run_case(model, length, keys, cfg, qwen_functions, capture=None):
    heads=32 if model=="qwen" else 24
    if capture:
        data,tensors=load_capture(capture)
        if tensors[0].ndim==3:
            tensors=[t.reshape(t.shape[0],t.shape[1],heads,128).transpose(1,2) for t in tensors]
        q,k,v=tensors
        length,keys=q.shape[-2],k.shape[-2]
    else:
        q,k,v=make_inputs(1,heads,length,keys,128,torch.float32,"packed_v" if model=="moss" else "interleaved")
    row={"model":model,"source":"captured" if capture else "synthetic_profiled_shape",
         "capture_path":str(capture) if capture else None,"layout":layouts((q,k,v)),
         "scale_mode":"post" if model=="qwen" else "sdpa", "accuracy":{},"isa":{}}
    if capture:row["capture_metadata"]=data.get("metadata",{})
    functions={}
    if model=="qwen":
        q3,k3,v3=[t.transpose(1,2).reshape(1,t.shape[-2],heads*128) for t in (q,k,v)]
        functions["attention_split"]=lambda:qwen_functions[0](q3,k3,v3,heads)
        functions["split_alpha"]=lambda:qwen_functions[1](q3,k3,v3,heads)
        def restore(y): return y.view(1,length,heads,128).transpose(1,2)
        def finish(y): return y.transpose(1,2).reshape(1,length,heads*128)
    else:
        def native():
            with sdpa_kernel([SDPBackend.MATH]):
                return F.scaled_dot_product_attention(q,k,v)
        functions["native_math"]=native
        functions["exact_math"]=lambda:exact_math(q,k,v,backend="rdna2")
        restore=finish=lambda y:y
    reference=restore(next(iter(functions.values()))()).detach().to(device="cpu",dtype=torch.float64)
    sample_rows=torch.linspace(0,length-1,min(4,length)).round().long().unique()
    fp64=reference64((q[:,:2],k[:,:2],v[:,:2]),False,rows=sample_rows)
    row["fp64_reference"]={"heads":[0,1],"query_rows":sample_rows.tolist()}
    for name,fn in functions.items():
        value=restore(fn())
        row["accuracy"][name]={"vs_native":errors(value,reference,TOL),
                               "vs_fp64_rows":errors(value[:,:2],fp64,TOL,sample_rows)}
    for block in cfg.query_blocks:
        name=f"rocblas_query{block}"
        def hybrid(block=block):return finish(rocblas_query_tiled(q,k,v,block,row["scale_mode"]))
        try:
            value=restore(hybrid())
            row["accuracy"][name]={"vs_native":errors(value,reference,TOL),
                                   "vs_fp64_rows":errors(value[:,:2],fp64,TOL,sample_rows)}
            row["accuracy"][name]["vs_native"]["bitwise"]=bool(torch.equal(
                value.detach().cpu().contiguous().view(torch.uint8),reference.float().contiguous().view(torch.uint8)))
            if all(x["passed"] for x in row["accuracy"][name].values()):functions[name]=hybrid
        except Exception as exc:
            row["accuracy"][name]={"error":f"{type(exc).__name__}: {exc}","passed":False}
    if cfg.include_old:
        for bq,bk in ((16,32),(32,64)):
            name=f"v04_{bq}x{bk}"
            def old(bq=bq,bk=bk):return finish(tiled_attention(q,k,v,block_q=bq,block_k=bk))
            value=restore(old())
            row["accuracy"][name]={"vs_native":errors(value,reference,TOL),
                                   "vs_fp64_rows":errors(value[:,:2],fp64,TOL,sample_rows)}
            row["isa"][name]=[old_kernel_info(q,k,v,bq,bk)]
            functions[name]=old
    for name in cfg.tiles:
        def candidate(name=name):
            return finish(fp32_attention(q,k,v,scale_mode=row["scale_mode"],**params(name)))
        try:
            value,kernels=fp32_attention(q,k,v,scale_mode=row["scale_mode"],return_kernels=True,**params(name))
            row["accuracy"][name]={"vs_native":errors(value,reference,TOL),
                                   "vs_fp64_rows":errors(value[:,:2],fp64,TOL,sample_rows)}
            row["isa"][name]=[kernel_info(x) for x in kernels]
            if all(x["passed"] for x in row["accuracy"][name].values()):
                functions[name]=candidate
        except Exception as exc:
            row["accuracy"][name]={"error":f"{type(exc).__name__}: {exc}","passed":False}
    row.update(measure(functions,cfg))
    for method in row["methods"].values():
        method["speedups"]={baseline:row["methods"][baseline]["median"]["wall_ms"]/method["median"]["wall_ms"]
                            for baseline in ("attention_split","split_alpha","native_math","exact_math") if baseline in row["methods"]}
    row["passed"]=all(name in functions for name in cfg.tiles+[f"rocblas_query{x}" for x in cfg.query_blocks])
    return row


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--models",nargs="+",choices=("qwen","moss"),default=["qwen","moss"])
    ap.add_argument("--qwen-runtime",type=Path,default=Path("~/DevSpace/qwen-image-21-rdna2-lab01/runtime/ComfyUI"))
    ap.add_argument("--qwen-rehearsal",type=Path,default=Path("~/DevSpace/rdna2-pro-research-20261102/rehearsal_qwen_20261106"))
    ap.add_argument("--tiles",nargs="+",choices=list(TILES),default=list(TILES))
    ap.add_argument("--qwen-keys",nargs="+",type=int,default=[4964,2408])
    ap.add_argument("--moss-lengths",nargs="+",type=int,default=[1061])
    ap.add_argument("--query-blocks",nargs="*",type=int,choices=(32,64,128),default=[32,64,128])
    ap.add_argument("--rounds",type=int,default=3)
    ap.add_argument("--iterations",type=int,default=3)
    ap.add_argument("--warmup",type=int,default=1)
    ap.add_argument("--include-old",action="store_true")
    ap.add_argument("--skip-small",action="store_true")
    ap.add_argument("--captured",type=Path)
    cfg=ap.parse_args()
    if cfg.output.exists():ap.error("Choose a new output; evidence is not overwritten")
    if cfg.captured and len(cfg.models)!=1:ap.error("Captured inputs require one --models choice")
    if min(cfg.rounds,cfg.iterations,cfg.warmup,*cfg.qwen_keys,*cfg.moss_lengths)<=0:ap.error("Positive lengths/repeats required")
    report={"schema_version":1,"started_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
            "seed":20261106,"tolerance_fixed_before_execution":TOL,"tiles":{n:TILES[n] for n in cfg.tiles},
            "rocblas_hybrid_query_blocks":cfg.query_blocks,
            "rounds":cfg.rounds,"iterations":cfg.iterations,"warmup":cfg.warmup,
            "memory_metric":"Incremental peak PyTorch allocated, excluding input/reference and non-allocator driver scratch.",
            "timing":"Whole ordinary call including validation/output allocation/layout conversion; compilation and input creation excluded.",
            "correctness":[],"benchmarks":[],"status":"started"}
    cfg.output.parent.mkdir(parents=True,exist_ok=True)
    def save():cfg.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    previous=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    status=1
    try:
        torch.set_num_threads(2);torch.manual_seed(report["seed"])
        props=torch.cuda.get_device_properties(0)
        report["environment"]={"torch":torch.__version__,"torch_git":torch.version.git_version,"hip":torch.version.hip,"triton":triton.__version__,
                               "arch":props.gcnArchName,"gpu":props.name}
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        qwen=qwen_baselines(cfg.qwen_runtime,cfg.qwen_rehearsal) if "qwen" in cfg.models else None
        sources=[Path(__file__),Path(__file__).parent/"rdna2_ops/attention.py",Path(__file__).parent/"rdna2_ops/fp32_attention.py"]
        if qwen:sources.extend([cfg.qwen_runtime/"comfy/ldm/modules/attention.py",cfg.qwen_rehearsal/"scripts/candidates.py"])
        report["source_sha256"]={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        save()
        with torch.inference_mode():
            if not cfg.skip_small:
                report["correctness"]=small_correctness(cfg.tiles);save()
                passed_names={r["tile"] for r in report["correctness"] if r["case"]=="tail_rectangular_packed_v" and r["passed"]}
                cfg.tiles=[name for name in cfg.tiles if name in passed_names]
                if not cfg.tiles:raise RuntimeError("No candidate passed small correctness")
                edge_failed=[r for r in report["correctness"] if r["case"]!="tail_rectangular_packed_v" and not r["passed"]]
                if edge_failed:raise RuntimeError("Algorithm edge correctness failed")
                print(json.dumps({"small_correctness":len(cfg.tiles),"cases":len(report["correctness"])}),flush=True)
            for model in cfg.models:
                cases=[(2400,s) for s in cfg.qwen_keys] if model=="qwen" else [(s,s) for s in cfg.moss_lengths]
                if cfg.captured:cases=[(0,0)]
                for length,keys in cases:
                    row=run_case(model,length,keys,cfg,qwen,cfg.captured);report["benchmarks"].append(row);save()
                    print(json.dumps({"model":model,"query":row["layout"]["q"]["shape"],"key":row["layout"]["k"]["shape"],
                                      "passed":row["passed"],"methods":{n:{"ms":r["median"]["wall_ms"],"peak_bytes":r["memory"]["incremental_peak_bytes"]}
                                                                         for n,r in row["methods"].items()}}),flush=True)
        all_passed=all(r["passed"] for r in report["benchmarks"]+report["correctness"])
        report["status"]="passed" if all_passed else "completed_with_candidate_failures"
        status=0
    except Exception as exc:
        report.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        print(report["error"],flush=True)
    finally:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous)
        report["exit_status"]=status;save()
        print(f"JSON: {cfg.output}",flush=True)
    return status


if __name__=="__main__":raise SystemExit(main())
