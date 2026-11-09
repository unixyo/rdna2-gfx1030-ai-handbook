#!/usr/bin/env python3
"""Reuse Applio's actual train.py without touching its config/data/checkpoints.

The absolute model_name redirects all trainer writes into a new research run.
Run startup, complete epochs, and fixed-reference audio with one GPU lease.
--db can point to a research FindDb seeded with only dilation=1 NORMAL results;
the full actual training process still uses FAST. This is not an operating patch.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import statistics
import sys
import time
import wave


def digest_file(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for part in iter(lambda:f.read(1<<20),b""): h.update(part)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--applio", type=Path, default=Path(__file__).resolve().parent/"results/applio_v08/applio_copy")
    p.add_argument("--dataset", default="voice_model_A")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--mode", choices=["FAST","NORMAL"], default="FAST")
    p.add_argument("--fold", action="store_true")
    p.add_argument("--selective-fold", action="store_true",
                   help="Only measured B8,C64,T6400,k7/11,d3/5 wins; process-local Conv1d boundary")
    p.add_argument("--observe-transpose", action="store_true",
                   help="Count actual ConvTranspose1d calls without changing their computation")
    p.add_argument("--compare", type=Path)
    p.add_argument("--profile-at",type=int,default=0)
    p.add_argument("--profile-steps",type=int,default=5)
    p.add_argument("--benchmark",choices=["native","true","false"],default="native")
    p.add_argument("--compile",default="")
    p.add_argument("--compile-native-rng",action="store_true",
                   help="Keep native random operators inside the compiled candidate")
    p.add_argument("--compile-eager-conv",action="store_true",
                   help="Keep measured Conv1d dispatch outside Dynamo while compiling its caller")
    p.add_argument("--save-every",type=int,default=10000)
    p.add_argument("--export-final",action="store_true")
    p.add_argument("--no-reference-audio",action="store_true")
    p.add_argument("--save-only-latest",action="store_true")
    p.add_argument("--save-every-weights",action="store_true")
    p.add_argument("--nonconv",action="store_true")
    p.add_argument("--dispatch",type=Path)
    args = p.parse_args()
    if args.fold and args.selective_fold: p.error("Choose blanket or measured selective fold")
    if args.compile_eager_conv and not (args.compile and args.dispatch):
        p.error("--compile-eager-conv requires --compile and --dispatch")
    if args.compare is not None: args.compare=args.compare.resolve()
    research = Path(__file__).resolve().parent
    root = args.applio.resolve(); out = args.out.resolve(); db = args.db.resolve()
    if not out.is_relative_to(research) or not db.is_relative_to(research):
        p.error("Outputs and DB must be inside the research folder")
    if out.exists(): p.error("Use a new run directory; refusing resume or overwrite")
    if args.epochs < 2: p.error("At least one warm epoch and one measured epoch required")
    source_data = root/"logs"/args.dataset
    gpre = root/"rvc/models/pretraineds/custom/G_KLM_HFG_40k.pth"
    dpre = root/"rvc/models/pretraineds/custom/D_KLM_HFG_40k.pth"
    sources = [root/"assets/config.json", source_data/"config.json", source_data/"filelist.txt",
               source_data/"model_info.json", root/"rvc/train/train.py",
               root/"rvc/lib/zluda.py", root/"rvc/lib/algorithm/fold_conv.py",
               root/"assets/config_template.json", root/"core.py", root/"rvc/train/lab_hooks.py",
               root/"rvc/train/data_utils.py", root/"rvc/lib/algorithm/synthesizers.py",
               root/"rvc/lib/algorithm/discriminators.py", root/"rvc/lib/algorithm/residuals.py",
               gpre,dpre]
    original_manifest=json.loads((research/"results/applio_v08/protected_before.json").read_text())
    protected_original={**original_manifest["legacy16"],**original_manifest["additional_target_and_source"]}
    protected = {str(x):digest_file(x) for x in sources}
    if any(digest_file(Path(x))!=h for x,h in protected_original.items()):
        raise RuntimeError("Protected operating file changed since v0.8 start")
    out.mkdir(parents=True); db.mkdir(parents=True, exist_ok=True)
    (out/"protected_before.json").write_text(json.dumps(protected,indent=2))
    cfg = json.loads((source_data/"config.json").read_text())
    cfg.pop("process_pids",None)
    (out/"config.json").write_text(json.dumps(cfg,indent=2))
    (out/"filelist.txt").write_bytes((source_data/"filelist.txt").read_bytes())
    (out/"model_info.json").write_bytes((source_data/"model_info.json").read_bytes())
    os.environ.update(MIOPEN_FIND_MODE=args.mode, MIOPEN_USER_DB_PATH=str(db),
                      MIOPEN_CUSTOM_CACHE_DIR=str(db.parent/"miopen_cache"),
                      APPLIO_DILATION_FOLD="1" if args.fold else "0",
                      APPLIO_COMPILE=args.compile, APPLIO_PROFILE_AT="0", APPLIO_AMP_LOG="0",
                      MPLBACKEND="Agg", MASTER_ADDR="127.0.0.1",
                      MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1="0",
                      MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53="0",PYTHONDONTWRITEBYTECODE="1")
    Path(os.environ["MIOPEN_CUSTOM_CACHE_DIR"]).mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(research/"results/applio_v08/triton_cache"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(out/"inductor_cache"))
    with socket.socket() as s:
        s.bind(("127.0.0.1",0)); os.environ["MASTER_PORT"] = str(s.getsockname()[1])
    os.chdir(root)
    sys.path[:0] = [str(root),str(root/"rvc/train")]
    # Original train.py's CLI is parsed on import. Its main()/cleanup is never called.
    sys.argv = [str(root/"rvc/train/train.py"),str(out),str(args.save_every),str(args.epochs),
                str(gpre),str(dpre),"0","8","40000",str(args.save_only_latest),str(args.save_every_weights),"False",
                "False","HiFi-GAN","False"]
    import numpy as np
    import torch
    # The original train.py __main__ selects spawn; keep its DataLoader worker
    # model even though this wrapper intentionally bypasses original main().
    start_method=torch.multiprocessing.get_start_method(allow_none=True)
    if start_method is None:
        torch.multiprocessing.set_start_method("spawn")
    elif start_method != "spawn":
        raise RuntimeError("Fresh process with spawn workers required")
    from gpu_guard_v06 import GpuAudit
    from bench_conv_catalog_v06 import db_inventory
    process_started=time.perf_counter()
    result = dict(schema="applio_epoch_v08", mode=args.mode, fold=args.fold, compile=args.compile, profile_at=args.profile_at, requested_benchmark=args.benchmark,
                  completed=False, harness_sha256=digest_file(Path(__file__)),
                  direct_dispatch_manifest=({"path":str(args.dispatch.resolve()), "sha256":digest_file(args.dispatch)} if args.dispatch else None),
                  compile_native_rng_requested=args.compile_native_rng,
                  compile_eager_conv_requested=args.compile_eager_conv,
                  cache_paths={k:os.environ[k] for k in ("TRITON_CACHE_DIR","TORCHINDUCTOR_CACHE_DIR")},
                  dataset=args.dataset, batch_size=8, pretrain="KLM HFG 40k",
                  valid_for_epoch_timing=not args.profile_at and not bool(os.environ.get("APPLIO_RDNA2_OPT_SNAPSHOT_DIR")),
                  profile_probe="metadata uses channels/kernel/parameter dtype; never evaluate parametrized weight getter",
                  torch_num_threads=torch.get_num_threads(),
                  selective_fold=args.selective_fold, selective_fold_hits={},
                  observe_transpose=args.observe_transpose, actual_transpose_calls={},
                  torch=torch.__version__, hip=torch.version.hip, protected_before=protected,
                  pretrain_sha256={str(gpre):protected[str(gpre)],str(dpre):protected[str(dpre)]},
                  db_before=db_inventory(db), epochs=[], steps=[],
                  quality_gate={"provisional_audio_gate":True,"wave_max_abs":0.001,
                                "wave_rmse":0.0001,"loss_median_relative":0.02,
                                "note":"Predeclared engineering screen, not user-approved perceptual equivalence. Require native-repeat floor too."})
    def state_digest(model):
        h=hashlib.sha256()
        for name,t in model.state_dict().items():
            h.update(name.encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    def audio(model, reference, tag):
        states=(torch.get_rng_state(),torch.cuda.get_rng_state(),random.getstate(),np.random.get_state())
        original_training=model.training; model.eval()
        torch.manual_seed(997131); random.seed(997131); np.random.seed(997131)
        phone,lengths,pitch,pitchf,sid=reference
        frames=min(160,int(lengths[0]))
        fixed=(phone[:1,:frames],torch.full_like(lengths[:1],frames),pitch[:1,:frames],
               pitchf[:1,:frames],sid[:1])
        reference_metadata=[{"shape":list(t.shape),"dtype":str(t.dtype),
            "sha256":hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
            for t in fixed]
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.float16):
            y=model.infer(*fixed)[0].float().cpu().flatten()
        torch.save(y,out/(tag+".pt"))
        pcm=(y.clamp(-1,1)*32767).round().to(torch.int16).numpy()
        with wave.open(str(out/(tag+".wav")),"wb") as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(40000);w.writeframes(pcm.tobytes())
        model.train(original_training)
        torch.set_rng_state(states[0]);torch.cuda.set_rng_state(states[1]);random.setstate(states[2]);np.random.set_state(states[3])
        row=dict(samples=y.numel(),finite=bool(torch.isfinite(y).all()),
                 sha256=hashlib.sha256(y.numpy().tobytes()).hexdigest(),peak=float(y.abs().max()),
                 reference_tensors=reference_metadata)
        if args.compare:
            control=torch.load(args.compare/(tag+".pt"),map_location="cpu",weights_only=True)
            control_sha256=hashlib.sha256(control.contiguous().numpy().tobytes()).hexdigest()
            a=control.double()
            b=y.double();err=b-a; mse=float(err.square().mean());power=float(a.square().mean())
            row["control_error"]=dict(bitwise=(control.dtype==y.dtype and control.shape==y.shape and control_sha256==row["sha256"]),
                value_equal=torch.equal(control,y),control_sha256=control_sha256,max_abs=float(err.abs().max()),
                rmse=mse**.5,nrmse=(mse/max(power,1e-30))**.5,
                snr_db=10*np.log10(max(power,1e-30)/max(mse,1e-30)),
                engineering_gate_pass=float(err.abs().max())<=.001 and mse**.5<=.0001)
        return row
    class ResultAudit(GpuAudit):
        def __exit__(self,typ,exc,tb):
            result["total_process_after_import_s"]=time.perf_counter()-process_started
            result["operating_protection"]={"legacy16_count":len(original_manifest["legacy16"]),
                "additional_count":len(original_manifest["additional_target_and_source"]),
                "changed":[x for x,h in protected_original.items() if digest_file(Path(x))!=h]}
            rv=super().__exit__(typ,exc,tb)
            result["gpu_audit"]=self.report()
            result["valid_for_timing"]=exc is None and self.report()["valid_for_timing"]
            result["protected_after"]={str(x):digest_file(x) for x in sources}
            result["protected_unchanged"]=result["protected_before"]==result["protected_after"]
            (out/"protected_after.json").write_text(json.dumps(result["protected_after"],indent=2))
            if exc is not None:result["error"]={"type":type(exc).__name__,"message":str(exc)}
            (out/"result.json").write_text(json.dumps(result,indent=2))
            return rv
    with ResultAudit() as audit:
        audit.ensure_clear()
        # Import may run Applio's existing native/fold startup probe; it is owned here.
        spec=importlib.util.spec_from_file_location("applio_train_owned_v06",root/"rvc/train/train.py")
        train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)
        # Paths were resolved from the absolute research directory at import.
        # Use a basename only for checkpoint/export names inside that directory.
        train.model_name=out.name
        original_conv_forward=torch.nn.Conv1d._conv_forward
        original_transpose=torch.nn.functional.conv_transpose1d
        if args.observe_transpose:
            def observe_transpose(input,weight,bias=None,stride=1,padding=0,output_padding=0,groups=1,dilation=1):
                key=str((tuple(input.shape),tuple(weight.shape),str(input.dtype),str(weight.dtype),
                         stride,padding,output_padding,groups,dilation))
                result["actual_transpose_calls"][key]=result["actual_transpose_calls"].get(key,0)+1
                return original_transpose(input,weight,bias,stride,padding,output_padding,groups,dilation)
            torch.nn.functional.conv_transpose1d=observe_transpose
        if args.selective_fold:
            from rvc.lib.algorithm.fold_conv import FoldConv1d
            # Use the existing fold implementation only at verified module shapes.
            # Cast/autocast, weight_norm, state_dict and autograd remain the caller's.
            def selective_forward(module,input,weight,bias):
                if (tuple(input.shape)==(8,64,6400) and tuple(weight.shape) in ((64,64,7),(64,64,11))
                    and module.dilation in ((3,),(5,)) and module.stride==(1,)
                    and module.groups==1 and module.padding_mode=="zeros"
                    and module.padding==(module.dilation[0]*(weight.shape[-1]-1)//2,)):
                    key=f"B8_C64_T6400_k{weight.shape[-1]}_d{module.dilation[0]}_{input.dtype}_{weight.dtype}"
                    result["selective_fold_hits"][key]=result["selective_fold_hits"].get(key,0)+1
                    return FoldConv1d._conv_forward(module,input,weight,bias)
                return original_conv_forward(module,input,weight,bias)
            torch.nn.Conv1d._conv_forward=selective_forward
        train.train_dtype=torch.float16
        if args.compile:
            import torch._inductor.config as inductor_config
            if args.compile_native_rng:
                inductor_config.fallback_random=True
            result["effective_inductor_fallback_random"]=inductor_config.fallback_random
        if args.benchmark!="native":
            torch.backends.cudnn.benchmark=args.benchmark=="true"
        result["effective_cudnn_benchmark"]=torch.backends.cudnn.benchmark
        result["safety_env"]={k:os.environ[k] for k in
            ["MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1","MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53"]}
        if args.nonconv:
            import v08_nonconv_opt
            nonconv_handle=v08_nonconv_opt.apply(train)
            result["nonconv"]=nonconv_handle.report
        dispatch_context=None
        if args.dispatch:
            import v08_direct_conv
            os.environ["APPLIO_RDNA2_DIRECT_CONV"]="1"
            result["direct_dispatch_hits"]={}
            dispatch_context=v08_direct_conv.measured_dispatch(args.dispatch,hits=result["direct_dispatch_hits"])
            dispatch_context.__enter__()
            if args.compile_eager_conv:
                # Preserve the measured AMP/custom-autograd boundary. Dynamo may
                # compile surrounding decoder work but never trace this dispatch.
                torch.nn.Conv1d._conv_forward=torch.compiler.disable(
                    torch.nn.Conv1d._conv_forward,recursive=True)
                result["compile_eager_conv_enabled"]=True
        
        random.seed(cfg["train"]["seed"]);np.random.seed(cfg["train"]["seed"])
        torch.manual_seed(cfg["train"]["seed"])
        result["device"]=torch.cuda.get_device_name()
        result["actual_functional_conv1d"]={"module":getattr(torch.nn.functional.conv1d,"__module__",None),
                                            "name":getattr(torch.nn.functional.conv1d,"__qualname__",str(torch.nn.functional.conv1d))}
        original=train.train_and_evaluate
        captured={};pending=[]
        prof=None
        conv_handles=[]
        conv_ranges={}
        result["conv_calls"]={}
        def conv_pre(module,inputs):
            if prof is None or not inputs:return
            x=inputs[0]
            trans=isinstance(module,torch.nn.modules.conv._ConvTransposeNd)
            weight_shape=([module.in_channels,module.out_channels//module.groups] if trans else
                          [module.out_channels,module.in_channels//module.groups])+list(module.kernel_size)
            parameter=next(module.parameters())
            meta={"kind":type(module).__name__,"input":list(x.shape),
                  "weight":weight_shape,"input_dtype":str(x.dtype),
                  "weight_dtype":str(parameter.dtype),"stride":list(module.stride),
                  "padding":list(module.padding),"dilation":list(module.dilation),
                  "groups":module.groups,"output_padding":list(getattr(module,"output_padding",[]))}
            key=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()[:12]
            row=result["conv_calls"].setdefault(key,{**meta,"count":0})
            row["count"]+=1
            ctx=torch.profiler.record_function("V08_CONV_"+key);ctx.__enter__()
            conv_ranges.setdefault(id(module),[]).append(ctx)
        def conv_post(module,inputs,output):
            stack=conv_ranges.get(id(module),[])
            if stack:stack.pop().__exit__(None,None,None)
        def save_profile(pf):
            rows=[]
            for e in pf.key_averages(group_by_input_shape=True):
                rows.append(dict(name=e.key,count=e.count,device_type=str(e.device_type),
                    input_shapes=e.input_shapes,cpu_us=e.cpu_time_total,
                    self_cpu_us=e.self_cpu_time_total,gpu_us=e.device_time_total,
                    self_gpu_us=e.self_device_time_total))
            (out/"profile_rows.json").write_text(json.dumps(rows,indent=2))
            pf.export_chrome_trace(str(out/"profile_trace.json"))
            (out/"profile.txt").write_text(pf.key_averages().table(sort_by="self_device_time_total",row_limit=100)
                +"\n\n"+pf.key_averages(group_by_input_shape=True).table(sort_by="self_device_time_total",row_limit=120))
            print("PROFILE_SAVED",out,flush=True)
        def after_step(step,scaler,experiment_dir,loss_d,loss_g,grad_d,grad_g,scale_before):
            nonlocal prof
            frame=sys._getframe(1)
            components=[frame.f_locals[k].detach().float() for k in ("loss_mel","loss_fm","loss_kl","loss_gen")]
            pending.append((step,torch.stack([loss_d.detach().float(),loss_g.detach().float(),
                torch.as_tensor(grad_d,device="cuda").float(),torch.as_tensor(grad_g,device="cuda").float()]),
                scale_before,scaler.get_scale(),torch.stack(components)))
            if args.profile_at and step==args.profile_at-1:
                torch.cuda.synchronize()
                prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],record_shapes=True)
                prof.__enter__()
            elif prof is not None and step==args.profile_at-1+args.profile_steps:
                torch.cuda.synchronize();prof.__exit__(None,None,None)
                save_profile(prof);prof=None
        train.lab_hooks.after_step=after_step
        def epoch(*v,**kw):
            audit.ensure_clear()
            rank,number,hps,nets,optims,loaders,writers,cache,_,_,device,device_id,reference,fn_loss,scaler=v
            captured.update(nets=nets,reference=reference,writers=writers,loaders=loaders)
            if number==1:
                if args.profile_at:
                    for net in nets:
                        for module in net.modules():
                            if isinstance(module,torch.nn.modules.conv._ConvNd):
                                conv_handles.append(module.register_forward_pre_hook(conv_pre))
                                conv_handles.append(module.register_forward_hook(conv_post))
                result["steps_per_epoch"]=len(loaders[0])
                result["dataset_rows"]=len(loaders[0].dataset)
                result["dataset_sampler_samples"]=loaders[0].batch_sampler.num_samples
                result["optimizer_groups"]=[{k:g.get(k) for k in ("foreach","fused","capturable","eps","weight_decay")} for opt in optims for g in opt.param_groups]
                result["initial_state_sha256"]=[state_digest(m) for m in nets]
                if not args.no_reference_audio:
                    result["initial_audio"]=audio(nets[0],reference,"initial_audio")
            pending.clear();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            before=scaler.get_scale();t=time.perf_counter()
            original(*v,**kw)
            torch.cuda.synchronize();elapsed=time.perf_counter()-t
            record=dict(epoch=number,seconds=elapsed,steps=len(pending),scale_before=before,
                        scale_after=scaler.get_scale(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
            for step,values,s0,s1,components in pending:
                vals=values.cpu().tolist()
                comp=components.cpu().tolist()
                result["steps"].append(dict(step=step,epoch=number,loss_d=vals[0],loss_g=vals[1],
                    mel=comp[0],fm=comp[1],kl=comp[2],adv=comp[3],
                    grad_norm_d=vals[2] if np.isfinite(vals[2]) else None,
                    grad_norm_g=vals[3] if np.isfinite(vals[3]) else None,
                    grad_norm_d_nonfinite=None if np.isfinite(vals[2]) else str(vals[2]),
                    grad_norm_g_nonfinite=None if np.isfinite(vals[3]) else str(vals[3]),
                    scale_before=s0,scale_after=s1,
                    scale_reduced=(s0 is not None and s1<s0)))
            record["scale_reductions"]=sum(x[2] is not None and x[3]<x[2] for x in pending)
            record["nonfinite_losses"]=sum(not np.isfinite(x[k]) for x in result["steps"] if x["epoch"]==number for k in ("loss_d","loss_g"))
            result["epochs"].append(record)
            print("EPOCH_RESULT",json.dumps(record),flush=True)
            (out/"result.json").write_text(json.dumps(result,indent=2))
        train.train_and_evaluate=epoch
        try:
            # Global total_epoch controls loop length. High custom_total_epoch suppresses
            # original completion export/os._exit without modifying the training algorithm.
            train.run(0,1,str(out),str(gpre),str(dpre),10000,args.save_every_weights,train.config,torch.device("cuda"),0)
            if not args.no_reference_audio:
                result["final_audio"]=audio(captured["nets"][0],captured["reference"],"final_audio")
            result["final_state_sha256"]=[state_digest(m) for m in captured["nets"]]
            if args.export_final:
                from rvc.train.process.extract_model import extract_model
                model=captured["nets"][0]
                export_path=out/"model_final.pth"
                extract_model(ckpt=model.state_dict(),sr=40000,name=out.name,
                    model_path=str(export_path),epoch=args.epochs,step=train.global_step,
                    hps=train.config,vocoder="HiFi-GAN")
                result["model_export"]={"path":str(export_path),"sha256":digest_file(export_path)}
                torch.save({"model":model.state_dict(),"config":cfg,"epoch":args.epochs,
                    "step":train.global_step},out/"generator_full.pth")
            for writer in captured.get("writers",[]):
                if writer: writer.close()
        finally:
            if prof is not None:
                prof.__exit__(None,None,None)
            if args.nonconv:
                nonconv_handle.restore()
            if dispatch_context is not None and hasattr(dispatch_context,"__exit__"):
                dispatch_context.__exit__(None,None,None)
            torch.nn.Conv1d._conv_forward=original_conv_forward
            torch.nn.functional.conv_transpose1d=original_transpose
            if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()
    result["gpu_audit"]=audit.report();result["valid_for_timing"]=audit.report()["valid_for_timing"]
    result["db_after"]=db_inventory(db)
    result["training_epoch_sum_s"]=sum(x["seconds"] for x in result["epochs"])
    last_steps=[x for x in result["steps"] if x["epoch"]>args.epochs*0.75]
    result["last_quarter_loss_medians"]={k:statistics.median(x[k] for x in last_steps) for k in ("mel","fm","kl","adv","loss_d","loss_g")}
    result["steady_epoch_median_s"]=statistics.median(x["seconds"] for x in result["epochs"][1:])
    if args.compare:
        baseline=json.loads((args.compare/"result.json").read_text())
        result["initial_states_match_control"]=result["initial_state_sha256"]==baseline["initial_state_sha256"]
        result["speedup"]=baseline["steady_epoch_median_s"]/result["steady_epoch_median_s"]
        for loss in ("loss_d","loss_g"):
            a=statistics.median(x[loss] for x in baseline["steps"])
            b=statistics.median(x[loss] for x in result["steps"])
            result[loss+"_median_relative_delta"]=(b-a)/max(abs(a),1e-30)
    result["completed"]=True
    (out/"result.json").write_text(json.dumps(result,indent=2))
    print("DONE",out,result["steady_epoch_median_s"],result["valid_for_timing"],flush=True)


if __name__=="__main__": main()
