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
    p.add_argument("--applio", type=Path, default=Path("~/DevSpace/applio-lab"))
    p.add_argument("--dataset", default="bench40k")
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
    args = p.parse_args()
    if args.fold and args.selective_fold: p.error("Choose blanket or measured selective fold")
    if args.compare is not None: args.compare=args.compare.resolve()
    research = Path(__file__).resolve().parent
    root = args.applio.resolve(); out = args.out.resolve(); db = args.db.resolve()
    if not out.is_relative_to(research) or not db.is_relative_to(research):
        p.error("Outputs and DB must be inside the research folder")
    if out.exists(): p.error("Use a new run directory; refusing resume or overwrite")
    if args.epochs < 2: p.error("At least one warm epoch and one measured epoch required")
    source_data = root/"logs"/args.dataset
    gpre = root/"rvc/models/pretraineds/hifi-gan/f0G40k.pth"
    dpre = root/"rvc/models/pretraineds/hifi-gan/f0D40k.pth"
    sources = [root/"assets/config.json", source_data/"config.json", source_data/"filelist.txt",
               source_data/"model_info.json", root/"rvc/train/train.py",
               root/"rvc/lib/zluda.py", root/"rvc/lib/algorithm/fold_conv.py",
               root/"lab/train_bench.sh", root/"core.py", root/"rvc/train/lab_hooks.py",
               root/"rvc/train/data_utils.py", root/"rvc/lib/algorithm/synthesizers.py",
               root/"rvc/lib/algorithm/discriminators.py", root/"rvc/lib/algorithm/residuals.py",
               gpre,dpre]
    protected = {str(x):digest_file(x) for x in sources}
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
                      APPLIO_COMPILE="", APPLIO_PROFILE_AT="0", APPLIO_AMP_LOG="0",
                      MPLBACKEND="Agg", MASTER_ADDR="127.0.0.1")
    Path(os.environ["MIOPEN_CUSTOM_CACHE_DIR"]).mkdir(parents=True,exist_ok=True)
    with socket.socket() as s:
        s.bind(("127.0.0.1",0)); os.environ["MASTER_PORT"] = str(s.getsockname()[1])
    os.chdir(root)
    sys.path[:0] = [str(root),str(root/"rvc/train")]
    # Original train.py's CLI is parsed on import. Its main()/cleanup is never called.
    sys.argv = [str(root/"rvc/train/train.py"),str(out),"10000",str(args.epochs),
                str(gpre),str(dpre),"0","8","40000","False","False","False",
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
    result = dict(schema="applio_epoch_v06", mode=args.mode, fold=args.fold,
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
        torch.backends.cudnn.benchmark=False
        random.seed(cfg["train"]["seed"]);np.random.seed(cfg["train"]["seed"])
        torch.manual_seed(cfg["train"]["seed"])
        result["device"]=torch.cuda.get_device_name()
        result["actual_functional_conv1d"]={"module":getattr(torch.nn.functional.conv1d,"__module__",None),
                                            "name":getattr(torch.nn.functional.conv1d,"__qualname__",str(torch.nn.functional.conv1d))}
        original=train.train_and_evaluate
        captured={};pending=[]
        def after_step(step,scaler,experiment_dir,loss_d,loss_g,grad_d,grad_g,scale_before):
            pending.append((step,torch.stack([loss_d.detach().float(),loss_g.detach().float(),
                torch.as_tensor(grad_d,device="cuda").float(),torch.as_tensor(grad_g,device="cuda").float()]),
                scale_before,scaler.get_scale()))
        train.lab_hooks.after_step=after_step
        def epoch(*v,**kw):
            audit.ensure_clear()
            rank,number,hps,nets,optims,loaders,writers,cache,_,_,device,device_id,reference,fn_loss,scaler=v
            captured.update(nets=nets,reference=reference,writers=writers,loaders=loaders)
            if number==1:
                result["initial_state_sha256"]=[state_digest(m) for m in nets]
                result["initial_audio"]=audio(nets[0],reference,"initial_audio")
            pending.clear();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            before=scaler.get_scale();t=time.perf_counter()
            original(*v,**kw)
            torch.cuda.synchronize();elapsed=time.perf_counter()-t
            record=dict(epoch=number,seconds=elapsed,steps=len(pending),scale_before=before,
                        scale_after=scaler.get_scale(),peak_allocated_bytes=torch.cuda.max_memory_allocated())
            for step,values,s0,s1 in pending:
                vals=values.cpu().tolist()
                result["steps"].append(dict(step=step,epoch=number,loss_d=vals[0],loss_g=vals[1],
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
            train.run(0,1,str(out),str(gpre),str(dpre),10000,False,train.config,torch.device("cuda"),0)
            result["final_audio"]=audio(captured["nets"][0],captured["reference"],"final_audio")
            result["final_state_sha256"]=[state_digest(m) for m in captured["nets"]]
            for writer in captured.get("writers",[]):
                if writer: writer.close()
        finally:
            torch.nn.Conv1d._conv_forward=original_conv_forward
            torch.nn.functional.conv_transpose1d=original_transpose
            if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()
    result["gpu_audit"]=audit.report();result["valid_for_timing"]=audit.report()["valid_for_timing"]
    result["db_after"]=db_inventory(db)
    result["steady_epoch_median_s"]=statistics.median(x["seconds"] for x in result["epochs"][1:])
    if args.compare:
        baseline=json.loads((args.compare/"result.json").read_text())
        result["initial_states_match_control"]=result["initial_state_sha256"]==baseline["initial_state_sha256"]
        result["speedup"]=baseline["steady_epoch_median_s"]/result["steady_epoch_median_s"]
        for loss in ("loss_d","loss_g"):
            a=statistics.median(x[loss] for x in baseline["steps"])
            b=statistics.median(x[loss] for x in result["steps"])
            result[loss+"_median_relative_delta"]=(b-a)/max(abs(a),1e-30)
    (out/"result.json").write_text(json.dumps(result,indent=2))
    print("DONE",out,result["steady_epoch_median_s"],result["valid_for_timing"],flush=True)


if __name__=="__main__": main()
