#!/usr/bin/env python3
"""BF16 Conv2d operator screen and the existing actual Comfy VAE A/B harness.

Use --micro first. Model mode reuses dtype_vae_v06.main without changing its file;
only its context-class binding is changed inside this owned process.
"""
from __future__ import annotations
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time


def micro(output):
    import torch
    import torch.nn.functional as F
    from gpu_guard_v06 import GpuAudit
    from rdna2_ops.bf16_conv_v06 import bf16_conv2d
    from research_comfy_v05 import device_activity_table
    native=F.conv2d
    shapes=[(384,128),(96,512),(192,256)]
    report={"status":"started","scope":"BF16 Conv2D forward; actual MOSS VAE shapes, synthetic reproducible inputs",
            "shape_source":"results/v06_bf16_vae_sdpa_fp32/results.json regular native operator input_shapes",
            "torch":torch.__version__,"hip":torch.version.hip,"rows":[],
            "miopen_find_mode":os.environ.get("MIOPEN_FIND_MODE"),
            "final_rgb_gate":{"max_abs":2/255,"rmse":.25/255},
            "operator_gate":{"finite":True,"nrmse":.002,"max_over_reference_rms":.05},
            "timing_scope":"Each candidate includes input/weight conversions and output/bias operations; no weight cache"}
    def err(a,b):
        d=a.float()-b.float();rms=float(b.float().square().mean().sqrt())
        rmse=float(d.square().mean().sqrt());maximum=float(d.abs().max())
        return {"finite":bool(torch.isfinite(a).all()),
                "bit_equal":(a.dtype==b.dtype and a.shape==b.shape and
                    torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))),
                "value_equal":torch.equal(a,b),
                "max_abs":maximum,"rmse":rmse,"nrmse":rmse/max(rms,1e-30),
                "max_over_reference_rms":maximum/max(rms,1e-30)}
    audit=GpuAudit()
    try:
        with audit,torch.inference_mode():
            audit.ensure_clear();torch.set_num_threads(2)
            torch.backends.cudnn.benchmark=False
            torch.backends.cuda.matmul.allow_tf32=False
            report["device"]=torch.cuda.get_device_name()
            for C,H in shapes:
                audit.ensure_clear()
                gen=torch.Generator(device="cpu").manual_seed(821+H)
                xcpu=torch.randn((1,C,H,H),generator=gen).bfloat16()
                wcpu=(torch.randn((C,C,3,3),generator=gen)/(9*C)**.5).bfloat16()
                bcpu=(torch.randn((C,),generator=gen)*.1).bfloat16()
                x,w,b=xcpu.cuda(),wcpu.cuda(),bcpu.cuda()
                funcs={"native":lambda:native(x,w,b,padding=1),
                       "fp32_bf16_bias":lambda:bf16_conv2d(x,w,b,padding=1,bias_rounding="bf16",native=native),
                       "fp32_bias":lambda:bf16_conv2d(x,w,b,padding=1,bias_rounding="fp32",native=native)}
                control=funcs["native"]()
                reconstruct=native(x,w,None,padding=1).add_(b.reshape(1,-1,1,1))
                row={"shape":[1,C,H,H],"weight":[C,C,3,3],"bias":[C],"padding":1,
                     "native_bias_reconstruction":err(reconstruct,control),"methods":{}}
                if not row["native_bias_reconstruction"]["bit_equal"]:
                    raise AssertionError("Observed native bias ordering differs from reconstructed source contract")
                del reconstruct
                values={k:fn() for k,fn in funcs.items()}
                for fn in funcs.values():
                    for _ in range(3):fn()
                samples={k:[] for k in funcs};gpu_samples={k:[] for k in funcs}
                for rnd in range(5):
                    for key in list(funcs)[::1 if rnd%2==0 else -1]:
                        fn=funcs[key];torch.cuda.synchronize()
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        t=time.perf_counter();start.record()
                        for _ in range(5):fn()
                        end.record();end.synchronize()
                        samples[key].append((time.perf_counter()-t)*1000/5)
                        gpu_samples[key].append(start.elapsed_time(end)/5)
                for key,fn in funcs.items():
                    torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
                    y=fn();torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()-base;del y
                    trace=output/f"C{C}_H{H}_{key}.trace.json"
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
                        fn();torch.cuda.synchronize()
                    prof.export_chrome_trace(str(trace))
                    e=err(values[key],control)
                    row["methods"][key]={"wall_ms":samples[key],"gpu_event_ms":gpu_samples[key],
                        "median_ms":statistics.median(samples[key]),"peak_extra_allocated_bytes":peak,
                        "native_error":e,"operator_screen_pass":e["finite"] and e["nrmse"]<=.002 and e["max_over_reference_rms"]<=.05,
                        "device_activities":device_activity_table(trace)}
                baseline=row["methods"]["native"]["median_ms"]
                for v in row["methods"].values():v["speedup"]=baseline/v["median_ms"]
                report["rows"].append(row)
                print(json.dumps({"shape":row["shape"],"times_ms":{k:v["median_ms"] for k,v in row["methods"].items()},
                                  "error":{k:v["native_error"] for k,v in row["methods"].items()}}),flush=True)
                (output/"results.json").write_text(json.dumps(report,indent=2,allow_nan=False))
                del x,w,b,control,values,xcpu,wcpu,bcpu;torch.cuda.empty_cache()
            report["status"]="measured"
    except Exception as exc:
        report.update(status="failed",error=f"{type(exc).__name__}: {exc}");raise
    finally:
        report["gpu_audit"]=audit.report()
        if not audit.report()["valid_for_timing"]:report["status"]="invalid_timing_discarded"
        (output/"results.json").write_text(json.dumps(report,indent=2,allow_nan=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--micro",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--project",type=Path)
    p.add_argument("--latent",type=Path)
    p.add_argument("--paths",choices=["regular","regular,tiled"],default="regular,tiled")
    p.add_argument("--load-mode",choices=["normal","lowvram"],default="normal")
    p.add_argument("--bias-rounding",choices=["bf16","fp32"],default="bf16")
    p.add_argument("--find-mode",choices=["FAST","NORMAL"],default="FAST")
    cfg=p.parse_args();root=Path(__file__).resolve().parent;output=cfg.output.resolve()
    if not output.is_relative_to(root):p.error("Output must remain in the research folder")
    if (output/"results.json").exists():p.error("Use a fresh output directory")
    output.mkdir(parents=True,exist_ok=True)
    os.environ["MIOPEN_FIND_MODE"]=cfg.find_mode
    os.environ["MIOPEN_USER_DB_PATH"]=str(output/"miopen_db")
    os.environ["MIOPEN_CUSTOM_CACHE_DIR"]=str(root/"results/conv_catalog_v06/vae_miopen_cache")
    for key in ("MIOPEN_USER_DB_PATH","MIOPEN_CUSTOM_CACHE_DIR"):
        Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
    if cfg.micro:
        micro(output);return
    if cfg.project is None or cfg.latent is None:p.error("Model mode requires --project and --latent")
    import dtype_vae_v06 as runner
    from rdna2_ops.bf16_conv_v06 import BF16Conv2dMode
    runner.BF16GemmMode=partial(BF16Conv2dMode,bias_rounding=cfg.bias_rounding)
    sys.argv=[str(root/"dtype_vae_v06.py"),"--project",str(cfg.project.resolve()),"--latent",str(cfg.latent.resolve()),
              "--output",str(output),"--method","fp32","--paths",cfg.paths,"--load-mode",cfg.load_mode]
    try:
        runner.main()
    finally:
        path=output/"results.json"
        if path.exists():
            report=json.loads(path.read_text())
            report.update(boundary="torch.nn.functional.conv2d",bias_rounding=cfg.bias_rounding,
                scope="Existing MOSS latent and actual BF16 VAE; Conv2D compute FP32 only, caller dtype/casting/offload retained",
                timing_scope="Native timings without observer; candidate includes functional hook and conversions, no weight cache")
            report["source_sha256"].update({n:hashlib.sha256((root/n).read_bytes()).hexdigest()
                for n in ("conv_vae_v06.py","rdna2_ops/bf16_conv_v06.py")})
            for row in report.get("rows",[]):
                if all(k in row for k in ("error","native_output","candidate_output")):
                    row["error"].setdefault("value_equal",row["error"]["bit_equal"])
                    row["error"]["bit_equal"]=(row["native_output"]["float32_sha256"]==
                                                row["candidate_output"]["float32_sha256"])
            path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":main()
