#!/usr/bin/env python3
"""Conv catalog measurements in an existing interpreter; explicit process-local DB.

Seed only dilation=1 with NORMAL, then replay FAST using the same research DB.
Run a FAST control with a different research DB. Never toggle FindMode per call.
All GPU invocations require the parent's serial lease and hold GpuAudit's lock.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time


def cases(which):
    def c(name, shape, weight, *, kind="conv1d", groups=1, dilation=1,
          stride=1, padding=0, origin="Applio HiFi-GAN 40k, segment=12800, B=8"):
        return dict(name=name, shape=shape, weight=weight, kind=kind,
                    groups=groups, dilation=dilation, stride=stride,
                    padding=padding, origin=origin)
    if which in ("resblock", "seed", "all_d1"):
        bases = [(64, 6400, 7), (64, 6400, 11), (32, 12800, 3),
                 (128, 3200, 11), (64, 6400, 3), (32, 12800, 11)]
        if which in ("seed", "all_d1"):
            bases = [(C, T, k) for C, T in [(256, 320), (128, 3200),
                                           (64, 6400), (32, 12800)] for k in (3, 7, 11)]
        return [c(f"res_C{C}_T{T}_k{k}_d{d}", (8, C, T), (C, C, k),
                  dilation=d, padding=d * (k - 1) // 2)
                for C, T, k in bases for d in ((1,) if which != "resblock" else (1, 3, 5))]
    if which == "patch_embed":
        return [c(f"patch_embed_N{N}", (N,3,2,16,16), (1152,3,2,16,16),
                  kind="conv3d", stride=(2,16,16),
                  origin="Existing Qwen3-VL exact-tile patch-embed; synthetic data, batch sweep")
                for N in (128,512,2400)]
    result = []
    for C, O, H, P in [(32,128,2134,2), (128,512,712,2), (512,1024,238,2)]:
        result.append(c(f"period_C{C}_O{O}_H{H}_P{P}", (8,C,H,P), (O,C,5,1),
                        kind="conv2d", stride=(3,1), padding=(2,0)))
    for C, O, T, G in [(16,64,12800,4), (64,256,3200,16), (256,1024,800,64)]:
        result.append(c(f"grouped_C{C}_O{O}_T{T}_G{G}", (8,C,T), (O,C//G,41),
                        groups=G, stride=4, padding=20))
    for C, T, k in [(32,12800,3), (64,6400,7), (128,3200,11)]:
        result.append(c(f"depthwise_C{C}_T{T}_k{k}", (8,C,T), (C,1,k), groups=C,
                        padding=k//2, origin="Synthetic depthwise analog of Applio residual shapes; model A/B not covered"))
    for C, O, T, k, s, p in [(512,256,32,16,10,3), (256,128,320,16,10,3),
                             (128,64,3200,4,2,1)]:
        result.append(c(f"transpose_C{C}_O{O}_T{T}_k{k}", (8,C,T), (C,O,k),
                        kind="conv_transpose1d", stride=s, padding=p))
    for C, O, D, H, W in [(16,32,1,32,32),(32,32,4,32,32),(64,32,8,16,16)]:
        result.append(c(f"conv3d_C{C}_O{O}_D{D}_H{H}_W{W}", (1,C,D,H,W), (O,C,3,3,3),
                        kind="conv3d", padding=1,
                        origin="Synthetic bounded Conv3D survey; no new model; model A/B not covered"))
    return [row for row in result if row["kind"] == "conv_transpose1d"] if which == "transpose" else result


def db_inventory(folder):
    rows = []
    for path in sorted(Path(folder).rglob("*")):
        if path.is_file():
            raw = path.read_bytes()
            rows.append(dict(path=str(path.relative_to(folder)), bytes=len(raw),
                             sha256=hashlib.sha256(raw).hexdigest()))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", choices=["resblock", "seed", "all_d1", "survey", "patch_embed", "transpose"], default="resblock")
    p.add_argument("--case", help="Optional exact case name; --list prints all")
    p.add_argument("--list", action="store_true")
    p.add_argument("--mode", choices=["FAST", "NORMAL"], default="FAST")
    p.add_argument("--db", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--candidate", choices=["native", "fold", "channels_last", "linear_exact"], default="native")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--compare", type=Path, help="Reference run output directory")
    p.add_argument("--no-profile", action="store_true")
    p.add_argument("--forward-only", action="store_true")
    p.add_argument("--with-bias", action="store_true")
    p.add_argument("--miopen-enabled", choices=["yes","no"], default="yes",
                   help="Explicit research/caller setting; record the actual Comfy flag separately")
    args = p.parse_args()
    selected = [c for c in cases(args.set) if not args.case or c["name"] == args.case]
    if args.list:
        print(json.dumps(selected, indent=2)); return
    if not selected or args.db is None or args.out is None:
        p.error("Nonempty cases, --db and --out are required")
    root = Path(__file__).resolve().parent
    for path in (args.db, args.out):
        if not path.resolve().is_relative_to(root):
            p.error("All written DB/results must remain under this research folder")
        path.mkdir(parents=True, exist_ok=True)
    if args.set == "seed" and (args.mode != "NORMAL" or args.candidate != "native"):
        p.error("seed means native NORMAL on dilation=1 only")
    os.environ["MIOPEN_FIND_MODE"] = args.mode
    os.environ["MIOPEN_USER_DB_PATH"] = str(args.db.resolve())
    os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = str((args.db.parent / "miopen_cache").resolve())
    Path(os.environ["MIOPEN_CUSTOM_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    import torch
    import torch.nn.functional as F
    from gpu_guard_v06 import GpuAudit
    from research_comfy_v05 import device_activity_table
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = args.miopen_enabled == "yes"
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(schema="conv_catalog_v06", mode=args.mode, candidate=args.candidate,
                  timing_scope="forward including conversions" if args.forward_only else "forward+dX+dW including conversions",
                  with_bias=args.with_bias,
                  cudnn_enabled=torch.backends.cudnn.enabled,
                  dtype=args.dtype, torch=torch.__version__, hip=torch.version.hip,
                  cases=[], db_before=db_inventory(args.db),
                  gate={"finite":True, "reference":"FP32 same native operation on identical rounded inputs",
                        "max_over_reference_rms":0.02, "nrmse":0.002,
                        "scope":"operator screening only; no model quality claim"})
    def error(a, b):
        bitwise = (a.dtype == b.dtype and a.shape == b.shape and
                   torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))
        value_equal = torch.equal(a,b)
        a, b = a.double(), b.double()
        d = a - b; rms = float(b.square().mean().sqrt())
        nr = float(d.square().mean().sqrt()) / max(rms, 1e-30)
        mx = float(d.abs().max()) / max(rms, 1e-30)
        return dict(bitwise=bitwise, value_equal=value_equal, finite=bool(torch.isfinite(a).all()),
                    max_abs=float(d.abs().max()), rmse=float(d.square().mean().sqrt()),
                    reference_rms=rms, nrmse=nr, max_over_reference_rms=mx,
                    screen_pass=bool(torch.isfinite(a).all()) and nr <= .002 and mx <= .02)
    class ResultAudit(GpuAudit):
        def __exit__(self, typ, exc, tb):
            result = super().__exit__(typ, exc, tb)
            report["gpu_audit"] = self.report()
            report["db_after"] = db_inventory(args.db)
            report["valid_for_timing"] = exc is None and self.report()["valid_for_timing"]
            if exc is not None: report["error"] = {"type":type(exc).__name__,"message":str(exc)}
            (args.out/"result.json").write_text(json.dumps(report,indent=2))
            return result
    with ResultAudit() as audit:
        audit.ensure_clear()
        report["device"] = torch.cuda.get_device_name()
        for case in selected:
            audit.ensure_clear()
            name = case["name"]
            report["active_case"] = name
            print("BEGIN", name, flush=True)
            gen = torch.Generator(device="cpu").manual_seed(63117)
            xcpu = torch.randn(case["shape"], generator=gen).to(dtype)
            fan_in = math.prod(case["weight"][1:])
            wcpu = (torch.randn(case["weight"], generator=gen) / math.sqrt(fan_in)).to(dtype)
            out_channels=case["weight"][1]*case["groups"] if case["kind"].startswith("conv_transpose") else case["weight"][0]
            bcpu=(torch.randn((out_channels,),generator=gen)*.1).to(dtype) if args.with_bias else None
            bias=bcpu.cuda() if bcpu is not None else None
            x, w = xcpu.cuda().requires_grad_(), wcpu.cuda().requires_grad_()
            kw = {k: case[k] for k in ("groups", "dilation", "stride", "padding")}
            native = getattr(F, case["kind"])
            def forward(xx, ww):
                if args.candidate == "linear_exact" and case["kind"] == "conv3d" and xx.shape[2:] == ww.shape[2:] and case["groups"] == 1 and case["padding"] == 0 and case["dilation"] == 1:
                    return F.linear(xx.flatten(1),ww.flatten(1),bias).reshape(xx.shape[0],ww.shape[0],1,1,1)
                if args.candidate == "fold" and case["kind"] == "conv1d" and case["dilation"] > 1 and case["groups"] == 1 and case["stride"] == 1:
                    d, pad = case["dilation"], case["padding"]
                    padded = F.pad(xx, (pad,pad))
                    B,C,T = padded.shape
                    padded = F.pad(padded, (0,(-T)%d))
                    folded = padded.reshape(B,C,-1,d).permute(0,3,1,2).reshape(B*d,C,-1)
                    y = F.conv1d(folded, ww,bias)
                    return y.reshape(B,d,y.shape[1],-1).permute(0,2,3,1).reshape(B,y.shape[1],-1)[...,:xx.shape[-1]]
                if args.candidate == "channels_last" and xx.ndim in (4,5):
                    fmt = torch.channels_last if xx.ndim == 4 else torch.channels_last_3d
                    return native(xx.contiguous(memory_format=fmt), ww.contiguous(memory_format=fmt),bias, **kw)
                return native(xx, ww,bias, **kw)
            cold = time.perf_counter()
            y = forward(x,w)
            dycpu = (torch.randn(y.shape, generator=gen) * .01).to(dtype)
            dy = dycpu.cuda()
            def step():
                yy = forward(x,w)
                dx,dw = torch.autograd.grad(yy,(x,w),dy)
                return yy,dx,dw
            def timed_step():
                if args.forward_only:
                    with torch.inference_mode(): return forward(x,w)
                return step()
            measured = step()
            torch.cuda.synchronize()
            cold_s = time.perf_counter() - cold
            del y
            if args.set == "seed":
                report["cases"].append(dict(case=case, first_forward_backward_s=cold_s))
                del measured, x,w,dy,xcpu,wcpu,dycpu
                gc.collect(); torch.cuda.empty_cache()
                (args.out/"result.json").write_text(json.dumps(report,indent=2))
                print("SEEDED",name,cold_s,flush=True)
                continue
            for _ in range(3): timed_step()
            torch.cuda.synchronize()
            timings, events = [], []
            for _ in range(args.rounds):
                a,b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(); t = time.perf_counter(); a.record()
                for _ in range(args.reps): timed_step()
                b.record(); b.synchronize()
                timings.append((time.perf_counter()-t)*1000/args.reps)
                events.append(a.elapsed_time(b)/args.reps)
            torch.cuda.reset_peak_memory_stats()
            base_mem = torch.cuda.memory_allocated()
            temp = timed_step(); torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()-base_mem
            del temp
            outputs = [t.detach().cpu() for t in measured]
            del measured
            torch.save(dict(y=outputs[0],dx=outputs[1],dw=outputs[2]), args.out/(name+".pt"))
            row = dict(case=case, cold_s=cold_s, ms_rounds=timings, event_ms_rounds=events,
                       median_ms=statistics.median(timings), event_median_ms=statistics.median(events),
                       peak_extra_allocated_bytes=peak)
            if not args.no_profile:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    timed_step(); torch.cuda.synchronize()
                trace = args.out/(name+"_trace.json")
                prof.export_chrome_trace(str(trace))
                row["device_activities"] = device_activity_table(trace)
            if args.set == "patch_embed":
                xf,wf = xcpu.double().requires_grad_(), wcpu.double().requires_grad_()
                yf = F.linear(xf.flatten(1),wf.flatten(1),bcpu.double() if bcpu is not None else None).reshape(x.shape[0],w.shape[0],1,1,1)
                dxf,dwf = torch.autograd.grad(yf,(xf,wf),dycpu.double())
                row["reference"] = "CPU FP64 equivalent exact-tile linear, identical rounded inputs"
            else:
                xf,wf = x.detach().float().requires_grad_(), w.detach().float().requires_grad_()
                yf = native(xf,wf,bias.float() if bias is not None else None,**kw)
                dxf,dwf = torch.autograd.grad(yf,(xf,wf),dy.float())
                row["reference"] = "GPU FP32 native convolution, identical rounded inputs"
            row["reference_error"] = {k:error(a,b.detach().cpu()) for k,a,b in zip(("y","dx","dw"),outputs,(yf,dxf,dwf))}
            if args.compare:
                control = torch.load(args.compare/(name+".pt"), map_location="cpu", weights_only=True)
                row["control_error"] = {k:error(a,control[k]) for k,a in zip(("y","dx","dw"),outputs)}
            report["cases"].append(row)
            print("RESULT",name,row["median_ms"],json.dumps(row["reference_error"]),flush=True)
            del xf,wf,yf,dxf,dwf,x,w,dy,outputs,xcpu,wcpu,dycpu
            gc.collect(); torch.cuda.empty_cache()
            (args.out/"result.json").write_text(json.dumps(report,indent=2))
    report["gpu_audit"] = audit.report()
    report["db_after"] = db_inventory(args.db)
    report["valid_for_timing"] = report["gpu_audit"]["valid_for_timing"]
    (args.out/"result.json").write_text(json.dumps(report,indent=2))
    print("DONE",args.out,report["valid_for_timing"],flush=True)


if __name__ == "__main__":
    main()
