"""Three-shape captured-input FP16 Attention comparison on existing gfx1030."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.profiler import profile, ProfilerActivity
import triton
from gpu_guard_v06 import GpuAudit
from research_comfy_v05 import profile_tables
from rdna2_ops.fp16_attention_v06 import fp16_attention, METHODS


def error(y, ref):
    y, ref = y.detach().cpu(), ref.detach().cpu()
    finite = bool(torch.isfinite(y).all())
    if not finite:
        return dict(finite=False, byte_equal=False, max_abs=None, rmse=None,
                    relative_rmse=None, operator_gate=False)
    d = y.double() - ref.double()
    rms = d.square().mean().sqrt().item()
    relative = rms / max(ref.double().square().mean().sqrt().item(), 1e-12)
    maximum = d.abs().max().item()
    return dict(finite=True, byte_equal=y.dtype == ref.dtype and torch.equal(y, ref),
                max_abs=maximum, rmse=rms, relative_rmse=relative,
                operator_gate=maximum <= .02 and relative <= .005)


def restore(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    result = []
    for i, name in enumerate(("q", "k", "v")):
        x = data[name]
        stride = data["metadata"]["strides"][i]
        dest = torch.empty_strided(x.shape, stride, device="cuda", dtype=x.dtype)
        dest.copy_(x)
        if dest.ndim == 3:
            heads = data["metadata"]["heads"]
            dest = dest.reshape(dest.shape[0], dest.shape[1], heads, -1).transpose(1,2)
        result.append(dest)
    return result, data


def native(q,k,v):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q,k,v)


def sampled_fp64(q,k,v):
    # Independent CPU FP64 reference on 2 heads x 4 rows, using every key.
    b = 0
    hs = [0, q.shape[1]-1]
    rows = sorted(set([0, q.shape[-2]//3, 2*q.shape[-2]//3, q.shape[-2]-1]))
    outputs = []
    for h in hs:
        qq = q[b,h,rows].double().cpu()
        kk, vv = k[b,h].double().cpu(), v[b,h].double().cpu()
        outputs.append(torch.softmax(qq @ kk.T * q.shape[-1]**-.5,-1) @ vv)
    return hs, rows, torch.stack(outputs)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--rounds",type=int,default=3)
    ap.add_argument("--repeat",type=int,default=2)
    cfg=ap.parse_args()
    if cfg.out.exists():
        raise FileExistsError("Use a fresh result file")
    cfg.out.parent.mkdir(parents=True,exist_ok=True)
    report={"status":"started", "contract":"FP16 BHLD forward, no mask/dropout/GQA; finite actual inputs",
            "reference":"native math SDPA with reduced-precision flag True; Anima manual_vt is speed baseline",
            "operator_gate":{"max_abs":.02,"relative_rmse":.005},
            "final_rgb_gate":{"max_abs":2/255,"rmse":.25/255,"status":"separate model A/B"},
            "timing":"3 warmups; alternating forward/reverse method order per round; all per-call packing/allocations included",
            "memory":"additional PyTorch allocated peak above inputs; driver/private allocation excluded",
            "rounds":cfg.rounds,"repeat":cfg.repeat,"shapes":[],
            "source_sha256":{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                for name in ("bench_fp16_attention_v06.py","rdna2_ops/fp16_attention_v06.py","gpu_guard_v06.py")}}
    def save():cfg.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    old=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
    audit=GpuAudit()
    try:
        with audit, torch.inference_mode():
            audit.ensure_clear()
            report["runtime"]={"torch":str(torch.__version__),"hip":torch.version.hip,
                "triton":triton.__version__,"device":torch.cuda.get_device_name(),
                "arch":torch.cuda.get_device_properties(0).gcnArchName,
                "reduced_precision_math":torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()}
            cases=[("self4096",4096,4096), ("cross512",4096,512), ("cropped_self2048",2048,2048)]
            for label, length, keys in cases:
                qkv,data=restore(cfg.capture_dir/("actual_qkv_4096_512.pt" if keys==512
                                               else "actual_qkv_4096_4096.pt"))
                q,k,v=qkv
                if length!=4096:
                    q,k,v=q[...,:length,:],k[...,:keys,:],v[...,:keys,:]
                row={"label":label,"q":list(q.shape),"k":list(k.shape),
                     "strides":[list(t.stride()) for t in (q,k,v)],
                     "data":"actual capture" if length==4096 else "prefix slice of actual capture; representative operator shape",
                     "input_stats":[{"max_abs":float(t.abs().max()),"finite":bool(torch.isfinite(t).all())} for t in (q,k,v)],
                     "methods":{}}
                functions={"native_math":lambda:native(q,k,v)}
                functions.update({m:(lambda m=m:fp16_attention(q,k,v,method=m)) for m in METHODS})
                reference=functions["native_math"]().cpu()
                manual=functions["manual_vt"]().cpu()
                hs, selected_rows, ref64=sampled_fp64(q,k,v)
                for name,fn in functions.items():
                    for _ in range(3):fn()
                    torch.cuda.synchronize()
                    y=fn()
                    sampled=torch.stack([y[0,h,selected_rows].double().cpu() for h in hs])
                    row["methods"][name]={"vs_native":error(y,reference),"vs_manual_vt":error(y,manual),
                        "vs_sampled_fp64":error(sampled,ref64),"samples":[]}
                    del y
                for rnd in range(cfg.rounds):
                    order=list(functions)
                    if rnd%2:order.reverse()
                    for name in order:
                        audit.ensure_clear()
                        fn=functions[name]
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize();start=time.perf_counter();begin.record()
                        for _ in range(cfg.repeat):y=fn()
                        end.record();torch.cuda.synchronize()
                        row["methods"][name]["samples"].append({"round":rnd,
                            "wall_ms":(time.perf_counter()-start)*1000/cfg.repeat,
                            "event_ms":begin.elapsed_time(end)/cfg.repeat})
                        del y
                for name,fn in functions.items():
                    result=row["methods"][name]
                    result["median_wall_ms"]=statistics.median(s["wall_ms"] for s in result["samples"])
                    result["median_event_ms"]=statistics.median(s["event_ms"] for s in result["samples"])
                    gc.collect();torch.cuda.synchronize();base=torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats();y=fn();torch.cuda.synchronize()
                    result["peak_additional_allocated_bytes"]=torch.cuda.max_memory_allocated()-base
                    del y
                    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
                        y=fn();torch.cuda.synchronize()
                    tables=profile_tables(prof)
                    result["kernel_profile"]=tables["device_activities"]
                    del y
                    if name.startswith("fused_") or "inplace" in name:
                        y,kernels=fp16_attention(q,k,v,method=name,return_kernels=True)
                        result["triton_resources"]=[{"registers":kernel.n_regs,"spills":kernel.n_spills,
                            "shared":kernel.metadata.shared,"asm_sha256":hashlib.sha256(kernel.asm["amdgcn"].encode()).hexdigest()}
                            for kernel in kernels]
                        del y,kernels
                baseline=row["methods"]["manual_vt"]["median_wall_ms"]
                for result in row["methods"].values():
                    result["speedup_vs_manual_vt"]=baseline/result["median_wall_ms"]
                report["shapes"].append(row);save()
                print("FP16_ATTENTION "+json.dumps({"shape":label,"times":{n:round(r["median_wall_ms"],4) for n,r in row["methods"].items()},
                       "errors":{n:r["vs_native"]["max_abs"] for n,r in row["methods"].items()}}),flush=True)
                del functions,qkv,q,k,v,reference,manual,ref64
                gc.collect();torch.cuda.empty_cache()
            audit.ensure_clear();report["status"]="measured"
    except Exception as exc:
        report["status"]="failed";report["error"]=repr(exc)
        raise
    finally:
        report["gpu_audit"]=audit.report()
        if not report["gpu_audit"]["valid_for_timing"]:
            report["status"]="discarded_overlap_or_unavailable_audit"
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(old)
        save()


if __name__=="__main__":main()
