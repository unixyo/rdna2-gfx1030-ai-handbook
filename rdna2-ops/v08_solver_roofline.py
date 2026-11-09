#!/usr/bin/env python3
"""CPU-only shape/direction roofline-style accounting for v0.8 solver traces.

Dense-equivalent FLOPs / register-dot2 throughput is an optimistic direct-conv
compute estimate, not a universal algorithmic lower bound (notably Winograd).
Minimum tensor traffic / streaming bandwidth assumes cold DRAM; a second
estimate uses the separately measured cache-resident copy bandwidth.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from v08_solver_select import owned, dump, output_shape, DIRECTIONS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result",type=Path,required=True)
    p.add_argument("--calibration",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    source=owned(a.result);calib=owned(a.calibration);out=owned(a.out)
    if out.exists():p.error("Use a new output directory")
    out.mkdir(parents=True)
    d=json.loads(source.read_text()); h=json.loads(calib.read_text())
    if not d.get("valid_for_timing"):raise ValueError("Timing audit must pass")
    tflops=h["dot2"]["measured_tflop_s"]
    dram=h["copy"]["measured_GB_s"]
    cache=h["copy_cached"]["measured_GB_s"]
    rows=[]
    for c in d["cases"]:
        case=c["case"];x=case["shape"];w=case["weight"];y=output_shape(case)
        trans="transpose" in case["kind"]
        sites=math.prod(x[2:] if trans else y[2:])
        nominal_flops=2*x[0]*sites*math.prod(w)
        minimum_tensor_bytes=2*(math.prod(x)+math.prod(w)+math.prod(y))
        compute=nominal_flops/(tflops*1e9)
        dram_ms=minimum_tensor_bytes/(dram*1e6)
        cache_ms=minimum_tensor_bytes/(cache*1e6)
        for direction in DIRECTIONS:
            q=c["directions"][direction]
            device=q.get("device_activities",{})
            kernels=device.get("rows",[])
            names=[k["name"] for k in kernels]
            winograd=any("Sp3AsmConv" in n or "Winograd" in n for n in names)
            im2col_ms=sum(k["self_gpu_ms"] for k in kernels
                         if any(z in k["name"].lower() for z in ("im2col","im2d2col","col2im","col2imd")))
            actual=q["event_median_ms"]
            direct_dram=max(compute,dram_ms);direct_cache=max(compute,cache_ms)
            r={"case":case["name"],"kind":case["kind"],"shape":x,"weight":w,
               "direction":direction,"dilation":case["dilation"],"groups":case["groups"],
               "event_median_ms":actual,"wall_median_ms":q["wall_median_ms"],
               "profile_activity_sum_ms":device.get("denominator_ms"),
               "dense_equivalent_GFLOP":nominal_flops/1e9,
               "minimum_tensor_MiB":minimum_tensor_bytes/(1<<20),
               "optimistic_direct_compute_ms":compute,
               "streaming_minimum_traffic_ms":dram_ms,
               "cached_minimum_traffic_ms":cache_ms,
               "direct_assumption_roofline_streaming_ms":direct_dram,
               "direct_assumption_roofline_cached_ms":direct_cache,
               "actual_over_direct_assumption_streaming":None if winograd else actual/direct_dram,
               "actual_over_direct_assumption_cached":None if winograd else actual/direct_cache,
               "direct_assumption_applies":not winograd,
               "why_not_lower_bound":"Winograd reduces arithmetic; dense-equivalent FLOPs are not executed FLOPs" if winograd else
                   "Optimistic direct-convolution assumptions; excludes extra traffic, LDS, occupancy and launch costs",
               "im2col_col2im_profile_ms":im2col_ms,
               "im2col_col2im_profile_fraction":im2col_ms/max(device.get("denominator_ms",0),1e-30),
               "largest_actual_kernel":max(kernels,key=lambda z:z["self_gpu_ms"])["name"] if kernels else None,
               "kernel_names":names,
               "requested_single_solver":q.get("requested_single_solver"),
               "observed_solver":"pending_native_log_crosscheck",
               "strict_gate_pass":q["reference_error"]["screen_pass"],
               "nrmse":q["reference_error"]["nrmse"],
               "max_over_reference_rms":q["reference_error"]["max_over_reference_rms"],
               "observed_forward_calls":case.get("observed_forward_calls"),
               "trace":q.get("trace")}
            rows.append(r)
    result={"schema":"v08_solver_direction_roofline","source":str(source),"source_sha256":digest(source),
            "calibration":str(calib),"calibration_sha256":digest(calib),
            "calibration_values":{"register_dot2_TFLOP_s":tflops,"streaming_copy_GB_s":dram,
                                  "cached_copy_GB_s":cache},
            "notes":["Scalar throughput is measured with independent register accumulators, not an achievable guarantee for all shapes.",
                     "DRAM minimum assumes streaming; tensors that fit Infinity Cache may exceed measured streaming bandwidth.",
                     "Winograd entries carry dense-equivalent FLOPs for comparison only and no direct lower-bound ratio.",
                     "Kernel/copy/memset activity sums are separate from event/wall and CPU operator attribution.",
                     "Each row is one isolated direction; microbench occurrence weights are not a whole-model timing forecast."],
            "case_count":len(d["cases"]),"direction_count":len(rows),
            "strict_pass_count":sum(r["strict_gate_pass"] for r in rows),"rows":rows}
    dump(out/"roofline.json",result)
    with (out/"shape_direction.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v
                             for k,v in r.items()})
    summary=["# v0.8 현재 FAST: shape·방향별 시간과 낙관적 하한 비교","",
             f"- {len(d['cases'])} shape, {len(rows)} 방향, strict gate {result['strict_pass_count']}/{len(rows)}.",
             f"- dot2 register calibration {tflops:.3f} TFLOP/s; streaming copy {dram:.3f} GB/s; cached copy {cache:.3f} GB/s.",
             "- 아래는 현재 시간 상위 20방향이다. 전체 432방향은 shape_direction.csv와 roofline.json.",
             "- dense FLOP 기반 수치는 직접 convolution 가정이다. Winograd는 연산량이 줄어 이 비율을 하한으로 해석하지 않는다.",
             "- DRAM bandwidth 기반 최소시간은 streaming 가정이다. 캐시 내 재사용·추가 im2col traffic·LDS·점유율·launch 비용을 분리해서 해석한다.",
             "- 이 표의 kernel 이름은 실제 trace다. solver 이름은 별도 native 로그 대조 전까지 미확정이다.","",
             "| shape | 방향 | 현재 event ms | direct compute ms | streaming traffic ms | cached traffic ms | direct 가정 대비 | im2col/col2im 비중 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows,key=lambda r:r["event_median_ms"],reverse=True)[:20]:
        ratio=r["actual_over_direct_assumption_streaming"]
        summary.append(f'| {r["case"]} | {r["direction"]} | {r["event_median_ms"]:.6f} | {r["optimistic_direct_compute_ms"]:.6f} | {r["streaming_minimum_traffic_ms"]:.6f} | {r["cached_minimum_traffic_ms"]:.6f} | {f"{ratio:.1f}×" if ratio is not None else "Winograd: 적용 안 함"} | {r["im2col_col2im_profile_fraction"]:.1%} |')
    (out/"ROOFLINE_KO.md").write_text("\n".join(summary)+"\n")
    print(json.dumps({"out":str(out),"cases":len(d["cases"]),"directions":len(rows),
                      "strict_pass":result["strict_pass_count"],"top5":[
        {k:r[k] for k in ("case","direction","event_median_ms","optimistic_direct_compute_ms",
                         "streaming_minimum_traffic_ms","actual_over_direct_assumption_streaming",
                         "largest_actual_kernel")}
        for r in sorted(rows,key=lambda r:r["event_median_ms"],reverse=True)[:5]]},indent=2))


if __name__=="__main__":main()
