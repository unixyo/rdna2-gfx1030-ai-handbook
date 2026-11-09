#!/usr/bin/env python3
"""CPU-only attribution and conditional roofline model for actual v1.0 profiles.

No torch/ROCm import. A logical byte estimate is not a DRAM counter. Dense
equivalent FLOPs are not executed instructions for Winograd/FFT. Model floors
must not be presented as measured irreducible time or promised speedups.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from v08_analyze_profile import concrete, merge_intervals, tensor_shapes

ROOT = Path(__file__).resolve().parent
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def numel(shape):
    return math.prod(shape) if isinstance(shape, list) and shape and all(isinstance(i, int) and i > 0 for i in shape) else 0


def vec(value, n, default):
    if value is None or value == [] or value == "":
        return [default] * n
    if isinstance(value, (int, float)):
        return [int(value)] * n
    if isinstance(value, (list, tuple)) and len(value) == n:
        return [int(x) for x in value]
    raise ValueError(f"Invalid spatial argument: {value!r}")


def dtype_info(types, index=0):
    name = str(types[index]).lower() if index < len(types) else ""
    if "half" in name or ("float16" in name and "bfloat" not in name):
        return "float16", 2
    if "bfloat" in name:
        return "bfloat16", 2
    if "double" in name or "float64" in name:
        return "float64", 8
    if "float" in name:
        return "float32", 4
    if "long" in name or "int64" in name:
        return "int64", 8
    if "int" in name:
        return "int32", 4
    if "bool" in name:
        return "bool", 1
    return "unknown", None


def conv_spec(event, backward=False):
    s = tensor_shapes(event)
    args = event.get("args", {})
    types = args.get("Input type", [])
    if backward:
        if len(s) < 3:
            return None
        x, w, y = s[1], s[2], s[0]
        stride, padding = concrete(event, 4), concrete(event, 5)
        dilation, transposed = concrete(event, 6), concrete(event, 7)
        outpad, groups = concrete(event, 8), concrete(event, 9)
        mask = concrete(event, 10)
        dtype, size = dtype_info(types, 1)
        directions = [d for d, active in zip(("dx", "dw", "db"), mask or (True, True, False)) if active]
    else:
        if len(s) < 2:
            return None
        x, w = s[0], s[1]
        stride, padding = concrete(event, 3), concrete(event, 4)
        dilation, transposed = concrete(event, 5), concrete(event, 6)
        outpad, groups = concrete(event, 7), concrete(event, 8)
        dtype, size = dtype_info(types, 0)
        y, directions = None, ["fwd"]
    if not numel(x) or not numel(w) or len(x) != len(w) or len(x) < 3:
        return None
    n = len(x) - 2
    stride, padding = vec(stride, n, 1), vec(padding, n, 0)
    dilation, outpad = vec(dilation, n, 1), vec(outpad, n, 0)
    transposed, groups = bool(transposed), int(groups or 1)
    if y is None:
        cout = w[1] * groups if transposed else w[0]
        if transposed:
            dims = [(a-1)*st-2*p+d*(k-1)+o+1 for a,st,p,d,k,o in zip(x[2:],stride,padding,dilation,w[2:],outpad)]
        else:
            dims = [(a+2*p-d*(k-1)-1)//st+1 for a,st,p,d,k in zip(x[2:],stride,padding,dilation,w[2:])]
        y = [x[0], cout, *dims]
    return dict(kind="conv_transpose" if transposed else f"conv{n}d", input=x,
                weight=w, output=y, stride=stride, padding=padding, dilation=dilation,
                groups=groups, transposed=transposed, dtype=dtype, element_bytes=size,
                directions=directions, implementation="native", split=0)


def direct_spec(event, result):
    name = event["name"]
    raw = name.removeprefix("V10_DIRECT_")
    catalog = result.get("direct_calls", {})
    if raw in catalog:
        d = dict(catalog[raw])
    else:
        direction, sep, key = raw.partition("_")
        if not sep or direction not in ("fwd", "dx", "dw"):
            return None
        try:
            parsed = json.loads(key)
        except json.JSONDecodeError:
            d = catalog.get(key, {})
            if not d:
                return None
            parsed = [d.get("input"), d.get("weight"), d.get("dilation", 1),
                      d.get("padding", 0), d.get("groups", 1)]
        x, w, dilation, padding, groups = parsed[:5]
        d = dict(direction=direction,input=x,weight=w,dilation=dilation,padding=padding,groups=groups)
    x, w = d.get("input"), d.get("weight")
    direction = d.get("direction")
    if not numel(x) or not numel(w):
        return None
    n = len(x)-2
    dilation, padding = vec(d.get("dilation"),n,1),vec(d.get("padding"),n,0)
    stride, outpad = vec(d.get("stride"),n,1),vec(d.get("output_padding"),n,0)
    groups = int(d.get("groups",1))
    transposed = "transpose" in d.get("kind", "")
    if transposed:
        spatial=[(a-1)*s-2*p+di*(k-1)+o+1 for a,s,p,di,k,o in zip(x[2:],stride,padding,dilation,w[2:],outpad)]
        co=w[1]*groups
    else:
        spatial=[(a+2*p-di*(k-1)-1)//s+1 for a,s,p,di,k in zip(x[2:],stride,padding,dilation,w[2:])]
        co=w[0]
    y=[x[0],co,*spatial]
    raw_impl=d.get("implementation","direct")
    impl={"direct_v08":"direct","fused_v10":"fused","candidate_v10":"candidate"}.get(raw_impl,raw_impl)
    config=d.get("config",{})
    split=int(d.get("split",config.get("split",1 if impl=="candidate" else 32))) if direction=="dw" else 0
    dtype,size=dtype_info([d.get("dtype","torch.float16")])
    return dict(kind="conv_transpose" if transposed else f"conv{n}d",input=x,weight=w,output=y,
                stride=stride,padding=padding,dilation=dilation,output_padding=outpad,groups=groups,
                transposed=transposed,dtype=dtype,element_bytes=size,directions=[direction],
                implementation=impl,split=split,config=config,
                source_dtype=d.get("source_dtype",d.get("dtype","torch.float16")),
                leaky_relu_on_load=d.get("leaky_relu_on_load",False),
                leaky_relu_gradient_mask=d.get("leaky_relu_gradient_mask",False),
                residual_add=d.get("residual_add",False),residual_dtype=d.get("residual_dtype"),
                residual_gradient_add=d.get("residual_gradient_add",False),
                residual_gradient_dtype=d.get("residual_gradient_dtype"))


def gemm_spec(event):
    s = tensor_shapes(event)
    s = s[1:3] if event["name"] == "aten::addmm" else s[:2]
    if len(s) != 2 or any(not numel(x) or len(x) < 2 for x in s):
        return None
    a, b = s
    if a[-1] != b[-2]:
        return None
    types = event.get("args", {}).get("Input type", [])
    dtype, size = dtype_info(types, 1 if event["name"] == "aten::addmm" else 0)
    batch = math.prod(a[:-2]) if len(a)>2 else 1
    return dict(kind="gemm", input=a, weight=b, output=[*a[:-2], a[-2], b[-1]],
                dtype=dtype, element_bytes=size, directions=["fwd"],
                implementation="native", flop_count=2*batch*a[-2]*a[-1]*b[-1])


def quantities(spec):
    """Return direct-equivalent math and minimum logical tensor bytes per call."""
    if not spec:
        return None, None, "unmapped"
    if spec["kind"] == "pointwise":
        return None, spec["minimum_bytes"], "logical elementwise inputs + output; DRAM residency unmeasured"
    x, w, y = spec["input"], spec["weight"], spec["output"]
    size = spec.get("element_bytes")
    if spec["kind"] == "gemm":
        return spec["flop_count"], size*(numel(x)+numel(w)+numel(y)) if size else None, "GEMM logical operands"
    n_dirs = sum(d in ("fwd", "dx", "dw") for d in spec["directions"])
    if n_dirs == 0:
        return None, None, "bias-only backward, not dot2-modeled"
    sites = math.prod(x[2:] if spec.get("transposed") else y[2:])
    flop = 2*x[0]*sites*numel(w)*n_dirs
    q = size*(numel(x)+numel(w)+numel(y))*n_dirs if size else None
    basis = "one read/write per logical tensor per direction; padding MACs retained"
    impl = spec.get("implementation")
    direction = spec["directions"][0] if len(spec["directions"])==1 else None
    if impl == "fused":
        _, sx = dtype_info([spec.get("source_dtype","torch.float16")])
        if sx is None:
            return flop,None,"fused source dtype unknown"
        if direction == "fwd":
            residual = spec.get("residual_add",False)
            _, sr = dtype_info([spec.get("residual_dtype")]) if residual else (None,0)
            if sr is None:
                return flop,None,"fused residual dtype unrecorded; cannot infer it from activation dtype"
            q=sx*numel(x)+2*numel(w)+(sr if residual else 2)*numel(y)+sr*numel(y)
        elif direction == "dx":
            # The implementation requires residual_grad dtype == source x dtype.
            q=2*numel(y)+2*numel(w)+2*sx*numel(x)
            if spec.get("residual_gradient_add",False):
                q+=sx*numel(x)
        elif direction == "dw":
            q=sx*numel(x)+2*numel(y)+2*numel(w)
        basis="fused logical tensor reads/writes incl source mask/residual/output dtype; activation math excluded from dot2 FLOPs"
    if direction == "dw" and impl in ("direct","fused","candidate") and q is not None:
        split=int(spec.get("split",32))
        algorithm=spec.get("config",{}).get("algorithm", "flat_dw" if impl=="candidate" else "v08_dw")
        scratch=impl in ("direct","fused") or split>1 or algorithm.startswith("v08_")
        if scratch:
            q+=8*split*numel(w)
            basis+="; includes FP32 split partial write and reduction read"
    return flop, q, basis


def pointwise_spec(event):
    # Only operations whose output size and elementary tensor access contract
    # are known. Reductions, TensorList foreach, casts and special functions
    # remain visible but unmodeled rather than guessing their traffic.
    allowed = {"aten::leaky_relu", "aten::leaky_relu_", "aten::leaky_relu_backward",
               "aten::relu", "aten::relu_", "aten::threshold_backward",
               "aten::add", "aten::add_", "aten::sub", "aten::sub_",
               "aten::mul", "aten::mul_", "aten::div", "aten::div_",
               "aten::neg", "aten::neg_", "aten::abs", "aten::abs_"}
    if event["name"] not in allowed:
        return None
    shapes = tensor_shapes(event)
    types = event.get("args", {}).get("Input type", [])
    tensors = [(shape, dtype_info(types, i)) for i, shape in enumerate(shapes) if numel(shape)]
    if not tensors or any(size is None for _, (_, size) in tensors):
        return None
    shape, _ = max(tensors, key=lambda item: numel(item[0]))
    dtype, size = max((ds for _, ds in tensors), key=lambda ds: ds[1])
    if {ds[0] for _, ds in tensors} == {"float16", "bfloat16"}:
        dtype, size = "float32", 4
    q = sum(numel(s) * ds for s, (_, ds) in tensors) + numel(shape) * size
    return dict(kind="pointwise", op=event["name"], shapes=[s for s, _ in tensors],
                dtype=dtype, element_bytes=size, minimum_bytes=q)


def family(spec, owner, optimizer):
    if spec:
        k = spec["kind"]
        if k == "pointwise":
            return family(None, spec["op"], spec.get("in_optimizer", optimizer))
        return f'{spec["implementation"]}_{k}' if k != "gemm" else "other_gemm"
    if optimizer:
        return "optimizer"
    if any(x in owner for x in ("leaky_relu", "relu")):
        return "activation"
    if any(x in owner for x in ("add", "sub", "mul", "div", "clamp")):
        return "elementwise"
    if any(x in owner for x in ("copy", "_to_copy", "contiguous", "clone")):
        return "copy_cast"
    if any(x in owner for x in ("norm", "sum", "mean", "var", "amax", "amin", "abs")):
        return "reduction_normalization"
    return "other_unmodeled"


def contexts(events, result):
    """Map each runtime launch correlation to its innermost computation scope."""
    per_thread = defaultdict(list)
    for i, e in enumerate(events):
        if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation", "cuda_runtime"):
            per_thread[(e.get("pid"), e.get("tid"))].append((i,e))
    external, correlations, scopes, parse_errors = {}, {}, {}, Counter()
    for cpu in per_thread.values():
        stack = []
        for i,e in sorted(cpu, key=lambda ie:(ie[1]["ts"],-ie[1].get("dur",0))):
            end = e["ts"]+e.get("dur",0)
            while stack and (e["ts"] >= stack[-1][0]-0.0001 or end > stack[-1][0]+0.1):
                stack.pop()
            current, optimizer, owner = stack[-1][1:] if stack else (None, False, "")
            name = e["name"]
            if name.startswith("Optimizer.step#"):
                optimizer = True
            if e.get("cat") == "cpu_op":
                owner = name
            spec = None
            attempted = False
            try:
                if name.startswith("V10_DIRECT_"):
                    attempted, spec = True, direct_spec(e, result)
                elif name == "aten::convolution_backward":
                    attempted, spec = True, conv_spec(e, True)
                elif name in ("aten::convolution", "aten::_convolution") and current is None:
                    attempted, spec = True, conv_spec(e)
                elif name in ("aten::mm","aten::bmm","aten::addmm") and current is None:
                    attempted, spec = True, gemm_spec(e)
                elif current is None:
                    spec = pointwise_spec(e)
            except (ValueError, TypeError, IndexError, KeyError) as error:
                parse_errors[f"{name}: {type(error).__name__} {error}"] += 1
            if spec:
                if spec["kind"] == "pointwise":
                    spec["in_optimizer"] = optimizer
                current = i
                scopes[i] = dict(spec=spec, event=name, kernel_events=[], optimizer=optimizer)
            elif attempted:
                parse_errors[f"{name}: no supported shape/dtype metadata"] += 1
            ext = e.get("args",{}).get("External id")
            if ext is not None:
                external[ext] = (current,optimizer,owner)
            correlation = e.get("args",{}).get("correlation")
            if e.get("cat") == "cuda_runtime" and correlation is not None:
                value = (current,optimizer,owner)
                if correlation in correlations and correlations[correlation] != value:
                    parse_errors["conflicting runtime correlation IDs"] += 1
                correlations[correlation] = value
            stack.append((end,current,optimizer,owner))
    return external, correlations, scopes, dict(parse_errors)


def render_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k:json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})


def review_capacity_model(analysis):
    """Keep logical-byte diagnostics, exclude unmeasured pointwise DRAM floors.

    Cache-resident elementwise work in the actual profile beats the cold-streaming
    tensor-byte estimate. A minimum logical tensor size cannot identify its
    off-chip traffic, even for rows which happen not to exceed the estimate.
    """
    c, s = analysis["constants"], analysis["scope"]
    multiplier = s["epochs"] * s["steps_per_epoch"] / s["profile_steps"] / 60e6
    families = {r["family"]: r for r in analysis["families"]}
    for r in families.values():
        for k in ("modeled_gpu_us", "rdna2_roof_us", "rtx5070_roof_us", "same_roof_fraction_capacity_saving_us", "roof_violation_count"):
            r[k] = 0
    violations = 0
    contradicted_cold_rows = 0
    for r in analysis["work_units"]:
        q = r["minimum_logical_tensor_bytes"]
        stream_r = q / (c["rdna2_streaming_gb_s"] * 1e3) if q is not None else None
        stream_n = q / (c["rtx5070_gb_s"] * 1e3) if q is not None else None
        r["rdna2_logical_bytes_streaming_us"] = stream_r
        r["rtx5070_logical_bytes_streaming_us"] = stream_n
        r["logical_streaming_time_exceeds_observed"] = bool(stream_r is not None and stream_r > r["gpu_us"])
        violations += r["logical_streaming_time_exceeds_observed"]
        if r["spec"]["kind"] == "pointwise":
            r["matched_roof_applicable"] = False
            r["roof_model"] = "logical-byte reference only; DRAM residency and compute bound unmeasured"
            r["roof_exceeds_observed"] = False
            for k in ("rdna2_direct_cold_roof_us", "rtx5070_direct_cold_roof_us",
                      "conditional_hardware_capacity_penalty_us", "headroom_to_optimistic_rdna2_roof_us",
                      "same_roof_fraction_rtx5070_us", "same_roof_fraction_capacity_saving_us"):
                r[k] = None
        elif (r.get("cold_streaming_assumption_rejected") or
              (r["matched_roof_applicable"] and r["roof_exceeds_observed"])):
            # Fused mask/residual operands can remain in cache. A cold-DRAM
            # reference contradicted by timing cannot enter identified paired
            # floor totals. Preserve its reference/diagnostic columns instead.
            contradicted_cold_rows += 1
            r["cold_streaming_assumption_rejected"] = True
            r["matched_roof_applicable"] = False
            r["roof_model"] = "cold-direct reference contradicted by observed timing; excluded from paired totals"
            for k in ("conditional_hardware_capacity_penalty_us", "headroom_to_optimistic_rdna2_roof_us",
                      "same_roof_fraction_rtx5070_us", "same_roof_fraction_capacity_saving_us"):
                r[k] = None
        if r["matched_roof_applicable"]:
            z = families[r["family"]]
            z["modeled_gpu_us"] += r["gpu_us"]
            z["rdna2_roof_us"] += r["rdna2_direct_cold_roof_us"]
            z["rtx5070_roof_us"] += r["rtx5070_direct_cold_roof_us"]
            z["same_roof_fraction_capacity_saving_us"] += r["same_roof_fraction_capacity_saving_us"] or 0
            z["roof_violation_count"] += r["roof_exceeds_observed"]
    for r in families.values():
        modeled = r["modeled_gpu_us"] > 0
        r.update(modeled_gpu_minutes=r["modeled_gpu_us"]*multiplier,
                 rdna2_cold_direct_roof_minutes=r["rdna2_roof_us"]*multiplier if modeled else None,
                 rtx5070_cold_direct_roof_minutes=r["rtx5070_roof_us"]*multiplier if modeled else None,
                 conditional_capacity_penalty_minutes=(r["rdna2_roof_us"]-r["rtx5070_roof_us"])*multiplier if modeled else None,
                 optimistic_headroom_minutes=(r["modeled_gpu_us"]-r["rdna2_roof_us"])*multiplier if modeled else None,
                 same_roof_fraction_capacity_saving_minutes=r["same_roof_fraction_capacity_saving_us"]*multiplier if modeled else None,
                 status="conditional FP16 direct model on covered subset" if modeled else "time measured; no validated paired-capacity floor")
    rows = analysis["families"]
    analysis["identification"].update(
        conditional_capacity_penalty_minutes=sum(r["conditional_capacity_penalty_minutes"] or 0 for r in rows),
        optimistic_covered_headroom_minutes=sum(r["optimistic_headroom_minutes"] or 0 for r in rows),
        same_roof_fraction_capacity_saving_minutes=sum(r["same_roof_fraction_capacity_saving_minutes"] or 0 for r in rows),
        paired_model_gpu_coverage_fraction=sum(r["modeled_gpu_us"] for r in rows)/analysis["accounting"]["kernel_sum_us"])
    analysis["model_review"] = dict(policy="exclude_pointwise_and_contradicted_cold_DRAM_from_paired_capacity_totals",
        logical_byte_streaming_exceeds_observed_work_units=violations,
        contradicted_cold_direct_rows_excluded=contradicted_cold_rows,
        explanation="Logical bytes are not DRAM counters; cache traffic must not be charged at streaming-DRAM bandwidth as an identified time floor.")
    note = "All pure elementwise cold-DRAM floors are excluded from paired-capacity sums; logical-byte bandwidth indicators remain diagnostic."
    if note not in analysis["limitations"]:
        analysis["limitations"].append(note)
    note = "Any additional cold-direct row whose reference floor exceeds its measured GPU time is excluded from paired totals; its math and logical-byte diagnostics remain visible."
    if note not in analysis["limitations"]:
        analysis["limitations"].append(note)
    # These split-dW kernel roles are known from the research source. Unlike a
    # native multi-kernel library call, their math/traffic can be apportioned
    # without guessing from a generic GEMM kernel name. Verify call coverage.
    by_id={r["id"]:r for r in analysis["work_units"]}
    for k in analysis["kernels"]:
        partial=k["name"] in ("_dw_partial","_dw_act_partial","_backward_weight_flat")
        reduction=k["name"] in ("_dw_reduce","_reduce_weight")
        if not (partial or reduction):
            continue
        related=[by_id[rid] for rid in k["work_unit_ids"]]
        if (not related or any(r["spec"].get("directions")!=["dw"] or
                              r["spec"].get("implementation") not in ("direct","fused","candidate") for r in related)
            or sum(r["profile_calls"] for r in related)!=k["count"]):
            continue
        flops=0;byte_count=0;adds=0;known=True
        for r in related:
            sp=r["spec"];calls=r["profile_calls"];split=int(sp.get("split",32))
            nx,nw,ny=(numel(sp[t]) for t in ("input","weight","output"))
            _,sx=dtype_info([sp.get("source_dtype","torch.float16")])
            if sx is None:
                known=False;break
            if partial:
                flops+=r["dense_equivalent_flops"]
                scratch=sp["implementation"] in ("direct","fused") or split>1 or sp.get("config",{}).get("algorithm","").startswith("v08_")
                byte_count+=calls*(sx*nx+2*ny+(4*split*nw if scratch else 2*nw))
            else:
                byte_count+=calls*(4*split*nw+2*nw)
                adds+=calls*(split-1)*nw
        if not known:
            continue
        k.update(dense_equivalent_flops=flops if partial else None,
                 fp32_reduction_additions=adds if reduction else None,
                 minimum_logical_tensor_bytes=byte_count,
                 dot2_utilization=flops/(k["gpu_us"]*c["rdna2_dot2_tflop_s"]*1e6) if partial else None,
                 minimum_bytes_over_streaming_bw=byte_count/(k["gpu_us"]*c["rdna2_streaming_gb_s"]*1e3),
                 metric_status="source-defined split-dW role; annotated calls match kernel count; logical bytes are not DRAM counters")


def write_outputs(out, analysis):
    out.mkdir(parents=True)
    dump(out/"gap_analysis.json", analysis)
    render_csv(out/"work_units.csv", analysis["work_units"])
    render_csv(out/"kernels.csv", analysis["kernels"])
    render_csv(out/"family_minutes.csv", analysis["families"])
    print(json.dumps(dict(out=str(out),work_units=len(analysis["work_units"]),kernels=len(analysis["kernels"]),
        kernel_gpu_ms=analysis["accounting"]["kernel_sum_us"]/1000,
        modeled_gpu_ms=sum(r["modeled_gpu_us"] for r in analysis["families"])/1000,
        parse_error_types=len(analysis["parsing_errors"]),
        capacity_model_minutes=analysis["identification"]["conditional_capacity_penalty_minutes"],
        headroom_model_minutes=analysis["identification"]["optimistic_covered_headroom_minutes"],
        steady_projected_minutes=analysis["scope"]["operating_steady_projected_minutes"]),ensure_ascii=False,indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    entry = ap.add_mutually_exclusive_group(required=True)
    entry.add_argument("--run",type=Path)
    entry.add_argument("--from-analysis",type=Path,help="Review cached attribution/add steady epoch, without reloading a large trace")
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--calibration",type=Path,default=ROOT/"results/applio_v08/hardware_calibration/result.json")
    ap.add_argument("--specs",type=Path,default=ROOT/"results/v10/rtx5070_sources.json")
    ap.add_argument("--steps",type=int,default=5)
    ap.add_argument("--steps-per-epoch",type=int,default=31)
    ap.add_argument("--epochs",type=int,default=100)
    ap.add_argument("--steady-epoch-s",type=float,help="Unprofiled operating-path measurement, never the profile epoch.")
    ap.add_argument("--diagnostic-only",action="store_true")
    args = ap.parse_args()
    source, out = (args.run or args.from_analysis).resolve(), args.out.resolve()
    if not source.is_relative_to(ROOT) or not out.is_relative_to(ROOT):
        ap.error("Input/output must be inside research rdna2-ops")
    if out.exists():
        ap.error("Preserve completed analyses: choose a new output directory")
    if min(args.steps,args.steps_per_epoch,args.epochs) < 1:
        ap.error("Step and epoch counts must be positive")
    if args.from_analysis:
        analysis = read(source)
        if analysis.get("schema") != "v10_conditional_roofline_gap":
            ap.error("Unsupported cached-attribution schema")
        if not analysis.get("valid_profile_for_timing") and not args.diagnostic_only:
            ap.error("Cached attribution has no valid timing audit")
        analysis.setdefault("revisions", []).append(dict(input=str(source),input_sha256=sha(source),
            previous_parser_sha256=analysis["parser_sha256"],reason="CPU model review / unprofiled steady-epoch attachment; original profile attribution preserved"))
        analysis["parser_sha256"] = sha(Path(__file__))
        if args.steady_epoch_s is not None:
            steady = args.steady_epoch_s*analysis["scope"]["epochs"]/60
            analysis["scope"].update(operating_steady_epoch_s=args.steady_epoch_s,operating_steady_projected_minutes=steady)
            analysis["accounting"]["steady_minus_projected_kernel_minutes"] = steady-analysis["accounting"]["projected_kernel_minutes"]
        review_capacity_model(analysis)
        write_outputs(out, analysis)
        return
    run = source
    result_path, trace_path = run/"result.json", run/"profile_trace.json"
    result, trace, cal, specs = read(result_path), read(trace_path), read(args.calibration), read(args.specs)
    valid = result.get("valid_for_timing") is True
    if not valid and not args.diagnostic_only:
        ap.error("Run audit has not passed; use diagnostic-only to inspect invalid or incomplete data")
    events = trace["traceEvents"]
    kernels = [e for e in events if e.get("ph")=="X" and e.get("cat")=="kernel" and e.get("dur",0)>0]
    gpu = [e for e in events if e.get("ph")=="X" and e.get("cat") in GPU_CATS and e.get("dur",0)>0]
    ext, correlations, scopes, errors = contexts(events,result)
    linkage = Counter()
    kernel_group = defaultdict(lambda:dict(count=0,gpu_us=0.0,unscoped_us=0.0,scope_ids=set(),families=Counter()))
    unscoped = defaultdict(lambda:dict(count=0,gpu_us=0.0))
    for e in kernels:
        correlation = e.get("args",{}).get("correlation")
        if correlation in correlations:
            sid, optimizer, owner = correlations[correlation]
            linkage["runtime_correlation"] += 1
        else:
            sid, optimizer, owner = ext.get(e.get("args",{}).get("External id"),(None,False,""))
            linkage["external_id_fallback"] += 1
        spec = scopes[sid]["spec"] if sid is not None else None
        fam = family(spec,owner,optimizer)
        if sid is not None:
            scopes[sid]["kernel_events"].append(e)
        else:
            unscoped[fam]["count"] += 1
            unscoped[fam]["gpu_us"] += e["dur"]
        row = kernel_group[e["name"]]
        row["count"]+=1
        row["gpu_us"]+=e["dur"]
        row["families"][fam]+=e["dur"]
        if sid is not None:
            row["scope_ids"].add(sid)
        else:
            row["unscoped_us"]+=e["dur"]
    rtf, rbw = cal["dot2"]["measured_tflop_s"],cal["copy"]["measured_GB_s"]
    ns = specs["nvidia_specification"]
    ntf, nbw = ns["fp16_tensor_fp32_accumulate_dense_tflop_s"],ns["memory_bandwidth_gb_s"]
    multiplier = args.epochs*args.steps_per_epoch/args.steps/60e6
    rows, by_spec = [], {}
    for sid, scope in scopes.items():
        es, spec = scope["kernel_events"],scope["spec"]
        if not es:
            continue
        key = json.dumps(spec,sort_keys=True)
        if key not in by_spec:
            by_spec[key] = dict(spec=spec, profile_calls=0,gpu_us=0.0, kernels=Counter(),scope_ids=[])
        r = by_spec[key]
        r["profile_calls"]+=1;r["gpu_us"]+=sum(e["dur"] for e in es);r["scope_ids"].append(sid)
        for e in es:r["kernels"][e["name"]]+=e["dur"]
    scope_to_row = {}
    for i,r in enumerate(sorted(by_spec.values(),key=lambda r:r["gpu_us"],reverse=True)):
        spec = r["spec"];f,q,q_basis=quantities(spec)
        f = f*r["profile_calls"] if f is not None else None
        q = q*r["profile_calls"] if q is not None else None
        winograd = any("winograd" in n.lower() or "sp3asmconv" in n.lower() or "fft" in n.lower() for n in r["kernels"])
        dense_eligible = bool(f is not None and q is not None and spec["dtype"]=="float16" and not winograd)
        memory_only = spec["kind"]=="pointwise" and q is not None
        eligible = dense_eligible or memory_only
        roof_r = max((f or 0)/(rtf*1e6),q/(rbw*1e3)) if eligible else None
        roof_n = max((f or 0)/(ntf*1e6),q/(nbw*1e3)) if eligible else None
        rid = f"w{i:04d}"
        row = dict(id=rid,family=family(spec,"",False),spec=spec,profile_calls=r["profile_calls"],
                   gpu_us=r["gpu_us"],dense_equivalent_flops=f,minimum_logical_tensor_bytes=q,
                   bytes_basis=q_basis,winograd_or_fft=winograd,matched_roof_applicable=eligible,
                   roof_model="cold-memory-only; arithmetic bound unmeasured" if memory_only else "cold-direct-FP16-FP32" if dense_eligible else "unmodeled",
                   dot2_dense_equivalent_utilization=f/(r["gpu_us"]*rtf*1e6) if dense_eligible else None,
                   minimum_bytes_over_streaming_bw=q/(r["gpu_us"]*rbw*1e3) if q is not None else None,
                   rdna2_direct_cold_roof_us=roof_r,rtx5070_direct_cold_roof_us=roof_n,
                   conditional_hardware_capacity_penalty_us=roof_r-roof_n if eligible else None,
                   headroom_to_optimistic_rdna2_roof_us=r["gpu_us"]-roof_r if eligible else None,
                   same_roof_fraction_rtx5070_us=r["gpu_us"]*roof_n/roof_r if eligible and roof_r>0 else None,
                   same_roof_fraction_capacity_saving_us=r["gpu_us"]*(1-roof_n/roof_r) if eligible and roof_r>0 else None,
                   roof_exceeds_observed=bool(eligible and roof_r>r["gpu_us"]),
                   observed_gpu_minutes_per_100_epochs=r["gpu_us"]*multiplier,
                   kernel_names=list(r["kernels"]))
        rows.append(row)
        for sid in r["scope_ids"]:scope_to_row[sid]=row
    kernel_rows=[]
    for name,r in sorted(kernel_group.items(),key=lambda kv:kv[1]["gpu_us"],reverse=True):
        related={scope_to_row[s]["id"]:scope_to_row[s] for s in r["scope_ids"] if s in scope_to_row}
        all_single = bool(related) and r["unscoped_us"]==0 and all(set(w["kernel_names"])=={name} for w in related.values())
        # Only one-kernel scopes permit exact per-kernel nominal-math attribution.
        f=sum(w["dense_equivalent_flops"] for w in related.values()) if all_single and all(w["dense_equivalent_flops"] is not None for w in related.values()) else None
        q=sum(w["minimum_logical_tensor_bytes"] for w in related.values()) if all_single and all(w["minimum_logical_tensor_bytes"] is not None for w in related.values()) else None
        eligible=all_single and all(w["dot2_dense_equivalent_utilization"] is not None for w in related.values())
        kernel_rows.append(dict(name=name,count=r["count"],gpu_us=r["gpu_us"],
            families=dict(r["families"]),work_unit_ids=list(related),
            dense_equivalent_flops=f,minimum_logical_tensor_bytes=q,
            dot2_utilization=f/(r["gpu_us"]*rtf*1e6) if eligible else None,
            minimum_bytes_over_streaming_bw=q/(r["gpu_us"]*rbw*1e3) if q is not None else None,
            metric_status="single-kernel scope; nominal math/bytes, not hardware counters" if all_single else
                          "time measured; multiple kernels or no exact shape attribution, see work_units.csv"))
    fams=defaultdict(lambda:dict(gpu_us=0.0,modeled_gpu_us=0.0,rdna2_roof_us=0.0,rtx5070_roof_us=0.0,same_roof_fraction_capacity_saving_us=0.0,roof_violation_count=0))
    for r in rows:
        z=fams[r["family"]];z["gpu_us"]+=r["gpu_us"]
        if r["matched_roof_applicable"]:
            z["modeled_gpu_us"]+=r["gpu_us"];z["rdna2_roof_us"]+=r["rdna2_direct_cold_roof_us"]
            z["rtx5070_roof_us"]+=r["rtx5070_direct_cold_roof_us"]
            z["same_roof_fraction_capacity_saving_us"]+=r["same_roof_fraction_capacity_saving_us"] or 0
            z["roof_violation_count"]+=r["roof_exceeds_observed"]
    for f,r in unscoped.items():fams[f]["gpu_us"]+=r["gpu_us"]
    family_rows=[]
    for fam,r in sorted(fams.items(),key=lambda kv:kv[1]["gpu_us"],reverse=True):
        modeled=r["modeled_gpu_us"]>0
        family_rows.append(dict(family=fam,**r,observed_gpu_minutes=r["gpu_us"]*multiplier,
            modeled_gpu_minutes=r["modeled_gpu_us"]*multiplier,
            rdna2_cold_direct_roof_minutes=r["rdna2_roof_us"]*multiplier if modeled else None,
            rtx5070_cold_direct_roof_minutes=r["rtx5070_roof_us"]*multiplier if modeled else None,
            conditional_capacity_penalty_minutes=(r["rdna2_roof_us"]-r["rtx5070_roof_us"])*multiplier if modeled else None,
            optimistic_headroom_minutes=(r["modeled_gpu_us"]-r["rdna2_roof_us"])*multiplier if modeled else None,
            same_roof_fraction_capacity_saving_minutes=r["same_roof_fraction_capacity_saving_us"]*multiplier if modeled else None,
            status="conditional model on covered subset only" if modeled else "time measured; floor unmodeled"))
    sum_us=sum(e["dur"] for e in kernels)
    union_us=sum(b-a for a,b in merge_intervals((e["ts"],e["ts"]+e["dur"]) for e in kernels))
    gpu_union=sum(b-a for a,b in merge_intervals((e["ts"],e["ts"]+e["dur"]) for e in gpu))
    projected=sum_us*multiplier
    steady=args.steady_epoch_s*args.epochs/60 if args.steady_epoch_s is not None else None
    analysis=dict(schema="v10_conditional_roofline_gap",valid_profile_for_timing=valid,
        diagnostic_only=args.diagnostic_only,
        source_files={k:dict(path=str(p),sha256=sha(p)) for k,p in
                      (("trace",trace_path),("result",result_path),("calibration",args.calibration),("specifications",args.specs))},
        parser_sha256=sha(Path(__file__)),
        constants=dict(rdna2_dot2_tflop_s=rtf,rdna2_streaming_gb_s=rbw,
                       rtx5070_fp16_fp32_dense_tflop_s=ntf,rtx5070_gb_s=nbw,
                       compute_ceiling_ratio=ntf/rtf,bandwidth_ceiling_ratio=nbw/rbw),
        scope=dict(profile_steps=args.steps,steps_per_epoch=args.steps_per_epoch,epochs=args.epochs,
                   operating_steady_epoch_s=args.steady_epoch_s,operating_steady_projected_minutes=steady),
        accounting=dict(kernel_sum_us=sum_us,kernel_union_us=union_us,gpu_activity_union_us=gpu_union,
                        kernel_overlap_fraction=(sum_us-union_us)/sum_us if sum_us else 0,
                        mapped_work_unit_gpu_us=sum(r["gpu_us"] for r in rows),
                        unmodeled_gpu_us=sum(r["gpu_us"] for r in unscoped.values()),
                        attributed_kernel_sum_us=sum(r["gpu_us"] for r in family_rows),
                        projected_kernel_minutes=projected,
                        steady_minus_projected_kernel_minutes=steady-projected if steady is not None else None),
        parsing_errors=errors,kernel_linkage=dict(linkage),work_units=rows,kernels=kernel_rows,families=family_rows,
        identification=dict(external_5070_minutes=12.0,matched_conditions_verified=False,
                            measured_irreducible_hardware_minutes=None,
                            measured_removable_software_minutes=None,
                            conditional_capacity_penalty_minutes=sum(r["conditional_capacity_penalty_minutes"] or 0 for r in family_rows),
                            same_roof_fraction_capacity_saving_minutes=sum(r["same_roof_fraction_capacity_saving_minutes"] or 0 for r in family_rows),
                            optimistic_covered_headroom_minutes=sum(r["optimistic_headroom_minutes"] or 0 for r in family_rows)),
        limitations=[
            "FLOPs are direct/dense-equivalent. Winograd/FFT is excluded from matched direct-roof sums.",
            "Bytes are logical tensor traffic estimates, not measured DRAM counters. Cold streaming is conditional.",
            "Elementwise floors model only tensor traffic; arithmetic and special hardware constraints are not measured.",
            "Native GEMM may materialize extra tensors. The minimal direct model is a counterfactual, not its actual traffic.",
            "Single-kernel scopes and source-verified split-dW roles permit per-kernel nominal metrics; other multi-kernel math stays at work-unit scope.",
            "Runtime correlation IDs identify individual launches; External id is only a fallback because several Triton launches can share one autograd External id.",
            "Profile GPU time is occurrence-scaled; unprofiled steady epoch is separate. Negative residual signals a projection mismatch.",
            "Kernel overlap makes sums non-additive in wall time. GPU annotations and CPU attributed times are not added.",
            "Neither ceiling headroom nor ideal-device floor difference identifies removable or irreducible real minutes.",
            "Same-roof-fraction projection additionally assumes identical fractional roofline utilization on both GPUs; this has not been measured.",
            "The external 12-minute report lacks matched conditions and has not been reproduced on an RTX 5070."])
    if abs(analysis["accounting"]["attributed_kernel_sum_us"]-sum_us)>max(1e-3,sum_us*1e-9):
        raise ValueError("Attribution did not partition the measured kernel time")
    review_capacity_model(analysis)
    write_outputs(out, analysis)


if __name__ == "__main__":
    main()

