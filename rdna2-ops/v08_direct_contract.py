#!/usr/bin/env python3
"""Validate measured Conv1d dispatch at the Applio AMP/weight_norm boundary.

This is a correctness experiment, not a timing benchmark. It compares native
FP16 AMP with the selected wrapper using FP32 weight_norm master parameters,
the captured module input dtype, and the same stored input/upstream gradient.
The existing FP32-reference operator experiments are reused, not repeated.
No operating Applio source, pretrain, optimizer, or model log is written.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAFETY = {"MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1": "0",
          "MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53": "0"}
GATE = {"finite": True, "nrmse": 0.002, "max_over_reference_rms": 0.02,
        "reference": "native FP16 AMP with the identical FP32 weight_norm parameters and stored input/grad-output",
        "scope": "AMP/cast/weight_norm/autograd contract; operator FP32-reference gates are separate"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def owned(path):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE):
        raise ValueError("Use an existing research path under " + str(HERE))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[],
                        help="Exact captured case name; repeat to check a subset")
    parser.add_argument("--seed", type=int, default=63117)
    args = parser.parse_args()
    manifest_path, db, out = map(owned, (args.manifest, args.db, args.out))
    manifest = json.loads(manifest_path.read_text())
    kernel_sha = sha(HERE / "v08_direct_conv.py")
    if (manifest.get("schema") != "applio_v08_direct_conv_dispatch" or
        manifest.get("strict_gate_pass") is not True or
        manifest.get("source_sha256") != kernel_sha):
        parser.error("A measured operator-pass manifest matching the current kernel source is required")
    selected = list(manifest.get("entries", {}).items())
    if args.case:
        available = {entry["case"]["name"] for _, entry in selected}
        missing = set(args.case) - available
        if missing:
            parser.error("Cases absent from this manifest: " + ", ".join(sorted(missing)))
        selected = [(key, entry) for key, entry in selected
                    if entry["case"]["name"] in args.case]
    if not selected or not db.is_dir():
        parser.error("Nonempty selected cases and an existing research FindDb directory are required")
    for _, entry in selected:
        variants = entry["case"].get("logical_boundary_variants", [])
        dtypes = {v.get("input_dtype") for v in variants}
        if not dtypes or not dtypes <= {"torch.float16", "torch.float32"}:
            parser.error("This boundary check requires captured FP16/FP32 module input dtypes")
    if out.exists():
        parser.error("Refusing to overwrite an existing run")
    out.mkdir(parents=True)
    cache = db.parent / "miopen_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.update(SAFETY)
    os.environ.update(MIOPEN_FIND_MODE="FAST", MIOPEN_USER_DB_PATH=str(db),
                      MIOPEN_CUSTOM_CACHE_DIR=str(cache), APPLIO_RDNA2_DIRECT_CONV="1")
    os.environ.setdefault("TRITON_CACHE_DIR", str(HERE / "results/applio_v08/triton_cache"))
    report = dict(schema="applio_v08_direct_conv_contract", completed=False,
                  valid_for_contract=False, valid_for_timing=False, gate=GATE,
                  source_sha256=kernel_sha, contract_tool_sha256=sha(__file__),
                  manifest=str(manifest_path), manifest_sha256=sha(manifest_path),
                  manifest_is_contract_probe=manifest.get("contract_probe_only", False),
                  db=str(db), seed=args.seed, safety_env=SAFETY, cases=[],
                  scope="Synthetic fixed-seed tensors at actual captured geometry/dtype, FP32 master parameters; no optimizer or real training steps")

    def save():
        temp = out / "result.json.tmp"
        temp.write_text(json.dumps(report, indent=2, allow_nan=False))
        temp.replace(out / "result.json")

    save()
    from gpu_guard_v06 import GpuAudit
    audit = GpuAudit()
    try:
        with audit:
            audit.ensure_clear()
            # Even importing torch can register a KFD client in this runtime.
            # Keep it inside the owned lease and overlap audit.
            import torch
            from torch.nn.utils.parametrizations import weight_norm
            from v08_direct_conv import measured_dispatch
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = False
            report.update(torch=torch.__version__, hip=torch.version.hip,
                          device=torch.cuda.get_device_name(),
                          arch=torch.cuda.get_device_properties(0).gcnArchName,
                          cudnn_benchmark=torch.backends.cudnn.benchmark)

            def errors(candidate, reference):
                if candidate is None or reference is None:
                    return dict(screen_pass=False, reason="Missing gradient")
                a, b = candidate.detach().double(), reference.detach().double()
                finite = bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
                delta = a - b
                rms = b.square().mean().sqrt().item()
                rmse = delta.square().mean().sqrt().item()
                max_abs = delta.abs().max().item()
                nrmse = rmse / max(rms, 1e-30)
                max_over_rms = max_abs / max(rms, 1e-30)
                clean = lambda v: v if math.isfinite(v) else None
                return dict(finite=finite, reference_finite=bool(torch.isfinite(b).all()),
                            reference_rms=clean(rms), rmse=clean(rmse), max_abs=clean(max_abs),
                            nrmse=clean(nrmse), max_over_reference_rms=clean(max_over_rms),
                            candidate_dtype=str(candidate.dtype), reference_dtype=str(reference.dtype),
                            screen_pass=finite and nrmse <= GATE["nrmse"] and
                            max_over_rms <= GATE["max_over_reference_rms"])

            def check_one(key, entry, input_dtype):
                case = entry["case"]
                xshape, wshape = case["shape"], case["weight"]
                groups = int(case["groups"])
                one = lambda value: int(value[0] if isinstance(value, list) else value)
                dilation, padding = one(case["dilation"]), one(case["padding"])
                dtype = {"torch.float16": torch.float16, "torch.float32": torch.float32}[input_dtype]
                rng = torch.Generator(device="cpu").manual_seed(args.seed)
                # Random constructor values are overwritten before weight_norm.
                original = torch.nn.Conv1d(xshape[1], wshape[0], wshape[2], stride=1,
                                           padding=padding, dilation=dilation,
                                           groups=groups, bias=True)
                if tuple(original.weight.shape) != tuple(wshape):
                    raise ValueError("Captured weight geometry differs from the module")
                xc = torch.randn(xshape, generator=rng).to(dtype)
                wc = torch.randn(wshape, generator=rng) / math.sqrt(math.prod(wshape[1:]))
                bc = torch.randn((wshape[0],), generator=rng) * 0.1
                to = xshape[2] + 2 * padding - dilation * (wshape[2] - 1)
                dy = (torch.randn((xshape[0], wshape[0], to), generator=rng) * 0.01).half().cuda()
                with torch.no_grad():
                    original.weight.copy_(wc)
                    original.bias.copy_(bc)
                original = weight_norm(original, name="weight", dim=0)
                native, candidate = copy.deepcopy(original).cuda(), copy.deepcopy(original).cuda()
                xn, xcand = xc.cuda().requires_grad_(True), xc.clone().cuda().requires_grad_(True)
                with torch.autocast("cuda", dtype=torch.float16):
                    yn = native(xn)
                yn.backward(dy)
                original_hook = torch.nn.Conv1d._conv_forward
                hits = {}
                with measured_dispatch(manifest_path, hits=hits):
                    with torch.autocast("cuda", dtype=torch.float16):
                        yc = candidate(xcand)
                    yc.backward(dy)
                torch.cuda.synchronize()
                metrics = dict(y=errors(yc, yn), dx=errors(xcand.grad, xn.grad))
                np, cp = dict(native.named_parameters()), dict(candidate.named_parameters())
                if set(np) != set(cp):
                    raise RuntimeError("Native/candidate parametrization names differ")
                for name in np:
                    metrics["grad:" + name] = errors(cp[name].grad, np[name].grad)
                master_fp32 = all(p.dtype == torch.float32 and p.grad is not None and
                                  p.grad.dtype == torch.float32 for p in [*np.values(), *cp.values()])
                restored = torch.nn.Conv1d._conv_forward is original_hook
                hit_count = hits.get(key, 0)
                matched_dtypes = (yn.dtype == yc.dtype == torch.float16 and
                                  xn.grad is not None and xcand.grad is not None and
                                  xn.grad.dtype == xcand.grad.dtype == dtype)
                return dict(key=key, name=case["name"], input_dtype=input_dtype,
                            directions=entry["directions"], output_dtype=str(yc.dtype),
                            fp32_master_parameters_and_gradients=master_fp32,
                            input_output_gradient_dtypes_match=matched_dtypes,
                            weight_norm_dim=0, parameter_names=sorted(np),
                            dispatch_hits=hit_count, hook_restored=restored,
                            metrics=metrics,
                            all_gates_pass=all(m["screen_pass"] for m in metrics.values()) and
                            master_fp32 and matched_dtypes and restored and hit_count == 1)

            for key, entry in selected:
                dtypes = sorted({v["input_dtype"] for v in entry["case"]["logical_boundary_variants"]})
                for dtype in dtypes:
                    audit.ensure_clear()
                    report["active_case"] = dict(name=entry["case"]["name"], input_dtype=dtype)
                    save()
                    print("BEGIN", entry["case"]["name"], dtype, flush=True)
                    result = check_one(key, entry, dtype)
                    report["cases"].append(result)
                    save()
                    print("CONTRACT", result["name"], dtype,
                          json.dumps(dict(pass_gate=result["all_gates_pass"],
                                          hits=result["dispatch_hits"],
                                          errors={name: {k: value.get(k) for k in
                                                  ("nrmse", "max_over_reference_rms", "screen_pass")}
                                                  for name, value in result["metrics"].items()})), flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
            report["completed"] = True
    except BaseException as exc:
        report["fatal_error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        report["gpu_audit"] = audit.report()
        report["valid_for_contract"] = (report["completed"] and bool(report["cases"]) and
            "fatal_error" not in report and audit.report()["valid_for_timing"] and
            all(row["all_gates_pass"] for row in report["cases"]))
        save()
    print("DONE", str(out / "result.json"), report["valid_for_contract"], flush=True)
    if not report["valid_for_contract"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
