#!/usr/bin/env python3
"""v0.8 nonconv gates and timings; run CUDA only while holding the root GPU lease.

Examples:
  .../.venv-rocm10/bin/python v08_nonconv_bench.py --device cpu --out results/applio_v08/nonconv_cpu.json
  .../.venv-rocm10/bin/python v08_nonconv_bench.py --device cuda --out results/applio_v08/nonconv_gpu.json --profile
  .../.venv-rocm10/bin/python v08_nonconv_bench.py --device cuda --only adam --snapshots results/applio_v08/snapshots/*.pt --out results/applio_v08/adam_actual.json

Slicing must be bitwise equal in forward and backward. AdamW keeps FP32 parameter
and moment tensors; same-step parameter/moment error must satisfy NRMSE<=2e-6 and
max/reference RMS<=2e-5, both aggregate and per tensor. In addition, parameter
update error uses the user's operator gate (0.002 / 0.02). No criteria are relaxed
after measurements. AMP overflow must leave parameters/moments/step byte-identical.
This is operator evidence; epoch and final listening judgments are separate.
"""
from __future__ import annotations
import argparse
import copy
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

# Every research process carries the same known-correct solver exclusions.
os.environ["MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1"] = "0"
os.environ["MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53"] = "0"

import torch
from v08_nonconv_opt import (
    _cpu_tree, grad_norm_device, grad_norm_foreach, optimizer_flags,
    slice_segments_gather,
)

GATES = dict(slice_bitwise=True, adam_nrmse=2e-6, adam_max_over_rms=2e-5,
             update_nrmse=0.002, update_max_over_rms=0.02,
             overflow_skip_bitwise=True)
HYPER = dict(lr=1e-4, betas=(0.8, 0.99), eps=1e-9, weight_decay=0.01)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def native_slice(x, starts, segment_size, dim):
    out = torch.zeros_like(x[:, :segment_size] if dim == 2 else x[:, :, :segment_size])
    for i in range(x.shape[0]):
        start = starts[i].item()
        if dim == 2:
            out[i] = x[i, start:start + segment_size]
        else:
            out[i] = x[i, :, start:start + segment_size]
    return out


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def timing(fn, device, repeats=10, trials=3):
    synchronize(device)
    start = time.perf_counter()
    fn()
    synchronize(device)
    first_ms = (time.perf_counter() - start) * 1000
    for _ in range(3):
        fn()
    synchronize(device)
    rows = []
    for _ in range(trials):
        if device == "cuda":
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        if device == "cuda":
            end.record()
        synchronize(device)
        wall = (time.perf_counter() - start) * 1000 / repeats
        rows.append(dict(wall_ms=wall, event_ms=begin.elapsed_time(end) / repeats if device == "cuda" else None))
    return dict(first_timed_invocation_ms=first_ms, trials=rows,
                warm_wall_ms=statistics.median(r["wall_ms"] for r in rows),
                warm_event_ms=statistics.median(r["event_ms"] for r in rows) if device == "cuda" else None,
                note="Timed after correctness; first timed invocation is not process cold. Event span includes host launch gaps; profiler kernel totals are separate.")


def metrics(refs, values, nrmse_gate, max_gate):
    total_e = total_r = max_abs = 0.0
    count = 0
    bitwise = finite = True
    worst_nrmse = worst_max = 0.0
    bad = []
    nonfinite = []
    missing = []
    for i, (a, b) in enumerate(zip(refs, values, strict=True)):
        if a is None or b is None:
            if a is not b:
                bad.append(i); bitwise = finite = False
                missing.append(i)
            continue
        a = a.detach().cpu()
        b = b.detach().cpu()
        bitwise &= (a.dtype == b.dtype and a.shape == b.shape and torch.equal(
            a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)))
        f = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        finite &= f
        if not f:
            bad.append(i)
            nonfinite.append(dict(index=i,
                reference_nan=int(torch.isnan(a).sum()),
                reference_posinf=int(torch.isposinf(a).sum()),
                reference_neginf=int(torch.isneginf(a).sum()),
                candidate_nan=int(torch.isnan(b).sum()),
                candidate_posinf=int(torch.isposinf(b).sum()),
                candidate_neginf=int(torch.isneginf(b).sum())))
            count += a.numel()
            continue
        aa, bb = a.double(), b.double()
        d = bb - aa
        er, rr = float(d.square().sum()), float(aa.square().sum())
        n = a.numel()
        mx = float(d.abs().max()) if n else 0.0
        rms = math.sqrt(rr / max(n, 1))
        nr = math.sqrt(er / max(rr, 1e-60))
        mr = mx / max(rms, 1e-30)
        if not f or nr > nrmse_gate or mr > max_gate:
            bad.append(i)
        worst_nrmse, worst_max = max(worst_nrmse, nr), max(worst_max, mr)
        total_e += er; total_r += rr; count += n; max_abs = max(max_abs, mx)
    agg_rms = math.sqrt(total_r / max(count, 1))
    nrmse = math.sqrt(total_e / max(total_r, 1e-60))
    max_over_rms = max_abs / max(agg_rms, 1e-30)
    return dict(bitwise=bool(bitwise), finite=bool(finite), numel=count,
                nrmse=nrmse if finite else None,
                max_abs=max_abs if finite else None,
                max_over_rms=max_over_rms if finite else None,
                worst_tensor_nrmse=worst_nrmse if finite else None,
                worst_tensor_max_over_rms=worst_max if finite else None,
                nonfinite_tensors=nonfinite, missing_tensor_indices=missing,
                failed_tensor_indices=bad, pass_gate=bool(finite and not bad and
                    nrmse <= nrmse_gate and max_over_rms <= max_gate))


def synthetic(shape, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shapes = [shape, (shape[0], 1, 1), (shape[0],)]
    params = [torch.randn(s, generator=generator).mul_(0.05) for s in shapes]
    grads = [torch.randn(s, generator=generator).mul_(0.02) for s in shapes]
    p = [torch.nn.Parameter(t.clone()) for t in params]
    opt = torch.optim.AdamW(p, **HYPER, foreach=False, fused=False)
    for _ in range(3):
        for pp, g in zip(p, grads):
            pp.grad = g.clone()
        opt.step()
    return dict(params=[pp.detach().clone() for pp in p], grads=grads,
                optimizer_state=_cpu_tree(opt.state_dict()),
                parameter_shapes=[list(t.shape) for t in p],
                note="Synthetic parameter values in actual Conv/weight-norm shapes; not a model update.")


def make_optimizer(snapshot, mode, device):
    params = [torch.nn.Parameter(x.to(device).clone()) for x in snapshot["params"]]
    sd = copy.deepcopy(snapshot["optimizer_state"])
    groups = []
    for g in sd["param_groups"]:
        g.update(optimizer_flags(mode))
        groups.append({**{k: v for k, v in g.items() if k != "params"},
                       "params": [params[i] for i in g["params"]]})
    optimizer = torch.optim.AdamW(groups, **optimizer_flags(mode))
    optimizer.load_state_dict(sd)
    for p, g in zip(params, snapshot["grads"]):
        p.grad = None if g is None else g.to(device).clone()
    return params, optimizer


def capture(params, optimizer):
    return dict(
        params=[p.detach().cpu().clone() for p in params],
        exp_avg=[_cpu_tree(optimizer.state[p].get("exp_avg")) for p in params],
        exp_avg_sq=[_cpu_tree(optimizer.state[p].get("exp_avg_sq")) for p in params],
        step=[_cpu_tree(optimizer.state[p].get("step")) for p in params],
    )


def compare_state(reference, candidate, initial):
    rows = {key: metrics(reference[key], candidate[key], GATES["adam_nrmse"],
                         GATES["adam_max_over_rms"])
            for key in ("params", "exp_avg", "exp_avg_sq", "step")}
    delta_a = [a - x for a, x in zip(reference["params"], initial)]
    delta_b = [b - x for b, x in zip(candidate["params"], initial)]
    rows["parameter_update"] = metrics(delta_a, delta_b, GATES["update_nrmse"],
                                      GATES["update_max_over_rms"])
    rows["pass_gate"] = all(row["pass_gate"] for row in rows.values())
    return rows


def profile_kernels(fn):
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA
    ]) as prof:
        fn()
        torch.cuda.synchronize()
    kernels = {}
    for event in prof.events():
        if str(event.device_type).endswith("CUDA"):
            row = kernels.setdefault(event.name, dict(name=event.name, calls=0, us=0.0))
            row["calls"] += 1
            row["us"] += float(event.device_time_total)
    return sorted(kernels.values(), key=lambda x: x["us"], reverse=True)


def amp_skip_check(device, mode):
    gen = torch.Generator(device="cpu").manual_seed(891)
    params = [torch.nn.Parameter(torch.randn((257, 17), generator=gen).to(device))]
    opt = torch.optim.AdamW(params, **HYPER, **optimizer_flags(mode))
    scaler = torch.amp.GradScaler(device, init_scale=128.0, growth_interval=2000)
    # Initialize through public scale API. Inject one known finite scaled gradient.
    scaler.scale(torch.ones((), device=device))
    params[0].grad = torch.randn(params[0].shape, generator=gen).to(device) * scaler.get_scale()
    scaler.unscale_(opt); scaler.step(opt); scaler.update()
    before = capture(params, opt)
    old_scale = scaler.get_scale()
    params[0].grad = torch.ones_like(params[0]) * old_scale
    params[0].grad.view(-1)[0] = float("inf")
    scaler.unscale_(opt); scaler.step(opt); scaler.update()
    after = capture(params, opt)
    states = {k: metrics(before[k], after[k], 0, 0) for k in before}
    return dict(state=states, scale_before=old_scale, scale_after=scaler.get_scale(),
                pass_gate=all(x["bitwise"] for x in states.values()) and scaler.get_scale() == old_scale / 2)


def run_adam(snapshot, label, args, save):
    references = {}
    rows = []
    for mode in ("native", "single", "foreach", "fused"):
        row = dict(mode=mode)
        try:
            params, optimizer = make_optimizer(snapshot, mode, args.device)
            checks = {}
            for step in range(1, args.steps + 1):
                if step == 1:
                    synchronize(args.device)
                    first_start = time.perf_counter()
                optimizer.step()
                if step == 1:
                    synchronize(args.device)
                    row["first_loaded_state_update_wall_ms"] = (time.perf_counter() - first_start) * 1000
                if step in {1, args.steps}:
                    current = capture(params, optimizer)
                    if mode == "native":
                        references[step] = current
                    checks[str(step)] = compare_state(references[step], current, snapshot["params"])
            row["same_step"] = checks
            row["timing"] = timing(optimizer.step, args.device, args.repeats, args.trials)
            if args.profile and args.device == "cuda":
                row["kernel_rows"] = profile_kernels(optimizer.step)
            row["amp_overflow"] = amp_skip_check(args.device, mode)
            row["pass_gate"] = all(v["pass_gate"] for v in checks.values()) and row["amp_overflow"]["pass_gate"]
            del params, optimizer
        except Exception as exc:
            row.update(pass_gate=False, error_type=type(exc).__name__, error=str(exc),
                       traceback=traceback.format_exc(limit=5))
        rows.append(row)
        save()
    native = next((x for x in rows if x["mode"] == "native" and "timing" in x), None)
    if native:
        for row in rows:
            if "timing" in row:
                row["speedup_wall"] = native["timing"]["warm_wall_ms"] / row["timing"]["warm_wall_ms"]
    return dict(label=label, parameter_shapes=snapshot["parameter_shapes"],
                parameter_count=len(snapshot["params"]),
                parameter_numel=sum(x.numel() for x in snapshot["params"]), rows=rows)


def run_slices(args):
    cases = [
        ("latent", (8, 192, 298), 32, 3),
        ("pitch", (8, 298), 32, 2),
        ("waveform", (8, 1, 119200), 12800, 3),
        ("mel_log", (8, 125, 298), 32, 3),
    ]
    rows = []
    for label, shape, length, dim in cases:
        for dtype in (torch.float16, torch.float32):
            generator = torch.Generator(device="cpu").manual_seed(444)
            x = torch.randn(shape, generator=generator).to(device=args.device, dtype=dtype).requires_grad_()
            starts = torch.randint(0, shape[-1] - length + 1, (8,), generator=generator).to(args.device)
            baseline = native_slice(x, starts, length, dim)
            grad = torch.randn(baseline.shape, generator=generator).to(device=args.device, dtype=dtype)
            dx = torch.autograd.grad(baseline, x, grad)[0]
            candidate = slice_segments_gather(x, starts, length, dim)
            dx2 = torch.autograd.grad(candidate, x, grad)[0]
            forward = metrics([baseline], [candidate], 0, 0)
            backward = metrics([dx], [dx2], 0, 0)
            a = timing(lambda: native_slice(x, starts, length, dim), args.device, args.repeats, args.trials)
            b = timing(lambda: slice_segments_gather(x, starts, length, dim), args.device, args.repeats, args.trials)
            rows.append(dict(label=label, shape=shape, length=length, dim=dim, dtype=str(dtype),
                forward=forward, backward=backward, native=a, gather=b,
                speedup_wall=a["warm_wall_ms"] / b["warm_wall_ms"],
                pass_gate=forward["bitwise"] and backward["bitwise"]))
    return rows


def run_norm(args):
    generator = torch.Generator(device="cpu").manual_seed(1911)
    params = []
    for shape in [(64, 64, 7)] * 12 + [(192, 192, 5)] * 4 + [(1024, 512, 5, 1)]:
        p = torch.nn.Parameter(torch.zeros(shape, device=args.device))
        p.grad = torch.randn(shape, generator=generator).to(args.device)
        params.append(p)
    baseline = grad_norm_device(params)
    rows = []
    for name, fn in (
        ("native_item", lambda: grad_norm_device(params).item()),
        ("device", lambda: grad_norm_device(params)),
        ("foreach", lambda: grad_norm_foreach(params)),
    ):
        value = torch.as_tensor(fn(), device=args.device)
        error = metrics([baseline], [value], 0.002, 0.02)
        rows.append(dict(mode=name, error=error, timing=timing(fn, args.device, args.repeats, args.trials)))
    return rows


def run_weight_norm_cache(args):
    """Compare actual D parameters with the process-local cache candidate.

    Fixed synthetic B8 waveform windows exercise both native training call sites:
    D loss parameter gradients, and G adversarial/feature-loss input gradients.
    The saved parameters come from the real pre-update D snapshot. No optimizer
    update, training-state mutation or audio-quality claim is made by this test.
    """
    from v08_nonconv_opt import apply
    root = args.applio_copy.resolve()
    sys.path.insert(0, str(root))
    from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator
    from rvc.train.losses import discriminator_loss, generator_loss, feature_loss

    snapshot = torch.load(args.wn_snapshot, map_location="cpu", weights_only=True)
    if snapshot.get("optimizer_index") != 1:
        raise ValueError("Weight-norm cache test requires the actual D optimizer snapshot")
    torch.manual_seed(63117)
    model = MultiPeriodDiscriminator(use_spectral_norm=False, checkpointing=False,
                                     version="v2").to(args.device).train()
    params = list(model.parameters())
    if len(params) != len(snapshot["params"]):
        raise ValueError("Actual D snapshot parameter count differs from native model")
    with torch.no_grad():
        for p, value in zip(params, snapshot["params"], strict=True):
            if p.shape != value.shape or value.dtype != torch.float32:
                raise ValueError("Actual D snapshot shape/dtype differs from native model")
            p.copy_(value)
    del snapshot
    generator = torch.Generator(device="cpu").manual_seed(63117)
    real = torch.randn((8, 1, 12800), generator=generator).mul_(0.05).to(args.device)
    fake = torch.randn((8, 1, 12800), generator=generator).mul_(0.05).to(
        device=args.device, dtype=torch.float16)
    scale = args.wn_grad_scale
    references = {}
    rows = []
    for mode in ("native", "cached"):
        handle = None
        if mode == "cached":
            options = dict(APPLIO_RDNA2_ADAMW="native", APPLIO_RDNA2_ADAMW_G="native",
                APPLIO_RDNA2_ADAMW_D="native", APPLIO_RDNA2_GRAD_NORM="native",
                APPLIO_RDNA2_SLICE_GATHER="0", APPLIO_RDNA2_D_WN_CACHE="1",
                APPLIO_RDNA2_SKIP_EMPTY_CACHE="0", APPLIO_RDNA2_OPT_SNAPSHOT_DIR="")
            train_stub = SimpleNamespace(config=SimpleNamespace(
                model=SimpleNamespace(use_spectral_norm=False)))
            with patch.dict(os.environ, options):
                handle = apply(train_stub)
        try:
            for stage in ("D_parameter_gradients", "G_input_gradient"):
                is_d = stage == "D_parameter_gradients"
                model.requires_grad_(is_d)

                def execute(capture_result=False):
                    model.zero_grad(set_to_none=True)
                    generated = fake.detach().requires_grad_(not is_d)
                    with torch.autocast("cuda", dtype=torch.float16):
                        real_scores, fake_scores, real_maps, fake_maps = model(real, generated)
                    loss = (discriminator_loss(real_scores, fake_scores)[0] if is_d else
                            generator_loss(fake_scores)[0] + feature_loss(real_maps, fake_maps))
                    (loss * scale).backward()
                    if capture_result:
                        gradients = ([p.grad for p in params] if is_d else [generated.grad])
                        return dict(
                            output=[x.detach().cpu().clone() for x in (*real_scores, *fake_scores)],
                            loss=[loss.detach().float().cpu().clone()],
                            gradient=[None if g is None else g.detach().float().div(scale).cpu()
                                      for g in gradients])

                current = execute(capture_result=True)
                if mode == "native":
                    references[stage] = current
                errors = {key: metrics(references[stage][key], current[key], 0.002, 0.02)
                          for key in ("output", "loss", "gradient")}
                row = dict(mode=mode, stage=stage, errors=errors,
                           pass_gate=all(value["pass_gate"] for value in errors.values()),
                           timing=timing(execute, args.device, args.repeats, args.trials))
                if args.profile:
                    row["kernel_rows"] = profile_kernels(execute)
                rows.append(row)
                del current
        finally:
            if handle is not None:
                handle.restore()
    for row in rows:
        native = next(x for x in rows if x["mode"] == "native" and x["stage"] == row["stage"])
        row["speedup_wall"] = native["timing"]["warm_wall_ms"] / row["timing"]["warm_wall_ms"]
    return dict(snapshot=str(args.wn_snapshot), snapshot_sha256=digest(args.wn_snapshot),
                parameter_count=len(params), parameter_numel=sum(p.numel() for p in params),
                input_shape=[8, 1, 12800], real_dtype="float32", fake_dtype="float16",
                autocast="float16", fixed_gradient_scale=scale,
                reference="Native D FP32 parameters under the same FP16 AMP; actual saved D weights, fixed synthetic waveforms",
                gate=dict(finite=True, nrmse=0.002, max_over_rms=0.02),
                rows=rows, pass_gate=all(row["pass_gate"] for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, nargs="*", default=[])
    parser.add_argument("--only", choices=["all", "adam", "gather", "norm", "wn"], default="all")
    parser.add_argument("--wn-snapshot", type=Path)
    parser.add_argument("--wn-grad-scale", type=float, default=128.0)
    parser.add_argument("--db-seed", type=Path)
    parser.add_argument("--applio-copy", type=Path,
                        default=Path(__file__).resolve().parent / "results/applio_v08/applio_copy")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    research = Path(__file__).resolve().parent
    args.out = args.out.resolve()
    if not args.out.is_relative_to(research) or args.out.exists():
        parser.error("Choose a new results file inside rdna2-ops")
    if args.steps < 1 or args.repeats < 1 or args.trials < 1:
        parser.error("Positive steps/repeats/trials required")
    if not math.isfinite(args.wn_grad_scale) or args.wn_grad_scale <= 0:
        parser.error("--wn-grad-scale must be finite and positive")
    if args.only == "wn":
        if args.device != "cuda" or args.wn_snapshot is None or args.db_seed is None:
            parser.error("--only wn requires --device cuda, --wn-snapshot and a research --db-seed")
        for path in (args.wn_snapshot, args.db_seed, args.applio_copy):
            if not path.resolve().is_relative_to(research) or not path.exists():
                parser.error("Weight-norm test inputs and code must exist inside rdna2-ops")
        db = args.out.with_name(args.out.stem + "_miopen_db")
        if db.exists():
            parser.error("Weight-norm test DB output already exists; choose a new --out")
        shutil.copytree(args.db_seed, db)
        os.environ.update(MIOPEN_FIND_MODE="FAST", MIOPEN_USER_DB_PATH=str(db),
                          MIOPEN_CUSTOM_CACHE_DIR=str(db / "kernel_cache"))
        (db / "kernel_cache").mkdir(exist_ok=True)
        torch.backends.cudnn.benchmark = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    result = dict(schema="applio_nonconv_bench_v08", device=args.device,
                  torch=str(torch.__version__), hip=torch.version.hip,
                  predeclared_gates=GATES, adam_hyperparameters=HYPER,
                  note="CPU results validate code only; CUDA warm operator timings and model acceptance are distinct.",
                  source_sha256={str(p): digest(p) for p in [Path(__file__), research / "v08_nonconv_opt.py"]},
                  adam=[], slices=[], norms=[], weight_norm_cache=None)
    def save():
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    save()
    if args.device == "cuda":
        from gpu_guard_v06 import GpuAudit
        context = GpuAudit()
    else:
        context = nullcontext()
    audit = None
    try:
        with context as audit:
            if audit:
                audit.ensure_clear()
                result["gpu_name"] = torch.cuda.get_device_name()
            if args.only in ("all", "gather"):
                result["slices"] = run_slices(args)
                save()
            if args.only in ("all", "norm"):
                result["norms"] = run_norm(args)
                save()
            if args.only == "wn":
                result["weight_norm_cache"] = run_weight_norm_cache(args)
                save()
            if args.only in ("all", "adam"):
                if args.snapshots:
                    for path in args.snapshots:
                        path = path.resolve()
                        if not path.is_relative_to(research):
                            raise ValueError("Use research-owned snapshots")
                        snapshot = torch.load(path, map_location="cpu", weights_only=True)
                        row = run_adam(snapshot, str(path), args, save)
                        row["snapshot_sha256"] = digest(path)
                        result["adam"].append(row)
                        save()
                else:
                    for i, shape in enumerate(((64, 64, 7), (192, 192, 5), (1024, 512, 5, 1))):
                        result["adam"].append(run_adam(synthetic(shape, 902 + i), str(shape), args, save))
                        save()
    except Exception as exc:
        result["error"] = dict(type=type(exc).__name__, message=str(exc),
                               traceback=traceback.format_exc())
        raise
    finally:
        result["gpu_audit"] = audit.report() if audit else None
        result["valid_for_gpu_timing"] = bool(audit and audit.report()["valid_for_timing"] and "error" not in result)
        save()
    print(json.dumps(dict(out=str(args.out), valid_for_gpu_timing=result["valid_for_gpu_timing"],
                          adam_rows=len(result["adam"]), slice_rows=len(result["slices"])), indent=2))


if __name__ == "__main__":
    main()
