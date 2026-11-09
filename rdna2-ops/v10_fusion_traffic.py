#!/usr/bin/env python3
"""One already-validated ResBlock pair for rocprofv3 GL2C/EA traffic.

Default is a CPU-only plan. --execute requires rocprofv3 --selected-regions,
--kernel-trace and a single --pmc pass. It does not rerun the full sweep,
operator gates, optimizer or epochs. --summarize is CPU-only postprocessing.
The measured interface is GL2C/EA, not identified physical GDDR6 traffic.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import copy
import csv
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import sys

# This helper module has standard-library imports only; no torch import here.
from bench_v10_fusion import pairs, sha, SAFETY

HERE = Path(__file__).resolve().parent
APP = Path('~/DevSpace/applio-lab')
SDK = APP / '.venv-rocm10/lib/python3.12/site-packages/_rocm_sdk_core'
CASE = 'B8_C32_T12800_K3_D1'
SCOPE = 'GL2C/EA interface request bytes; not identified physical DRAM bytes'


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def flag(name):
    return os.environ.get(name, '').lower() in ('1', 'true', 'yes', 'on')


def preflight(args):
    manifest_path, selection_path, evidence_path = (p.resolve() for p in
        (args.manifest, args.selection, args.evidence))
    manifest, selection, evidence = map(read, (manifest_path, selection_path, evidence_path))
    case = next(c for c in pairs(manifest) if c['name'] == CASE)
    row = next(c for c in evidence['cases'] if c['case']['name'] == CASE
               and c['input_dtype'] == 'float16')
    sources = {name: sha(HERE / name) for name in
               ('bench_v10_fusion.py', 'v08_direct_conv.py', 'v10_fused_resblock.py')}
    checks = {
        'completed_evidence': evidence.get('completed') is True,
        'valid_contract_evidence': evidence.get('valid_for_contract') is True,
        'valid_timing_evidence': evidence.get('valid_for_timing') is True,
        'row_pass': row.get('all_gates_pass') is True,
        'source_matches_evidence': sources['v10_fused_resblock.py'] == evidence['source_sha256'],
        'v08_source_matches': sources['v08_direct_conv.py'] == evidence['v08_source_sha256'],
        'benchmark_source_matches': sources['bench_v10_fusion.py'] == evidence['benchmark_sha256'],
        'manifest_matches': sha(manifest_path) == evidence['v08_manifest_sha256'],
        'selection_matches_source': selection.get('strict_gate_pass') is True
            and selection['source_sha256'] == sources['v10_fused_resblock.py']
            and selection['v08_manifest_sha256'] == sha(manifest_path),
        'selection_matches_evidence': selection['evidence_sha256'] == sha(evidence_path),
        'selected_fp16_pair': case['pair_key'] in selection['pair_keys']
            and 'torch.float16' in selection['input_dtypes_by_pair'][case['pair_key']],
        'seed_matches': args.seed == evidence['seed'],
        'roctx_library_exists': args.roctx_library.resolve().is_file(),
    }
    if not all(checks.values()):
        raise ValueError('Evidence preflight failed: ' + str(checks))
    expected = row['timing'][args.arm]['profile']['kernel_count']
    if expected != {'B': 28, 'B_prime': 22}[args.arm]:
        raise ValueError('Unexpected prior kernel-count evidence')
    return dict(case=case, arm=args.arm, seed=args.seed, warmup=args.warmup,
                measured_pairs=args.reps, label=f'{CASE}_float16_{args.arm}',
                expected_kernel_count_per_pair=expected,
                evidence=str(evidence_path), evidence_sha256=sha(evidence_path),
                manifest=str(manifest_path), manifest_sha256=sha(manifest_path),
                selection=str(selection_path), selection_sha256=sha(selection_path),
                sources=sources, checks=checks, requested_counters=args.counters,
                memory_scope=SCOPE, safety_env=SAFETY,
                prior_logical_traffic_model=row['logical_traffic_model'],
                prior_profile=row['timing'][args.arm]['profile']['trace'])


def run(args, plan):
    out, db = args.out.resolve(), args.db.resolve()
    if out.exists() or not out.is_relative_to(HERE) or not db.is_relative_to(HERE) or not db.is_dir():
        raise ValueError('Use a new research output directory and an existing research DB')
    requested = set(args.counters)
    active = set(os.environ.get('ROCPROF_COUNTERS', '').removeprefix('pmc:').split())
    if (not flag('ROCPROF_SELECTED_REGIONS') or not flag('ROCPROF_COUNTER_COLLECTION')
        or not flag('ROCPROF_KERNEL_TRACE')
        or flag('ROCPROF_SELECTED_REGIONS_REF_COUNT')
        or os.environ.get('ROCPROF_COUNTER_GROUPS') or active != requested):
        raise ValueError('Use --selected-regions without ref-count, --kernel-trace, and one matching --pmc pass')
    out.mkdir(parents=True)
    (out / 'miopen_cache').mkdir()
    os.environ.update(SAFETY)
    os.environ.update(MIOPEN_FIND_MODE='FAST', MIOPEN_USER_DB_PATH=str(db),
                      MIOPEN_CUSTOM_CACHE_DIR=str(out / 'miopen_cache'),
                      APPLIO_RDNA2_DIRECT_CONV='1', APPLIO_DILATION_FOLD='0',
                      PYTORCH_TUNABLEOP_ENABLED='0', PYTORCH_TUNABLEOP_TUNING='0')
    os.environ.setdefault('TRITON_CACHE_DIR', str(HERE / 'results/v10/triton_cache'))
    sys.path.insert(0, str(args.applio.resolve()))
    report = dict(schema='applio_v10_fusion_gl2c_traffic', **plan, pid=os.getpid(),
                  driver_sha256=sha(__file__), completed=False, valid_for_timing=False,
                  valid_for_input_contract=False, valid_for_counters=False,
                  counter_validation='Pending CPU CSV postprocessing after rocprofv3 exits',
                  db=str(db), roctx_library=str(args.roctx_library.resolve()))
    dump(out / 'PREDECLARED_GATES.json', dict(
        evidence_checks=plan['checks'], requested_counters=args.counters,
        measured_pairs=args.reps, expected_kernel_count=plan['expected_kernel_count_per_pair'] * args.reps,
        required=['finite output/gradients', 'unchanged input/master hashes',
                  'same inputs and masters across arms/passes', 'exclusive GPU audit',
                  'exact recorded kernel count', 'one finite nonnegative metric per dispatch'],
        memory_scope=SCOPE, timing_is_not_a_performance_sample=True))
    dump(out / 'result.json', report)
    from gpu_guard_v06 import GpuAudit
    audit = GpuAudit()
    try:
        with audit:
            audit.ensure_clear()
            roctx = ctypes.CDLL(str(args.roctx_library.resolve()))
            for name in ('roctxProfilerPause', 'roctxProfilerResume'):
                getattr(roctx, name).argtypes = [ctypes.c_uint64]
                getattr(roctx, name).restype = ctypes.c_int
            roctx.roctxRangeStartA.argtypes = [ctypes.c_char_p]
            roctx.roctxRangeStartA.restype = ctypes.c_uint64
            roctx.roctxRangeStop.argtypes = [ctypes.c_uint64]
            roctx.roctxRangeStop.restype = None

            def control(name):
                rc = getattr(roctx, name)(0)  # All threads, including autograd workers.
                if rc != 0:
                    raise RuntimeError(f'{name}(0) returned {rc}')

            control('roctxProfilerPause')
            import torch
            import v08_direct_conv as base
            import v10_fused_resblock as fusion
            from rvc.lib.algorithm.residuals import ResBlock
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = False
            control('roctxProfilerPause')
            torch.manual_seed(args.seed)
            rng = torch.Generator(device='cpu').manual_seed(args.seed)
            case = plan['case']
            original = ResBlock(case['shape'][1], case['weight'][2], (case['dilation'],))
            # Exact construction in bench_v10_fusion.one; only the selected arm is materialized.
            with torch.no_grad():
                for conv in [*original.convs1, *original.convs2]:
                    v = conv.parametrizations.weight.original1
                    values = torch.randn(v.shape, generator=rng) / math.sqrt(math.prod(v.shape[1:]))
                    v.copy_(values)
                    conv.parametrizations.weight.original0.copy_(values.square().sum((1, 2), keepdim=True).sqrt())
                    conv.bias.copy_(torch.randn(conv.bias.shape, generator=rng) * .1)
            xc = torch.randn(case['shape'], generator=rng).to(torch.float16)
            xc.view(-1)[::127] = 0
            dyc = (torch.randn(case['shape'], generator=rng) * .01).to(torch.float16)

            def tensor_hash(t):
                t = t.detach().contiguous().cpu()
                h = hashlib.sha256(str((tuple(t.shape), str(t.dtype))).encode())
                h.update(t.numpy().tobytes())
                return h.hexdigest()

            def state_hash(model):
                h = hashlib.sha256()
                for name, t in sorted(model.state_dict().items()):
                    h.update(name.encode()); h.update(tensor_hash(t).encode())
                return h.hexdigest()

            input_hashes = dict(x=tensor_hash(xc), dy=tensor_hash(dyc), masters=state_hash(original))
            model = copy.deepcopy(original).cuda()
            x = xc.clone().cuda().requires_grad_(True)
            dy = dyc.cuda()
            hits, fusion_hits = {}, {}

            def invoke():
                model.zero_grad(set_to_none=True)
                x.grad = None
                with torch.autocast('cuda', dtype=torch.float16):
                    y = model(x)
                y.backward(dy)
                return y

            with ExitStack() as stack:
                stack.enter_context(base.measured_dispatch(args.manifest, hits=hits))
                if args.arm == 'B_prime':
                    stack.enter_context(fusion.fused_resblocks(args.manifest,
                        selection=args.selection, hits=fusion_hits))
                for _ in range(args.warmup):
                    invoke()
                torch.cuda.synchronize()
                audit.ensure_clear()
                before_hits = dict(hits)
                before_fusion = dict(fusion_hits)
                control('roctxProfilerResume')
                range_id = roctx.roctxRangeStartA(plan['label'].encode())
                try:
                    for _ in range(args.reps):
                        y = invoke()
                    torch.cuda.synchronize()
                finally:
                    roctx.roctxRangeStop(range_id)
                    control('roctxProfilerPause')
                audit.ensure_clear()
            # All copies, hashing and finite checks occur after global pause.
            named_grads = {name: p.grad for name, p in model.named_parameters()}
            tensors = {'y': y, 'dx': x.grad, **named_grads}
            finite = {name: t is not None and bool(torch.isfinite(t.detach().cpu()).all())
                      for name, t in tensors.items()}
            fp32_masters = all(p.dtype == torch.float32 and p.grad is not None
                               and p.grad.dtype == torch.float32 for p in model.parameters())
            after_hashes = dict(x=tensor_hash(x), dy=tensor_hash(dy), masters=state_hash(model))
            measured_hits = {k: v - before_hits.get(k, 0) for k, v in hits.items()}
            measured_fusion = {k: v - before_fusion.get(k, 0) for k, v in fusion_hits.items()}
            route_ok = (sum(measured_hits.values()) == 2 * args.reps if args.arm == 'B'
                        else measured_fusion.get(case['pair_key'], 0) == args.reps)
            report.update(input_hashes=input_hashes, after_hashes=after_hashes,
                output_hashes={name: tensor_hash(t) for name, t in tensors.items() if t is not None},
                finite=finite, fp32_masters_and_gradients=fp32_masters,
                input_output_gradient_fp16=x.dtype == y.dtype == x.grad.dtype == torch.float16,
                measured_direct_hits=measured_hits, measured_fusion_hits=measured_fusion,
                expected_route_pass=route_ok, torch=torch.__version__, hip=torch.version.hip,
                input_construction='bench_v10_fusion.py one(), same fixed CPU RNG order',
                measurement_scope='warmup paused; one global resume/pause window; one arm per process')
            report['valid_for_input_contract'] = (all(finite.values()) and fp32_masters
                and input_hashes == after_hashes and route_ok and report['input_output_gradient_fp16'])
            report['completed'] = report['valid_for_input_contract']
            if not report['completed']:
                raise RuntimeError('Measured pair input/dtype/finite/route contract failed')
    except BaseException as exc:
        report['fatal_error'] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        report['gpu_audit'] = audit.report()
        report['completed'] = report['completed'] and report['gpu_audit']['valid_for_timing']
        dump(out / 'result.json', report)
    print(json.dumps(dict(completed=report['completed'], arm=args.arm,
                         result=str(out / 'result.json'), counter_validation=report['counter_validation'])))
    if not report['completed']:
        raise SystemExit(2)


def summarize(out, counter_dir):
    out, counter_dir = out.resolve(), counter_dir.resolve()
    if not out.is_relative_to(HERE):
        raise ValueError('Summary output must remain in the research folder')
    report = read(out / 'result.json')
    if (report.get('schema') != 'applio_v10_fusion_gl2c_traffic'
        or not report['completed'] or not report['gpu_audit']['valid_for_timing']):
        raise ValueError('Run did not complete with an exclusive GPU audit')
    expected = report['expected_kernel_count_per_pair'] * report['measured_pairs']
    pid = str(report['pid'])
    kernel_rows = []
    counter_rows = []
    for pattern, rows in (('*_kernel_trace.csv', kernel_rows), ('*_counter_collection.csv', counter_rows)):
        files = sorted(counter_dir.rglob(pattern))
        if not files:
            raise ValueError('Missing rocprofv3 CSV: ' + pattern)
        for path in files:
            with path.open(newline='') as f:
                rows.extend(csv.DictReader(f))
    def dispatch(row):
        if str(row['Process_Id']) != pid:
            raise ValueError('Unexpected process in counter output')
        return str(row['Dispatch_Id'])
    kernels = {dispatch(r): r['Kernel_Name'] for r in kernel_rows}
    if len(kernels) != len(kernel_rows) or len(kernels) != expected:
        raise ValueError(f'Collected {len(kernels)} kernels; expected {expected}. Warmup/boundaries may be wrong')
    values = {name: {} for name in report['requested_counters']}
    for row in counter_rows:
        name, key = row['Counter_Name'], dispatch(row)
        if name not in values or key not in kernels or key in values[name]:
            raise ValueError('Unexpected/duplicate counter row')
        value = float(row['Counter_Value'])
        if not math.isfinite(value) or value < 0:
            raise ValueError('Counter value must be finite and nonnegative')
        values[name][key] = value
    for name, rows in values.items():
        if set(rows) != set(kernels):
            raise ValueError(f'{name}: missing per-kernel counters')
        if sum(rows.values()) <= 0:
            raise ValueError(f'{name}: all-zero traffic is not accepted as validated measurement for this pair')
    result = dict(schema='applio_v10_fusion_gl2c_summary', valid_for_counters=True,
                  arm=report['arm'], input_hashes=report['input_hashes'],
                  output_hashes=report['output_hashes'], sources=report['sources'],
                  run_sha256=sha(out / 'result.json'), run=str(out / 'result.json'),
                  counter_dir=str(counter_dir), measured_pairs=report['measured_pairs'],
                  observed_kernel_count=len(kernels), expected_kernel_count=expected,
                  kernel_names=dict(Counter(kernels.values())), memory_scope=SCOPE,
                  counters={name: dict(unit='KiB', sum=sum(rows.values()),
                      bytes_per_pair=sum(rows.values()) * 1024 / report['measured_pairs'])
                      for name, rows in values.items()},
                  comparison_requirement='B/B_prime and read/write passes must have identical input_hashes and source SHAs',
                  timing_scope='Counter-instrumented execution; do not use it to replace existing speed measurements')
    target = out / 'traffic_summary.json'
    if target.exists():
        raise ValueError('Refusing to overwrite an existing traffic summary')
    dump(target, result)
    print(json.dumps(result, indent=2, allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--summarize', type=Path)
    p.add_argument('--counter-dir', type=Path)
    p.add_argument('--arm', choices=('B', 'B_prime'), default='B')
    p.add_argument('--out', type=Path)
    p.add_argument('--manifest', type=Path, default=HERE / 'results/applio_v08/direct_selected_final/dispatch.json')
    p.add_argument('--selection', type=Path, default=HERE / 'results/v10/fusion_full/selection.json')
    p.add_argument('--evidence', type=Path, default=HERE / 'results/v10/fusion_full/result.json')
    p.add_argument('--db', type=Path, default=HERE / 'results/v10/fusion_bench_db')
    p.add_argument('--applio', type=Path, default=APP)
    p.add_argument('--roctx-library', type=Path, default=SDK / 'lib/librocprofiler-sdk-roctx.so.1')
    p.add_argument('--seed', type=int, default=20261108)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--reps', type=int, default=1)
    p.add_argument('--counters', nargs='+', choices=('FETCH_SIZE', 'WRITE_SIZE'), default=['FETCH_SIZE', 'WRITE_SIZE'])
    args = p.parse_args()
    if args.summarize:
        if args.execute or not args.counter_dir:
            p.error('--summarize requires --counter-dir and cannot use --execute')
        summarize(args.summarize, args.counter_dir)
        return
    if args.warmup < 1 or args.reps < 1 or len(set(args.counters)) != len(args.counters):
        p.error('Positive warmup/reps and distinct counter names required')
    plan = preflight(args)
    if not args.execute:
        print(json.dumps(dict(mode='cpu_plan_only', **plan), indent=2))
        return
    if not args.out:
        p.error('--execute requires --out')
    run(args, plan)


if __name__ == '__main__':
    main()
