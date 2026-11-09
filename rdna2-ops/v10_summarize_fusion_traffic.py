#!/usr/bin/env python3
"""CPU-only validation of frozen v10 traffic driver output; never loads a GPU module.

The installed rocprofv3 kernel CSV has Thread_Id but no Process_Id. Establish its
process through an exact dispatch identity/time join to the PID-bearing counter
CSV, then require the PID-bearing ROCTX range to contain every dispatch.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCOPE = 'GL2C/EA interface request bytes; not identified physical DRAM bytes'
DRIVER_SHA = 'b236e122cd424dd57ff1d95bf6ce1df0707d657af950b148d85a22f415f47acb'
IDENTITY = ('Dispatch_Id', 'Agent_Id', 'Queue_Id', 'Kernel_Id', 'Kernel_Name',
            'Start_Timestamp', 'End_Timestamp', 'Correlation_Id', 'Thread_Id')
METRICS = ('FETCH_SIZE', 'WRITE_SIZE')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(HERE), 'Output must remain in the research folder')
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def csv_rows(directory, suffix):
    files = sorted(directory.glob('*_' + suffix + '.csv'))
    require(len(files) == 1, f'Expected one {suffix} CSV, got {len(files)}')
    with files[0].open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(rows, f'Empty {suffix} CSV')
    return rows, dict(path=str(files[0]), sha256=sha(files[0]), rows=len(rows))


def timestamps(row):
    start, end = int(row['Start_Timestamp']), int(row['End_Timestamp'])
    require(0 < start < end, 'Invalid trace timestamp interval')
    return start, end


def validate(run, counter_dir):
    run, counter_dir = Path(run).resolve(), Path(counter_dir).resolve()
    report = read(run / 'result.json')
    require(report.get('schema') == 'applio_v10_fusion_gl2c_traffic', 'Wrong run schema')
    require(report.get('completed') is True and report.get('valid_for_input_contract') is True,
            'Incomplete run or failed input contract')
    audit = report['gpu_audit']
    require(audit.get('valid_for_timing') is True and not audit['errors']
            and not audit['foreign_clients'], 'Failed exclusive GPU audit')
    require(report['driver_sha256'] == DRIVER_SHA == sha(HERE / 'v10_fusion_traffic.py'),
            'Frozen driver SHA mismatch')
    require(report['checks'] and all(v is True for v in report['checks'].values()),
            'Source/evidence preflight failed')
    require(report['finite'] and all(v is True for v in report['finite'].values())
            and report['fp32_masters_and_gradients'] is True
            and report['input_output_gradient_fp16'] is True
            and report['expected_route_pass'] is True, 'Finite/dtype/route contract failed')
    require(report['input_hashes'] == report['after_hashes'], 'Input/master hashes changed')
    require(report['memory_scope'] == SCOPE, 'Unexpected memory scope')
    for name, digest in report['sources'].items():
        require(sha(HERE / name) == digest, f'Source changed: {name}')
    for key in ('evidence', 'manifest', 'selection'):
        require(sha(report[key]) == report[key + '_sha256'], f'{key} changed')

    arm, pairs = report['arm'], report['measured_pairs']
    require(arm in ('B', 'B_prime') and isinstance(pairs, int) and pairs > 0, 'Invalid arm/reps')
    expected_per_pair = 28 if arm == 'B' else 22
    require(report['expected_kernel_count_per_pair'] == expected_per_pair, 'Wrong count contract')
    expected = expected_per_pair * pairs
    requested = report['requested_counters']
    require(requested and len(set(requested)) == len(requested)
            and set(requested) <= set(METRICS), 'Unexpected requested counters')
    kernels, kfile = csv_rows(counter_dir, 'kernel_trace')
    counters, cfile = csv_rows(counter_dir, 'counter_collection')
    markers, mfile = csv_rows(counter_dir, 'marker_api_trace')
    require(len(kernels) == expected, f'Collected {len(kernels)} kernels, expected {expected}')
    pid = str(report['pid'])
    require(all(r['Process_Id'] == pid for r in counters + markers), 'Foreign counter/marker PID')

    ranges = [r for r in markers if r['Domain'] == 'MARKER_CORE_RANGE_API']
    require(len(ranges) == 1 and ranges[0]['Function'] == report['label'], 'Wrong/multiple ROCTX ranges')
    range_start, range_end = timestamps(ranges[0])
    controls = sorted([r for r in markers if r['Domain'] == 'MARKER_CONTROL_API'],
                      key=lambda r: int(r['Start_Timestamp']))
    require([r['Function'] for r in controls] == ['roctxProfilerPause', 'roctxProfilerPause',
                                                'roctxProfilerResume', 'roctxProfilerPause'],
            'Unexpected global collection controls')
    require(len(markers) == 5, 'Unexpected marker rows')
    require(timestamps(controls[1])[1] < timestamps(controls[2])[0]
            and timestamps(controls[2])[1] <= range_start < range_end
            and range_end <= timestamps(controls[3])[0], 'Marker collection window invalid')

    by_dispatch = {}
    for row in kernels:
        key = row['Dispatch_Id']
        require(row['Kind'] == 'KERNEL_DISPATCH' and key not in by_dispatch,
                'Wrong kind or duplicate dispatch')
        require('Process_Id' not in row or row['Process_Id'] == pid, 'Foreign kernel PID')
        start, end = timestamps(row)
        require(range_start <= start < end <= range_end, 'Kernel outside named measured range')
        by_dispatch[key] = row
    values = {name: {} for name in requested}
    for row in counters:
        name, key = row['Counter_Name'], row['Dispatch_Id']
        require(name in values and key in by_dispatch and key not in values[name],
                'Unexpected/duplicate counter row')
        reference = by_dispatch[key]
        require(all(row[field] == reference[field] for field in IDENTITY),
                f'Counter/kernel identity or timestamp mismatch: {key}')
        value = float(row['Counter_Value'])
        require(math.isfinite(value) and value >= 0, 'Nonfinite/negative counter')
        values[name][key] = value
    for name, rows in values.items():
        require(set(rows) == set(by_dispatch), f'{name}: missing dispatch counters')
        require(math.isfinite(math.fsum(rows.values())) and math.fsum(rows.values()) > 0,
                f'{name}: invalid or all-zero traffic')

    return dict(schema='applio_v10_fusion_gl2c_summary_v2', valid_for_counters=True,
        arm=arm, seed=report['seed'], case=report['case'], input_hashes=report['input_hashes'],
        output_hashes=report['output_hashes'], sources=report['sources'],
        driver_sha256=report['driver_sha256'], summarizer_sha256=sha(__file__),
        controls={k: report[k] for k in ('seed', 'warmup', 'measured_pairs', 'case', 'sources',
            'driver_sha256', 'evidence_sha256', 'manifest_sha256', 'selection_sha256',
            'torch', 'hip', 'safety_env', 'db', 'memory_scope')},
        run=str(run / 'result.json'), run_sha256=sha(run / 'result.json'), pid=report['pid'],
        counter_dir=str(counter_dir), csv_files=dict(kernel=kfile, counter=cfile, marker=mfile),
        join_fields=list(IDENTITY), kernel_pid_source='Exact join to counter Process_Id',
        marker=dict(label=report['label'], pid=report['pid'], start_ns=range_start, end_ns=range_end),
        measured_pairs=pairs, observed_kernel_count=len(kernels), expected_kernel_count=expected,
        kernel_names=dict(Counter(r['Kernel_Name'] for r in kernels)), memory_scope=SCOPE,
        counters={name: dict(unit='KiB', sum=math.fsum(rows.values()),
                            bytes_per_pair=math.fsum(rows.values()) * 1024 / pairs)
                  for name, rows in values.items()},
        valid_for_timing=False,
        timing_scope='Counter-instrumented execution; existing speed measurements remain authoritative')


def compare(paths):
    require(len(paths) == 4, 'Comparison requires exactly four arm/metric summaries')
    items = [read(path) for path in paths]
    reference = items[0]
    arms = {'B': {}, 'B_prime': {}}
    evidence = []
    for path, item in zip(paths, items):
        require(item.get('schema') == 'applio_v10_fusion_gl2c_summary_v2'
                and item.get('valid_for_counters') is True, 'Invalid summary')
        require(sha(item['run']) == item['run_sha256'], 'Run changed after summarizing')
        for entry in item['csv_files'].values():
            require(sha(entry['path']) == entry['sha256'], 'CSV changed after summarizing')
        for key in ('controls', 'input_hashes', 'output_hashes', 'summarizer_sha256'):
            require(item[key] == reference[key], f'Cross-pass mismatch: {key}')
        require(len(item['counters']) == 1, 'Each comparison pass must collect one metric')
        name = next(iter(item['counters']))
        arm = item['arm']
        require(arm in arms and name in METRICS and name not in arms[arm], 'Duplicate/wrong arm metric')
        arms[arm][name] = item['counters'][name]['bytes_per_pair']
        evidence.append(dict(path=str(Path(path).resolve()), sha256=sha(path), arm=arm, metric=name))
    require(all(set(a) == set(METRICS) for a in arms.values()), 'Missing arm/metric')
    table = {}
    for name in (*METRICS, 'TOTAL'):
        before = sum(arms['B'].values()) if name == 'TOTAL' else arms['B'][name]
        after = sum(arms['B_prime'].values()) if name == 'TOTAL' else arms['B_prime'][name]
        table[name] = dict(B_bytes_per_pair=before, B_prime_bytes_per_pair=after,
            saved_bytes_per_pair=before-after, reduction_percent=100*(before-after)/before)
    return dict(schema='applio_v10_fusion_gl2c_comparison', valid_for_counters=True,
        memory_scope=SCOPE, controls=reference['controls'], input_hashes=reference['input_hashes'],
        output_hashes=reference['output_hashes'], evidence=evidence, metrics=table,
        kernel_count_per_pair={'B': 28, 'B_prime': 22},
        limitation='One measured pair per arm/metric, separate processes; no repeat uncertainty estimate. '
                   'GL2C/EA request bytes are not identified physical DRAM bytes. '
                   'Counter timestamps are not performance samples.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--counter-dir', type=Path)
    parser.add_argument('--compare', nargs=4, type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.compare:
        require(args.run is None and args.counter_dir is None and args.out is not None,
                '--compare requires --out and excludes --run/--counter-dir')
        result, target = compare(args.compare), args.out
    else:
        require(args.run is not None and args.counter_dir is not None, 'Need --run and --counter-dir')
        result, target = validate(args.run, args.counter_dir), args.out or args.run / 'traffic_summary.json'
    save_new(target, result)
    print(json.dumps(dict(out=str(target), valid_for_counters=result['valid_for_counters'],
                         counters=result.get('counters'), metrics=result.get('metrics')), allow_nan=False))


if __name__ == '__main__':
    main()
