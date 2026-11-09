#!/usr/bin/env python3
"""Read-only Strata v1.0 prerequisites; never imports a GPU runtime or starts a server.

The model cache census uses PROT_NONE/mincore without touching mapped pages.
Large protected inputs are inventoried, not hashed, by this preparation tool.
"""
from __future__ import annotations

import argparse
import ast
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DEV = Path('~/DevSpace')
LAB = DEV / 'strata-prefill-v07-lab-20261107'
EXE = LAB / 'build-v07-gfx1030/strata'
CONFIG = DEV / 'strata_v01403_port/prod-promoted/context-verified.private.json'
UNIT = Path('~/.config/systemd/user/strata-verified-256k.service')
SHADOW = DEV / 'strata_v01403_port/diag-guard-wide-corpus-20261009/shadow-canary-v2'
SWAP_LIMIT_BYTES = 256 << 20


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def service_state():
    env = dict(os.environ, XDG_RUNTIME_DIR='/run/user/1000',
               DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
    command = ['systemctl', '--user', 'show', 'strata-verified-256k.service',
               '--property=ActiveState,SubState,MainPID,FragmentPath,DropInPaths']
    p = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
    if p.returncode:
        return {'query_failed': True, 'exit_code': p.returncode}
    return dict(line.split('=', 1) for line in p.stdout.splitlines() if '=' in line)


def pressure():
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        parts = value.split()
        if len(parts) == 2 and parts[1] == 'kB':
            mem[key] = int(parts[0]) * 1024
    vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    return dict(meminfo_bytes=mem, mem_available_bytes=mem['MemAvailable'],
                swap_used_bytes=mem['SwapTotal'] - mem['SwapFree'],
                pswpin_pages=int(vm['pswpin']), pswpout_pages=int(vm['pswpout']),
                page_bytes=os.sysconf('SC_PAGE_SIZE'),
                memory_psi=Path('/proc/pressure/memory').read_text().strip(),
                io_psi=Path('/proc/pressure/io').read_text().strip())


def kfd_clients():
    result = []
    for path in sorted(Path('/sys/class/kfd/kfd/proc').iterdir()):
        if not path.name.isdigit():
            continue
        try:
            result.append(dict(pid=int(path.name), queues=len(list((path / 'queues').iterdir())),
                               name=Path('/proc', path.name, 'comm').read_text().strip()))
        except FileNotFoundError:
            continue
    return result


def listening_ports():
    result = set()
    for name in ('tcp', 'tcp6'):
        for line in Path('/proc/net', name).read_text().splitlines()[1:]:
            parts = line.split()
            if parts[3] == '0A':
                result.add(int(parts[1].rsplit(':', 1)[1], 16))
    return result


def large_processes():
    result = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = dict(line.split(':', 1) for line in (path / 'status').read_text().splitlines())
            row = {key: int(fields.get(key, '0 kB').split()[0]) * 1024
                   for key in ('VmRSS', 'RssAnon', 'RssFile', 'RssShmem', 'VmSwap')}
            if row['VmRSS'] >= (512 << 20) or row['VmSwap'] >= (128 << 20):
                result.append(dict(pid=int(path.name), name=fields['Name'].strip(), **row))
        except (OSError, ValueError):
            continue
    return sorted(result, key=lambda row: -row['VmRSS'])


def mincore_file(path):
    """Count resident file bytes without a read, page fault, madvise, or cache flush."""
    page = os.sysconf('SC_PAGE_SIZE')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
    libc.mmap.restype = ctypes.c_void_p
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
    libc.mincore.restype = ctypes.c_int
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.munmap.restype = ctypes.c_int
    resident = resident_bytes = 0
    with path.open('rb', buffering=0) as f:
        before = os.fstat(f.fileno())
        for offset in range(0, before.st_size, 1 << 30):
            length = min(1 << 30, before.st_size - offset)
            pages = (length + page - 1) // page
            # PROT_NONE=0, MAP_PRIVATE=2: no mapping is ever dereferenced.
            address = libc.mmap(None, length, 0, 2, f.fileno(), offset)
            if address == ctypes.c_void_p(-1).value:
                raise OSError(ctypes.get_errno(), 'mmap(PROT_NONE) failed', str(path))
            try:
                vector = (ctypes.c_ubyte * pages)()
                if libc.mincore(address, length, vector):
                    raise OSError(ctypes.get_errno(), 'mincore failed', str(path))
                count = sum(v & 1 for v in vector)
                resident += count
                resident_bytes += count * page
                if length % page and vector[-1] & 1:
                    resident_bytes -= page - length % page
            finally:
                if libc.munmap(address, length):
                    raise OSError(ctypes.get_errno(), 'munmap failed', str(path))
        after = os.fstat(f.fileno())
    stable = (before.st_size, before.st_mtime_ns, before.st_ino) == (
        after.st_size, after.st_mtime_ns, after.st_ino)
    return dict(path=str(path), bytes=before.st_size, resident_pages=resident,
                total_pages=(before.st_size + page - 1) // page,
                resident_bytes=resident_bytes,
                resident_fraction=resident_bytes / before.st_size if before.st_size else None,
                stable_while_observed=stable,
                owned_by_current_uid=before.st_uid == os.getuid(),
                writable_by_current_uid=os.access(path, os.W_OK))


def protection_inventory(config, service):
    legacy = json.loads((ROOT / 'results/strata_v07/operating_after.json').read_text())['files']
    current = json.loads((ROOT / 'results/v10/protected_before.json').read_text())['files']
    paths = {Path(p): {'legacy_v07'} for p in legacy}
    def add(path, reason):
        path = Path(path)
        if path.is_dir():
            for child in path.rglob('*'):
                if child.is_file() and '__pycache__' not in child.parts:
                    paths.setdefault(child, set()).add(reason)
        else:
            paths.setdefault(path, set()).add(reason)
    for path in (UNIT, CONFIG, DEV / 'strata_v01403_port/serve_context_verified.sh', config['exe']):
        add(path, 'operating_control')
    for path in Path(str(UNIT) + '.d').glob('*'):
        if path.is_file():
            add(path, 'systemd_drop_in')
    if service.get('FragmentPath'):
        add(service['FragmentPath'], 'live_systemd_fragment')
    args = config['args']
    flags = {'--pack', '--native', '--ple-gguf', '--expert-profile', '--mtp',
             '--mtp-draft-vocab', '--native-dense-gguf'}
    for i, value in enumerate(args[:-1]):
        if value in flags:
            add(args[i + 1], 'runtime_' + value[2:])
    if isinstance(config.get('tokenizer'), str):
        add(config['tokenizer'], 'operating_tokenizer')
    add(DEV / 'strata-rocm-feasibility/packs/sc117-iq3s-abliterated-20261002/tokenizer',
        'research_server_tokenizer')
    for path in (SHADOW / 'serve').rglob('*.py'):
        if '__pycache__' not in path.parts:
            add(path, 'research_server_import_source')
    rows = []
    for path, reasons in sorted(paths.items()):
        row = dict(path=str(path), reasons=sorted(reasons),
                   in_v10_applio_baseline=str(path) in current,
                   in_v07_baseline=str(path) in legacy)
        try:
            stat = path.stat()
            row.update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                       resolved_path=str(path.resolve()), exists=path.is_file())
            old = legacy.get(str(path), {})
            row['legacy_stat_equal'] = bool(old and old.get('bytes') == stat.st_size
                                            and old.get('mtime_ns') == stat.st_mtime_ns)
        except OSError as exc:
            row.update(exists=False, error=type(exc).__name__)
        rows.append(row)
    return dict(recommended_files=rows, count=len(rows),
                bytes=sum(row.get('bytes', 0) for row in rows),
                legacy_count=len(legacy),
                in_v10_applio_baseline=sum(row['in_v10_applio_baseline'] for row in rows),
                hash_comparison_performed=False,
                note='Stat equality is not SHA equality. A new before/after SHA inventory is required before any D GPU run.')


def build_and_sources():
    old = json.loads((ROOT / 'results/strata_v07/build_incremental_record.json').read_text())
    origin = json.loads((ROOT / 'results/strata_v07/worktree_provenance.json').read_text())
    files = [EXE, LAB / 'build-v07-gfx1030/strata-pinned-baseline']
    expected = [old['strata']['sha256'], origin['baseline_copy_sha256']]
    builds = []
    for path, prior in zip(files, expected):
        actual = digest(path)
        builds.append(dict(path=str(path), bytes=path.stat().st_size, sha256=actual,
                           recorded_v07_sha256=prior, matches_v07=actual == prior))
    names = ['strata_prefill_v07.py', 'strata_server_v07.py', 'strata_audit_v07.py',
             'strata_summarize_v07.py', 'gpu_guard_v06.py', 'v10_strata_preflight.py']
    source = {name: digest(ROOT / name) for name in names}
    tree = ast.parse((ROOT / 'strata_prefill_v07.py').read_text())
    assignments = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'swap_limit_bytes' for t in node.targets)]
    fixed = len(assignments) == 1 and ast.dump(assignments[0], include_attributes=False) == ast.dump(
        ast.parse('256 << 20', mode='eval').body, include_attributes=False)
    return dict(builds=builds, source_sha256=source, fixed_256_mib_guard_verified=fixed)


def commands(prefix, port):
    common = ['env', '-u', 'STRATA_MMQ_V07_TRACE', 'PYTHONDONTWRITEBYTECODE=1',
              'HSA_DISABLE_COREDUMP_ON_EXCEPTION=1', 'MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0',
              'MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53=0', 'OMP_NUM_THREADS=2', '/usr/bin/python3', '-B', '-u',
              str(ROOT / 'strata_prefill_v07.py'), '--exe', str(EXE),
              '--prime-rows', '390', '--rows', '390', '--repeats', '1', '--max-new', '32', '--port', str(port)]
    sequence = [('a1', 0), ('b1', 1), ('b2', 1), ('a2', 0), ('a3', 0), ('b3', 1),
                ('timing_a', 0), ('timing_b', 1), ('routing_b', 1)]
    result = []
    for label, j in sequence:
        out = ROOT / 'results/v10' / f'{prefix}_{label}'
        argv = common + ['--out', str(out), '--arm', f'v10-warm-{label}-j{j}',
                         '--env', f'STRATA_MMQ_TYPICAL_J={j}']
        if label.startswith('timing'):
            argv += ['--timing']
        if label.startswith('routing'):
            argv += ['--env', 'STRATA_MMQ_V07_TRACE=1']
        result.append(dict(label=label, argv=argv, shell=shlex.join(argv), output_exists=out.exists(),
                           eligible_for_throughput_median=label[0] in 'ab',
                           optional=label.startswith('routing')))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--port', type=int, default=18487)
    p.add_argument('--run-prefix', default='strata_warm_01')
    args = p.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / 'results/v10') or out.exists():
        p.error('Choose a new JSON output within results/v10')
    if not 1024 <= args.port <= 65535 or args.port == 8080:
        p.error('Use a separate nonprivileged research port')
    if not re.fullmatch(r'[A-Za-z0-9_]+', args.run_prefix):
        p.error('The run prefix must contain only letters, digits and underscores')
    before = pressure()
    service = service_state()
    config = json.loads(CONFIG.read_text())
    inventory = protection_inventory(config, service)
    caches, errors = [], []
    for row in inventory['recommended_files']:
        if row.get('exists') and row.get('bytes', 0) >= (1 << 20) and any(
                reason.startswith('runtime_') or reason == 'legacy_v07' for reason in row['reasons']):
            try:
                caches.append(mincore_file(Path(row['path'])))
            except OSError as exc:
                errors.append(dict(path=row['path'], error=type(exc).__name__, errno=exc.errno))
    sources = build_and_sources()
    clients = kfd_clients()
    ports = listening_ports()
    after_service = service_state()
    checks = dict(service_inactive=all(s.get('ActiveState') == 'inactive' and s.get('SubState') == 'dead'
                                      and s.get('MainPID') == '0' for s in (service, after_service)),
                  mem_available_at_least_14_gib=before['mem_available_bytes'] >= (14 << 30),
                  ports_unused=8080 not in ports and args.port not in ports,
                  no_current_kfd_clients=not clients,
                  existing_build_matches=all(row['matches_v07'] for row in sources['builds']),
                  fixed_256_mib_guard_verified=sources['fixed_256_mib_guard_verified'],
                  config_has_no_routing_trace='STRATA_MMQ_V07_TRACE' not in config.get('env', {}),
                  mincore_succeeded=not errors,
                  recommended_inputs_present=all(row['exists'] for row in inventory['recommended_files']))
    record = dict(schema='strata_v10_cpu_preflight', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  phase='CPU preparation only; no new GPU/server execution',
                  service_before=service, service_after=after_service,
                  before=before, after=pressure(), cache_method='PROT_NONE/MAP_PRIVATE/mincore; mapped pages never touched',
                  file_cache=caches, cache_errors=errors, large_processes=large_processes(),
                  kfd_clients=clients, ports={str(port): port in ports for port in (8080, args.port)},
                  sources=sources, protection=inventory, checks=checks,
                  prerequisite_checks_pass=all(checks.values()),
                  blocked_by=[name for name, passed in checks.items() if not passed],
                  gpu_execution=False, server_execution=False, operating_service_actions=[],
                  torch_imported='torch' in sys.modules, rocm_imported=False,
                  guard=dict(swap_out_limit_bytes=SWAP_LIMIT_BYTES, interval_seconds=1, increasing_samples=2,
                             minimum_runtime_mem_available_bytes=2 << 30,
                             scope='System pswpout since research process start; stop only owned research group'),
                  commands=commands(args.run_prefix, args.port),
                  note='Passing prerequisites is not a warm-timing result. Recheck before each arm under root GPU ownership. Do not repeat an unchanged failed resource state.')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(out=str(out), prerequisite_checks_pass=record['prerequisite_checks_pass'],
                          blocked_by=record['blocked_by'], mem_available_gib=before['mem_available_bytes'] / (1 << 30),
                          swap_used_gib=before['swap_used_bytes'] / (1 << 30),
                          cache_files=len(caches), protected_candidates=inventory['count'],
                          gpu_execution=False, server_execution=False, torch_imported=record['torch_imported']),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
