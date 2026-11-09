"""Build only this package's shared HIP library using an existing compiler."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hipcc", default=os.environ.get("HIPCC") or shutil.which("hipcc"))
    args = ap.parse_args()
    if not args.hipcc:
        ap.error("provide --hipcc pointing to the existing ROCm HIP compiler")
    root = Path(__file__).parent
    src = root / "csrc" / "group_norm_silu.hip"
    out = root / "lib" / "librdna2_ops.so"
    out.parent.mkdir(exist_ok=True)
    cmd = [args.hipcc, "-O3", "-std=c++17", "-shared", "-fPIC", "--offload-arch=gfx1030", "-ffp-contract=off", "-Rpass-analysis=kernel-resource-usage", str(src), "-o", str(out)]
    version = subprocess.run([args.hipcc, "--version"], capture_output=True, text=True, check=True).stdout
    t0 = time.perf_counter()
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    (out.parent / "build.log").write_text(run.stdout + run.stderr)
    result = {"command": cmd, "compiler_version": version, "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "seconds": time.perf_counter() - t0, "returncode": run.returncode, "target": "gfx1030"}
    if run.returncode == 0:
        result["library_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    (out.parent / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if run.returncode:
        print(run.stderr, flush=True)
        raise SystemExit(run.returncode)


if __name__ == "__main__":
    main()
