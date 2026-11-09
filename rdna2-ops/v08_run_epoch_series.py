#!/usr/bin/env python3
"""Run an explicitly declared sequence of fresh research-only Applio epochs.

The coordinator uses only the standard library. Each training child owns the
existing GPU guard. Completed runs are preserved and never silently rerun.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PYTHON = "~/DevSpace/applio-lab/.venv-rocm10/bin/python"


def owned(value):
    p = Path(value)
    p = (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("Research path required: " + str(p))
    return p


def write(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan_path = owned(args.plan)
    plan = json.loads(plan_path.read_text())
    status_path = plan_path.with_name(plan_path.stem + "_status.json")
    if status_path.exists():
        raise FileExistsError("Inspect prior status; refusing automatic rerun: " + str(status_path))
    state = dict(schema="applio_v08_epoch_series", plan=str(plan_path), completed=False, runs=[])
    write(status_path, state)
    for run in plan["runs"]:
        out, db, source = map(owned, (run["out"], run["db"], run["db_seed"]))
        if out.exists() or db.exists():
            raise FileExistsError("Fresh output and DB required: " + run["name"])
        shutil.copytree(source, db)
        env = os.environ.copy()
        for key in list(env):
            if key.startswith("APPLIO_RDNA2_") or key in ("LD_PRELOAD", "TORCH_LOGS", "TORCHINDUCTOR_CACHE_DIR"):
                del env[key]
        env.update(run.get("env", {}))
        env["TRITON_CACHE_DIR"] = str(ROOT / "results/applio_v08/triton_cache")
        command = [PYTHON, str(ROOT / "applio_epoch_v08.py"), "--out", str(out),
                   "--db", str(db), "--epochs", str(run["epochs"]), "--no-reference-audio"]
        command += run.get("args", [])
        log = out.with_name(out.name + ".console.log")
        row = dict(name=run["name"], out=str(out), db=str(db), log=str(log), command=command,
                   environment_overrides=run.get("env", {}), status="running",
                   started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        state["runs"].append(row)
        write(status_path, state)
        print("EPOCH_RUN_START", run["name"], flush=True)
        start = time.monotonic()
        with log.open("x") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            row["pid"] = process.pid
            write(status_path, state)
            code = process.wait()
        row.update(exit_code=code, elapsed_wall_s=time.monotonic()-start,
                   ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        result_file = out / "result.json"
        result = json.loads(result_file.read_text()) if result_file.exists() else {}
        row["valid"] = (code == 0 and result.get("completed") is True and
                        result.get("valid_for_timing") is True and
                        result.get("valid_for_epoch_timing") is True and
                        result.get("protected_unchanged") is True and
                        result.get("operating_protection", {}).get("changed") == [] and
                        len(result.get("epochs", [])) == run["epochs"] and
                        all(e["nonfinite_losses"] == 0 for e in result.get("epochs", [])))
        row["status"] = "completed" if row["valid"] else "failed_requires_inspection"
        if row["valid"]:
            row.update(first_epoch_s=result["epochs"][0]["seconds"],
                       steady_epoch_median_s=result["steady_epoch_median_s"],
                       training_epoch_sum_s=result["training_epoch_sum_s"],
                       scale_reductions=sum(e["scale_reductions"] for e in result["epochs"]),
                       last_quarter_loss_medians=result["last_quarter_loss_medians"])
        write(status_path, state)
        print("EPOCH_RUN_DONE", json.dumps({k:v for k,v in row.items()
              if k not in ("command", "environment_overrides", "last_quarter_loss_medians")}), flush=True)
        if not row["valid"]:
            raise RuntimeError("Run failed; inspect " + str(log))
    state["completed"] = True
    write(status_path, state)
    print("EPOCH_SERIES_DONE", status_path, flush=True)


if __name__ == "__main__":
    main()
