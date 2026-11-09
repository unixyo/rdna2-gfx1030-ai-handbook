"""Guarded read-only runner for existing Anima/Qwen research harnesses.

--plan is CPU-only. --run activates a private research process, private caches
and CSV copies. This wrapper changes neither model/lab source nor operating CSV.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

from v10_tunable_inventory import ANIMA_CSV, QWEN_CSV, sha
ROOT = Path(__file__).resolve().parent
ANIMA = Path("~/DevSpace/anima-rdna2-lab01")
QWEN_RUNNER = ROOT.parent / "rehearsal_qwen_20261106/scripts/rehearsal_tunableop.py"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["anima", "qwen"], required=True)
    p.add_argument("--csv", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tunable", choices=["on", "off"], default="on")
    p.add_argument("--modes", help="Defaults: Anima base,cand,cand,cand; Qwen base,tbase,tbase,base")
    p.add_argument("--steps", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--port", type=int, default=8477)
    p.add_argument("--trace-csv-hits", action="store_true", help="Exact C++ lookup hits by stage; diagnostic timing")
    p.add_argument("--profile-conditioning", action="store_true", help="Also capture text-encoder GPU kernel traces")
    p.add_argument("--run", action="store_true")
    a = p.parse_args()
    a.out = a.out.resolve()
    if a.model == "anima":
        runner = ANIMA / "scripts/anima_lab.py"
        csv_path = a.csv or ANIMA_CSV
        steps, seed = a.steps or 30, a.seed if a.seed is not None else 20261107
        modes = a.modes or "base,cand,cand,cand"
        args = ["--cand", "manual_vt", "--width", "1024", "--height", "1024"]
        workflow = ANIMA / "workflows/anima_t2i_1024_api.json"
        runtime_main = Path("~/DevSpace/moss-image-rdna2-lab01/runtime/ComfyUI/main.py")
    else:
        runner = QWEN_RUNNER
        csv_path = a.csv or QWEN_CSV
        steps, seed = a.steps or 25, a.seed if a.seed is not None else 20261106
        modes = a.modes or ("base,tbase,tbase,base" if a.tunable == "on" else "base,base,base")
        args = ["--tuning", "0", "--candidate", "none"]
        workflow = Path("~/DevSpace/qwen-image-21-rdna2-lab01/workflows/request-gemini-edit-640x960-api.json")
        runtime_main = Path("~/DevSpace/qwen-image-21-rdna2-lab01/runtime/ComfyUI/main.py")
    csv_path = csv_path.resolve()
    args += ["--out", str(a.out), "--modes", modes, "--steps", str(steps),
             "--seed", str(seed), "--port", str(a.port)]
    record = {"status": "cpu_plan", "model": a.model, "runner": str(runner), "runner_sha256": sha(runner),
              "workflow": str(workflow), "workflow_sha256": sha(workflow),
              "python": sys.executable, "csv": str(csv_path.resolve()), "csv_sha256": sha(csv_path),
              "args": args, "tunable": a.tunable,
              "runtime": "~/DevSpace/moss-image-rdna2-lab01/.venv-moss/bin/python",
              "completed": False, "valid_for_timing": False}
    record["runtime_main"] = str(runtime_main)
    diagnostic = a.trace_csv_hits or a.profile_conditioning
    record["timing_scope"] = "instrumented_diagnostic" if diagnostic else "runner_denoise_timer_condition_hashes_outside"
    record["conditioning_capture"] = {"requested": True, "trace_csv_hits": diagnostic,
                                        "profile_conditioning": a.profile_conditioning}
    if not a.run:
        print(json.dumps(record, indent=2))
        return 0
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "tunableop_results0.csv").write_bytes(csv_path.read_bytes())
    private_env = {
        "HF_HUB_OFFLINE": "1", "PYTHONNOUSERSITE": "1",
        "HSA_DISABLE_COREDUMP_ON_EXCEPTION": "1",
        "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1": "0",
        "MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53": "0",
        "PYTORCH_TUNABLEOP_TUNING": "0",
        "PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED": "0",
        "PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE": "0",
        "PYTORCH_TUNABLEOP_FILENAME": str(a.out / "tunableop_results.csv"),
        "PYTORCH_TUNABLEOP_RECORD_UNTUNED": "1",
        "PYTORCH_TUNABLEOP_UNTUNED_FILENAME": str(a.out / "missing.csv"),
    }
    for key, relative in (("XDG_CACHE_HOME", "cache/xdg"), ("TRITON_CACHE_DIR", "cache/triton"),
                          ("TORCH_EXTENSIONS_DIR", "cache/extensions"),
                          ("MIOPEN_USER_DB_PATH", "cache/miopen_db"),
                          ("MIOPEN_CUSTOM_CACHE_DIR", "cache/miopen_kernels")):
        dest = a.out / relative
        dest.mkdir(parents=True)
        private_env[key] = str(dest)
    # Qwen has per-mode API toggles; an ENABLED env override would break its off control.
    os.environ.pop("PYTORCH_TUNABLEOP_ENABLED", None)
    os.environ.pop("PYTORCH_TUNABLEOP_VERBOSE", None)
    os.environ.pop("PYTORCH_TUNABLEOP_VERBOSE_FILENAME", None)
    if diagnostic:
        private_env["PYTORCH_TUNABLEOP_VERBOSE"] = "3"
        private_env["PYTORCH_TUNABLEOP_VERBOSE_FILENAME"] = str(a.out / "tunable_verbose.log")
    if a.model == "anima":
        private_env["PYTORCH_TUNABLEOP_ENABLED"] = "1" if a.tunable == "on" else "0"
    os.environ.update(private_env)
    sys.dont_write_bytecode = True
    record["env"] = private_env
    # RecordUntuned=1 skips automatic CSV loading in this torch build. The imported
    # harness must load the private CSV explicitly before its first GEMM.
    from gpu_guard_v06 import GpuAudit
    audit = GpuAudit()
    rc = 1
    capture = None
    previous_profile = sys.getprofile()
    original_run_path = runpy.run_path
    def isolated_run_path(path_name, *path_args, **path_kwargs):
        if Path(path_name).resolve() == runtime_main.resolve():
            for flag, subdir in (("--user-directory", "user"), ("--temp-directory", "temp")):
                dest = a.out / subdir
                dest.mkdir(exist_ok=True)
                if flag in sys.argv:
                    index = sys.argv.index(flag)
                    sys.argv[index + 1] = str(dest)
                else:
                    sys.argv.extend([flag, str(dest)])
            record["runtime_argv"] = list(sys.argv)
            record["runtime_directories_isolated"] = True
        return original_run_path(path_name, *path_args, **path_kwargs)
    def after_instrument(frame, event, arg):
        nonlocal capture
        if event == "return" and frame.f_code.co_name == "instrument" and Path(frame.f_code.co_filename).resolve() == runner.resolve():
            sys.setprofile(previous_profile)
            from v10_tunable_capture import install
            capture = install(frame.f_locals["nodes"], frame.f_globals["state"], a.out,
                              trace_csv_hits=diagnostic, profile_conditioning=a.profile_conditioning)
    began = time.perf_counter()
    try:
        with audit:
            audit.ensure_clear()
            import torch
            import torch.cuda.tunable as tunable
            tunable.tuning_enable(False)
            loaded = tunable.read_file(str(a.out / "tunableop_results0.csv"))
            if not loaded:
                raise RuntimeError("Private CSV validators did not load")
            record["runtime_torch"] = torch.__version__
            record["loaded_csv_rows"] = len(tunable.get_results())
            record["status"] = "running"
            (a.out / "protocol.json").write_text(json.dumps(record, indent=2) + "\n")
            sys.argv = [str(runner), *args]
            sys.setprofile(after_instrument)
            runpy.run_path = isolated_run_path
            try:
                original_run_path(str(runner), run_name="__main__")
                rc = 0
            except SystemExit as exc:
                rc = int(exc.code or 0)
    except Exception:
        record["error"] = traceback.format_exc()
    finally:
        sys.setprofile(previous_profile)
        runpy.run_path = original_run_path
        record["exit_code"] = rc
        record["elapsed_s"] = time.perf_counter() - began
        record["gpu_audit"] = audit.report()
        runs_path = a.out / "runs.json"
        record["runs"] = json.loads(runs_path.read_text()) if runs_path.exists() else []
        record["conditioning_capture"].update({"installed": bool(capture),
                                                "path": str(a.out / "conditioning_capture.json"),
                                                "clip_calls": len(capture["conditioning_calls"]) if capture else 0,
                                                "sampler_calls": len(capture["sampler_calls"]) if capture else 0})
        record["completed"] = (rc == 0 and bool(record["runs"]) and all(r.get("success") for r in record["runs"])
                               and bool(capture) and bool(capture["sampler_calls"]))
        record["valid_for_timing"] = record["completed"] and audit.report()["valid_for_timing"]
        record["valid_for_speed_comparison"] = record["valid_for_timing"] and not diagnostic
        record["status"] = "completed" if record["completed"] else "incomplete"
        images = []
        from PIL import Image
        for path in sorted((a.out / "images").glob("*.png")):
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                images.append({"path": str(path), "size": list(rgb.size),
                               "file_sha256": sha(path), "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})
        record["images"] = images
        record["input_csv_unchanged"] = sha(csv_path) == record["csv_sha256"]
        record["runner_sha256_after"] = sha(runner)
        record["runner_unchanged"] = record["runner_sha256_after"] == record["runner_sha256"]
        record["workflow_unchanged"] = sha(workflow) == record["workflow_sha256"]
        if (not record["input_csv_unchanged"] or not record["runner_unchanged"] or not record["workflow_unchanged"]
                or not record.get("runtime_directories_isolated")):
            record["completed"] = False
            record["valid_for_timing"] = False
            record["valid_for_speed_comparison"] = False
            record["status"] = "source_changed_or_runtime_directory_isolation_missing"
        (a.out / "protocol.json").write_text(json.dumps(record, indent=2) + "\n")
    return 0 if record["completed"] and record["valid_for_timing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

