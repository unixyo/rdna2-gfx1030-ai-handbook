"""Run actual operating core.py train under a descendant-aware GpuAudit."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import statistics
import subprocess
import time
from gpu_guard_v06 import GpuAudit
from v10_prepare import sha

ROOT = Path(__file__).resolve().parent
APP = Path("~/DevSpace/applio-lab")
PY = APP / ".venv-rocm10/bin/python"

class OwnedTreeAudit(GpuAudit):
    def __init__(self):
        super().__init__()
        self.queue_samples = []
        self.max_queues = 0
        self.max_clients = 0
        self.last_queues = None
    def sample(self):
        todo, seen = [os.getpid()], set()
        while todo:
            pid = todo.pop()
            if pid in seen:
                continue
            seen.add(pid)
            self.allow_pid(pid)
            try:
                todo.extend(int(x) for x in Path(f"/proc/{pid}/task/{pid}/children").read_text().split())
            except OSError:
                pass
        super().sample()
        clients = {}
        try:
            for proc in self.kfd_path.iterdir():
                try:
                    clients[proc.name] = len(list((proc / "queues").iterdir()))
                except OSError:
                    pass
        except OSError:
            pass
        queues = sum(clients.values())
        self.max_queues = max(self.max_queues, queues)
        self.max_clients = max(self.max_clients, len(clients))
        if clients != self.last_queues:
            self.queue_samples.append({"unix_s": time.time(), "clients": clients, "queues": queues})
            self.last_queues = clients
    def report(self):
        return {**super().report(), "max_queues": self.max_queues, "max_clients": self.max_clients,
                "queue_transitions": self.queue_samples, "queue_limit": 7,
                "queues_pass": self.max_queues <= 7}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--arm", choices=["B", "B_prime"], required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--profile-at", type=int, default=0)
    p.add_argument("--fusion-selection", type=Path)
    p.add_argument("--native-selection", type=Path)
    a = p.parse_args()
    if (a.arm == "B") == bool(a.fusion_selection or a.native_selection):
        p.error("B must have no new selections; B_prime requires a validated selection")
    out = a.out.resolve()
    if not out.is_relative_to(ROOT / "results/v10") or out.exists():
        p.error("Use a new output inside results/v10")
    if not a.model_name.startswith("v10_") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for c in a.model_name):
        p.error("Use a fresh v10_ model name")
    model = APP / "logs" / a.model_name
    if model.exists():
        p.error("Model name already exists; no cleanup/resume is performed")
    if json.loads((APP / "assets/config.json").read_text()).get("precision") != "fp16":
        p.error("Operating precision is not fp16; no operating config edit is permitted")
    if not (ROOT / "results/v10/protected_before.json").exists():
        p.error("Current protection baseline required")
    out.mkdir(parents=True)
    audit = OwnedTreeAudit()
    tracked = [APP / x for x in ["core.py", "rvc/train/train.py", "rvc/train/lab_hooks.py", "assets/config.json", "run-applio.sh", "lab/train_bench.sh"]]
    tracked += [x for x in (APP / "rvc/train/rdna2_v08").rglob("*") if x.is_file()]
    before = {str(x): sha(x) for x in tracked}
    result = {"schema": "applio_v10_operating_run", "arm": a.arm, "completed": False,
              "profile_at": a.profile_at, "epochs_requested": a.epochs, "model_dir": str(model),
              "protected_before": before, "valid_for_epoch_timing": not a.profile_at}
    source_files = [ROOT / name for name in ("v10_run_operating.py", "v10_core_hook.py",
        "v10_inject/sitecustomize.py", "gpu_guard_v06.py", "v10_prepare.py")]
    if a.fusion_selection:
        source_files += [ROOT / "v10_fused_resblock.py", a.fusion_selection.resolve()]
    if a.native_selection:
        source_files += [ROOT / "v10_native_conv.py", a.native_selection.resolve()]
    result["research_source_sha256"] = {str(path): sha(path) for path in source_files}
    for path in source_files:
        relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path(path.name)
        snapshot = out / "source" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, snapshot)
    cmd = [str(PY), "core.py", "train", "--model-name", a.model_name,
           "--vocoder", "HiFi-GAN", "--save-every-epoch", "100", "--total-epoch", str(a.epochs),
           "--sample-rate", "40000", "--batch-size", "8", "--gpu", "0", "--pretrained",
           "--custom-pretrained", "--g-pretrained-path", str(APP / "rvc/models/pretraineds/custom/G_KLM_HFG_40k.pth"),
           "--d-pretrained-path", str(APP / "rvc/models/pretraineds/custom/D_KLM_HFG_40k.pth"),
           "--index-algorithm", "Auto"]
    result["command"] = cmd
    proc = None
    started = time.perf_counter()
    try:
        with audit:
            audit.ensure_clear()
            model.mkdir()
            for name in ("config.json", "filelist.txt"):
                shutil.copyfile(APP / "logs/voice_model_A" / name, model / name)
            shutil.copytree(APP / "rvc/train/rdna2_v08/miopen_db_seed", out / "miopen_db")
            env = os.environ.copy()
            settings = {"APPLIO_RDNA2_V08": "1", "APPLIO_WORKER_NO_GPU": "1", "APPLIO_PROFILE_AT": str(a.profile_at),
                        "APPLIO_PROFILE_STEPS": "5", "APPLIO_AMP_LOG": "0", "APPLIO_COMPILE": "",
                        "APPLIO_V10_OUTPUT": str(out), "MIOPEN_USER_DB_PATH": str(out / "miopen_db"),
                        "MIOPEN_FIND_MODE": "FAST", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1": "0",
                        "MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53": "0", "HSA_DISABLE_COREDUMP_ON_EXCEPTION": "1",
                        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                        "PYTHONPATH": str(ROOT / "v10_inject") + os.pathsep + str(ROOT)}
            settings.update(APPLIO_RDNA2_ADAMW="native", APPLIO_RDNA2_ADAMW_G="native",
                            APPLIO_RDNA2_ADAMW_D="single", APPLIO_RDNA2_SLICE_GATHER="1",
                            APPLIO_RDNA2_GRAD_NORM="foreach", APPLIO_RDNA2_D_WN_CACHE="0",
                            APPLIO_RDNA2_SKIP_EMPTY_CACHE="1", APPLIO_RDNA2_DIRECT_CONV="1",
                            PYTORCH_TUNABLEOP_ENABLED="0")
            for key in ("APPLIO_V10_FUSION_SELECTION", "APPLIO_V10_NATIVE_SELECTION", "APPLIO_RDNA2_V10_NATIVE", "APPLIO_DILATION_FOLD", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
                env.pop(key, None)
            if a.fusion_selection:
                settings["APPLIO_V10_FUSION_SELECTION"] = str(a.fusion_selection.resolve())
            if a.native_selection:
                settings["APPLIO_V10_NATIVE_SELECTION"] = str(a.native_selection.resolve())
                settings["APPLIO_RDNA2_V10_NATIVE"] = "1"
            env.update(settings)
            result["explicit_env"] = settings
            (out / "command.json").write_text(json.dumps(result, indent=2))
            with (out / "run.log").open("w") as log:
                proc = subprocess.Popen(cmd, cwd=APP, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                audit.allow_pid(proc.pid)
                while proc.poll() is None:
                    time.sleep(0.5)
                    if audit.foreign or audit.errors:
                        raise RuntimeError("Foreign KFD client/audit error; timing invalid")
                    if time.perf_counter() - started > a.epochs * 90 + 600:
                        raise TimeoutError("Operating training exceeded bounded research runtime")
                result["process_exit_code"] = proc.returncode
            state_path = out / "training_state.json"
            if state_path.exists():
                result.update(json.loads(state_path.read_text()))
            exports = list(model.glob(f"{a.model_name}_{a.epochs}e_*s.pth"))
            result["model_exports"] = [{"path": str(x), "bytes": x.stat().st_size, "sha256": sha(x)} for x in exports]
            result["training_completed"] = bool(proc.returncode == 0 and len(result.get("epochs", [])) == a.epochs
                and len(result.get("steps", [])) == a.epochs * result.get("steps_per_epoch", -1))
            result["export_completed"] = bool(exports)
            result["completed"] = result["training_completed"] and result["export_completed"]
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    finally:
        result["process_wall_s"] = time.perf_counter() - started
        result["gpu_audit"] = audit.report()
        result["protected_after"] = {str(x): sha(x) for x in tracked}
        result["protected_unchanged"] = before == result["protected_after"]
        result["valid_for_timing"] = bool(result.get("completed") and audit.report()["valid_for_timing"] and audit.report()["queues_pass"] and result["protected_unchanged"])
        result["valid_for_epoch_timing"] = result["valid_for_timing"] and not a.profile_at
        steady = [e["seconds"] for e in result.get("epochs", []) if e["epoch"] >= 3]
        result["steady_epoch_median_s"] = statistics.median(steady) if steady and not a.profile_at else None
        result["index_note"] = "Only config/filelist copied per protocol. Post-training index command has no extracted/ input; training completion is checked from epochs and final model, not index status."
        (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        print(json.dumps({k: result.get(k) for k in ("completed", "valid_for_timing", "process_wall_s", "steady_epoch_median_s", "error")}), flush=True)
    if not result["completed"] or not result["valid_for_timing"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
