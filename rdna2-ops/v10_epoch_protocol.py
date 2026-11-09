"""Serial actual-entry B/B-prime protocol; completed runs are never repeated."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results/v10"

def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["short", "final"], required=True)
    parser.add_argument("--tag", default="20261108")
    args = parser.parse_args()
    if not args.tag.replace("_", "").isalnum():
        parser.error("Use an alphanumeric run tag")
    gates = json.loads((BASE / "PREDECLARED_GATES.json").read_text())
    out = BASE / (args.phase + "_epoch_protocol")
    out.mkdir(exist_ok=True)
    fusion = BASE / "fusion_full/selection.json"
    native = BASE / "native_selected/dispatch.json"
    for path in (fusion, native):
        if not path.is_file():
            parser.error("Validated selection is missing: " + str(path))
    if args.phase == "final":
        short = json.loads((BASE / "short_epoch_protocol/result.json").read_text())
        if not short.get("meaningful_speedup_pass"):
            parser.error("The short operating epoch gate must pass before 100 epochs")
        if not (BASE / "FINAL_ETA_KO.md").is_file():
            parser.error("Record and announce the final training ETA first")
    epochs = gates["epoch"]["epochs_per_short_run"] if args.phase == "short" else gates["final"]["epochs_each"]
    arms = ["B", "B_prime", "B", "B_prime"] if args.phase == "short" else ["B", "B_prime"]
    result = {"schema": "applio_v10_operating_epoch_protocol", "phase": args.phase,
              "epochs_each": epochs, "arms": arms, "completed": False, "runs": [],
              "selection_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (fusion, native)}}
    output = out / "result.json"
    save(output, result)
    for index, arm in enumerate(arms):
        number = index // 2 + 1
        label = ("Bprime" if arm == "B_prime" else "B") + str(number)
        name = args.phase + "_" + label
        run_out = BASE / name
        command = [sys.executable, "-B", str(ROOT / "v10_run_operating.py"), "--out", str(run_out),
                   "--model-name", "v10_" + name + "_" + args.tag, "--arm", arm, "--epochs", str(epochs)]
        if arm == "B_prime":
            command += ["--fusion-selection", str(fusion), "--native-selection", str(native)]
        result["active_run"] = {"label": label, "out": str(run_out), "command": command}
        save(output, result)
        existing = run_out / "result.json"
        if existing.is_file():
            row = json.loads(existing.read_text())
            if not row.get("completed") or not row.get("valid_for_epoch_timing"):
                raise RuntimeError("Existing incomplete/invalid run needs inspection; it will not be repeated: " + str(existing))
        else:
            with (out / (label + ".launcher.log")).open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                while process.poll() is None:
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        progress_path = run_out / "training_state.json"
                        if progress_path.is_file():
                            progress = json.loads(progress_path.read_text())
                            completed_epochs = progress.get("epochs", [])
                            if completed_epochs:
                                latest = completed_epochs[-1]
                                print("PROGRESS", json.dumps({"label": label, "epoch": latest["epoch"],
                                    "target": epochs, "last_epoch_s": latest["seconds"],
                                    "nonfinite_losses": sum(e["nonfinite_losses"] for e in completed_epochs)}), flush=True)
                if process.returncode:
                    raise subprocess.CalledProcessError(process.returncode, command)
            row = json.loads(existing.read_text())
        if not row.get("completed") or not row.get("valid_for_epoch_timing"):
            raise RuntimeError("Operating training did not complete with valid timing: " + str(existing))
        if row.get("epochs_requested") != epochs or row.get("arm") != arm:
            raise RuntimeError("Existing run does not match this predeclared protocol")
        if arm == "B_prime" and any(row.get("research_source_sha256", {}).get(path) != expected
                                    for path, expected in result["selection_sha256"].items()):
            raise RuntimeError("Existing run used a different validated selection")
        if any(any(step.get(key) is None or not math.isfinite(step[key]) for key in
                   ("loss_d", "loss_g", "mel", "fm", "kl", "adv")) for step in row["steps"]):
            raise RuntimeError("Nonfinite operating loss")
        if arm == "B_prime" and not (sum(row.get("fusion_hits", {}).values()) and sum(row.get("native_hits", {}).values())):
            raise RuntimeError("B-prime must exercise both validated selections")
        result["runs"].append({"label": label, "arm": arm, "result_path": str(existing),
            "result_sha256": hashlib.sha256(existing.read_bytes()).hexdigest(),
            "steady_epoch_median_s": row["steady_epoch_median_s"],
            "steady_epoch_seconds": [e["seconds"] for e in row["epochs"] if e["epoch"] >= 3],
            "sum_epoch_seconds": sum(e["seconds"] for e in row["epochs"]),
            "process_wall_s": row["process_wall_s"], "initial_state_sha256": row["initial_state_sha256"],
            "model_exports": row["model_exports"], "max_queues": row["gpu_audit"]["max_queues"],
            "scale_reductions": sum(e["scale_reductions"] for e in row["epochs"]), "finite_losses": True})
        save(output, result)
        print("COMPLETED", label, row["steady_epoch_median_s"], flush=True)
    initials = [r["initial_state_sha256"] for r in result["runs"]]
    result["initial_states_identical"] = all(v == initials[0] for v in initials)
    if not result["initial_states_identical"]:
        raise RuntimeError("Initial G/D states differ between arms")
    medians = {arm: statistics.median([value for row in result["runs"] if row["arm"] == arm
                                      for value in row["steady_epoch_seconds"]]) for arm in ("B", "B_prime")}
    result["pooled_steady_medians_s"] = medians
    result["steady_speedup"] = medians["B"] / medians["B_prime"]
    result["adjacent_pair_speedups"] = [result["runs"][i]["steady_epoch_median_s"] /
        result["runs"][i+1]["steady_epoch_median_s"] for i in range(0, len(arms), 2)]
    result["meaningful_speedup_pass"] = bool(result["steady_speedup"] >= gates["epoch"]["meaningful_speedup_min"]
        and all(value > 1 for value in result["adjacent_pair_speedups"]))
    result["completed"] = True
    result.pop("active_run", None)
    save(output, result)
    print(json.dumps({key: result[key] for key in ("completed", "pooled_steady_medians_s", "steady_speedup", "meaningful_speedup_pass")}), flush=True)

if __name__ == "__main__":
    main()
