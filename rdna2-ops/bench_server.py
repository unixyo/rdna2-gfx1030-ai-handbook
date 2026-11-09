#!/usr/bin/env python3
"""Run the existing shell launcher and original API workflow, sequentially."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--project", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--modes", nargs="+", default=["legacy:native", "view:native", "view:auto"])
ap.add_argument("--seeds", nargs="+", type=int, default=[20261105])
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--reference", type=Path)
ap.add_argument("--port", type=int, default=8299)
ap.add_argument("--same-server", action="store_true")
ap.add_argument("--width", type=int)
ap.add_argument("--height", type=int)
args = ap.parse_args()
project, out = args.project.resolve(), args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
base = f"http://127.0.0.1:{args.port}"


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.load(response)


report = {"launcher": str(project / "scripts/start_moss_comfy.sh"),
          "workflow": "sft_fast4_int8_512", "seeds": args.seeds, "same_server": args.same_server,
          "width_override": args.width, "height_override": args.height,
          "acceptance_fixed_before_execution": {"exact_paths": "latent/RGB byte equality", "rgb_max": 2/255, "rgb_rmse": .25/255},
          "experiments": []}
reference = args.reference.resolve() if args.reference else None
if reference is None and args.modes[0].split(":")[1] != "native":
    raise ValueError("The first mode must use native attention when creating a reference")
groups = [args.modes] if args.same_server else [[mode] for mode in args.modes]
for modes in groups:
    layout, attention = modes[0].split(":")
    run_dir = out / ("shared_server" if args.same_server else layout + "_" + attention)
    run_dir.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    env = dict(os.environ, MOSS_PORT=str(args.port), MOSS_RDNA2_INT8_PATCH="1",
               MOSS_RDNA2_INT8_LAYOUT=layout, MOSS_RDNA2_ATTENTION=attention,
               MOSS_RDNA2_MEASURE_DIR=str(run_dir),
               MOSS_RDNA2_INT8_STATS="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    if reference is not None:
        env["MOSS_RDNA2_REFERENCE_DIR"] = str(reference)
    else:
        env.pop("MOSS_RDNA2_REFERENCE_DIR", None)
    experiments = {mode: {"mode": mode, "runs": []} for mode in modes}
    report["experiments"].extend(experiments.values())
    with (run_dir / "server.log").open("w") as log:
        proc = subprocess.Popen(["/usr/bin/bash", str(project / "scripts/start_moss_comfy.sh")],
                                cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(240):
                if proc.poll() is not None:
                    raise RuntimeError(f"Server exited {proc.returncode}; inspect {run_dir / 'server.log'}")
                try:
                    request("/system_stats")
                    break
                except Exception:
                    time.sleep(.5)
            else:
                raise TimeoutError("Owned server did not become ready")
            index = 0
            for seed in args.seeds:
                schedule = [(mode, True, 0) for mode in modes]
                for trial in range(args.rounds):
                    order = modes if trial % 2 == 0 else modes[::-1]
                    schedule.extend((mode, False, trial + 1) for mode in order)
                for mode, warmup, trial in schedule:
                    layout, attention = mode.split(":")
                    experiment = experiments[mode]
                    index += 1
                    control = run_dir / "control.json"
                    control.with_suffix(".tmp").write_text(json.dumps({"layout": layout, "attention": attention}))
                    control.with_suffix(".tmp").replace(control)
                    graph = json.loads((project / "workflows/sft_fast4_int8_512.json").read_text())
                    graph["8"]["inputs"]["seed"] = seed
                    latent_nodes = [n for n in graph.values() if n["class_type"] == "EmptyFlux2LatentImage"]
                    if args.width is not None or args.height is not None:
                        if len(latent_nodes) != 1:
                            raise ValueError("Expected one EmptyFlux2LatentImage node")
                        if args.width is not None:
                            latent_nodes[0]["inputs"]["width"] = args.width
                        if args.height is not None:
                            latent_nodes[0]["inputs"]["height"] = args.height
                    graph["11"]["inputs"]["filename_prefix"] = f"rdna2_production/{layout}_{attention}_{seed}_{trial}"
                    start = time.perf_counter()
                    prompt = request("/prompt", {"prompt": graph, "client_id": "rdna2-production-validation"})
                    prompt_id = prompt["prompt_id"]
                    for _ in range(3000):
                        history = request("/history/" + prompt_id)
                        if prompt_id in history:
                            item = history[prompt_id]
                            if item["status"]["status_str"] != "success":
                                raise RuntimeError(json.dumps(item["status"]))
                            break
                        time.sleep(.1)
                    else:
                        raise TimeoutError("Actual workflow did not finish")
                    row = json.loads((run_dir / f"run{index:03d}.json").read_text())
                    if args.width is not None and row["image"]["shape"][2] != args.width:
                        raise AssertionError("Requested image width did not reach the actual decoder")
                    if args.height is not None and row["image"]["shape"][1] != args.height:
                        raise AssertionError("Requested image height did not reach the actual decoder")
                    row.update(warmup=warmup, request_wall_s=time.perf_counter()-start, prompt_id=prompt_id)
                    experiment["runs"].append(row)
                    (out / "server_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps({"mode": mode, "warmup": row["warmup"], "seed": seed,
                                      "stage_s": sum(row["stage_wall_s"].values()),
                                      "pass": row["accuracy_pass"], "image": row["image"]}), flush=True)
                    if attention in {"native", "auto"} and not row["accuracy_pass"]:
                        raise RuntimeError("Exact path failed byte equality; preserved result for diagnosis")
            for experiment in experiments.values():
                samples = [sum(r["stage_wall_s"].values()) for r in experiment["runs"] if not r["warmup"]]
                experiment["median_stage_s"] = statistics.median(samples)
                experiment["accuracy_pass"] = all(r["accuracy_pass"] for r in experiment["runs"])
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=45)
    if reference is None:
        reference = run_dir / "reference"
    (out / "server_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"completed": True, "output": str(out / "server_comparison.json")}), flush=True)
