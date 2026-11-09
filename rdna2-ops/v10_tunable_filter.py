"""Create a research-only TunableOp filter candidate; does not import torch."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from v10_tunable_inventory import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, default=Path(__file__).parent / "results/v10/tunable_cpu/inventory.json")
    p.add_argument("--csv-index", type=int, required=True)
    p.add_argument("--policy", choices=["explicit", "fp32_split_default", "all_split_default"], default="explicit")
    p.add_argument("--entry", type=int, action="append", default=[])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    info = json.loads(a.inventory.read_text())["csvs"][a.csv_index]
    source = Path(info["path"])
    if sha(source) != info["sha256"]:
        raise ValueError("Source CSV changed since inventory")
    if a.out.resolve() == source.resolve() or a.out.exists():
        raise ValueError("Output must be a new research file")
    selected = set(a.entry)
    for e in info["entries"]:
        split = (e.get("tensile") or {}).get("sizeMapping", {}).get("globalSplitU", 1) > 1
        if split and (a.policy == "all_split_default" or (a.policy == "fp32_split_default" and e["dtype"] == "float")):
            selected.add(e["id"])
    if not selected or not selected <= set(range(len(info["entries"]))):
        raise ValueError("No selected entries or unknown entry")
    changes = []
    keys = {}
    for e in info["entries"]:
        if e["id"] in selected:
            keys[(e["op"], e["signature"])] = e
            changes.append({"id": e["id"], "op": e["op"], "signature": e["signature"],
                            "old_solution": e["solution"], "new_solution": "Default",
                            "policy": a.policy,
                            "validation": "candidate only; performance and determinism not measured"})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with source.open(newline="") as src, a.out.open("w", newline="") as dst:
        writer = csv.writer(dst, lineterminator="\n")
        for row in csv.reader(src):
            if tuple(row[:2]) in keys:
                row[2] = "Default"
            writer.writerow(row)
    report = {"source": str(source), "source_sha256": info["sha256"],
              "output": str(a.out), "output_sha256": sha(a.out), "changes": changes,
              "status": "unvalidated_filter_candidate", "gpu_run": False}
    a.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

