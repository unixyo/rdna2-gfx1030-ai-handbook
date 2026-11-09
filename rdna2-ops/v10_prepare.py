"""CPU-only current-baseline protection snapshot for the v1.0 experiments."""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/v10"
APP = Path("~/DevSpace/applio-lab")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["before", "after"])
    a = p.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    target = OUT / ("protected_" + a.mode + ".json")
    if target.exists():
        raise SystemExit("Preserving existing snapshot: " + str(target))
    old = json.loads((ROOT / "results/applio_v08/protected_before.json").read_text())
    old_paths = {**old["legacy16"], **old["additional_target_and_source"]}
    if a.mode == "before":
        paths = set(old_paths)
        paths.update(str(APP / x) for x in ["rvc/train/train.py", "rvc/train/lab_hooks.py", "run-applio.sh", "lab/train_bench.sh"])
        paths.update(str(x) for x in (APP / "rvc/train/rdna2_v08").rglob("*") if x.is_file())
        paths.update(str(x) for x in Path("~/.config/miopen").rglob("*") if x.is_file())
        paths.update([str(ROOT / "RDNA2_OPTIMIZATION_GUIDE_KO.md"), str(ROOT.parent / "RDNA2_OP_CATALOG_KO.md")])
    else:
        before = json.loads((OUT / "protected_before.json").read_text())
        paths = set(before["files"])
    files, sizes, errors = {}, {}, {}
    for name in sorted(paths):
        try:
            files[name] = sha(name)
            sizes[name] = Path(name).stat().st_size
        except OSError as exc:
            errors[name] = str(exc)
    report = {"schema": "rdna2_v10_protection", "mode": a.mode,
              "utc": datetime.now(timezone.utc).isoformat(), "files": files,
              "bytes": sum(sizes.values()), "count": len(files), "errors": errors}
    if a.mode == "before":
        report["historical_v08_differences"] = [x for x, digest in old_paths.items() if files.get(x) != digest]
        report["baseline_note"] = "Current 2026-11-08 rollout is the protected baseline. Historical v08 differences are not reverted."
    else:
        report["changed"] = [x for x, digest in before["files"].items() if x in files and files[x] != digest]
        report["missing"] = sorted(set(before["files"]) - set(files))
        report["unchanged"] = not report["changed"] and not report["missing"] and not errors
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, ensure_ascii=False))

if __name__ == "__main__":
    main()
