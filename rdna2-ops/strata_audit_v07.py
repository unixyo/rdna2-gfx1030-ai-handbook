#!/usr/bin/env python3
"""Read-only SHA protection for Strata v0.7. Never controls the operating service."""
import argparse, datetime, hashlib, json, os, pathlib, subprocess, time

ROOT = pathlib.Path("~/DevSpace")
UNIT = pathlib.Path("~/.config/systemd/user/strata-verified-256k.service")
CONFIG = ROOT / "strata_v01403_port/prod-promoted/context-verified.private.json"
SCRIPT = ROOT / "strata_v01403_port/serve_context_verified.sh"
PACKS = ROOT / "strata-rocm-feasibility/packs"

def state():
    env = dict(os.environ, XDG_RUNTIME_DIR="/run/user/1000", DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/1000/bus")
    p = subprocess.run(["systemctl", "--user", "show", "strata-verified-256k.service",
                        "--property=ActiveState,SubState,MainPID,FragmentPath,DropInPaths"],
                       env=env, text=True, capture_output=True, check=True)
    return dict(line.split("=", 1) for line in p.stdout.splitlines() if "=" in line)

def items():
    cfg = json.loads(CONFIG.read_text())
    args = cfg["args"]
    actual_pack = pathlib.Path(args[args.index("--pack") + 1])
    files = {UNIT, CONFIG, SCRIPT, pathlib.Path(cfg["exe"])}
    files.update(p for p in pathlib.Path(str(UNIT) + ".d").glob("*") if p.is_file())
    for base in (PACKS, actual_pack):
        files.update(p for p in base.rglob("*") if p.is_file())
    return sorted(files), [str(PACKS), str(actual_pack)]

def sha(p):
    before = p.stat()
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    after = p.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("File changed while hashing: " + str(p))
    return {"sha256": h.hexdigest(), "bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns, "is_symlink": p.is_symlink(),
            "resolved_path": str(p.resolve())}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--compare")
    opt = ap.parse_args()
    os.nice(12)
    start = time.time(); files, roots = items()
    record = {"schema": "strata_operating_sha_v07", "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "service_before": state(), "pack_roots": roots, "files": {}}
    for i, p in enumerate(files):
        record["files"][str(p)] = sha(p)
        print(json.dumps({"i": i + 1, "n": len(files), "path": str(p), "bytes": record["files"][str(p)]["bytes"]}), flush=True)
    record["service_after"] = state(); record["elapsed_s"] = time.time() - start
    if opt.compare:
        old = json.loads(pathlib.Path(opt.compare).read_text())["files"]
        new = record["files"]
        record["comparison"] = {"added": sorted(set(new) - set(old)), "missing": sorted(set(old) - set(new)),
                                "changed": [p for p in old.keys() & new.keys() if old[p]["sha256"] != new[p]["sha256"]],
                                "unchanged_count": sum(old[p]["sha256"] == new[p]["sha256"] for p in old.keys() & new.keys())}
        record["comparison"]["pass"] = not any(record["comparison"][k] for k in ("added", "missing", "changed"))
    out = pathlib.Path(opt.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(out), "files": len(files), "bytes": sum(x["bytes"] for x in record["files"].values()),
                      "elapsed_s": record["elapsed_s"], "comparison": record.get("comparison")}), flush=True)

if __name__ == "__main__": main()
