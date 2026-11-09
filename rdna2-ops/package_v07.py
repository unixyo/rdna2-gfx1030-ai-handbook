#!/usr/bin/env python3
"""Create a bounded v0.7 evidence package without models, captures or service edits."""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../rdna2-ops-v07-20261107.zip")
    ap.add_argument("--transport", default="../.delivery_v07/archive.b64")
    args = ap.parse_args()
    ops = Path(__file__).resolve().parent
    root = ops.parent
    old = json.loads((root / "RDNA2_PACKAGE_V06_MANIFEST.json").read_text())
    previous = {r["path"]: r for r in old["files"]}
    chosen = set(previous)
    chosen.discard("rdna2-ops/SHA256SUMS")
    chosen.update(["RDNA2_PACKAGE_V06_MANIFEST.json", "PRO_TASK_V07_KO.md",
                   "REVIEW_V06_CLAUDE_KO.md"])
    for folder, pattern in [(ops, "*v07*.py"), (ops / "rdna2_ops", "*v07*.py"),
                            (ops / "research_configs", "*v07*.json"),
                            (ops / "results", "V07*.md"),
                            (ops / "results", "V07*.json")]:
        chosen.update(str(p.relative_to(root)) for p in folder.glob(pattern) if p.is_file())
    omitted = []
    for name in ["int4_v07", "sparse_v07", "strata_v07", "v07_operating_files", "v07_package_sync"]:
        for path in (ops / "results" / name).rglob("*"):
            if not path.is_file():
                continue
            rel = str(path.relative_to(root))
            reason = None
            if "__pycache__" in path.parts or path.suffix in {".pt", ".db", ".lock", ".pyc", ".allow"}:
                reason = "large tensor or runtime cache; original remains on research PC"
            elif name == "strata_v07" and (path.name == "config.json" or (path.name == "result.json" and path.parent.name.startswith(("pinned_stage_", "research_stage_")))):
                reason = "operating-derived configuration or its identifying stage envelope; raw original remains on research PC"
            elif name == "v07_package_sync" and path.name in {"release_verification.json", "transfer.b64"}:
                reason = "generated transfer or self-referential release receipt"
            if reason:
                omitted.append({"path": rel, "bytes": path.stat().st_size, "reason": reason})
            else:
                chosen.add(rel)
    escaped = sorted(p for p in chosen if not (root / p).resolve().is_relative_to(root))
    if escaped:
        raise SystemExit(json.dumps({"paths_outside_research_root": escaped}))
    missing = sorted(p for p in chosen if not (root / p).is_file())
    if missing:
        raise SystemExit(json.dumps({"missing_selected_files": missing}, ensure_ascii=False))
    records = [{"path": p, "bytes": (root / p).stat().st_size, "sha256": digest(root / p)}
               for p in sorted(chosen)]
    checksum_path = ops / "SHA256SUMS"
    checksum_path.write_text("".join(r["sha256"] + "  " + os.path.relpath(root / r["path"], ops)
                                    + "\n" for r in records), encoding="utf-8")
    records.append({"path": "rdna2-ops/SHA256SUMS", "bytes": checksum_path.stat().st_size,
                    "sha256": digest(checksum_path)})
    records.sort(key=lambda r: r["path"])
    manifest = {
        "schema": "rdna2_package_manifest_v07",
        "package_version": "0.7.0", "catalog_version": "0.3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pc_research_root": str(root), "entry": "RDNA2_OP_CATALOG_KO.md",
        "file_count_excluding_manifest": len(records),
        "scope_notes": [
            "Only listed files are packaged; no new model downloads or installed environment changes.",
            "V06 manifest is an immutable historical snapshot, not a checksum claim for current edited files.",
            "V07 manifest and rdna2-ops/SHA256SUMS describe current selected bytes. The manifest excludes itself.",
            "Large PT captures, model weights, compiler caches, and operating-derived Strata configs remain on the research PC.",
            "Raw failed candidates and incomplete model verdicts are preserved; inclusion is not adoption.",
            "Blind prompts, public panels, original judge responses, private mapping and unblinded verdict are evidence artifacts."
        ],
        "omitted_new_files": sorted(omitted, key=lambda r: r["path"]),
        "files": records
    }
    manifest_path = root / "RDNA2_PACKAGE_V07_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    members = [r["path"] for r in records] + [manifest_path.name]
    out = (ops / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in sorted(members):
            z.write(root / rel, rel)
    mismatches = []
    with zipfile.ZipFile(out) as z:
        if z.testzip() is not None:
            raise RuntimeError("ZIP CRC failure")
        for r in records:
            if hashlib.sha256(z.read(r["path"])).hexdigest() != r["sha256"]:
                mismatches.append(r["path"])
    if mismatches:
        raise SystemExit(json.dumps({"zip_hash_mismatches": mismatches}))
    encoded = (ops / args.transport).resolve()
    encoded.parent.mkdir(parents=True, exist_ok=True)
    encoded.write_bytes(base64.b64encode(out.read_bytes()))
    receipt = {
        "schema": "rdna2_v07_package_release_verification",
        "archive": str(out), "archive_bytes": out.stat().st_size, "archive_sha256": digest(out),
        "manifest": str(manifest_path), "manifest_sha256": digest(manifest_path),
        "file_count_excluding_manifest": len(records), "zip_member_count": len(members),
        "zip_hash_mismatches": mismatches,
        "transfer_representation": "base64 of exact full archive",
        "transfer_path": str(encoded.relative_to(root)), "transfer_bytes": encoded.stat().st_size,
        "transfer_sha256": digest(encoded),
        "omitted_new_file_count": len(omitted),
        "base_archive_sha256": "3a483fcd1a42b255feb498f94e29838153b2c4af4262cf1c30a8092837737285"
    }
    (ops / "results" / "v07_package_sync" / "release_verification.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))
if __name__ == "__main__":
    main()
