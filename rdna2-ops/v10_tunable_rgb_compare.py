"""CPU-only RGB integrity and pixel-error comparison for completed model runs."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import PIL
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def arg(p, key):
    a = p.get("args", [])
    return a[a.index(key) + 1] if key in a else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    recipe = json.loads(a.recipe.read_text())
    result = {"cpu_only": True, "torch_imported": "torch" in sys.modules,
              "pillow_version": PIL.__version__, "script_sha256": sha(Path(__file__)),
              "recipe_sha256": sha(a.recipe), "models": {}, "comparisons": [], "errors": []}
    requests = {}
    for label, relative in recipe["models"].items():
        d = (ROOT / relative).resolve()
        pp = d / "protocol.json"
        p = json.loads(pp.read_text())
        if not (p.get("completed") and p.get("valid_for_timing")):
            result["errors"].append(f"{label}: incomplete or invalid protocol")
        record = {"protocol": str(pp), "protocol_sha256": sha(pp), "model": p.get("model"),
                  "seed": arg(p, "--seed"), "steps": arg(p, "--steps"), "images": []}
        for row in p.get("images", []):
            path = Path(row["path"])
            match = re.match(r"^(?:anima|rehearsal)_(\d+)_(.+?)_\d+_?\.png$", path.name)
            if not match:
                result["errors"].append(f"unrecognized generated filename: {path}")
                continue
            with Image.open(path) as source:
                rgb = source.convert("RGB")
                raw_sha, size = hashlib.sha256(rgb.tobytes()).hexdigest(), list(rgb.size)
                rgb.close()
            file_sha = sha(path)
            good = (file_sha == row.get("file_sha256") and raw_sha == row.get("rgb_sha256")
                    and size == row.get("size"))
            item = {"request_index": int(match[1]), "mode": match[2], "path": str(path),
                    "file_sha256": file_sha, "rgb_sha256": raw_sha, "size": size,
                    "matches_protocol": good}
            record["images"].append(item)
            requests.setdefault((label, int(match[1])), []).append(item)
            if not good:
                result["errors"].append(f"{label}: PNG or RGB metadata mismatch: {path.name}")
        record["all_match_protocol"] = bool(record["images"]) and all(
            x["matches_protocol"] for x in record["images"])
        result["models"][label] = record
    for pair in recipe["pairs"]:
        left_key, right_key = tuple(pair["left"]), tuple(pair["right"])
        left, right = requests[left_key], requests[right_key]
        if len(left) != 1 or len(right) != 1:
            raise ValueError(f"Pair requires one image per request: {pair['label']}")
        left, right = left[0], right[0]
        with Image.open(left["path"]) as im:
            x = im.convert("RGB")
        with Image.open(right["path"]) as im:
            y = im.convert("RGB")
        if x.size != y.size:
            raise ValueError(f"Different image sizes: {pair['label']}")
        diff = ImageChops.difference(x, y)
        hist = diff.histogram()
        count = sum(hist)
        assert count == x.width * x.height * 3
        maximum = max((i % 256 for i, n in enumerate(hist) if n), default=0)
        rmse = math.sqrt(sum((i % 256) ** 2 * n for i, n in enumerate(hist)) / count)
        lm, rm = result["models"][left_key[0]], result["models"][right_key[0]]
        result["comparisons"].append({
            "label": pair["label"], "left": pair["left"], "right": pair["right"],
            "scope": pair.get("scope"), "size": list(x.size), "rgb_channel_samples": count,
            "same_model_seed_steps": all(lm[k] == rm[k] for k in ("model", "seed", "steps")),
            "max_abs_8bit": maximum, "rmse_8bit": rmse, "rmse_normalized": rmse / 255,
            "different_channels": sum(n for i, n in enumerate(hist) if i % 256 != 0),
            "bitwise_equal_rgb": left["rgb_sha256"] == right["rgb_sha256"],
            "both_match_protocol": left["matches_protocol"] and right["matches_protocol"]})
        diff.close()
        x.close()
        y.close()
    result["completed"] = (not result["errors"] and not result["torch_imported"]
                           and all(x["both_match_protocol"] for x in result["comparisons"]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"completed": result["completed"],
                      "images": sum(len(m["images"]) for m in result["models"].values()),
                      "comparisons": result["comparisons"], "errors": result["errors"]}, indent=2))
    return 0 if result["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

