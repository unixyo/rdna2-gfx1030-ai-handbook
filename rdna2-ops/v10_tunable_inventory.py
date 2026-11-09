"""CPU-only CSV/Tensile inventory. No torch, HIP or rocBLAS library imports."""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parent
SITE = Path("~/DevSpace/moss-image-rdna2-lab01/.venv-moss/lib/python3.12/site-packages")
LIBRARY = SITE / "_rocm_sdk_libraries/lib/rocblas/library"
QWEN_CSV = ROOT.parent / "rehearsal_qwen_20261106/results/tunableop25/tunableop_results0.csv"
ANIMA_CSV = Path("~/DevSpace/anima-rdna2-lab01/results/tunableop1/tunableop_results0.csv")
PARAM_RE = re.compile(r"([ntc])([ntc])_(\d+)_(\d+)_(\d+)(?:_B_(\d+))?_ld_(\d+)_(\d+)_(\d+)$")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MsgPack:
    """Decode the scalar/list/map subset used by installed Tensile metadata."""
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def take(self, n):
        if self.pos + n > len(self.data):
            raise ValueError("Truncated msgpack")
        ret = self.data[self.pos:self.pos+n]
        self.pos += n
        return ret

    def number(self, fmt):
        return struct.unpack(">" + fmt, self.take(struct.calcsize(fmt)))[0]

    def read(self):
        c = self.take(1)[0]
        if c < 128:
            return c
        if c >= 224:
            return c - 256
        if 160 <= c <= 191:
            return self.take(c & 31).decode()
        if 144 <= c <= 159:
            return [self.read() for _ in range(c & 15)]
        if 128 <= c <= 143:
            return {self.read(): self.read() for _ in range(c & 15)}
        if c == 192:
            return None
        if c in (194, 195):
            return c == 195
        if c in (202, 203):
            return self.number("f" if c == 202 else "d")
        fmt = {204: "B", 205: "H", 206: "I", 207: "Q",
               208: "b", 209: "h", 210: "i", 211: "q"}.get(c)
        if fmt:
            return self.number(fmt)
        if c in (196, 197, 198, 217, 218, 219):
            fmt = {196: "B", 197: "H", 198: "I", 217: "B", 218: "H", 219: "I"}[c]
            data = self.take(self.number(fmt))
            return data.decode() if c >= 217 else data
        if c in (220, 221):
            return [self.read() for _ in range(self.number("H" if c == 220 else "I"))]
        if c in (222, 223):
            return {self.read(): self.read() for _ in range(self.number("H" if c == 222 else "I"))}
        raise ValueError(f"Unsupported msgpack tag {c:#x} at {self.pos - 1}")


def read_csv(path):
    validators, entries = {}, []
    with Path(path).open(newline="") as fp:
        for line, row in enumerate(csv.reader(fp), 1):
            if not row:
                continue
            if row[0] == "Validator":
                validators[row[1]] = row[2]
                continue
            op, signature, solution, *elapsed = row
            m = PARAM_RE.fullmatch(signature)
            if not m:
                raise ValueError(f"Unsupported signature at {path}:{line}: {signature}")
            ta, tb, mm, nn, kk, batch, lda, ldb, ldc = m.groups()
            dtype = op.split("TunableOp_", 1)[1].rsplit("_", 1)[0]
            e = dict(id=len(entries), line=line, op=op, signature=signature,
                     solution=solution, csv_ms=float(elapsed[0]) if elapsed else None,
                     transa=ta, transb=tb, m=int(mm), n=int(nn), k=int(kk),
                     batch=int(batch) if batch else 1, batched=batch is not None,
                     lda=int(lda), ldb=int(ldb), ldc=int(ldc), dtype=dtype)
            e["flops"] = 2 * e["batch"] * e["m"] * e["n"] * e["k"]
            e["reconstruction"] = {
                "known": ["dtype", "m", "n", "k", "transpose", "lda", "ldb", "ldc", "batch_count"],
                "not_in_csv": ["input_values", "alpha", "beta", "batch_strides", "storage_offset", "pointer_alignment"],
                "canonical_probe": "column-major matrices; alpha=1 beta=0; packed batches unless explicit strides",
            }
            entries.append(e)
    return validators, entries


def load_tensile(directory):
    index, sources = {}, []
    for path in sorted(Path(directory).glob("*gfx1030.dat")):
        raw = path.read_bytes()
        reader = MsgPack(raw)
        data = reader.read()
        if reader.pos != len(raw):
            raise ValueError(f"Trailing bytes in {path}")
        solutions = data.get("solutions", [])
        sources.append({"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                        "solutions": len(solutions)})
        for solution in solutions:
            key = solution["index"]
            record = {k: solution.get(k) for k in
                      ("index", "name", "problemType", "sizeMapping", "problemPredicate")}
            record["source"] = str(path)
            if key in index and index[key]["name"] != record["name"]:
                raise ValueError(f"Conflicting Tensile index {key}")
            index[key] = record
    return index, sources


def inventory(csvs, library):
    index, sources = load_tensile(library)
    report = {"status": "cpu_inventory_only_gpu_not_run", "library": str(library),
              "mapping_source": {
                  "repository": "ROCm/rocm-libraries", "commit": "3d8f334b",
                  "reserved": 10, "formula": "tensile_index = -rocblas_solution_id - 11",
                  "files": ["results/v10/tunable_cpu/sources/tensile_host.cpp",
                            "results/v10/tunable_cpu/sources/definitions.hpp"],
                  "warning": "IDs in [-10,-1] are reserved/default space; do not map as Tensile."
              }, "sources": sources, "unique_tensile_solutions": len(index), "csvs": []}
    for path in csvs:
        validators, entries = read_csv(path)
        for e in entries:
            if e["solution"].startswith("Gemm_Rocblas_"):
                sid = int(e["solution"].rsplit("_", 1)[1])
                e["rocblas_solution_id"] = sid
                if sid < -10:
                    ti = -sid - 11
                    e["tensile_index"] = ti
                    e["tensile"] = index.get(ti)
                    e["mapping_status"] = "mapped" if ti in index else "index_missing_from_installed_gfx1030_db"
                else:
                    e["mapping_status"] = "reserved_or_other_backend_requires_runtime_trace"
            else:
                e["mapping_status"] = "Default"
            mapped = e.get("tensile")
            if mapped:
                sm = mapped["sizeMapping"]
                e["split_or_atomic_candidate"] = (sm.get("globalSplitU", 1) > 1 or
                                                  sm.get("streamKAtomic", 0) != 0)
                e["causal_status"] = "metadata_only_not_nondeterminism_proof"
        counts = collections.Counter(e["mapping_status"] for e in entries)
        gsu = collections.Counter(str(e["tensile"]["sizeMapping"].get("globalSplitU"))
                                  for e in entries if e.get("tensile"))
        report["csvs"].append({"path": str(path), "sha256": sha(path),
                               "validators": validators, "entries": entries,
                               "counts": dict(counts), "globalSplitU_counts": dict(gsu)})
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, action="append")
    ap.add_argument("--library", type=Path, default=LIBRARY)
    ap.add_argument("--out", type=Path, default=ROOT / "results/v10/tunable_cpu/inventory.json")
    a = ap.parse_args()
    report = inventory(a.csv or [QWEN_CSV, ANIMA_CSV], a.library)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(a.out), "unique_tensile_solutions": report["unique_tensile_solutions"],
                      "csvs": [{k: x[k] for k in ("path", "counts", "globalSplitU_counts")}
                               for x in report["csvs"]]}, indent=2))


if __name__ == "__main__":
    main()

