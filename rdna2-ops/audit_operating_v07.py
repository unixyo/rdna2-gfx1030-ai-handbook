"""v0.7 protected-file digest audit. Reads operating files; writes research JSON only."""
from __future__ import annotations
import argparse, hashlib, json, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEV = HERE.parent.parent
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024), b""): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=["before","after"],required=True)
    a=p.parse_args()
    old=json.loads((HERE/"results/v06_operating_files_audit.json").read_text())
    previous={str(DEV/x["path"]):x["after_sha256"] for x in old["files"]}
    applio=json.loads((HERE/"results/conv_catalog_v06/epoch_filtered_transpose/protected_after.json").read_text())
    previous.update(applio)
    rows=[]
    for name, prev in sorted(previous.items()):
        f=Path(name)
        rows.append({"path":name,"sha256":sha(f),"bytes":f.stat().st_size,"v06_after_sha256":prev})
    result={"schema":"rdna2-v07-operating-audit-1","phase":a.phase,
            "checked_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "scope":"15 MOSS/Qwen/Anima +16 Applio operating paths; Strata audited separately",
            "count":len(rows),"files":rows,
            "preexisting_changes_vs_v06":[r["path"] for r in rows if r["sha256"]!=r["v06_after_sha256"]]}
    dest=HERE/"results/v07_operating_files"
    dest.mkdir(parents=True,exist_ok=True)
    if a.phase=="after":
        bef=json.loads((dest/"before.json").read_text())
        bh={r["path"]:r["sha256"] for r in bef["files"]}
        result["unchanged_since_v07_before"]=all(bh.get(r["path"])==r["sha256"] for r in rows)
        result["changed_since_v07_before"]=[r["path"] for r in rows if bh.get(r["path"])!=r["sha256"]]
    out=dest/(a.phase+".json")
    if a.phase=="before" and out.exists(): raise RuntimeError("Before audit already exists; do not replace")
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="files"},ensure_ascii=False))
if __name__=="__main__":main()
