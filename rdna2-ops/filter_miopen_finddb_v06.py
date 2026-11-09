#!/usr/bin/env python3
"""Copy research FindDb, excluding one observed-bad FP16 Conv1d forward solver.

Never edits the source DB or an operating MIOpen DB. The resulting DB still
requires independent numeric checks of every seeded shape before a model run.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--source-sha256",required=True)
    cfg=p.parse_args();root=Path(__file__).resolve().parent
    source=cfg.source.resolve();out=cfg.output_dir.resolve()
    if not source.is_relative_to(root) or not out.is_relative_to(root):
        p.error("Only DBs under the research folder are allowed")
    if not source.name.endswith(".ufdb.txt"):p.error("Expected User FindDb text file")
    if out.exists():p.error("Use a new output directory")
    raw=source.read_bytes();before=hashlib.sha256(raw).hexdigest()
    if before!=cfg.source_sha256:raise RuntimeError("Source DB changed")
    removed=[];lines=[]
    for line in raw.decode().splitlines():
        key,sep,values=line.partition("=")
        entries=values.split(";")
        # PyTorch's Conv1d descriptor is NCHW with H=1 and filter H=1.
        parts=key.split("-")
        conv1d_key=len(parts)>=5 and parts[1]=="1" and parts[3].startswith("1x")
        if sep and conv1d_key and key.endswith("-NCHW-FP16-F"):
            keep=[v for v in entries if not v.startswith("ConvHipImplicitGemmV4R1Fwd:")]
            if len(keep)!=len(entries):
                removed.append({"key":key,"removed":[v for v in entries if v not in keep]})
                if not keep:continue  # Missing key uses ordinary FAST fallback.
                line=key+"="+";".join(keep)
        lines.append(line)
    if not removed:raise RuntimeError("No matching entry; refuse an ineffective candidate")
    out.mkdir(parents=True)
    result=("\n".join(lines)+"\n").encode()
    target=out/source.name;target.write_bytes(result)
    timefile=source.with_name(source.name+".time")
    if timefile.exists():(out/timefile.name).write_bytes(timefile.read_bytes())
    manifest={"source":str(source),"source_sha256":before,"output":str(target),
              "output_sha256":hashlib.sha256(result).hexdigest(),"removed":removed,
              "scope":"Only seeded NCHW FP16 Conv1d forward keys; B/W records unchanged",
              "status":"candidate_requires_numeric_validation",
              "source_unchanged":hashlib.sha256(source.read_bytes()).hexdigest()==before}
    (out/"filter_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":main()
