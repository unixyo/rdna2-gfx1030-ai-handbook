#!/usr/bin/env python3
"""CPU-only verification of Radial Attention equation (4), no model/download."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path

def visible(i,j,k,l,s):
    distance=max(abs(i-j),1)
    factor=2**int(math.floor(math.log2(distance)))
    return ((factor<=s and abs(k-l)+1<=s/factor) or
            (abs(i-j)%math.ceil(factor/s)==0 and k==l))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    rows=[]
    for frames in (1,2,4,8):
        for spatial in (8,32):
            count=sum(visible(i,j,k,l,spatial) for i in range(frames)
                      for j in range(frames) for k in range(spatial) for l in range(spatial))
            rows.append({"frames":frames,"tokens_per_frame":spatial,"kept":count,
                         "total":frames*frames*spatial*spatial,
                         "density":count/(frames*frames*spatial*spatial)})
    assert all(r["density"]==1. for r in rows if r["frames"]<=2)
    report={"kind":"CPU equation verification; not a GPU or video-model measurement",
            "source":"https://arxiv.org/html/2506.19852v1#S4.SS2",
            "source_equation":4,"first_frame_sink_added":False,"rows":rows,
            "implication":"Single-frame image and two adjacent frames give an all-visible mask; no sparse compute saved by this unmodified formula",
            "model_scope":"Qwen reference images are not declared video frames; this verifies only the f=1/2 limiting case",
            "time_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(rows))
if __name__=="__main__":main()
