#!/usr/bin/env python3
"""Export physical Conv descriptors from a completed Applio v0.8 module capture.

CPU/stdlib only. Module boundary FP32 inputs under FP16 AMP remain visible in
metadata; benchmark cases collapse equivalent geometry into one FP16 descriptor.
This does not claim that a module hook directly observes the internal GPU dtype.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from v08_solver_select import HERE, normalize_case, case_id, db_key, load_db, owned, dump, sha

CURRENT_DB_NAME = "gfx1030_40.HIP.3_6_0_8d1ae90e.ufdb.txt"


def case_name(c):
    x, w = c["shape"], c["weight"]
    dims = "T" + str(x[2]) if len(x) == 3 else "H" + str(x[2]) + "_W" + str(x[3])
    px = lambda v: "x".join(map(str,v))
    return (f'{c["kind"]}_B{x[0]}_C{x[1]}_{dims}_O{w[1]*c["groups"] if "transpose" in c["kind"] else w[0]}'
            f'_K{px(w[2:])}_S{px(c["stride"])}_D{px(c["dilation"])}_G{c["groups"]}_P{px(c["padding"])}'
            + (f'_OP{px(c["output_padding"])}' if any(c["output_padding"]) else ""))


def is_resblock(c):
    return (c["kind"] == "conv1d" and c["groups"] == 1 and
            c["shape"][1] == c["weight"][0] == c["weight"][1] and
            (c["shape"][1], c["shape"][2]) in ((256,320),(128,3200),(64,6400),(32,12800)) and
            c["weight"][2] in (3,7,11) and c["dilation"][0] in (1,3,5) and
            c["stride"] == [1] and
            c["padding"] == [c["dilation"][0]*(c["weight"][2]-1)//2])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--baseline-db",type=Path,required=True,
                   help="Research copy of current native FindDb directory, possibly with multiple runtimes")
    p.add_argument("--single-runtime-dir",type=Path,required=True)
    a=p.parse_args()
    profile=owned(a.profile); out=owned(a.out); source=owned(a.baseline_db)/CURRENT_DB_NAME
    single=owned(a.single_runtime_dir)
    out.mkdir(parents=True,exist_ok=True)
    names=("conv_cases.json","conv_cases_resblock.json","conv_cases_critical.json","v08_solver_case_manifest.json")
    for name in names:
        if (out/name).exists():
            p.error("Refusing overwrite: "+str(out/name))
    data=json.loads(profile.read_text())
    calls=data["conv_calls"]
    grouped={}; excluded=[]
    kind_map={"Conv1d":"conv1d","Conv2d":"conv2d","ConvTranspose1d":"conv_transpose1d",
              "ConvTranspose2d":"conv_transpose2d"}
    for key, meta in calls.items():
        bare=meta["kind"].removeprefix("Parametrized")
        if meta["input"][0] != 8 or bare not in kind_map:
            excluded.append({"capture_key":key,**meta})
            continue
        c=normalize_case({"kind":kind_map[bare],"shape":meta["input"],"weight":meta["weight"],
                          **{k:meta[k] for k in ("stride","padding","dilation","groups","output_padding")}})
        cid=case_id(c)
        if cid not in grouped:
            c["name"]=case_name(c)
            c.update(origin="Actual voice_model_A KLM49 HFG FP16 AMP B8 training capture",
                     benchmark_dtype="fp16",
                     physical_dtype_basis="FP16 training autocast contract; module hook records logical boundary dtypes",
                     source_capture_keys=[],logical_boundary_variants=[],observed_forward_calls=0)
            grouped[cid]=c
        r=grouped[cid]
        r["source_capture_keys"].append(key)
        r["logical_boundary_variants"].append({"capture_key":key,"module_kind":meta["kind"],
             "input_dtype":meta["input_dtype"],"weight_dtype":meta["weight_dtype"],
             "count":meta["count"]})
        r["observed_forward_calls"]+=meta["count"]
    cases=list(grouped.values())
    # Priority is a stable export ordering only, not a claim based on GPU times.
    cases.sort(key=lambda c:(not is_resblock(c),-c["observed_forward_calls"],c["name"]))
    resblock=[c for c in cases if is_resblock(c)]
    important={(64,6400,7),(64,6400,11),(32,12800,3),(128,3200,11),
               (64,6400,3),(32,12800,11)}
    critical=[c for c in resblock if (c["shape"][1],c["shape"][2],c["weight"][2]) in important]
    db=load_db(source)
    if single.exists():
        target=single/CURRENT_DB_NAME
        if not target.exists() or sha(target)!=sha(source):
            p.error("Existing single-runtime DB differs; choose new directory")
        if len(list(single.glob("*.ufdb.txt")))!=1:
            p.error("Existing directory contains other runtime DBs")
        copied=False
    else:
        single.mkdir(parents=True)
        target=single/CURRENT_DB_NAME
        shutil.copy2(source,target)
        side=source.with_name(source.name+".time")
        if side.exists():shutil.copy2(side,single/side.name)
        copied=True
    common={"schema":"v08_actual_conv_cases","source_profile":str(profile),
            "source_sha256":sha(profile),"effective_cudnn_benchmark":data.get("effective_cudnn_benchmark"),
            "dtype_note":"Logical FP32/FP16 module inputs are retained as variants, geometry collapsed for FP16 autocast benchmarking; kernel dtype must be verified in actual trace.",
            "scope":"B8 captured training descriptors; bias/weight_norm outside MIOpen not included in solver microbench"}
    for name, subset in zip(names[:3],(cases,resblock,critical)):
        dump(out/name,{**common,"cases":subset})
    covered={d:sum(db_key(c,d) in db for c in cases) for d in ("fwd","dx","dw")}
    manifest={**common,"input_capture_records":len(calls),"unique_B8_geometry":len(cases),
              "collapsed_records":sum(len(c["source_capture_keys"]) for c in cases)-len(cases),
              "excluded_count":len(excluded),"excluded":excluded,
              "by_kind":dict(Counter(c["kind"] for c in cases)),
              "resblock_cases":len(resblock),"critical_cases":len(critical),
              "current_DB_key_coverage":covered,
              "single_runtime_db":str(target),"single_runtime_sha256":sha(target),
              "source_db_sha256":sha(source),"copied_new":copied,
              "outputs":{name:{"path":str(out/name),"sha256":sha(out/name)} for name in names[:3]},
              "GPU_executed":False}
    dump(out/names[3],manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ("excluded","outputs")},indent=2))
    print("outputs",json.dumps(manifest["outputs"],indent=2))


if __name__=="__main__":main()
