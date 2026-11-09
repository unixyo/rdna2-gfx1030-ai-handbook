#!/usr/bin/env python3
"""Run the v0.7 actual-weight MMQ screen under the shared GPU lease audit."""
import argparse, hashlib, json, os, pathlib, subprocess, sys, time
from gpu_guard_v06 import GpuAudit
LAB=pathlib.Path("~/DevSpace/strata-prefill-v07-lab-20261107")
PACK="<storage>/models/SC117-IQ3_XXS/pack-strata-v0142"
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--cases")
    opt=ap.parse_args(); out=pathlib.Path(opt.out).resolve();out.mkdir(parents=True,exist_ok=False)
    exe=LAB/"build-v07-gfx1030/hip_prefill_mmq_v07"
    args=[str(exe),PACK]+([str(pathlib.Path(opt.cases).resolve())] if opt.cases else [])
    result={"command":args,"exe_sha256":hashlib.sha256(exe.read_bytes()).hexdigest(),
            "started_unix_s":time.time(),"prewarm":"3 launches per hint; 7 alternating rounds, 20 launches each"}
    readfd,writefd=os.pipe()
    gate="import os,sys; fd=int(sys.argv[1]); os.read(fd,1); os.close(fd); os.execv(sys.argv[2],sys.argv[2:])"
    env=dict(os.environ,HIP_VISIBLE_DEVICES="0",LD_LIBRARY_PATH="~/DevSpace/rocm10-gfx1030/install/lib")
    env.pop("STRATA_MMQ_V07_TRACE",None)
    try:
        with GpuAudit() as guard:
            guard.ensure_clear()
            with (out/"operator.jsonl").open("w") as stdout,(out/"stderr.log").open("w") as stderr:
                proc=subprocess.Popen([sys.executable,"-c",gate,str(readfd),*args],
                                      pass_fds=(readfd,),stdout=stdout,stderr=stderr,env=env,cwd=LAB)
                guard.allow_pid(proc.pid)
                os.close(readfd);readfd=-1;os.write(writefd,b"1");os.close(writefd);writefd=-1
                result["pid"]=proc.pid;result["exit"]=proc.wait()
            guard.ensure_clear()
        result["guard"]=guard.report()
    finally:
        if readfd>=0:os.close(readfd)
        if writefd>=0:os.close(writefd)
        result["ended_unix_s"]=time.time()
        (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)
    raise SystemExit(result.get("exit",1))
if __name__=="__main__":main()
