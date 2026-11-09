#!/usr/bin/env python3
"""CPU-only offline build in the dedicated Strata worktree; no executable is run."""
import hashlib, json, os, pathlib, subprocess, time
ROOT=pathlib.Path("~/DevSpace")
LAB=ROOT/"strata-prefill-v07-lab-20261107"
OUT=ROOT/"rdna2-pro-research-20261102/rdna2-ops/results/strata_v07"
BUILD=LAB/"build-v07-gfx1030"
ROCM=ROOT/"rocm10-gfx1030/install"
GGML=ROOT/"strata_v01403_port/build-hip-gfx1030-rocm10-juud/_deps/strata_llamacpp-src"
os.nice(12)
env=dict(os.environ, CARGO_NET_OFFLINE="true", CMAKE_BUILD_PARALLEL_LEVEL="2")
commands=[
 ["/usr/bin/cmake","-S",str(LAB),"-B",str(BUILD),"-G","Ninja",
  "-DCMAKE_BUILD_TYPE=Release","-DSTRATA_ENABLE_HIP=ON","-DSTRATA_ENABLE_CUDA=OFF",
  "-DSTRATA_HIP_GFX906=OFF","-DSTRATA_PREFILL_MMQ=ON","-DSTRATA_NATIVE_EXPERTS=ON",
  "-DSTRATA_MMQ_KQUANTS=OFF","-DSTRATA_ORCA_Q4KS_MMQ=OFF","-DSTRATA_BUILD_TESTS=ON",
  "-DSTRATA_PORTABLE=OFF","-DCMAKE_HIP_ARCHITECTURES=gfx1030",
  "-DCMAKE_C_COMPILER="+str(ROCM/"lib/llvm/bin/clang"),
  "-DCMAKE_CXX_COMPILER="+str(ROCM/"lib/llvm/bin/clang++"),
  "-DCMAKE_HIP_COMPILER="+str(ROCM/"lib/llvm/bin/clang++"),
  "-DCMAKE_PREFIX_PATH="+str(ROCM),"-DSTRATA_GGML_DIR="+str(GGML),
  "-DFETCHCONTENT_FULLY_DISCONNECTED=ON","-DFETCHCONTENT_UPDATES_DISCONNECTED=ON"],
 ["/usr/bin/cmake","--build",str(BUILD),"--target","strata","hip_prefill_mmq_v07","-j","2"]
]
meta={"started_unix_s":time.time(),"nice":os.getpriority(os.PRIO_PROCESS,0),
      "offline":True,"commands":commands,"gpu_execution":False,"steps":[]}
def save(): (OUT/"build_record.json").write_text(json.dumps(meta,indent=2)+"\n")
save()
for index,cmd in enumerate(commands):
 logpath=OUT/("configure_v07.log" if index==0 else "build_v07.log")
 print(json.dumps({"stage":index,"command":cmd,"log":str(logpath)}),flush=True)
 with logpath.open("w") as log:
  p=subprocess.Popen(cmd,cwd=LAB,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (OUT/"build_control.json").write_text(json.dumps({"pid":p.pid,"pgid":p.pid,"stage":index,"started_unix_s":time.time()})+"\n")
  status=p.wait()
 meta["steps"].append({"stage":index,"exit":status,"log":str(logpath)})
 save()
 if status:
  print(logpath.read_text()[-8000:],flush=True)
  raise SystemExit(status)
exe=BUILD/"strata"
meta.update(ended_unix_s=time.time(),exe=str(exe),exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
save()
print(json.dumps({"build_done":True,"exe":str(exe),"sha256":meta["exe_sha256"]}),flush=True)
