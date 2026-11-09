#!/usr/bin/env python3
"""Summarize saved Strata v0.7 runs; CPU-only, never launches a model."""
import argparse, hashlib, json, pathlib, re, statistics
PHASES=("embed+steps","hc read","gdn","qsa proj","qsa indexer","qsa select","qsa attn",
        "router+shared","host grouping","gather","wait copy","dequant","gemm gate/up",
        "gemm down","combine","ple","kv stage","gdn conv+gates","gdn recurrence","gdn out proj")
PHASE_RE=re.compile("("+"|".join(re.escape(x) for x in sorted(PHASES,key=len,reverse=True))+r") ([0-9.]+) \(([0-9.]+)%\)")
HEADER=re.compile(r"strata prefill timing: (\d+) tokens, GPU timeline ([0-9.]+) ms, wall ([0-9.]+) ms, host staging ([0-9.]+) ms:")
SERVE=re.compile(r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in ([0-9.]+) ms \(([0-9.]+) tok/s\)")
COPY=re.compile(r"strata prefill copy v07: chunk=(\d+) T=(\d+) experts=(\d+) bytes=(\d+) ring=(\d+)")
IO=re.compile(r"OS read ([0-9.]+) MB from storage \((\d+) major faults\) for ([0-9.]+) MB")
PHASE_IO=re.compile(r"phase I/O \(process-wide\): prefill ([0-9.]+) MB read, (\d+) major faults; decode ([0-9.]+) MB read, (\d+) major faults")
def parse_log(text):
    reqs=[]; stages=[]; copies=[]; phase_io={}
    for line in text.splitlines():
        if m:=HEADER.search(line):
            stages.append({"tokens":int(m[1]),"timeline_ms":float(m[2]),"wall_ms":float(m[3]),
                           "host_staging_cumulative_ms":float(m[4]),
                           "phases_ms":{p:float(v) for p,v,_ in PHASE_RE.findall(line)}})
        if m:=COPY.search(line):
            copies.append(dict(zip(("chunk","tokens","experts","bytes","ring"),map(int,m.groups()))))
        if m:=PHASE_IO.search(line):
            phase_io={"prefill_read_MB":float(m[1]),"prefill_major_faults":int(m[2]),
                      "decode_read_MB":float(m[3]),"decode_major_faults":int(m[4]),
                      "scope":"request start to prefill end, then prefill end to decode end; process-wide counters"}
        if m:=SERVE.search(line):
            reqs.append({"prompt_tokens":int(m[1]),"reused":int(m[2]),"read":int(m[3]),
                         "prompt_ms":float(m[4]),"tok_s":float(m[5]),"stages":stages,"copies":copies,"phase_io":phase_io})
            stages=[];copies=[];phase_io={}
        if (m:=IO.search(line)) and reqs:
            reqs[-1]["legacy_storage_read_MB"]=float(m[1]);reqs[-1]["legacy_major_faults"]=int(m[2]);reqs[-1]["legacy_expert_read_MB"]=float(m[3])
            reqs[-1]["legacy_io_scope"]="engine process start to first request completion" if len(reqs)==1 else "previous request completion to this request completion"
    return reqs
def main():
    ap=argparse.ArgumentParser();ap.add_argument("runs",nargs="+");ap.add_argument("--out",required=True)
    o=ap.parse_args();summary={"note":"Phase intervals on one stream are disjoint. They include idle gaps. Do not add host staging, GPU timeline and wall; host staging is cumulative. First/cold and primed fresh remain separate.","runs":[]}
    for run in o.runs:
        p=pathlib.Path(run);r=json.loads((p/"result.json").read_text())
        entry={k:r.get(k) for k in ("arm","exe_sha256","timing_enabled","ready_s","error","service_before","service_after",
                    "resource_abort","resource_valid_for_timing","post_observation_swap_gate")}
        entry.update(path=str(p.resolve()),result_sha256=hashlib.sha256((p/"result.json").read_bytes()).hexdigest(),
                     guard_valid=r.get("guard",{}).get("valid_for_timing"),owned_pids_remaining=r.get("owned_pids_remaining"))
        entries=parse_log((p/"engine.log").read_text()) if (p/"engine.log").exists() else []
        requests=r.get("requests",[]);native=r.get("native_requests",[])
        for i,req in enumerate(entries):
            if i<len(requests):
                req.update({k:requests[i].get(k) for k in ("label","phase","first_request","prewarm_in_same_process","api_s","request_sha256")})
                req["swap_in_pages_delta"]=requests[i]["after"]["pswpin"]-requests[i]["before"]["pswpin"]
                req["swap_out_pages_delta"]=requests[i]["after"]["pswpout"]-requests[i]["before"]["pswpout"]
                req["swap_out_bytes_delta"]=req["swap_out_pages_delta"]*requests[i].get("swap_page_bytes",4096)
                ib=requests[i].get("owned_io_before",{}).get("processes",{})
                ia=requests[i].get("owned_io_after",{}).get("processes",{})
                req["http_boundary_native_io"]=[]
                for pid,b in ib.items():
                    if pid in ia and "strata-prefill-v07-lab" in b.get("exe",""):
                        a=ia[pid]
                        req["http_boundary_native_io"].append({"pid":int(pid),
                            "read_bytes":a["io"]["read_bytes"]-b["io"]["read_bytes"],
                            "major_faults":a["major_faults"]-b["major_faults"],
                            "scope":"HTTP request boundary; prefill plus generated tokens, excluding engine startup"})
            if i<len(native):
                req["generated_ids"]=native[i].get("token_ids");req["prompt_digest"]=native[i].get("prompt_ids_sha256")
            total=sum(s["timeline_ms"] for s in req["stages"])
            phases={name:sum(s["phases_ms"].get(name,0) for s in req["stages"]) for name in PHASES}
            req["phase_total_ms"]=total
            req["phase_rank"]=[{"phase":n,"ms":v,"fraction":v/total if total else 0} for n,v in sorted(phases.items(),key=lambda q:-q[1]) if v]
            req["issued_h2d_bytes"]=sum(c["bytes"] for c in req["copies"]) if req["copies"] else None
        entry["requests"]=entries;summary["runs"].append(entry)
    # Only pair identical request JSON and prompt-token digests. Model equality is separate from timing.
    groups={}
    for run in summary["runs"]:
        for req in run["requests"]:
            key=req.get("request_sha256")
            if key:groups.setdefault(key,[]).append((run["arm"],req))
    summary["same_prompt_comparisons"]=[]
    for rows in groups.values():
        if len(rows)<2:continue
        a,ar=rows[0]
        for b,br in rows[1:]:
            summary["same_prompt_comparisons"].append({"a":a,"b":b,"tokens":ar["prompt_tokens"],
                "prompt_digest_equal":ar.get("prompt_digest")==br.get("prompt_digest"),
                "generated_ids_equal":bool(ar.get("generated_ids")) and ar.get("generated_ids")==br.get("generated_ids"),
                "fresh_both":ar["reused"]==br["reused"]==0,
                "a_over_b_prompt_ms":ar["prompt_ms"]/br["prompt_ms"]})
    out=pathlib.Path(o.out);out.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    for run in summary["runs"]:
        print(json.dumps({"arm":run["arm"],"guard_valid":run["guard_valid"],
              "requests":[{k:q.get(k) for k in ("label","phase","prompt_tokens","reused","prompt_ms","tok_s","phase_io","legacy_storage_read_MB","legacy_io_scope","issued_h2d_bytes")} for q in run["requests"]]}))
if __name__=="__main__":main()
