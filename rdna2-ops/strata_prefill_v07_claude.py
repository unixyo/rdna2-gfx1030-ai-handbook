#!/usr/bin/env python3
"""Run an isolated, guarded Strata prefill arm. No production service control."""
import argparse, copy, datetime, hashlib, json, os, pathlib, signal, socket, subprocess
import threading, time, urllib.error, urllib.request
from gpu_guard_v06 import GpuAudit
from strata_audit_v07 import state

ROOT = pathlib.Path("~/DevSpace")
PKG = pathlib.Path(__file__).resolve().parent
LAB = ROOT / "strata-prefill-v07-lab-20261107"
PY = ROOT / "strata-rocm-feasibility/.stage6-venv/bin/python"
OPERATING_CONFIG = ROOT / "strata_v01403_port/prod-promoted/context-verified.private.json"

def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

def service_off():
    s = state()
    if s.get("ActiveState") != "inactive" or s.get("MainPID") != "0":
        raise RuntimeError("Operating Strata is not inactive; model experiment refused: " + str(s))
    return s

def pressure():
    result = {}
    for kind in ("memory", "io"):
        result[kind + "_psi"] = pathlib.Path("/proc/pressure/" + kind).read_text().strip()
    mem = dict((x.split(":")[0], int(x.split()[1]) * 1024) for x in pathlib.Path("/proc/meminfo").read_text().splitlines())
    result["mem_available"] = mem["MemAvailable"]
    result["mem_breakdown_bytes"] = {k: mem.get(k, 0) for k in ("MemFree", "Cached", "AnonPages", "Active(anon)", "Inactive(anon)", "Unevictable", "Mlocked", "PageTables")}
    result["swap_used"] = mem["SwapTotal"] - mem["SwapFree"]
    vm = dict(x.split() for x in pathlib.Path("/proc/vmstat").read_text().splitlines())
    result.update(pswpin=int(vm["pswpin"]), pswpout=int(vm["pswpout"]))
    return result

def get(url, data=None, timeout=2):
    payload = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.load(f)

def prompt(rows, nonce):
    # Same body across compared arms; nonce varies only between repetitions to prevent prefix reuse.
    records = "".join(f"TEST-{i:05d} expert={i*17%512} group={i%4} miss={i%7} data_pages={i*37%999}\n"
                      for i in range(rows))
    return [{"role": "system", "content": f"{nonce}: You are a GPU performance engineer. Answer in English, with detailed technical explanations."},
            {"role": "user", "content": "Analyze the synthetic expert-cache trace. Explain at least 20 distinct optimizations, tradeoffs and cautions in detail. Use numbered sections.\n" + records}]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--exe", default=str(LAB / "build-v07-gfx1030/strata-pinned-baseline"))
    ap.add_argument("--rows", default="390,800,4800"); ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--prime-rows", default="", help="Untimed requests in the same process, before measured fresh requests")
    ap.add_argument("--max-new", type=int, default=32); ap.add_argument("--port", type=int, default=18477)
    ap.add_argument("--timing", action="store_true"); ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--chunk-cap", type=int); ap.add_argument("--arm", default="baseline")
    ap.add_argument("--swap-limit-mib", type=int, default=256, help="Claude copy: abort threshold for growing swap-out since process start")
    opt = ap.parse_args(); out = pathlib.Path(opt.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if not LAB.is_dir(): raise RuntimeError("Dedicated worktree is missing")
    if not pathlib.Path(opt.exe).resolve().is_relative_to(LAB): raise RuntimeError("Only the dedicated research build may run")
    config = json.loads(OPERATING_CONFIG.read_text()); service_before = service_off()
    for port in (8080, opt.port):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError("Port is already in use: " + str(port))
    before = pressure()
    if before["mem_available"] < (14 << 30): raise RuntimeError("Insufficient available RAM for the existing model")
    cfg = copy.deepcopy(config); cfg["exe"] = opt.exe; cfg["cwd"] = str(LAB)
    cfg["port"] = opt.port; cfg["log"] = str(out / "engine.log")
    cfg["sampling"] = dict(cfg.get("sampling", {}), temperature=0.0, top_p=1.0, top_k=20)
    cfg["env"] = dict(cfg.get("env", {}))
    for k in ("STRATA_PREFILL_TIMING", "STRATA_DECODE_TIMING", "STRATA_VERIFY_PROFILE", "STRATA_WINDOW_TRACE", "STRATA_PHASE_IO"):
        cfg["env"].pop(k, None)
    if opt.timing:
        cfg["env"]["STRATA_PREFILL_TIMING"] = "1"
        cfg["env"]["STRATA_PHASE_IO"] = "1"
    else:
        cfg["env"].pop("STRATA_PHASE_IO", None)
    for item in opt.env:
        k, v = item.split("=", 1); cfg["env"][k] = v
    if opt.chunk_cap is not None:
        cfg["args"][cfg["args"].index("--prefill") + 1] = "auto:" + str(opt.chunk_cap)
    cfg_path = out / "config.json"; write(cfg_path, cfg)
    pid_dir = out / "pids"; pid_dir.mkdir()
    token_log = out / "token_ids.jsonl"
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HIP_VISIBLE_DEVICES="0",
               STRATA_CHAT_PACK=str(ROOT / "strata-rocm-feasibility/packs/sc117-iq3s-abliterated-20261002/tokenizer"),
               STRATA_V07_PID_DIR=str(pid_dir), STRATA_V07_TOKEN_LOG=str(token_log))
    for k in ("STRATA_PREFILL_TIMING", "STRATA_DECODE_TIMING", "STRATA_VERIFY_PROFILE", "STRATA_WINDOW_TRACE", "STRATA_PHASE_IO"):
        env.pop(k, None)
    result = {"arm": opt.arm, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "service_before": service_before, "before": before, "timing_enabled": opt.timing,
              "exe": str(pathlib.Path(opt.exe).resolve()),
              "exe_sha256": hashlib.sha256(pathlib.Path(opt.exe).read_bytes()).hexdigest(),
              "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
              "requests": [], "owned_pids": [], "ownership_errors": [], "resources": []}
    server = None; gate_stop = threading.Event(); mon_stop = threading.Event()
    active_request = {}
    swap_page_bytes = os.sysconf("SC_PAGE_SIZE")
    swap_limit_bytes = opt.swap_limit_mib << 20
    guard = GpuAudit()
    def owned_io_snapshot():
        observed = {}
        for owned in result["owned_pids"]:
            procdir = pathlib.Path("/proc") / str(owned["pid"])
            try:
                fields = (procdir / "stat").read_text().rsplit(")", 1)[1].split()
                observed[str(owned["pid"])] = {
                    "exe": owned["exe"],
                    "io": dict((k, int(v)) for k, v in
                               (line.split(": ") for line in (procdir / "io").read_text().splitlines())),
                    "minor_faults": int(fields[7]), "major_faults": int(fields[9])}
            except (OSError, ValueError):
                pass
        return {"unix_s": time.time(), "scope": "owned processes at the HTTP request boundary", "processes": observed}
    def grant_children():
        seen = set()
        while not gate_stop.wait(0.01):
            for path in pid_dir.glob("*.json"):
                if path.name in seen: continue
                try:
                    row = json.loads(path.read_text()); pid = int(row["pid"])
                    stat = (pathlib.Path("/proc") / str(pid) / "stat").read_text()
                    fields = stat[stat.rfind(")") + 2:].split()
                    ppid = int(fields[1]); pgid = int(fields[2])
                    if server is None or pgid != server.pid or ppid != int(row["ppid"]):
                        raise RuntimeError("PID is outside owned research process group")
                    guard.allow_pid(pid)
                    result["owned_pids"].append(row)
                    (pid_dir / (str(pid) + ".allow")).write_text("owned research process\n")
                    seen.add(path.name)
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
                except Exception as exc:
                    result["ownership_errors"].append(repr(exc)); gate_stop.set(); return
    def monitor():
        last_swap_pages = before["pswpout"]
        increasing_after_limit = 0
        while not mon_stop.wait(1):
            row = pressure(); row["unix_s"] = time.time()
            row["swap_out_since_process_start_bytes"] = max(0, row["pswpout"] - before["pswpout"]) * swap_page_bytes
            row["swap_out_since_request_start_bytes"] = (
                max(0, row["pswpout"] - active_request["pswpout"]) * swap_page_bytes
                if "pswpout" in active_request else None)
            if row["swap_out_since_process_start_bytes"] > swap_limit_bytes and row["pswpout"] > last_swap_pages:
                increasing_after_limit += 1
            else:
                increasing_after_limit = 0
            last_swap_pages = row["pswpout"]
            row["owned_process_io"] = {}
            for owned in result["owned_pids"]:
                procdir = pathlib.Path("/proc") / str(owned["pid"])
                try:
                    row["owned_process_io"][str(owned["pid"])] = {
                        "io": dict((k, int(v)) for k, v in
                                   (line.split(": ") for line in (procdir / "io").read_text().splitlines())),
                        "wait_channel": (procdir / "wchan").read_text().strip()}
                    status = dict(line.split(":", 1) for line in (procdir / "status").read_text().splitlines())
                    row["owned_process_io"][str(owned["pid"])]["memory_bytes"] = {
                        k: int(status.get(k, "0 kB").split()[0]) * 1024
                        for k in ("VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem", "VmSwap", "VmPin", "VmLck")}
                except (OSError, ValueError):
                    pass
            result["resources"].append(row)
            if increasing_after_limit >= 2:
                result["resource_abort"] = "resource-contaminated: swap-out kept growing beyond %d MiB since process start" % opt.swap_limit_mib
                result["resource_abort_sample"] = row
                if server is not None and server.poll() is None: os.killpg(server.pid, signal.SIGTERM)
                return
            if len(result["resources"]) % 5 == 0:
                running_service = state()
                row["operating_service"] = running_service
                if running_service.get("ActiveState") != "inactive" or running_service.get("MainPID") != "0":
                    result["resource_abort"] = "Operating service became active; research arm invalidated"
                    if server is not None and server.poll() is None: os.killpg(server.pid, signal.SIGTERM)
                    return
            if row["mem_available"] < (2 << 30):
                result["resource_abort"] = "MemAvailable below 2 GiB"
                if server is not None and server.poll() is None: os.killpg(server.pid, signal.SIGTERM)
                return
    log = (out / "server.log").open("w")
    t0 = time.monotonic()
    guard.__enter__()
    try:
        guard.ensure_clear()
        server = subprocess.Popen([str(PY), "-u", str(PKG / "strata_server_v07.py"), "--engine", "strata",
                                   "--config", str(cfg_path), "--host", "127.0.0.1", "--port", str(opt.port)],
                                  env=env, cwd=LAB, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        guard.allow_pid(server.pid)
        gate_thread = threading.Thread(target=grant_children, daemon=True); gate_thread.start()
        mon_thread = threading.Thread(target=monitor, daemon=True); mon_thread.start()
        deadline = time.monotonic() + 600
        while True:
            if server.poll() is not None: raise RuntimeError("Research server exited before ready")
            if result["ownership_errors"]: raise RuntimeError(str(result["ownership_errors"]))
            try:
                get(f"http://127.0.0.1:{opt.port}/health"); break
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                if time.monotonic() > deadline: raise RuntimeError("Research server readiness timeout")
                time.sleep(0.5)
        result["ready_s"] = time.monotonic() - t0
        print(json.dumps({"ready_s": result["ready_s"], "out": str(out)}), flush=True)
        sequence = [("prime", i, int(x)) for i, x in enumerate(opt.prime_rows.split(",")) if x]
        sequence += [("measured", rep, int(x)) for rep in range(opt.repeats) for x in opt.rows.split(",")]
        for phase, rep, rows in sequence:
                service_off(); guard.ensure_clear()
                nonce = f"v07-prime-{rows}-{rep}" if phase == "prime" else f"v07-{rows}-{rep}"
                messages = prompt(rows, nonce)
                request = {"model": cfg["model_name"], "messages": messages, "max_tokens": opt.max_new,
                           "temperature": 0, "top_p": 1, "top_k": 20, "seed": 20261107}
                label = ("" if phase == "measured" else "prime_") + f"rows{rows}_r{rep}"
                write(out / (label + "_request.json"), request)
                sample = {"label": label, "rows": rows, "repeat": rep, "phase": phase,
                          "prewarm_in_same_process": any(x["phase"] == "prime" for x in result["requests"]),
                          "first_request": not result["requests"],
                          "before": pressure(), "started_unix_s": time.time(),
                          "request_sha256": hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()}
                sample["owned_io_before"] = owned_io_snapshot()
                sample["swap_page_bytes"] = swap_page_bytes
                active_request.update(label=label, pswpout=sample["before"]["pswpout"])
                result["incomplete_request"] = sample
                write(out / "result.json", result)
                ts = time.monotonic()
                response = get(f"http://127.0.0.1:{opt.port}/v1/chat/completions", request, timeout=1200)
                sample.update(api_s=time.monotonic() - ts, after=pressure(), usage=response.get("usage"),
                              owned_io_after=owned_io_snapshot())
                write(out / (label + "_response.json"), response)
                result["requests"].append(sample)
                result.pop("incomplete_request", None)
                write(out / "result.json", result)
                print(json.dumps({"request": label, "api_s": sample["api_s"], "usage": sample["usage"]}), flush=True)
        guard.ensure_clear()
    except Exception as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc); raise
    finally:
        # The research process group may outlive its Python server after an error.
        if server is not None:
            try: os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL); server.wait(timeout=10)
        # The guard covers owned child shutdown as well as request execution.
        for row in result["owned_pids"]:
            pid = int(row["pid"]); deadline = time.monotonic() + 10
            while pathlib.Path("/proc", str(pid)).exists() and time.monotonic() < deadline:
                time.sleep(0.05)
        result["owned_pids_remaining"] = [int(row["pid"]) for row in result["owned_pids"]
                                         if pathlib.Path("/proc", str(row["pid"])).exists()]
        if result["owned_pids_remaining"] and server is not None:
            try: os.killpg(server.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            result["cleanup_required_kill"] = True
        gate_stop.set(); mon_stop.set(); log.close()
        guard.__exit__(None, None, None)
        result["after"] = pressure(); result["service_after"] = state()
        result["guard"] = guard.report(); result["wall_s"] = time.monotonic() - t0
        result["resource_valid_for_timing"] = not bool(result.get("resource_abort"))
        result["post_observation_swap_gate"] = {"limit_bytes": swap_limit_bytes,
            "consecutive_increasing_one_second_samples": 2,
            "introduced_after": "pinned_stage_9k_first_retry", "retroactive": False}
        result["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if token_log.exists():
            result["native_requests"] = [json.loads(x) for x in token_log.read_text().splitlines()]
        write(out / "result.json", result)
        print(json.dumps({"out": str(out), "error": result.get("error"),
                          "guard_valid": result["guard"]["valid_for_timing"], "service": result["service_after"]}), flush=True)

if __name__ == "__main__": main()
