#!/usr/bin/env python3
"""Research-only Strata server entry, subprocess GPU gate and raw token capture."""
from pathlib import Path
import hashlib, json, os, sys, time

if len(sys.argv) > 1 and sys.argv[1] == "--gate":
    directory = Path(os.environ["STRATA_V07_PID_DIR"])
    pid = os.getpid()
    request = directory / (str(pid) + ".json")
    request.write_text(json.dumps({"pid": pid, "ppid": os.getppid(), "exe": sys.argv[2],
                                   "requested_unix_s": time.time()}) + "\n")
    deadline = time.monotonic() + 120
    while not (directory / (str(pid) + ".allow")).exists():
        if time.monotonic() > deadline:
            raise RuntimeError("Research GPU ownership handshake timed out")
        time.sleep(0.02)
    os.execvpe(sys.argv[2], sys.argv[2:], os.environ)

SHADOW = "~/DevSpace/strata_v01403_port/diag-guard-wide-corpus-20261009/shadow-canary-v2"
sys.path.insert(0, SHADOW)
import serve.server as server

old_popen = server.popen
gate_path = str(Path(__file__).resolve())
def guarded_popen(what, args, **kwargs):
    env = dict(kwargs.get("env") or os.environ)
    env["STRATA_V07_PID_DIR"] = os.environ["STRATA_V07_PID_DIR"]
    kwargs["env"] = env
    return old_popen(what, [sys.executable, gate_path, "--gate", *args], **kwargs)
server.popen = guarded_popen

old_generate = server.StrataEngine.generate
def captured_generate(self, ids, max_new, sampling, cancel, embeddings=None):
    seq = []; start = time.time()
    try:
        for token in old_generate(self, ids, max_new, sampling, cancel, embeddings=embeddings):
            if token is not None: seq.append(int(token))
            yield token
    finally:
        row = {"started_unix_s": start, "ended_unix_s": time.time(),
               "prompt_tokens": len(ids),
               "prompt_ids_sha256": hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest(),
               "max_new": max_new, "sampling": sampling, "token_ids": seq,
               "token_ids_sha256": hashlib.sha256(json.dumps(seq, separators=(",", ":")).encode()).hexdigest(),
               "native": dict(self.last)}
        with open(os.environ["STRATA_V07_TOKEN_LOG"], "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
server.StrataEngine.generate = captured_generate
server.main()
