#!/usr/bin/env python3
"""Blind A/B of two final v1.0 Applio models on one user-supplied vocal (Claude, 2026-11-10).

Same native inference path and settings as v10_audio_eval.py (research applio_copy,
VoiceConverter, KLM-SVM embedder, RMVPE, index 0.3, protect 0.33), except pitch=0
for this vocal. One input file instead of the four TTS sentences. The input and the
outputs stay in the research folder (mode 0600) and are not published.

  .venv-rocm10/bin/python v10_vocal_eval.py --input VOCAL.wav --out results/v10/audio_eval_vocal \
      --model B=...pth --model B_prime=...pth
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import contextlib
import datetime
import html
import json
import os
from pathlib import Path
import random
import secrets
import shutil
import sys
import time
import uuid

import v08_audio_eval as native

RESEARCH = Path(__file__).resolve().parent
BASE = RESEARCH / "results/v10"
ARMS = ("B", "B_prime")
LABELS = ("A", "B")
SETTINGS = {**native.SETTINGS, "pitch": 0}
SAFE_ENV = {**native.SAFE_ENV, "HSA_DISABLE_COREDUMP_ON_EXCEPTION": "1",
            "PYTORCH_TUNABLEOP_ENABLED": "0", "PYTORCH_TUNABLEOP_TUNING": "0"}
sha, write_json = native.sha, native.write_json


def render(out, run_id, labels, input_name, seconds):
    players = []
    for label in LABELS:
        data = base64.b64encode((out / "blind" / f"vocal_{label}.wav").read_bytes()).decode()
        players.append(f'<div class="p"><h3>{label}</h3><audio controls preload="none" '
                       f'src="data:audio/wav;base64,{data}"></audio></div>')
    opts = lambda vals: "".join(f'<option value="{v}">{t}</option>' for v, t in vals)
    pref = opts([("", "선택"), ("A", "A"), ("B", "B"), ("same", "같음/구분 안 됨")])
    anom = opts([("", "선택"), ("A", "A"), ("B", "B"), ("none", "없음"), ("undecided", "모르겠음")])
    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>보컬 익명 청취</title>
<style>:root{{--bg:#fafafa;--fg:#1d1d1f;--card:#fff;--line:#d9d9de}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151517;--fg:#ececf0;--card:#1f1f23;--line:#38383f}}}}
body{{background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,sans-serif;margin:0;padding:16px}}
main{{max-width:760px;margin:auto}}section{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:16px 0}}
.ps{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}}audio{{width:100%}}
label{{display:block;margin:10px 0}}select,textarea{{width:100%;font:inherit;padding:6px;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:8px}}
button{{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg)}}</style></head>
<body><main><h1>보컬 익명 청취</h1>
<p>같은 보컬({html.escape(input_name)}, {seconds:.1f}초)을 두 모델로 바꾼 결과다. A/B와 모델의 대응은 무작위로 정했고 이 페이지에는 없다. run_id <code>{run_id}</code></p>
<section><div class="ps">{''.join(players)}</div>
<label>더 자연스럽게 들린 쪽<select id="preference">{pref}</select></label>
<label>이상한 소리(쇳소리·끊김·음정 흔들림 등)가 더 있는 쪽<select id="anomaly">{anom}</select></label>
<label>메모(구간·초 단위로 적으면 좋음)<textarea id="notes" rows="4"></textarea></label></section>
<button id="save">판정 JSON 저장</button></main>
<script>const runId={json.dumps(run_id)};
document.getElementById("save").onclick=()=>{{const j={{schema:"applio_v10_vocal_blind_judgment",run_id:runId,
timestamp:new Date().toISOString(),preference:document.getElementById("preference").value,
anomaly:document.getElementById("anomaly").value,notes:document.getElementById("notes").value}};
const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(j,null,2)],{{type:"application/json"}}));
a.download="applio_v10_vocal_judgment_"+runId+".json";a.click();}};</script></body></html>"""
    (out / "LISTEN_VOCAL_BLIND.html").write_text(page, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", action="append", default=[], metavar="B|B_prime=PATH")
    p.add_argument("--applio-copy", type=Path, default=native.COPY)
    p.add_argument("--db-seed", type=Path, default=BASE / "audio_eval/miopen_db")
    p.add_argument("--seed", type=int, default=997131)
    a = p.parse_args()
    models = dict(item.split("=", 1) for item in a.model)
    if set(models) != set(ARMS):
        p.error("Give exactly B=... and B_prime=...")
    models = {k: Path(v).resolve() for k, v in models.items()}
    out = a.out.resolve()
    if not out.is_relative_to(BASE) or out.exists():
        p.error("Use a new output directory under results/v10")
    native_root = native.inside_research(a.applio_copy)
    out.mkdir(parents=True)
    (out / "input").mkdir(mode=0o700)
    (out / "raw").mkdir(mode=0o700)
    (out / "blind").mkdir(mode=0o700)
    src = out / "input" / a.input.name
    shutil.copy2(a.input, src)
    os.chmod(src, 0o600)
    shutil.copytree(a.db_seed, out / "miopen_db")
    (out / "miopen_cache").mkdir()
    for key in tuple(os.environ):
        if key.startswith(("APPLIO_RDNA2_", "APPLIO_V10_")):
            del os.environ[key]
    os.environ.update(SAFE_ENV)
    os.environ.update(MIOPEN_FIND_MODE="FAST", MIOPEN_USER_DB_PATH=str(out / "miopen_db"),
                      MIOPEN_CUSTOM_CACHE_DIR=str(out / "miopen_cache"), APPLIO_DILATION_FOLD="0",
                      APPLIO_COMPILE="", MPLBACKEND="Agg")
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    order = list(ARMS)
    secrets.SystemRandom().shuffle(order)
    label_to_arm = dict(zip(LABELS, order))
    write_json(out / "blind_key.json", dict(run_id=run_id, label_to_training_arm=label_to_arm), private=True)
    report = dict(schema="applio_v10_vocal_eval", run_id=run_id, completed=False, settings=SETTINGS,
                  input=dict(name=a.input.name, bytes=src.stat().st_size, sha256=sha(src)),
                  models={k: dict(path=str(v), sha256=sha(v)) for k, v in models.items()},
                  source_sha256=sha(__file__), seed=a.seed, rows=[])
    save = lambda: write_json(out / "result.json", report, private=True)
    save()
    from gpu_guard_v06 import GpuAudit
    audit = GpuAudit()
    cwd = Path.cwd()
    try:
        with audit:
            audit.ensure_clear()
            os.chdir(native_root)
            sys.path.insert(0, str(native_root))
            with (out / "inference.log").open("w", encoding="utf-8") as log, \
                    contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                import numpy as np
                import soundfile as sf
                import torch
                from rvc.infer.infer import VoiceConverter
                hooks = {"Conv1d": torch.nn.Conv1d._conv_forward.__module__,
                         "Conv2d": torch.nn.Conv2d._conv_forward.__module__,
                         "ConvTranspose1d": torch.nn.ConvTranspose1d.forward.__module__}
                if any(m != "torch.nn.modules.conv" for m in hooks.values()):
                    raise RuntimeError("Training convolution hooks leaked into native inference")
                torch.backends.cudnn.benchmark = False
                converter = VoiceConverter()
                converter.load_hubert("custom", str(native_root / native.EMBEDDER))
                converter.last_embedder_model = "custom"
                for arm in ARMS:
                    audit.ensure_clear()
                    random.seed(a.seed); np.random.seed(a.seed)
                    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
                    target = out / "raw" / f"{arm}.wav"
                    torch.cuda.synchronize(); t = time.perf_counter()
                    with torch.autocast("cuda", enabled=False):
                        converter.convert_audio(audio_input_path=str(src), audio_output_path=str(target),
                                                model_path=str(models[arm]),
                                                index_path=str(native_root / native.INDEX),
                                                embedder_model_custom=str(native_root / native.EMBEDDER), **SETTINGS)
                    torch.cuda.synchronize()
                    metrics, samples = native.wave_metrics(target, np, sf)
                    os.chmod(target, 0o600)
                    dtypes = dict(Counter(str(q.dtype) for q in converter.net_g.parameters() if q.is_floating_point()))
                    row = dict(arm=arm, seconds=time.perf_counter() - t, net_g_dtypes=dtypes, **metrics)
                    row["finite"] = bool(row.get("finite", True)) and bool(np.isfinite(samples).all())
                    report["rows"].append(row)
                    save()
            report["gpu_audit"] = audit.report()
    finally:
        os.chdir(cwd)
    for label, arm in label_to_arm.items():
        shutil.copy2(out / "raw" / f"{arm}.wav", out / "blind" / f"vocal_{label}.wav")
    rows = report["rows"]
    report["completed"] = len(rows) == 2 and all(r["finite"] for r in rows)
    report["valid_for_listening"] = report["completed"] and report["gpu_audit"].get("valid_for_timing", False)
    save()
    render(out, run_id, label_to_arm, a.input.name, rows[0]["samples_per_channel"] / rows[0]["sample_rate"])
    print(json.dumps(dict(run_id=run_id, completed=report["completed"], valid=report["valid_for_listening"],
                          page=str(out / "LISTEN_VOCAL_BLIND.html"),
                          rows=[{k: r[k] for k in ("seconds", "sample_rate", "samples_per_channel", "finite", "net_g_dtypes")} for r in rows])))


if __name__ == "__main__":
    main()
