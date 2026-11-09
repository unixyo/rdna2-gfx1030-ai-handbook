#!/usr/bin/env python3
"""Native Applio listening evaluation for v0.8; research files only.

CPU preparation uses the Python standard library and never imports torch:
  /usr/bin/python3 v08_audio_eval.py --prepare-assets
  /usr/bin/python3 v08_audio_eval.py --preflight

Run --execute only with the coordinator's GPU lease and an unused output folder.
--model must name A, A_prime, and B final inference checkpoints. Native inference
is FP32, irrespective of training AMP. Training candidate patches are not applied.
The anonymous label key stays in blind_key.json, separate from the public page.
"""
from __future__ import annotations
import argparse
import base64
import contextlib
import datetime
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import secrets
import shutil
import sys
import time
import traceback
import uuid
import wave

RESEARCH = Path(__file__).resolve().parent
BASE = RESEARCH / "results/applio_v08"
SOURCE = Path("~/DevSpace/applio-lab")
COPY = BASE / "applio_copy"
EMBEDDER = "rvc/models/embedders/embedders_custom/KLM-SVM"
INDEX = "logs/voice_model_A/voice_model_A.index"
SENTENCES = "lab/tts_carrier/out/sentences.json"
ASSETS = {
    EMBEDDER + "/pytorch_model.bin": 378346807,
    EMBEDDER + "/config.json": None,
    "rvc/models/predictors/rmvpe.pt": 181184272,
    INDEX: 66515819,
    **{f"lab/tts_carrier/out3/scene/s{i}.wav": None for i in range(4)},
    SENTENCES: None,
}
NATIVE_FILES = (
    "assets/config.json", "rvc/infer/infer.py", "rvc/infer/pipeline.py",
    "rvc/lib/utils.py", "rvc/lib/predictors/f0.py", "rvc/configs/config.py",
    "rvc/lib/algorithm/synthesizers.py", "rvc/lib/zluda.py",
    "rvc/configs/40000.json",
)
ARMS = ("A", "A_prime", "B")
LABELS = ("A", "B", "C")
SETTINGS = dict(
    pitch=2, f0_method="rmvpe", index_rate=0.3, volume_envelope=0.3,
    protect=0.33, hop_length=128, split_audio=False, f0_autotune=False,
    f0_autotune_strength=1.0, embedder_model="custom", clean_audio=False,
    clean_strength=0.5, export_format="WAV", post_process=False, resample_sr=0,
    sid=0, proposed_pitch=False, proposed_pitch_threshold=155.0,
)
SAFE_ENV = {
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1": "0",
    "MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53": "0",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
}
os.environ.update(SAFE_ENV)
sys.dont_write_bytecode = True


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj, private=False):
    path = Path(path)
    temp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def inside_research(path):
    path = Path(path).resolve()
    if not path.is_relative_to(RESEARCH):
        raise ValueError("Research output path required: " + str(path))
    return path


def checked_paths(source, copy):
    source, copy = Path(source).resolve(), inside_research(copy)
    if source == copy or copy.is_relative_to(source):
        raise ValueError("Operating source and research copy must be distinct")
    return source, copy


def same_copy(source, target):
    expected = sha(source)
    if target.exists():
        if not target.is_file() or target.is_symlink() or sha(target) != expected:
            raise RuntimeError("Existing research file differs; refusing overwrite: " + str(target))
        return expected, False
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.resolve().is_relative_to(RESEARCH):
        raise RuntimeError("Copy destination resolves outside research")
    temp = target.with_name("." + target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with source.open("rb") as src, temp.open("xb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        if sha(temp) != expected or sha(source) != expected:
            raise RuntimeError("Source changed or copy checksum failed: " + str(source))
        # No competing writer is expected; refuse rather than overwrite if one appears.
        if target.exists():
            raise RuntimeError("Copy destination appeared concurrently: " + str(target))
        os.link(temp, target)
    finally:
        if temp.exists():
            temp.unlink()
    return expected, True


def prepare_assets(source, copy):
    source, copy = checked_paths(source, copy)
    manifest_path = BASE / "copy_manifest.json"
    before = manifest_path.read_bytes()
    manifest = json.loads(before)
    existing = {row["copy"]: row for row in manifest}
    rows = []
    for relative, size in ASSETS.items():
        src = source / relative
        if not src.is_file():
            raise FileNotFoundError("Existing local asset required; no download: " + str(src))
        if size is not None and src.stat().st_size != size:
            raise RuntimeError("Unexpected asset size: " + str(src))
    for relative in ASSETS:
        src, dst = source / relative, copy / relative
        digest, added = same_copy(src, dst)
        row = dict(source=str(src), copy=str(dst), bytes=src.stat().st_size, sha256=digest)
        if str(dst) in existing:
            if existing[str(dst)] != row:
                raise RuntimeError("Existing copy manifest entry differs: " + str(dst))
        else:
            manifest.append(row)
        rows.append({**row, "new_copy": added})
    if manifest_path.read_bytes() != before:
        raise RuntimeError("Shared copy manifest changed concurrently; assets preserved, rerun preparation")
    write_json(manifest_path, manifest)
    report = dict(schema="applio_v08_audio_assets", assets=rows, downloads=0,
                  source_read_only=True, torch_imported="torch" in sys.modules,
                  manifest_sha256=sha(manifest_path))
    write_json(BASE / "audio_assets_manifest.json", report)
    print(json.dumps(dict(mode="prepare_assets", files=len(rows),
                          new_copies=sum(row["new_copy"] for row in rows),
                          bytes=sum(row["bytes"] for row in rows), downloads=0,
                          torch_imported=report["torch_imported"]), ensure_ascii=False))
    return report


def preflight(source, copy, models):
    source, copy = checked_paths(source, copy)
    rows = {}
    for relative in (*ASSETS, *NATIVE_FILES):
        src, dst = source / relative, copy / relative
        if not src.is_file() or not dst.is_file():
            raise FileNotFoundError("Required local source/copy is absent: " + relative)
        if not dst.resolve().is_relative_to(copy):
            raise RuntimeError("Asset/code resolves outside research copy: " + str(dst))
        expected = ASSETS.get(relative)
        if expected is not None and src.stat().st_size != expected:
            raise RuntimeError("Unexpected local asset size: " + relative)
        if sha(src) != sha(dst):
            raise RuntimeError("Native source and research copy differ: " + relative)
        rows[relative] = dict(bytes=dst.stat().st_size, sha256=sha(dst))
    texts = json.loads((copy / SENTENCES).read_text())
    if len(texts) != 4 or not all(isinstance(x, str) for x in texts):
        raise RuntimeError("Four existing sentence texts required")
    for i in range(4):
        with wave.open(str(copy / f"lab/tts_carrier/out3/scene/s{i}.wav"), "rb") as wav:
            rows[f"lab/tts_carrier/out3/scene/s{i}.wav"]["wave"] = dict(
                sample_rate=wav.getframerate(), channels=wav.getnchannels(),
                frames=wav.getnframes(), sample_width_bytes=wav.getsampwidth())
    model_rows = {}
    for arm, model in models.items():
        model = inside_research(model)
        if not model.is_file():
            raise FileNotFoundError("Final exported checkpoint is absent: " + str(model))
        model_rows[arm] = dict(path=str(model), bytes=model.stat().st_size, sha256=sha(model))
    return dict(schema="applio_v08_audio_preflight", assets_and_native_sources=rows,
                models=model_rows, sentences=texts, inference_settings=SETTINGS,
                native_inference_dtype="float32; no AMP or training candidate patch",
                final_model_format_validation="Native weights_only load during execute",
                downloads=0, torch_imported="torch" in sys.modules)


def load_native_model_audited(converter, model_class, record):
    """Observe one native model load without changing its arguments or result."""
    absent = object()
    own_method = vars(model_class).get("load_state_dict", absent)
    original = getattr(model_class, "load_state_dict")
    observed_models = []
    record.update(calls=[], native_weight_load_complete=False)
    get_vc_returned = False

    def observed_load(module, *pos, **kw):
        state = pos[0] if pos else kw.get("state_dict")
        call = dict(
            input_state_key_count=len(state) if state is not None else None,
            model_state_key_count=len(module.state_dict()),
            strict=kw.get("strict", pos[1] if len(pos) > 1 else True),
            assign=kw.get("assign", pos[2] if len(pos) > 2 else False),
            returned=False, missing_keys=None, unexpected_keys=None)
        record["calls"].append(call)
        observed_models.append(module)
        try:
            result = original(module, *pos, **kw)
        except BaseException as exc:
            call["error"] = dict(type=type(exc).__name__, message=str(exc))
            raise
        call.update(returned=True, missing_keys=list(result.missing_keys),
                    unexpected_keys=list(result.unexpected_keys))
        return result

    setattr(model_class, "load_state_dict", observed_load)
    try:
        converter.get_vc(record["model_path"], 0)
        get_vc_returned = True
    except BaseException as exc:
        record["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        if own_method is absent:
            delattr(model_class, "load_state_dict")
        else:
            setattr(model_class, "load_state_dict", own_method)
        target = getattr(converter, "net_g", None)
        for module, call in zip(observed_models, record["calls"]):
            call["target_inference_model"] = module is target
        target_calls = [call for call in record["calls"] if call["target_inference_model"]]
        checkpoint = getattr(converter, "cpt", None)
        record.update(
            loaded_model_path=getattr(converter, "loaded_model", None),
            checkpoint_epoch=checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
            checkpoint_step=checkpoint.get("step") if isinstance(checkpoint, dict) else None,
            get_vc_returned=get_vc_returned,
            observed_load_count=len(record["calls"]),
            target_model_load_count=len(target_calls))
        record["native_weight_load_complete"] = bool(
            get_vc_returned and record["loaded_model_path"] == record["model_path"] and
            len(record["calls"]) == 1 and len(target_calls) == 1 and
            target_calls[0]["returned"] and not target_calls[0]["missing_keys"] and
            not target_calls[0]["unexpected_keys"])
    if not record["native_weight_load_complete"]:
        message = "Native weight load incomplete; inspect model_loads in result.json"
        record["error"] = dict(type="RuntimeError", message=message)
        raise RuntimeError(message)


def protected_operating():
    original = json.loads((BASE / "protected_before.json").read_text())
    paths = {**original["legacy16"], **original["additional_target_and_source"]}
    changed = [path for path, expected in paths.items() if sha(path) != expected]
    return dict(legacy16_count=len(original["legacy16"]),
                additional_count=len(original["additional_target_and_source"]),
                changed=changed, hashes=paths)


def peak97_metrics(x, rate, np):
    """Read-only CPU diagnostics for finite [samples, channels] audio.

    Both callers check finiteness first. Runs stay within each channel and use
    the whole measured signal's peak, preserving the original WAV definition.
    These relative-peak runs alone do not classify clipping or audio quality.
    """
    if x.ndim != 2:
        raise ValueError("Peak-run diagnostics require [samples, channels] audio")
    rate = float(rate)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Positive finite sample rate required for run duration")
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    counts, longest, marked = [], [], 0
    for channel in x.T:
        mask = np.abs(channel) >= 0.97 * peak if peak > 0 else np.zeros(channel.shape, dtype=bool)
        marked += int(mask.sum())
        edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
        lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        qualifying = lengths[lengths >= 6]
        counts.append(int(qualifying.size))
        longest.append(int(qualifying.max()) if qualifying.size else 0)
    return dict(peak=peak, peak97_run_count_ge6=sum(counts),
                peak97_run_count_ge6_by_channel=counts,
                peak97_longest_run_ge6_samples=max(longest, default=0),
                peak97_longest_run_ge6_ms=1000 * max(longest, default=0) / rate,
                peak97_sample_fraction=marked / x.size if x.size else 0.0,
                saturation_note="abs >= 0.97 * measured signal global peak, contiguous >= 6 samples; diagnostic, not a clipping or quality classification")


def wave_metrics(path, np, sf):
    x, rate = sf.read(str(path), dtype="float64", always_2d=True)
    finite = bool(np.isfinite(x).all())
    if not finite:
        raise RuntimeError("Nonfinite samples in output WAV")
    runs = peak97_metrics(x, rate, np)
    peak = runs["peak"]
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    info = sf.info(str(path))
    return dict(path=str(path), sha256=sha(path), sample_rate=int(rate),
                channels=int(x.shape[1]), samples_per_channel=int(x.shape[0]),
                duration_s=float(x.shape[0] / rate), subtype=info.subtype,
                finite=finite, rms=rms,
                crest_db=20 * math.log10(peak / rms) if rms > 0 else None,
                rms_dbfs=20 * math.log10(rms) if rms > 0 else None,
                **runs), x


def pair_metrics(a, b, np):
    if a.shape != b.shape:
        return dict(comparable=False, a_shape=list(a.shape), b_shape=list(b.shape),
                    reason="Different lengths/channels; no truncation or alignment applied")
    err = b - a
    rmse = float(np.sqrt(np.mean(err * err)))
    rms = float(np.sqrt(np.mean(a * a)))
    return dict(comparable=True, value_equal=bool(np.array_equal(a, b)),
                max_abs=float(np.max(np.abs(err))), rmse=rmse,
                nrmse=rmse / rms if rms > 0 else None,
                snr_db=20 * math.log10(rms / rmse) if rms > 0 and rmse > 0 else None)


def render_html(out):
    out = inside_research(out)
    public = json.loads((out / "blind_manifest.json").read_text())
    cards = []
    for row in public["sentences"]:
        i = row["sentence_index"]
        players = []
        for label in LABELS:
            item = row["audio"][label]
            path = out / item["file"]
            if not path.resolve().is_relative_to(out):
                raise RuntimeError("Public WAV path leaves run folder")
            if sha(path) != item["sha256"]:
                raise RuntimeError("Public WAV changed: " + str(path))
            uri = "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
            players.append('<div class="player"><b>' + label + '</b><audio controls preload="none" src="' + uri + '"></audio></div>')
        preference = '<option value="undecided">아직 판단하지 않음</option><option value="same">의미 있는 차이 없음</option>'
        anomaly = '<option value="undecided">아직 판단하지 않음</option><option value="none">하나만 이상하게 들리진 않음</option>'
        for label in LABELS:
            preference += '<option value="' + label + '">' + label + '</option>'
            anomaly += '<option value="' + label + '">' + label + '</option>'
        cards.append('<section><h2>문장 ' + str(i + 1) + '</h2><p class="sentence">' +
                     html.escape(row["text"]) + '</p><div class="players">' + ''.join(players) +
                     '</div><div class="answers"><label>더 자연스럽게 들린 쪽<select data-field="preference" data-sentence="' +
                     str(i) + '">' + preference + '</select></label><label>다른 둘과 달리 이상하게 들린 쪽<select data-field="anomaly" data-sentence="' +
                     str(i) + '">' + anomaly + '</select></label></div><label>메모<textarea data-field="notes" data-sentence="' +
                     str(i) + '" rows="3" placeholder="어느 구간에서, 어떤 차이가 났는지"></textarea></label></section>')
    template = """<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Applio 세 모델 익명 청취</title>
<style>
:root{color-scheme:light;font-family:system-ui,sans-serif;background:#f3f1eb;color:#23322d}
body{max-width:1100px;margin:0 auto;padding:32px 20px 60px;line-height:1.6}
h1{font-size:clamp(26px,4vw,38px);line-height:1.2;margin-bottom:16px}
h2{font-size:20px;margin-top:0}.intro{max-width:800px;color:#40534c}.small{font-size:13px;color:#50675e}
section,.closing{background:white;border:1px solid #d8e0d8;border-radius:14px;padding:24px;margin:24px 0}
.sentence{font-size:18px}.players{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
.player{background:#f2f7f3;border-radius:10px;padding:12px}.player b{display:block;margin-bottom:8px}
audio{width:100%}.answers{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:20px 0}
label{display:block;font-size:14px}select,textarea{box-sizing:border-box;width:100%;display:block;margin-top:6px;padding:10px;border:1px solid #b9c9c0;border-radius:6px;font:inherit}
button{border:0;border-radius:8px;padding:12px 20px;background:#21634c;color:white;font:inherit;cursor:pointer}
#status{margin-left:12px;color:#456052}@media(max-width:700px){.players,.answers{grid-template-columns:1fr}body{padding:24px 14px}section{padding:18px}}
</style>
<h1>같은 네 문장, 세 모델</h1>
<p class="intro">A · B · C는 한 번 무작위로 정한 익명 이름이야. 모든 문장에서 같은 글자는 같은 모델을 뜻해.
발음·숨소리·음색·잡음·갈라짐을 편하게 비교해 줘. 어느 쪽이 후보인지 맞히는 시험은 아니야.
잘 모르겠거나 차이가 없으면 그대로 남겨도 좋아.</p>
<p class="small">입력·피치·검색 인덱스·변환 설정과 문장별 난수 seed는 같아. 완성 WAV에 개별 음량 정규화를 추가하지 않았어.
자동 재생은 없고, 하나를 재생하면 나머지는 멈춰. 선택과 메모는 아래에서 JSON 파일로 저장할 수 있어.</p>
__CARDS__
<div class="closing"><label>전체 메모<textarea id="overall" rows="4" placeholder="특정 모델만 어색했는지, 혹은 반복 학습 사이의 차이 정도인지"></textarea></label>
<p><button id="download" type="button">판정과 메모 저장</button><span id="status" role="status"></span></p>
<p class="small">아직 사용자 판정 전이야. 파형 수치만으로 음질 합격을 정하지 않아. 이 페이지에는 모델 이름 대응표가 들어 있지 않아.</p></div>
<script>
const runId=__RUN_ID__;
const storeKey="applio_v08_blind_"+runId;
const fields=Array.from(document.querySelectorAll("[data-field]"));
function collect(){
 const sentences=[0,1,2,3].map(i=>({sentence_index:i}));
 fields.forEach(el=>sentences[Number(el.dataset.sentence)][el.dataset.field]=el.value);
 return {schema:"applio_v08_human_blind_judgment",run_id:runId,
 timestamp:new Date().toISOString(),sentences,overall_notes:document.getElementById("overall").value};
}
function saveLocal(){try{localStorage.setItem(storeKey,JSON.stringify(collect()));}catch(e){}}
try{
 const old=JSON.parse(localStorage.getItem(storeKey)||"null");
 if(old&&old.run_id===runId){fields.forEach(el=>{const row=old.sentences[Number(el.dataset.sentence)];if(row&&typeof row[el.dataset.field]==="string")el.value=row[el.dataset.field];});
 document.getElementById("overall").value=old.overall_notes||"";}
}catch(e){}
fields.forEach(el=>el.addEventListener("input",saveLocal));
document.getElementById("overall").addEventListener("input",saveLocal);
const audios=Array.from(document.querySelectorAll("audio"));
audios.forEach(a=>a.addEventListener("play",()=>audios.forEach(b=>{if(a!==b)b.pause();})));
document.getElementById("download").addEventListener("click",()=>{
 const blob=new Blob([JSON.stringify(collect(),null,2)],{type:"application/json;charset=utf-8"});
 const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;
 a.download="applio_v08_judgment_"+runId+".json";document.body.appendChild(a);a.click();a.remove();
 setTimeout(()=>URL.revokeObjectURL(url),1000);document.getElementById("status").textContent="판정 파일을 저장했어.";
});
</script></html>
"""
    text = template.replace("__CARDS__", ''.join(cards)).replace("__RUN_ID__", json.dumps(public["run_id"]))
    (out / "LISTEN_BLIND.html").write_text(text, encoding="utf-8")
    return out / "LISTEN_BLIND.html"


def execute(args, models):
    if set(models) != set(ARMS):
        raise ValueError("--execute requires exactly A, A_prime, and B final checkpoints")
    if args.db_seed is None:
        raise ValueError("--execute requires an existing research --db-seed; no operating DB writes")
    out = inside_research(args.out)
    db_seed = inside_research(args.db_seed)
    if not db_seed.is_dir():
        raise FileNotFoundError("Existing research DB seed required: " + str(db_seed))
    if out.exists():
        raise FileExistsError("Use a fresh run output directory; refusing overwrite or rerandomization")
    initial = preflight(args.source, args.applio_copy, models)
    protection = protected_operating()
    if protection["changed"]:
        raise RuntimeError("Operating protected files changed since v0.8 start")
    out.mkdir(parents=True)
    raw_dir = out / "raw"
    raw_dir.mkdir()
    (out / "blind").mkdir()
    db = out / "miopen_db"
    shutil.copytree(db_seed, db)
    (out / "miopen_cache").mkdir()
    for key in tuple(os.environ):
        if key.startswith("APPLIO_RDNA2_"):
            del os.environ[key]
    os.environ.update(SAFE_ENV)
    os.environ.update(
        MIOPEN_FIND_MODE="FAST", MIOPEN_USER_DB_PATH=str(db),
        MIOPEN_CUSTOM_CACHE_DIR=str(out / "miopen_cache"),
        APPLIO_DILATION_FOLD="0", APPLIO_COMPILE="", APPLIO_PROFILE_AT="0",
        APPLIO_AMP_LOG="0", MPLBACKEND="Agg",
    )
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    shuffled = list(ARMS)
    secrets.SystemRandom().shuffle(shuffled)
    label_to_arm = dict(zip(LABELS, shuffled))
    write_json(out / "blind_key.json", dict(run_id=run_id, label_to_training_arm=label_to_arm), private=True)
    write_json(out / "preflight.json", initial)
    report = dict(schema="applio_v08_audio_eval", run_id=run_id, state="running",
                  inference_settings=SETTINGS, models=initial["models"], rows=[],
                  model_loads=[], native_weight_load_complete=False,
                  source_assets=initial["assets_and_native_sources"], downloads=0,
                  quality_judgment="pending_user_blind_listening",
                  timing_note="Instrumented native quality evaluation; not a speed benchmark",
                  operating_before=protection, native_patches_applied=False,
                  sentence_seeds=[args.seed + i for i in range(4)],
                  output_normalization="Native VoiceConverter pipeline only; no per-output extra normalization",
                  gate={"raw_and_wav_finite": True, "native_weight_load_complete": True,
                        "quality": "User judgment; B-only abnormal audio fails. A vs A_prime is repeat floor.",
                        "wave_metrics": "Supplementary; no numerical quality acceptance threshold"},
                  environment={key: os.environ[key] for key in (*SAFE_ENV, "MIOPEN_FIND_MODE",
                      "MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR",
                      "APPLIO_DILATION_FOLD", "APPLIO_COMPILE")})
    write_json(out / "result.json", report)
    from gpu_guard_v06 import GpuAudit
    audit = GpuAudit()
    native_root = Path(args.applio_copy).resolve()
    try:
        with audit:
            audit.ensure_clear()
            os.chdir(native_root)
            sys.path.insert(0, str(native_root))
            with (out / "inference.log").open("w", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    import numpy as np
                    import soundfile as sf
                    import torch
                    from rvc.infer.infer import VoiceConverter
                    from rvc.lib.algorithm.synthesizers import Synthesizer
                    if not torch.cuda.is_available():
                        raise RuntimeError("Expected the existing ROCm GPU runtime")
                    report["runtime"] = dict(torch=torch.__version__, hip=torch.version.hip,
                        gpu=torch.cuda.get_device_name(0),
                        gfx=str(torch.cuda.get_device_properties(0).gcnArchName),
                        cudnn_benchmark=torch.backends.cudnn.benchmark)
                    converter = VoiceConverter()
                    converter.load_hubert("custom", str(native_root / EMBEDDER))
                    converter.last_embedder_model = "custom"
                    wave_data = {}
                    for arm in ARMS:
                        arm_dir = raw_dir / arm
                        arm_dir.mkdir()
                        load_record = dict(training_arm=arm, model_path=str(models[arm]),
                            model_sha256=initial["models"][arm]["sha256"],
                            model_sha256_source="Preflight SHA256; source file rechecked after evaluation")
                        report["model_loads"].append(load_record)
                        try:
                            load_native_model_audited(converter, Synthesizer, load_record)
                        finally:
                            report["native_weight_load_complete"] = bool(
                                len(report["model_loads"]) == len(ARMS) and
                                all(row.get("native_weight_load_complete", False)
                                    for row in report["model_loads"]))
                            write_json(out / "result.json", report)
                        original_infer = converter.net_g.infer
                        raw_probe = []
                        def inspected_infer(*pos, **kw):
                            result = original_infer(*pos, **kw)
                            signal = result[0].detach().float().cpu()
                            finite = bool(torch.isfinite(signal).all())
                            raw_probe.append(dict(finite=finite, samples=signal.numel(),
                                peak=float(signal.abs().max()) if signal.numel() else 0.0,
                                rms=float(signal.double().square().mean().sqrt()) if signal.numel() else 0.0))
                            if not finite:
                                raise RuntimeError("Native generator produced nonfinite float output")
                            # Native Synthesizer.infer returns NCT; pipeline.py
                            # consumes result[0][0, 0]. The NSF output has C=1.
                            if signal.ndim != 3 or tuple(signal.shape[:2]) != (1, 1):
                                raise RuntimeError("Expected native generator output [1, 1, samples]")
                            raw_samples = signal[0].numpy().T  # detached CPU [T, C]
                            sample_rate = int(converter.tgt_sr)
                            raw_probe[-1].update(peak97_metrics(raw_samples, sample_rate, np))
                            raw_probe[-1].update(
                                sample_rate=sample_rate,
                                sample_rate_source="VoiceConverter.tgt_sr (checkpoint config[-1])",
                                tensor_shape=list(signal.shape), tensor_layout="NCT",
                                channel_axis=1, sample_axis=2,
                                channels=int(raw_samples.shape[1]),
                                samples_per_channel=int(raw_samples.shape[0]),
                                measurement_point="Native generator output after decoder tanh; before pipeline padding trim, volume-envelope adjustment, peak scaling, and PCM WAV encoding",
                                abs_ge1_sample_fraction=(float(np.count_nonzero(np.abs(raw_samples) >= 1.0) / raw_samples.size)
                                                         if raw_samples.size else 0.0),
                                abs_ge1_note="abs >= 1.0 in native generator float output; auxiliary diagnostic, not a clipping or quality-failure classification")
                            return result
                        converter.net_g.infer = inspected_infer
                        try:
                            for i in range(4):
                                audit.ensure_clear()
                                seed = args.seed + i
                                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                                torch.cuda.manual_seed_all(seed)
                                raw_probe.clear()
                                target = arm_dir / f"s{i}.wav"
                                torch.cuda.synchronize()
                                start = time.perf_counter()
                                converter.convert_audio(
                                    audio_input_path=str(native_root / f"lab/tts_carrier/out3/scene/s{i}.wav"),
                                    audio_output_path=str(target), model_path=str(models[arm]),
                                    index_path=str(native_root / INDEX),
                                    embedder_model_custom=str(native_root / EMBEDDER), **SETTINGS)
                                torch.cuda.synchronize()
                                elapsed = time.perf_counter() - start
                                metrics, samples = wave_metrics(target, np, sf)
                                if not raw_probe:
                                    raise RuntimeError("No native generator float output observed")
                                row = dict(training_arm=arm, sentence_index=i, seed=seed,
                                    conversion_instrumented_s=elapsed, raw_generator=list(raw_probe), **metrics)
                                report["rows"].append(row)
                                wave_data[(arm, i)] = (samples, metrics["sample_rate"])
                                write_json(out / "result.json", report)
                        finally:
                            converter.net_g.infer = original_infer
                    report["wave_comparisons"] = []
                    for i in range(4):
                        pairs = {}
                        for a, b in (("A", "A_prime"), ("A", "B"), ("A_prime", "B")):
                            x, sx = wave_data[(a, i)]; y, sy = wave_data[(b, i)]
                            pairs[a + "_vs_" + b] = pair_metrics(x, y, np) if sx == sy else dict(
                                comparable=False, reason="Different sample rates", sample_rates=[sx, sy])
                        floor = pairs["A_vs_A_prime"].get("rmse")
                        ratios = {key: (value.get("rmse") / floor if floor and value.get("comparable") else None)
                                  for key, value in pairs.items() if key != "A_vs_A_prime"}
                        report["wave_comparisons"].append(dict(sentence_index=i, pairs=pairs,
                            B_rmse_over_native_repeat=ratios,
                            note="Same native pipeline and seed, no time alignment; supplementary to listening"))
                    audit.ensure_clear()
        public = dict(schema="applio_v08_blind_audio", run_id=run_id, labels=list(LABELS),
                      key_excluded=True, sentences=[])
        for i in range(4):
            row = dict(sentence_index=i, text=initial["sentences"][i], audio={})
            for label in LABELS:
                src = raw_dir / label_to_arm[label] / f"s{i}.wav"
                relative = f"blind/sentence_{i + 1:02d}_{label}.wav"
                digest, _ = same_copy(src, out / relative)
                row["audio"][label] = dict(file=relative, sha256=digest)
            public["sentences"].append(row)
        write_json(out / "blind_manifest.json", public)
        render_html(out)
        report["state"] = "audio_ready_user_judgment_pending"
    except BaseException as exc:
        report["state"] = "failed"
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        report["gpu_audit"] = audit.report()
        report["operating_after"] = protected_operating()
        report["operating_unchanged"] = not report["operating_after"]["changed"]
        try:
            after = preflight(args.source, args.applio_copy, models)
            report["source_model_assets_unchanged"] = (
                initial["models"] == after["models"] and
                initial["assets_and_native_sources"] == after["assets_and_native_sources"])
        except Exception as exc:
            report["source_model_assets_unchanged"] = False
            report["after_preflight_error"] = str(exc)
        report["valid_for_listening"] = bool(
            report["state"] == "audio_ready_user_judgment_pending" and
            report["native_weight_load_complete"] and
            audit.report()["valid_for_timing"] and report["operating_unchanged"] and
            report["source_model_assets_unchanged"])
        write_json(out / "result.json", report)
    if not report["valid_for_listening"]:
        raise RuntimeError("Guard/protection failed; outputs retained but invalid")
    print(json.dumps(dict(state=report["state"], wav_files=len(report["rows"]),
        public_page=str(out / "LISTEN_BLIND.html"), valid_for_listening=True,
        quality_judgment="pending_user", downloads=0), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-assets", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--render-only", action="store_true")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--applio-copy", type=Path, default=COPY)
    parser.add_argument("--out", type=Path, default=BASE / "audio_eval")
    parser.add_argument("--model", action="append", default=[], metavar="ARM=FINAL_MODEL_PATH")
    parser.add_argument("--db-seed", type=Path)
    parser.add_argument("--seed", type=int, default=997131)
    args = parser.parse_args()
    models = {}
    for item in args.model:
        arm, sep, value = item.partition("=")
        if not sep or arm not in ARMS or arm in models:
            parser.error("Each --model must be one unique A, A_prime, or B assignment")
        models[arm] = Path(value).resolve()
    if models and set(models) != set(ARMS):
        parser.error("Provide all three model arms together")
    if args.prepare_assets:
        prepare_assets(args.source, args.applio_copy)
    elif args.preflight:
        report = preflight(args.source, args.applio_copy, models)
        write_json(BASE / "audio_preflight.json", report)
        print(json.dumps(dict(mode="preflight", ready=True, model_count=len(models),
            files=len(report["assets_and_native_sources"]), torch_imported=report["torch_imported"],
            final_models_pending=not models, downloads=0), ensure_ascii=False))
    elif args.render_only:
        print(json.dumps(dict(public_page=str(render_html(args.out))), ensure_ascii=False))
    else:
        execute(args, models)


if __name__ == "__main__":
    main()
