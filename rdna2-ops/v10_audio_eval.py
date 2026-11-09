#!/usr/bin/env python3
"""Native FP32 blind listening for two completed operating-path v1.0 models.

CPU-only --preflight/--help never import torch. --execute owns GpuAudit before
the first GPU/runtime import and uses existing v0.8 native code/assets read-only.
The public artifact contains two anonymous labels; blind_key.json and technical
results stay outside the public page. Numerical audio metrics are diagnostics.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import contextlib
import datetime
import hashlib
import html
import json
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

# This existing module imports only the standard library until execute().
# Reuse its audited native-load observer and audio diagnostics unchanged.
import v08_audio_eval as native

RESEARCH = Path(__file__).resolve().parent
BASE = RESEARCH / "results/v10"
LEGACY = RESEARCH / "results/applio_v08"
SOURCE = native.SOURCE
COPY = LEGACY / "applio_copy"
ARMS = ("B", "B_prime")
LABELS = ("A", "B")
SETTINGS = dict(native.SETTINGS)
SAFE_ENV = {**native.SAFE_ENV, "HSA_DISABLE_COREDUMP_ON_EXCEPTION": "1",
            "PYTORCH_TUNABLEOP_ENABLED": "0", "PYTORCH_TUNABLEOP_TUNING": "0",
            "PYTORCH_TUNABLEOP_RECORD_UNTUNED": "0"}
sha, write_json = native.sha, native.write_json


def output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(BASE):
        raise ValueError("Use a research output under results/v10")
    return path


def model_input(path, source):
    """Read research exports or fresh operating v10 model exports, never write."""
    path, source = Path(path).resolve(), Path(source).resolve()
    research_export = path.is_relative_to(BASE)
    operating_export = path.parent.parent == source / "logs" and path.parent.name.startswith("v10_")
    if not (research_export or operating_export) or path.suffix != ".pth" or not path.is_file():
        raise ValueError("A final .pth under results/v10 or operating logs/v10_* is required: " + str(path))
    return path


def preflight(source, copy, models):
    source, copy = native.checked_paths(source, copy)
    prior = json.loads((LEGACY / "audio_assets_manifest.json").read_text())
    prior_hashes = {str(Path(row["copy"]).relative_to(COPY)): row["sha256"] for row in prior["assets"]}
    rows = {}
    required = tuple(dict.fromkeys((*native.ASSETS, *native.NATIVE_FILES,
                                  "rvc/lib/algorithm/residuals.py",
                                  "rvc/lib/algorithm/generators/hifigan_nsf.py")))
    for relative in required:
        src, dst = source / relative, copy / relative
        if not src.is_file() or not dst.is_file() or not dst.resolve().is_relative_to(copy):
            raise FileNotFoundError("Existing native source/copy required: " + relative)
        digest, source_digest = sha(dst), sha(src)
        if digest != source_digest:
            raise RuntimeError("Native source/research copy differ: " + relative)
        if relative in prior_hashes and digest != prior_hashes[relative]:
            raise RuntimeError("Asset differs from the four-sentence v0.8 evaluation: " + relative)
        expected_size = native.ASSETS.get(relative)
        if expected_size is not None and dst.stat().st_size != expected_size:
            raise RuntimeError("Unexpected existing asset size: " + relative)
        rows[relative] = dict(bytes=dst.stat().st_size, sha256=digest)
    sentences = json.loads((copy / native.SENTENCES).read_text())
    if len(sentences) != 4 or not all(isinstance(s, str) for s in sentences):
        raise ValueError("Exactly the four existing sentence texts are required")
    for i in range(4):
        relative = f"lab/tts_carrier/out3/scene/s{i}.wav"
        with wave.open(str(copy / relative), "rb") as wav:
            rows[relative]["wave"] = dict(sample_rate=wav.getframerate(), channels=wav.getnchannels(),
                                          frames=wav.getnframes(), sample_width_bytes=wav.getsampwidth())
    model_rows = {}
    for arm, path in models.items():
        path = model_input(path, source)
        model_rows[arm] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
    if len({row["path"] for row in model_rows.values()}) != len(model_rows):
        raise ValueError("Two distinct model export paths are required")
    return dict(schema="applio_v10_audio_preflight", models=model_rows, sentences=sentences,
                assets_and_native_sources=rows, inference_settings=SETTINGS,
                reused_v08_helpers_sha256=sha(RESEARCH / "v08_audio_eval.py"),
                native_inference_dtype="FP32, no AMP, no training candidate hooks",
                final_checkpoint_validation="Native weights_only load, 100 epochs/3100 steps, one speaker and KLM-SVM during execute",
                downloads=0, torch_imported="torch" in sys.modules)


def protected_operating():
    baseline_path = BASE / "protected_before.json"
    baseline = json.loads(baseline_path.read_text())
    if baseline.get("schema") != "rdna2_v10_protection" or baseline.get("mode") != "before" or baseline.get("errors"):
        raise RuntimeError("A clean current v1.0 protection baseline is required")
    current, errors = {}, {}
    for path in baseline["files"]:
        try:
            current[path] = sha(path)
        except OSError as exc:
            errors[path] = str(exc)
    changed = [path for path, expected in baseline["files"].items()
               if path in current and current[path] != expected]
    missing = sorted(set(baseline["files"]) - set(current))
    return dict(baseline=str(baseline_path), baseline_sha256=sha(baseline_path),
                checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                count=len(current), changed=changed, missing=missing, errors=errors,
                unchanged=not changed and not missing and not errors, hashes=current)


def render_html(out):
    out = output_path(out)
    public = json.loads((out / "blind_manifest.json").read_text())
    if public.get("schema") != "applio_v10_blind_audio" or public.get("labels") != list(LABELS):
        raise ValueError("A two-label v1.0 public manifest is required")
    cards = []
    for row in public["sentences"]:
        index = row["sentence_index"]
        players = []
        for label in LABELS:
            item = row["audio"][label]
            path = out / item["file"]
            if not path.resolve().is_relative_to(out / "blind") or sha(path) != item["sha256"]:
                raise RuntimeError("Public WAV path or checksum mismatch")
            uri = "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
            players.append(f'<div class="player"><b>{label}</b><audio controls preload="none" src="{uri}"></audio></div>')
        cards.append(
            f'<section><h2>문장 {index + 1}</h2><p class="sentence">{html.escape(row["text"])}</p>'
            '<div class="players">' + ''.join(players) + '</div><div class="answers">'
            f'<label>더 자연스럽게 들린 쪽<select data-field="preference" data-sentence="{index}">'
            '<option value="undecided">아직 판단하지 않음</option><option value="same">의미 있는 차이 없음</option>'
            '<option value="A">A</option><option value="B">B</option></select></label>'
            f'<label>어색하게 들린 쪽<select data-field="anomaly" data-sentence="{index}">'
            '<option value="undecided">아직 판단하지 않음</option><option value="none">둘 다 특별히 어색하지 않음</option>'
            '<option value="A">A만</option><option value="B">B만</option><option value="both">둘 다</option></select></label></div>'
            f'<label>메모<textarea data-field="notes" data-sentence="{index}" rows="3" '
            'placeholder="발음·숨소리·음색·잡음·갈라짐 등, 차이가 들린 구간"></textarea></label></section>')
    template = '''<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Applio 두 모델 익명 청취 · v1.0</title>
<style>
:root{color-scheme:light;font-family:system-ui,sans-serif;background:#f3f1eb;color:#23322d}
body{max-width:1020px;margin:0 auto;padding:32px 20px 60px;line-height:1.6}
h1{font-size:clamp(26px,4vw,38px);line-height:1.2}h2{font-size:20px;margin-top:0}
.intro{max-width:800px;color:#40534c}.small{font-size:13px;color:#50675e}
section,.closing{background:white;border:1px solid #d8e0d8;border-radius:14px;padding:24px;margin:24px 0}
.sentence{font-size:18px}.players,.answers{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
.player{background:#f2f7f3;border-radius:10px;padding:14px}.player b{display:block;margin-bottom:8px}
audio{width:100%}.answers{margin:20px 0}label{display:block;font-size:14px}
select,textarea{box-sizing:border-box;width:100%;display:block;margin-top:6px;padding:10px;border:1px solid #b9c9c0;border-radius:6px;font:inherit}
button{border:0;border-radius:8px;padding:12px 20px;background:#21634c;color:white;font:inherit;cursor:pointer}
#status{margin-left:12px;color:#456052}@media(max-width:700px){.players,.answers{grid-template-columns:1fr}body{padding:24px 14px}section{padding:18px}}
</style>
<h1>같은 네 문장, 두 모델</h1>
<p class="intro">A · B는 한 번 무작위로 정한 익명 이름이야. 모든 문장에서 같은 글자는 같은 모델을 뜻해.
발음·숨소리·음색·잡음·갈라짐을 편하게 비교해 줘. 잘 모르겠거나 차이가 없으면 그대로 남겨도 좋아.</p>
<p class="small">입력·피치·검색 인덱스·변환 설정과 문장별 난수 seed는 같아. 완성 WAV에 개별 음량 정규화를 추가하지 않았어.
자동 재생은 없고, 하나를 재생하면 나머지는 멈춰. 선택과 메모는 아래에서 JSON 파일로 저장할 수 있어.</p>
__CARDS__
<div class="closing"><label>전체 메모<textarea id="overall" rows="4" placeholder="네 문장을 함께 들었을 때의 인상"></textarea></label>
<p><button id="download" type="button">판정과 메모 저장</button><span id="status" role="status"></span></p>
<p class="small">이 페이지에는 모델 이름 대응표가 들어 있지 않아. 파형 수치로 음질 판정을 대신하지 않고 네 청취 결과를 기록해.</p></div>
<script>
const runId=__RUN_ID__;
const storeKey="applio_v10_blind_"+runId;
const fields=Array.from(document.querySelectorAll("[data-field]"));
function collect(){const sentences=[0,1,2,3].map(i=>({sentence_index:i}));
 fields.forEach(el=>sentences[Number(el.dataset.sentence)][el.dataset.field]=el.value);
 return {schema:"applio_v10_human_blind_judgment",run_id:runId,timestamp:new Date().toISOString(),
 sentences,overall_notes:document.getElementById("overall").value};}
function saveLocal(){try{localStorage.setItem(storeKey,JSON.stringify(collect()));}catch(e){}}
try{const old=JSON.parse(localStorage.getItem(storeKey)||"null");if(old&&old.run_id===runId){
 fields.forEach(el=>{const row=old.sentences[Number(el.dataset.sentence)];if(row&&typeof row[el.dataset.field]==="string")el.value=row[el.dataset.field];});
 document.getElementById("overall").value=old.overall_notes||"";}}catch(e){}
fields.forEach(el=>el.addEventListener("input",saveLocal));document.getElementById("overall").addEventListener("input",saveLocal);
const audios=Array.from(document.querySelectorAll("audio"));audios.forEach(a=>a.addEventListener("play",()=>audios.forEach(b=>{if(a!==b)b.pause();})));
document.getElementById("download").addEventListener("click",()=>{
 const blob=new Blob([JSON.stringify(collect(),null,2)],{type:"application/json;charset=utf-8"});
 const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="applio_v10_judgment_"+runId+".json";
 document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
 document.getElementById("status").textContent="판정 파일을 저장했어.";
});
</script></html>'''
    rendered = template.replace("__CARDS__", ''.join(cards)).replace("__RUN_ID__", json.dumps(public["run_id"]))
    (out / "LISTEN_BLIND.html").write_text(rendered, encoding="utf-8")
    return out / "LISTEN_BLIND.html"


def execute(args, models):
    if set(models) != set(ARMS):
        raise ValueError("Exactly B and B_prime final exports are required")
    if args.db_seed is None:
        raise ValueError("Use an existing research --db-seed")
    out, db_seed = output_path(args.out), native.inside_research(args.db_seed)
    if out.exists() or not db_seed.is_dir():
        raise ValueError("A fresh output directory and an existing research DB directory are required")
    initial = preflight(args.source, args.applio_copy, models)
    protection = protected_operating()
    if not protection["unchanged"]:
        raise RuntimeError("Protected operating files differ from the current v1.0 baseline")
    models = {arm: Path(initial["models"][arm]["path"]) for arm in ARMS}
    out.mkdir(parents=True)
    raw_dir = out / "raw"
    raw_dir.mkdir(mode=0o700)
    (out / "blind").mkdir()
    shutil.copytree(db_seed, out / "miopen_db")
    (out / "miopen_cache").mkdir()
    for key in tuple(os.environ):
        if key.startswith(("APPLIO_RDNA2_", "APPLIO_V10_")):
            del os.environ[key]
    os.environ.update(SAFE_ENV)
    os.environ.update(MIOPEN_FIND_MODE="FAST", MIOPEN_USER_DB_PATH=str(out / "miopen_db"),
                      MIOPEN_CUSTOM_CACHE_DIR=str(out / "miopen_cache"), APPLIO_DILATION_FOLD="0",
                      APPLIO_COMPILE="", APPLIO_PROFILE_AT="0", APPLIO_AMP_LOG="0", MPLBACKEND="Agg")
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    shuffled = list(ARMS)
    secrets.SystemRandom().shuffle(shuffled)
    label_to_arm = dict(zip(LABELS, shuffled))
    write_json(out / "blind_key.json", dict(run_id=run_id, label_to_training_arm=label_to_arm), private=True)
    write_json(out / "preflight.json", initial, private=True)
    report = dict(schema="applio_v10_audio_eval", run_id=run_id, state="running", completed=False,
                  inference_settings=SETTINGS, models=initial["models"], rows=[], model_loads=[],
                  source_assets=initial["assets_and_native_sources"], downloads=0,
                  source_sha256=sha(__file__), reused_v08_helpers_sha256=initial["reused_v08_helpers_sha256"],
                  native_weight_load_complete=False, native_fp32_verified=False, same_generator_inputs=False,
                  expected_epoch=args.expected_epoch, expected_steps=args.expected_steps,
                  quality_judgment="pending_user_blind_listening", operating_before=protection,
                  native_training_candidate_patches_applied=False,
                  sentence_seeds=[args.seed + i for i in range(4)],
                  output_normalization="Native VoiceConverter only; no additional per-output normalization",
                  timing_note="Instrumented native quality evaluation; no inference speed claim",
                  gate=dict(raw_and_wav_finite=True, native_weight_load_complete=True,
                            native_fp32=True, same_generator_inputs=True,
                            quality="User blind listening decides; no numerical quality acceptance threshold"),
                  environment={key: os.environ[key] for key in (*SAFE_ENV, "MIOPEN_FIND_MODE", "MIOPEN_USER_DB_PATH",
                                                               "MIOPEN_CUSTOM_CACHE_DIR", "APPLIO_DILATION_FOLD", "APPLIO_COMPILE")})
    save = lambda: write_json(out / "result.json", report, private=True)
    save()
    from gpu_guard_v06 import GpuAudit
    audit = GpuAudit()
    native_root = Path(args.applio_copy).resolve()
    original_cwd = Path.cwd()
    try:
        with audit:
            audit.ensure_clear()
            os.chdir(native_root)
            sys.path.insert(0, str(native_root))
            with (out / "inference.log").open("w", encoding="utf-8") as log:
                os.chmod(out / "inference.log", 0o600)
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    import numpy as np
                    import soundfile as sf
                    import torch
                    from rvc.infer.infer import VoiceConverter
                    from rvc.lib.algorithm.synthesizers import Synthesizer
                    if not torch.cuda.is_available() or torch.version.hip is None:
                        raise RuntimeError("The existing ROCm GPU runtime is required")
                    hooks = {"Conv1d": torch.nn.Conv1d._conv_forward.__module__,
                             "Conv2d": torch.nn.Conv2d._conv_forward.__module__,
                             "ConvTranspose1d": torch.nn.ConvTranspose1d.forward.__module__}
                    if any(module != "torch.nn.modules.conv" for module in hooks.values()):
                        raise RuntimeError("Training convolution dispatch hooks leaked into native inference")
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cuda.matmul.allow_tf32 = False
                    report["runtime"] = dict(torch=torch.__version__, hip=torch.version.hip,
                        gpu=torch.cuda.get_device_name(0), gfx=torch.cuda.get_device_properties(0).gcnArchName,
                        native_conv_method_modules=hooks, cudnn_benchmark=torch.backends.cudnn.benchmark)
                    converter = VoiceConverter()
                    converter.load_hubert("custom", str(native_root / native.EMBEDDER))
                    converter.last_embedder_model = "custom"
                    wave_data = {}

                    def fp32_parameters(module):
                        counts = Counter(str(p.dtype) for p in module.parameters() if p.is_floating_point())
                        if not counts or set(counts) != {"torch.float32"}:
                            raise RuntimeError("Native inference parameters are not all FP32")
                        return dict(counts)

                    report["hubert_parameter_dtypes"] = fp32_parameters(converter.hubert_model)
                    for arm in ARMS:
                        audit.ensure_clear()
                        arm_dir = raw_dir / arm
                        arm_dir.mkdir(mode=0o700)
                        load_record = dict(training_arm=arm, model_path=str(models[arm]),
                                           model_sha256=initial["models"][arm]["sha256"])
                        report["model_loads"].append(load_record)
                        try:
                            native.load_native_model_audited(converter, Synthesizer, load_record)
                            cpt = converter.cpt
                            metadata_ok = (load_record["checkpoint_epoch"] == args.expected_epoch and
                                           load_record["checkpoint_step"] == args.expected_steps and
                                           cpt.get("vocoder") == "HiFi-GAN" and cpt.get("version") == "v2" and
                                           bool(cpt.get("f0")) and cpt.get("speakers_id") == 1 and
                                           cpt.get("embedder_model") == native.EMBEDDER and
                                           int(converter.tgt_sr) == 40000 and int(converter.n_spk) == 1)
                            load_record.update(metadata_contract_pass=metadata_ok,
                                checkpoint_metadata={k:cpt.get(k) for k in ("epoch", "step", "vocoder", "version", "f0", "speakers_id", "embedder_model")},
                                generator_parameter_dtypes=fp32_parameters(converter.net_g),
                                native_fp32=True)
                            if not metadata_ok:
                                raise RuntimeError("Final checkpoint epoch/steps or one-speaker KLM-SVM metadata mismatch")
                        finally:
                            report["native_weight_load_complete"] = len(report["model_loads"]) == 2 and all(
                                row.get("native_weight_load_complete", False) and row.get("metadata_contract_pass", False)
                                for row in report["model_loads"])
                            report["native_fp32_verified"] = len(report["model_loads"]) == 2 and all(
                                row.get("native_fp32", False) for row in report["model_loads"])
                            save()
                        original_infer = converter.net_g.infer
                        raw_probe = []

                        def inspected_infer(*pos, **kw):
                            if torch.is_autocast_enabled("cuda"):
                                raise RuntimeError("Native listening inference must not use AMP")
                            inputs = []
                            for name, value in [(f"arg{i}", v) for i, v in enumerate(pos)] + sorted(kw.items()):
                                if torch.is_tensor(value):
                                    if value.is_floating_point() and value.dtype != torch.float32:
                                        raise RuntimeError("Native generator floating input is not FP32")
                                    cpu = value.detach().cpu().contiguous()
                                    inputs.append(dict(name=name, shape=list(cpu.shape), dtype=str(cpu.dtype),
                                                       sha256=hashlib.sha256(cpu.numpy().tobytes()).hexdigest()))
                                else:
                                    inputs.append(dict(name=name, scalar=value))
                            rng_state = dict(
                                torch_cpu_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                                torch_cuda_sha256=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
                                numpy_sha256=hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
                                python_sha256=hashlib.sha256(repr(random.getstate()).encode()).hexdigest())
                            result = original_infer(*pos, **kw)
                            signal = result[0].detach()
                            if signal.dtype != torch.float32 or signal.ndim != 3 or tuple(signal.shape[:2]) != (1, 1):
                                raise RuntimeError("Expected native FP32 generator output [1,1,samples]")
                            signal = signal.cpu()
                            if not bool(torch.isfinite(signal).all()) or signal.numel() == 0:
                                raise RuntimeError("Native generator produced empty or nonfinite audio")
                            samples = signal[0].numpy().T
                            rate = int(converter.tgt_sr)
                            raw_probe.append(dict(finite=True, samples=signal.numel(), generator_inputs=inputs,
                                generator_rng_state=rng_state,
                                dtype=str(signal.dtype), rms=float(signal.double().square().mean().sqrt()),
                                sample_rate=rate, tensor_shape=list(signal.shape), tensor_layout="NCT",
                                channels=int(samples.shape[1]), samples_per_channel=int(samples.shape[0]),
                                measurement_point="Native decoder tanh output before padding trim, volume envelope, peak scaling and PCM encoding",
                                abs_ge1_sample_fraction=float(np.count_nonzero(np.abs(samples) >= 1.0) / samples.size),
                                abs_ge1_note="Auxiliary diagnostic; not a clipping or quality-failure classification",
                                **native.peak97_metrics(samples, rate, np)))
                            return result

                        converter.net_g.infer = inspected_infer
                        try:
                            for i in range(4):
                                audit.ensure_clear()
                                seed = args.seed + i
                                random.seed(seed)
                                np.random.seed(seed)
                                torch.manual_seed(seed)
                                torch.cuda.manual_seed_all(seed)
                                raw_probe.clear()
                                target = arm_dir / f"s{i}.wav"
                                torch.cuda.synchronize()
                                started = time.perf_counter()
                                with torch.autocast("cuda", enabled=False):
                                    converter.convert_audio(
                                        audio_input_path=str(native_root / f"lab/tts_carrier/out3/scene/s{i}.wav"),
                                        audio_output_path=str(target), model_path=str(models[arm]),
                                        index_path=str(native_root / native.INDEX),
                                        embedder_model_custom=str(native_root / native.EMBEDDER), **SETTINGS)
                                torch.cuda.synchronize()
                                elapsed = time.perf_counter() - started
                                metrics, samples = native.wave_metrics(target, np, sf)
                                os.chmod(target, 0o600)
                                if not raw_probe or metrics["sample_rate"] != 40000 or metrics["channels"] != 1 or metrics["samples_per_channel"] <= 0:
                                    raise RuntimeError("Missing native generator observation or unexpected WAV format")
                                report["rows"].append(dict(training_arm=arm, sentence_index=i, seed=seed,
                                    conversion_instrumented_s=elapsed, raw_generator=list(raw_probe), **metrics))
                                wave_data[(arm, i)] = (samples, metrics["sample_rate"])
                                save()
                        finally:
                            converter.net_g.infer = original_infer
                    report["wave_comparisons"] = []
                    report["generator_input_comparisons"] = []
                    for i in range(4):
                        x, sx = wave_data[("B", i)]
                        y, sy = wave_data[("B_prime", i)]
                        comparison = native.pair_metrics(x, y, np) if sx == sy else dict(comparable=False, reason="Different sample rates")
                        report["wave_comparisons"].append(dict(sentence_index=i, B_vs_B_prime=comparison,
                            note="Supplementary waveform diagnostic; no alignment/truncation, no repeat-floor or numerical quality gate"))
                        rows = {row["training_arm"]: row for row in report["rows"] if row["sentence_index"] == i}
                        sigs = {arm: [dict(inputs=probe["generator_inputs"], rng=probe["generator_rng_state"])
                                      for probe in rows[arm]["raw_generator"]] for arm in ARMS}
                        same = sigs["B"] == sigs["B_prime"] and rows["B"]["seed"] == rows["B_prime"]["seed"]
                        report["generator_input_comparisons"].append(dict(sentence_index=i, identical=same,
                            chunks={arm: len(sigs[arm]) for arm in ARMS}))
                    report["same_generator_inputs"] = all(row["identical"] for row in report["generator_input_comparisons"])
                    if not report["same_generator_inputs"]:
                        raise RuntimeError("Generator inputs or per-sentence seeds differ between the two arms")
                    audit.ensure_clear()
        public = dict(schema="applio_v10_blind_audio", run_id=run_id, labels=list(LABELS), key_excluded=True, sentences=[])
        for i in range(4):
            row = dict(sentence_index=i, text=initial["sentences"][i], audio={})
            for label in LABELS:
                src = raw_dir / label_to_arm[label] / f"s{i}.wav"
                relative = f"blind/sentence_{i + 1:02d}_{label}.wav"
                digest, _ = native.same_copy(src, out / relative)
                os.chmod(out / relative, 0o644)
                row["audio"][label] = dict(file=relative, sha256=digest)
            public["sentences"].append(row)
        write_json(out / "blind_manifest.json", public)
        render_html(out)
        report.update(state="audio_ready_user_judgment_pending", completed=True)
    except BaseException as exc:
        report.update(state="failed", error=dict(type=type(exc).__name__, message=str(exc)))
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        os.chmod(out / "error.txt", 0o600)
        raise
    finally:
        os.chdir(original_cwd)
        if sys.path and sys.path[0] == str(native_root):
            sys.path.pop(0)
        report["gpu_audit"] = audit.report()
        report["operating_after"] = protected_operating()
        report["operating_unchanged"] = report["operating_after"]["unchanged"]
        try:
            after = preflight(args.source, args.applio_copy, models)
            report["source_model_assets_unchanged"] = initial["models"] == after["models"] and initial["assets_and_native_sources"] == after["assets_and_native_sources"]
        except Exception as exc:
            report["source_model_assets_unchanged"] = False
            report["after_preflight_error"] = str(exc)
        report["valid_for_listening"] = bool(report["completed"] and report["native_weight_load_complete"] and
            report["native_fp32_verified"] and report["same_generator_inputs"] and len(report["rows"]) == 8 and
            audit.report()["valid_for_timing"] and report["operating_unchanged"] and report["source_model_assets_unchanged"])
        save()
    if not report["valid_for_listening"]:
        raise RuntimeError("Technical checks failed; retained files are not valid blind-listening evidence")
    print(json.dumps(dict(state=report["state"], wav_files=len(report["rows"]), public_page=str(out / "LISTEN_BLIND.html"),
                         valid_for_listening=True, quality_judgment="pending_user", downloads=0), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--render-only", action="store_true")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--applio-copy", type=Path, default=COPY)
    parser.add_argument("--out", type=Path, default=BASE / "audio_eval")
    parser.add_argument("--model", action="append", default=[], metavar="B|B_prime=FINAL_MODEL_PATH")
    parser.add_argument("--db-seed", type=Path)
    parser.add_argument("--seed", type=int, default=997131)
    parser.add_argument("--expected-epoch", type=int, default=100)
    parser.add_argument("--expected-steps", type=int, default=3100)
    args = parser.parse_args()
    models = {}
    for item in args.model:
        arm, sep, value = item.partition("=")
        if not sep or arm not in ARMS or arm in models:
            parser.error("Use one unique B=... and one B_prime=... assignment")
        models[arm] = Path(value).resolve()
    if models and set(models) != set(ARMS):
        parser.error("Provide both model exports together")
    if args.preflight:
        report = preflight(args.source, args.applio_copy, models)
        protection = protected_operating()
        print(json.dumps(dict(mode="preflight", ready=protection["unchanged"], model_count=len(models),
            files=len(report["assets_and_native_sources"]), torch_imported=report["torch_imported"],
            final_models_pending=not models, protected_count=protection["count"],
            protected_changed=len(protection["changed"]), protected_missing=len(protection["missing"]),
            downloads=0), ensure_ascii=False))
        if not protection["unchanged"]:
            raise SystemExit(2)
    elif args.render_only:
        print(json.dumps(dict(public_page=str(render_html(args.out))), ensure_ascii=False))
    else:
        execute(args, models)


if __name__ == "__main__":
    main()
