# v1.0 조건부 100 epoch B/B′ 익명 청취 준비

## 현재 상태

`rdna2-ops/v10_audio_eval.py`를 준비했다. GPU 추론은 아직 실행하지 않았다. 이 도구는 실제 운영 경로 B′의 steady epoch 이득이 주 작업의 사전 기준을 넘고, 운영 경로 100 epoch B·B′가 모두 끝난 다음에 사용한다. 100 epoch 시작 직전의 예상 시간 안내와 학습 실행은 주 작업이 담당한다.

CPU 확인 결과:

- 소스 전체 AST 파싱과 `--help`: 통과.
- `python3 -B v10_audio_eval.py --preflight`: `ready=true`.
- 기존 로컬 소재/추론 코드 20개 확인, `torch_imported=false`, 다운로드 0.
- v1.0 보호 baseline의 1022개 파일: 변경 0, 누락 0.
- 최종 모델 두 개는 아직 인자로 주지 않았다(`final_models_pending=true`). native checkpoint load, GPU 추론, 실제 청취 HTML 생성은 미검증 상태다.
- v0.8 source, 소재, 운영 코드/venv는 변경하지 않았다.

## 같은 소재와 설정

v0.8의 `audio_assets_manifest.json` SHA와 동일한 다음 기존 파일을 `results/applio_v08/applio_copy/`에서 읽는다. 파일을 새로 생성하거나 다운로드하지 않는다.

| 항목 | 연구 사본 내부 경로 | 크기 |
|---|---|---:|
| KLM-SVM | `rvc/models/embedders/embedders_custom/KLM-SVM/pytorch_model.bin` | 378,346,807 bytes |
| RMVPE | `rvc/models/predictors/rmvpe.pt` | 181,184,272 bytes |
| 기존 검색 index | `logs/voice_model_A/voice_model_A.index` | 66,515,819 bytes |
| 원본 네 문장 | `lab/tts_carrier/out3/scene/s0.wav`–`s3.wav` | 기존 SHA 유지 |
| 문장 텍스트 | `lab/tts_carrier/out/sentences.json` | 기존 SHA 유지 |

네 문장은 다음과 같다.

1. 선생님, 오늘 당번은 나야. 필요한 거 있으면 말해줘.
2. 닭꼬치, 하나만 부탁해. 소금에 구운 것으로.
3. ......괜찮아. 조금 놀랐을 뿐이야. 신경 쓰지 않아도 돼.
4. 그 서류는 내가 정리해 둘게. 선생님은 잠깐 쉬어.

추론 설정은 v0.8의 기존 `SETTINGS`를 복사해 쓴다. pitch=2, f0=rmvpe, index_rate=0.3, volume_envelope=0.3, protect=0.33, sid=0, 문장별 seed=997131+i이다. split_audio/autotune/clean/post_process는 꺼져 있고 최종 WAV에 별도 음량 정규화를 추가하지 않는다.

학습 최적화 hook은 추론에 적용하지 않는다. native Conv 메서드의 소속 module이 PyTorch 기본인지 검사하고, AMP와 TunableOp를 끈다. 실제 Hubert/생성기 floating parameter와 생성기 입력/출력 dtype이 FP32인지 확인한다.

## 입력 모델과 엄격한 로드 확인

`--model B=... --model B_prime=...` 두 개가 필요하다. 경로는 다음 중 하나여야 하며 파일은 읽기만 한다.

- 연구 `results/v10/` 아래의 `.pth`.
- 운영 `applio-lab/logs/v10_*/` 바로 아래의 새 모델 `.pth`.

기존 모델 이름에 cleanup을 적용하는 기능은 없다. 두 arm에 같은 파일 경로를 주는 것도 거부한다.

native `VoiceConverter.get_vc`가 실제로 호출하는 `Synthesizer.load_state_dict`를 관찰한다. 대상 inference model에 정확히 한 번 로드되어야 하고 missing/unexpected key가 모두 0이어야 한다. native 호출은 `strict=False`이지만 결과를 이 기준으로 검사하므로 누락을 허용하지 않는다. 로드 인자나 model state를 바꾸는 보정은 하지 않는다.

기본 metadata 조건은 epoch=100, step=3100, 40 kHz, HiFi-GAN, v2, pitch guidance, 1 speaker, KLM-SVM이다. 실행 전후 모델 SHA도 같아야 한다. 원래 모델 경로를 직접 읽으므로 운영 export를 복사하며 실수로 다른 파일을 선택하는 단계를 추가하지 않는다.

## 동일 입력과 품질 판정

Python/NumPy/PyTorch CPU·GPU seed를 각 문장 전에 동일하게 재설정한다. native 생성기 호출 직전에 다음 값도 기록한다.

- 특징, 길이, coarse/fine pitch, speaker 입력 tensor의 shape/dtype/SHA.
- PyTorch CPU·GPU, NumPy, Python 난수 상태의 SHA.
- 각 문장이 나뉜 chunk 수와 호출 순서.

두 모델에서 이 입력·난수 상태와 seed가 같아야 청취 파일을 유효하게 만든다. 파형이 달라도 입력이나 난수 차이 때문인 비교가 섞이지 않게 하는 조건이다.

native decoder tanh 직후와 최종 WAV의 finite 여부를 모두 검사한다. peak/RMS/crest, 상대 peak 97%의 연속 구간, 파형 RMSE·NRMSE·SNR은 참고로만 기록한다. 시간 정렬이나 길이 잘라 맞추기는 하지 않는다. 이번 두 arm만으로 A/A′ 반복 학습 바닥을 새로 측정했다고 주장하지 않는다. 최종 음질 채택은 사용자의 익명 청취 판정이다.

## 익명 자료와 배포 범위

실행마다 한 번 무작위로 A/B 라벨을 정하고 네 문장 전체에 같은 대응을 유지한다. 출력 폴더가 이미 있으면 덮어쓰기나 재추첨을 하지 않는다.

| 파일 | 역할 | 사용자 청취 전달 |
|---|---|---|
| `LISTEN_BLIND.html` | 네 문장 × 두 모델, WAV 내장, 판정/메모 JSON 저장 | 전달 |
| `blind/sentence_01_A.wav` 등 8개 | 익명 WAV | 필요 시 전달 |
| `blind_manifest.json` | 익명 파일명/SHA와 문장 | 전달 가능 |
| `blind_key.json` | A/B와 실제 B/B′ 대응 | 청취 판정 전 노출하지 않음 |
| `result.json`, `preflight.json`, `raw/`, `inference.log` | 모델 경로, 원자료, 기술 확인 | 공개 청취 묶음에서 제외 |

`blind_key.json`과 기술 JSON/log는 mode 0600, raw 폴더는 0700으로 저장한다. 같은 사용자 계정에서 파일을 직접 열 수 없도록 차단한다는 뜻은 아니다. 공개 HTML에는 대응표, 원본 model path, training arm 또는 기술 result 내용을 넣지 않는다.

기술 검사가 끝나도 `quality_judgment`는 `pending_user_blind_listening`으로 남는다. `valid_for_listening=true`는 비교 자료의 기술적 유효성이며 음질 합격 판정이 아니다.

## 재현 명령

소재와 보호 상태 확인만 하는 CPU 명령:

```bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
python3 -B v10_audio_eval.py --preflight
```

두 100 epoch 운영 export가 생긴 뒤에만 실행한다. 아래 model path는 실제 완료 파일명으로 바꾼다.

```bash
~/DevSpace/applio-lab/.venv-rocm10/bin/python -B \
  ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops/v10_audio_eval.py \
  --execute \
  --model B=~/DevSpace/applio-lab/logs/v10_B100_NAME/v10_B100_NAME_100e_3100s.pth \
  --model B_prime=~/DevSpace/applio-lab/logs/v10_Bprime100_NAME/v10_Bprime100_NAME_100e_3100s.pth \
  --db-seed ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops/results/applio_v08/solver_selected_final_db \
  --out ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops/results/v10/audio_eval \
  --seed 997131
```

GPU lease는 스크립트 내부 `GpuAudit`가 잡는다. 외부에서 같은 flock을 중복 획득해 감싸지 않는다. 다른 GPU 작업이 끝난 뒤 실행한다. MIOpen DB/cache는 새 출력 폴더에 만들며 `~/.config/miopen`은 바꾸지 않는다.

이미 만들어진 익명 WAV와 manifest로 HTML만 다시 만드는 CPU 명령은 `--render-only --out ...`이다. 이 경로는 대응표를 읽거나 라벨을 재추첨하지 않는다.

보호 검사는 옛 v0.8 SHA가 아닌 `results/v10/protected_before.json`을 기준으로 실행 전후에 수행한다. v0.8 helper는 기존 `v08_audio_eval.py`의 표준 라이브러리 함수, native load 관찰, 파형 진단만 재사용하며 그 source SHA를 결과에 기록한다.
