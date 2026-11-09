# RDNA2 연구 v0.8 지시문: Applio RVC 학습 해체·최적화 (GPT Pro)

작성: 2026-11-07, Claude. 사용자 요청.

## 0. 왜 하는가

Applio는 사용자가 AI를 처음 시작한 프로그램이다. 쓸모를 따지는 과제가 아니라, 이 PC에서 할 수 있는 만큼 끝까지 빠르게 만드는 과제다. 기준 작업은 사용자의 목소리 학습 `voice_model_A`이다(231조각, 40k HiFi-GAN, KLM49 HFG pretrain, fp16, 배치 8, 31 step/epoch). 이 PC에서 100 epoch에 약 35–38분이 걸린다(2026-09-28 기록). 사용자가 본 다른 사용자의 RTX 5070(12 GB)은 같은 작업이 약 12분이었다고 한다(외부 사용자 보고, 미검증).

목표는 "이 PC의 같은 학습을 같은 품질로 얼마나 줄일 수 있는가"를 끝까지 밝히는 것이다. 5070과의 차이 중 줄일 수 없는 부분(행렬 유닛 부재)과 줄일 수 있는 부분(라이브러리 선택, 우회로, 합칠 수 있는 연산)을 나눠서 보고한다.

## 1. 이미 확인한 것 (다시 조사하지 않아도 됨)

### 1.1 환경

- 장치: RX 6900 XT, gfx1030. WMMA·FP8·BF16 dot 없음. dot2/dot4/dot8 있음.
- 학습 런타임: `~/DevSpace/applio-lab/.venv-rocm10/bin/python`, PyTorch 2.11.0+rocm10.0.0, Triton 3.8.0, MIOpen DB key 3_6_0_8d1ae90e. CK grouped conv 라이브러리는 gfx1030용이 없다(로그 경고). hipBLASLt는 gfx1030 미지원.
- 운영 기본값: `MIOPEN_FIND_MODE=FAST`, fp16 AMP(가중치·AdamW 상태 FP32), 배치 8, `cudnn.benchmark=True`.
- 2026-11-07부터 `run-applio.sh`, `lab/train_bench.sh`에 MIOpen 안전 변수 두 개가 기본으로 들어갔다: `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0`(FP16 forward 오답 solver 차단), `MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53=0`(FP32 dW NaN solver 차단). FAST 4 epoch에서 속도 변화 없음(9.337 → 9.359 s), NaN 0. 옛 이름 `..._OCL_WRW53`은 무효다.

### 1.2 GPU 시간 분해 (2026-09-27, fp16 b8 FAST, step 20–24, `lab/results/profile_base_fp16_b8.txt`)

5 step GPU 시간 2.96 s(0.59 s/step). epoch 9.3 s ≈ 16 step × 0.59 s라 GPU가 병목이다.

| 연산(연산 행) | 비중 |
|---|---:|
| `aten::convolution_backward` | 56.0% |
| `aten::miopen_convolution`(forward) | 25.2% |
| AdamW step | 5.6% |
| weight_norm | 1.2% |

| 커널(위와 겹침) | 비중 | 의미 |
|---|---:|---|
| `miopenSp3AsmConv_v30_3_1_gfx10_fp16_dot2_f2x3/f3x2` | 19.0% + 11.8% | Winograd, dot2 |
| `Im2d2Col_v2` + `Col2Im2dU` | 10.1% + 3.3% | im2col 우회로(펼치기·접기) |
| `Cijk_..._HHS_...` rocBLAS/Tensile GEMM 여러 개 | 약 25% | im2col 뒤 GEMM |

상위 shape는 HiFi-GAN ResBlock Conv1d다(채널 적고 길이 긺, 배치 8).

| input | weight | 방향 포함 시간(30회) |
|---|---|---:|
| [8,64,6400] | [64,64,7] | 141.5 ms |
| [8,64,6400] | [64,64,11] | 114.8 ms |
| [8,32,12800] | [32,32,3] | 91.3 ms |
| [8,128,3200] | [128,128,11] | 82.2 ms |
| [8,64,6400] | [64,64,3] | 81.1 ms |
| [8,32,12800] | [32,32,11] | 60.9 ms |
| [8,128,3200] | [128,128,7] | 60.2 ms |
| [8,32,12800] | [32,32,7] | 56.2 ms |
| [8,384,298] | [384,192,5] | 41.9 ms(56회) |
| [8,128,3200] | [128,128,3] | 39.8 ms |

dilation 1/3/5 변형, MRF ResBlock 구조, ConvTranspose1D 업샘플, period discriminator Conv2D([8,512,95,5]×[1024,512,5,1] 등)가 함께 있다. 전체 표는 위 원본 파일의 `by shape` 절에 있다. 현재 환경으로 다시 프로파일해 최신 분해를 먼저 만든다.

### 1.3 이미 시도한 것과 결과

| 시도 | 결과 | 근거 |
|---|---|---|
| fp16 AMP | fp32 대비 epoch 1.5배, VRAM −1.8 GiB | `applio-lab/lab/RESULTS_2026-09-27.md` |
| 배치 16 | 샘플당 더 느림(초당 13 → 10샘플) | 같은 파일 |
| `--cache-data-in-gpu` | 무이득 | 같은 파일 |
| dilation fold(일괄) | GPU 시간 −6%, 12 epoch 실측 −1%(노이즈) | 같은 파일. v0.6 18 shape 합계 1.004× |
| `torch.compile`(dec/d/enc 조합) | GPU 시간 최대 −10%, epoch 실측 미확정, 첫 epoch +15–30 s | 같은 파일 |
| d1 12 shape NORMAL seed → FAST replay(필터 DB) | 연산 1.811×, 실제 4 epoch 1.106×(V4R1 차단 변수 방식 1.092×). 학습 음성 동등성 미확정이라 운영 보류 | `rdna2-ops/results/conv_catalog_v06/CONV_CATALOG_V06_KO.md`, `REVIEW_V06_CLAUDE_KO.md` |
| 무필터 NORMAL seed | 4 shape forward 오답(V4R1). 차단 변수로 해결 | 같은 문서 |
| selective fold(C64/T6400/k7·11/d3·5) | 필터 DB 위 추가 1.013× | 같은 문서 |
| ConvTranspose1D FindDb 추가 | 연산 1.09–1.30×, 모델 0.998× | 같은 문서 |
| period Conv2D NORMAL | 2/3 shape 오답(V4R1과 같은 원인) | 같은 문서 |
| 직접 Triton GEMM(다른 과제) | rocBLAS 대비 4–6배 느림 | 가이드 §9.3 |

## 2. 이번 범위 (우선순위 순)

### 2.1 최신 분해와 상한 계산

- 현재 런타임·안전 변수·FAST로 `voice_model_A` 학습 몇 step을 다시 프로파일한다. 연산 행과 커널 행을 나눠 집계한다(중복 합산 금지).
- shape × 방향(forward / dX / dW)별로 MIOpen이 실제로 고른 solver와 시간을 표로 만든다(`MIOPEN_LOG_LEVEL`·`MIOPEN_ENABLE_LOGGING` 등은 연구 프로세스에서만).
- 각 shape의 이론 하한(FLOP / dot2 FP16 실측 처리량, 바이트 / 메모리 대역폭)과 현재 시간의 비를 계산해 "어디가 하드웨어 한계에 가깝고 어디가 아닌가"를 먼저 정한다. 이 표가 이후 과제 순서를 정한다.

### 2.2 im2col 우회로 제거: 직접 Conv1d 커널

- im2col + GEMM으로 가는 shape·방향에 대해, 펼치기 없이 계산하는 Conv1d 커널(Triton implicit GEMM 또는 HIP)을 forward / dX / dW 각각 설계한다. FP16 입력, FP32 누산, `tl.dot`이 `v_dot2c_f32_f16`으로 내려가는지 ISA로 확인한다.
- dilation 1/3/5, 커널 3/7/11, 채널 32/64/128/192/384, 길이 298–12800, 배치 8을 포함한다.
- 기준선은 shape·방향마다 (a) 현재 FAST, (b) 필터 seeded DB 중 빠른 쪽이다. 기준선을 이기는 shape만 골라 쓰는 선택 dispatch를 만든다.
- Winograd(Sp3AsmConv)가 이미 잘 하는 shape는 건드리지 않는 것이 기본이다. 이기는 근거가 있으면 예외로 한다.

### 2.3 solver 선택의 완성

- 필터 DB 방식을 dX·dW까지 넓히고, shape·방향별로 가장 빠르면서 정확한 solver를 고른 DB를 만든다. 오답 solver(V4R1 Fwd, WrW53 등)를 걸러내는 자동 검증 절차를 함께 만든다.
- MIOpen FindMode는 프로세스 처음 한 번만 읽힌다(소스 확인). Conv마다 환경변수를 바꾸지 않는다. seed는 NORMAL 별도 프로세스, 학습은 FAST 새 프로세스에서 그 DB를 읽는 방식이다.

### 2.4 conv 밖의 남은 시간

- AdamW(5.6%): `fused`·`foreach` 경로, 상태 dtype, 단일 커널 융합. 수치 동등성(같은 step의 가중치) 확인.
- weight_norm·leaky_relu·add 등 원소별 연산: conv 앞뒤와 합칠 수 있는지.
- 데이터 로더·host 대기·GradScaler 동기화 등 GPU가 노는 구간이 있는지(프로파일 trace의 빈 구간).
- `torch.compile`은 eager 이득과 분리해 판정한다. 첫 epoch 비용을 포함해 100 epoch 전체로 본다.

### 2.5 그 밖의 아이디어

- 위에 없는 방법도 근거가 있으면 시도한다. 단, 학습 의미(손실 함수, 배치, segment 길이, 학습률, epoch 수, 데이터)를 바꾸는 방법은 "다른 학습"이므로 별도 표로 분리하고 채택 후보로 섞지 않는다.

## 3. 판정 기준

- **연산:** 고정 seed 입력에서 y / dX / dW 모두 finite, FP32 참조 대비 NRMSE ≤ 0.002, max error / reference RMS ≤ 0.02(v0.6과 같음). 결과를 보기 전에 정하고 완화하지 않는다.
- **epoch:** `rdna2-ops/applio_epoch_v06.py`로 같은 세션에서 기준과 후보를 교대 실행. steady epoch 중앙값, NaN·scale 감소 횟수, loss 중앙값 상대 차, 보호 파일 16개 전후 SHA.
- **최종(채택 조건):** `voice_model_A` 100 epoch를 3회 학습한다. A(현재 운영), A′(현재 운영 반복, 원래 흔들림 크기), B(후보 전체). 같은 입력 문장 몇 개를 세 모델로 변환해 이름을 가린 A/B/C로 사용자에게 들려준다. 판정 기준은 지각 기준이 기본이다(카탈로그 §1): B만 이상하게 들리지 않으면 통과. 객관 지표(마지막 25% 구간 mel/fm/kl/adv 중앙값, 변환 결과 포화 지표)는 참고로 함께 낸다. A와 A′의 차이를 넘는 B의 차이만 의미 있는 차이로 본다.
- 100 epoch 3회는 약 2시간 GPU를 쓴다. 시작 전에 사용자에게 시간을 알린다. 학습 체크포인트·데이터는 연구 폴더에 복사본으로 만든다.

## 4. 지켜야 할 것

- Applio 운영 파일을 바꾸지 않는다: `run-applio.sh`, `lab/train_bench.sh`, `assets/config.json`, `rvc/` 소스, `logs/<모델>`, pretrained 가중치. 보호 16개의 전후 SHA를 기록한다. `--cleanup`이 붙은 기존 모델 이름으로 학습하지 않는다(기존 결과 삭제 위험).
- 작업은 `~/DevSpace/rdna2-pro-research-20261102/` 아래에서 한다. 필요하면 Applio 소스를 연구 폴더로 복사해 opt-in 패치(`APPLIO_RDNA2_*` 환경변수, 기본 꺼짐)로 만든다.
- 새 패키지는 운영 venv에 설치하지 않는다. Triton 3.8을 유지한다. 100 GB 미만 다운로드는 허용(받은 것의 파일명·출처·크기를 보고에 적는다).
- MIOpen 안전 변수 두 개를 모든 연구 프로세스에 지정한다.
- GPU 작업은 직렬로 하고 `gpu_guard`를 쓴다. 겹친 측정은 폐기한다.
- 연산 성공, epoch 성공, 100 epoch 품질 성공을 다른 판정으로 남긴다.

## 5. 산출물

- 한국어 보고서 `rdna2-ops/results/applio_v08/APPLIO_V08_KO.md`: 최신 분해, shape·방향별 solver 표, 하드웨어 한계 대비 표, 성공·실패·미검증 칸.
- 재사용 코드(opt-in)와 적용 방법, 롤백 방법.
- 카탈로그 §4.3 Conv 행과 §4.7 학습 행, 가이드 §9.14 이후 근거 절 갱신안.
- 끝 보고에 **Claude가 재현할 핵심 숫자 3–5개**를 명령과 함께 붙인다.
- 마지막에 "5070 12분과의 차이 중 줄일 수 없는 몫과 줄인 몫"을 한 문단으로 정리한다.

같이 힘내자 GPT.
