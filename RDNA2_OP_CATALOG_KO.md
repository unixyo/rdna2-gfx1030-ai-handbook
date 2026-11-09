# RX 6900 XT (gfx1030) 연산 카탈로그

작성: 2026-11-10 KST, v0.8 / 실험 묶음 rdna2-ops v1.2(FSR 4.1.2 INT8 provider 확인과 exact 셰이더 추가 후보, 가이드 §9.23, 아래 §4.8). 이전: v0.7 / rdna2-ops v1.1(Applio B+native 채택, 가이드 §9.22). 이전: v0.6 / rdna2-ops v1.0(v0.9 + GPT Pro v1.0 연구: Applio 융합·남은 Conv 교체, TunableOp 비결정 원인, Strata 재개 판단. 가이드 §9.22·§9.20·§9.18). **Applio B′는 익명 청취에서 4문장 중 3문장 기존 B 선호·B′ 쪽 아티팩트 지적, 1문장 같음(2026-11-09 사용자)으로 보류했다.** 이전: v0.5 / rdna2-ops v0.9(Applio 학습 v0.8 연구·운영 롤아웃, 가이드 §9.21). 그 이전: v0.4 / rdna2-ops v0.8(TunableOp·FlashAttention·판정 기준 변경). 이 문서는 RDNA2 최적화 통합집의 **입구**다. 모델이 아니라 연산 부품 단위로 "되는가, 얼마나 빠른가, 원본과 얼마나 다른가"를 기록한다. 세부 실험과 레시피는 [`rdna2-ops/RDNA2_OPTIMIZATION_GUIDE_KO.md`](rdna2-ops/RDNA2_OPTIMIZATION_GUIDE_KO.md)(이하 가이드)에 있고, 이 표는 그 결과를 연산별로 다시 묶는다.

## 0. 사용법

1. 새 모델(LLM·디퓨전·GAN 등)을 받으면 가이드 §2의 절차로 프로파일하고, 실제로 호출되는 연산·dtype·shape·백엔드 함수를 확인한다.
2. 상위 비용 연산을 아래 표의 행에서 찾는다.
3. 채워진 칸(✅)은 해당 레시피를 적용한다. 그 모델에서도 최종 출력 A/B를 다시 한다. 빈칸(⬜)이나 미검증(❔)은 다음 연구 과제다.
4. 새 결과는 이 표의 칸을 갱신하고 가이드에 근거 절을 추가한다.

### 0.1 새 모델 흐름

```text
새 모델
 ├─ 1. 실제 계산 dtype 확인
 │     ComfyUI는 RDNA2에서 BF16을 끈다(should_use_bf16=False).
 │     FP16을 지원하는 모델은 FP16, 아니면 FP32로 돈다(MOSS·Qwen은 FP32, Anima는 FP16).
 ├─ 2. 프로파일 (가이드 §2) → 상위 비용 연산 3개와 그 dtype·shape·layout
 ├─ 3. 각 연산을 §4에서 찾는다
 │     ├─ ✅  레시피 적용 → 같은 seed·입력으로 최종 출력 A/B (엄격 기준)
 │     ├─ ⚠️  속도 우선일 때만 후보. 엄격 실패면 지각 평가를 따로 받는다
 │     └─ ❔/⬜ §7 최소 실험 → 결과를 칸에 기록
 └─ 4. 적용 전 가이드 §10 체크리스트에서 해당 항목만 확인
```

### 0.2 먼저 확인할 공통 설정

| 설정 | 언제 | 근거 |
|---|---|---|
| Triton 3.8 고정 | 항상. 3.9부터 RDNA2 제거 | §3 |
| `MIOPEN_FIND_MODE=FAST` | Conv가 많은 학습·추론의 기본. 기본 find는 첫 호출에서 느린 solver를 고를 수 있다 | §4.3 |
| `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0` | NORMAL을 쓰거나 NORMAL로 FindDb를 시드할 때. FAST라도 NORMAL로 만든 DB를 재생하면 해당한다 | §4.3, [검수 §2.1](REVIEW_V06_CLAUDE_KO.md). FP16 forward 오답 차단 |
| `MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53=0` | FP32 Conv backward를 NORMAL로 찾을 때, FP32 정확도 참조를 만들 때 | §4.3, [검수 §2.3](REVIEW_V06_CLAUDE_KO.md). dW NaN 차단. 옛 이름 `..._OCL_WRW53`은 무효 |
| `torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)` | ComfyUI 기본값. FP16 4D Attention 경로의 전제 | §4.2 |
| 새 커널 경로 prewarm | 서버형 실행에서 첫 요청 성능이 중요할 때. 실제 shape로 몇 회 미리 실행 | §4.2, [검수 §2.2](REVIEW_V06_CLAUDE_KO.md) |
| 학습 중 GPU를 잡은 프로세스 수 확인 | spawn DataLoader 워커를 쓰는 학습. 워커가 학습 스크립트를 다시 import하며 GPU 큐를 만들면 학습 커널이 전부 느려진다(Applio 22 → 15 s/epoch). `/sys/class/kfd/kfd/proc/*/queues`로 센다 | §4.7, 가이드 §9.21 |
| 모델 A/B는 같은 프로세스 안에서 | 항상. MOSS는 텍스트 인코더 출력이 프로세스마다 달라 native끼리도 RGB RMSE 30/255 안팎이 갈라진다. 이전 실행 이미지와의 바이트 비교는 무효 | [검수 v0.7 §2.1](REVIEW_V07_CLAUDE_KO.md). Qwen·Anima는 프로세스 간 결정적 확인 |

환경변수와 prewarm은 측정한 런타임(PyTorch 2.11/2.12, ROCm 7.2/10.0/10.2)에서 확인했다. 운영 런처에 들어간 것은 §6에 모델별로 적는다(Applio는 2026-11-08부터 FAST·두 차단 변수·v0.8 묶음이 기본).

### 0.3 찾는 키

표를 찾는 키는 **연산 → 실제 계산 dtype → batch/M/N/K 또는 공간 shape → stride/layout → bias·scale·양자화 경계 → forward/backward → offload/캐시 조건**이다. 같은 모델 이름이어도 이 키가 다르면 다른 칸의 검증이 필요하다. 반대로 다른 모델이어도 키가 같으면 해당 레시피를 첫 비교 후보로 재사용한다. 모델 이름은 최종 출력 검증 사례이며 자동 dispatch 조건이 아니다.

## 1. 칸 기록 규칙

| 필드 | 값 |
|---|---|
| 상태 | ✅ 측정 범위에서 채택 가능 / ⚠️ 동작하지만 속도 또는 품질 실패 / ❌ 불가 / ❔ 검증·판정 일부 미완료(연산만 통과·모델 미검증 등) / ⬜ 미시험 |
| 정확도 등급 | **비트 일치** / **수학적 동일**(반올림 순서만 다름) / **근사**(값이 실제로 달라짐) |
| 속도 | 기준선 대비 배속. 기준선은 그 dtype·shape에서 실제로 쓰는 기존 경로(예: rocBLAS, native math) |
| 증거 | A 실측 / B 소스 확인 / C 공식 문서. 가이드 절 또는 결과 파일 |
| 판정 기준 | **기본은 지각 기준**(2026-11-07 사용자 결정): 같은 seed의 원본과 나란히 놓고 사용자 눈에 차이가 없으면 채택 가능. 엄격 기준(최종 RGB 최대 2/255, RMSE 0.25/255, 또는 비트 일치)은 참고 수치로 계속 기록하고, 재현·회귀 검사처럼 바이트가 같아야 하는 용도에서만 채택 조건으로 쓴다. 인물 복제·구도 붕괴처럼 눈에 보이는 결함은 속도와 관계없이 실패다. 오디오·학습 등은 해당 출력 계약을 별도로 먼저 정한다 |

## 2. 하드웨어 기능표 (우회 가능성의 출발점)

| 기능 | gfx1030 | 대체 방향 | 근거 |
|---|---|---|---|
| WMMA/MFMA 행렬 코어 | **없음** | wave32 SIMT 타일링 + dot2/dot4 | C, [CAPABILITY_MATRIX](CAPABILITY_MATRIX.md) |
| FP8 연산 | **없음** | FP8 저장값을 FP16/INT8로 풀어서 계산 | C |
| FP4/MXFP4/NVFP4 가속 | **없음** | 직접 unpack + 그룹 스케일 | C |
| BF16 dot | **없음**. BF16 dtype 연산 자체는 기존 라이브러리로 실행 가능 | native BBS와 FP32 계산 비교, 범위·출력 오차가 허용되면 FP16 재표현 | C, A(가이드 §9.12). 명령 부재와 API 미지원은 구분 |
| FP32 dot 가속 | **없음**(FMA만) | FP32 커널은 레지스터·타일 설계가 병목 | A(가이드 §9.7) |
| FP16 dot2 `v_dot2c_f32_f16` | 있음 | FP16 GEMM·Attention | A(컴파일·ISA 확인) |
| INT8 dot4 `v_dot4c_i32_i8` | 있음 | INT8 GEMM, 양자화 QK | A |
| INT4 dot8 `v_dot8_i32_i4` | 있음 | 선형 INT4 가중치, A8 보존은 nibble 분해·보정 필요 | A(실행·정수 oracle·ISA), 가이드 §9.16. 성능 성공을 뜻하지 않음 |
| sudot4(부호 혼합 INT8) | **없음** | signed로 변환 후 보정 | A(컴파일 오류) |
| LDS 64 KiB/작업그룹 | 물리 한계 | 타일 상한 | C |
| TMA/cp.async | **없음** | 일반 load + barrier, buffer→LDS 일부 | C |

## 3. 라이브러리 공백표 (직접 만들어 메울 대상)

| 라이브러리 | gfx1030 상태 | 현재 대응 | 근거 |
|---|---|---|---|
| PyTorch Flash/Efficient SDPA (CK, AOTriton) | 거절 | math SDPA 또는 직접 tiled 커널 | A(가이드 §9.4) |
| hipBLASLt | PyTorch GEMM 요청은 경고 후 hipBLAS fallback. MIOpen FP32 Conv3D의 FAST miss는 Lt 라이브러리 부재로 실패하는 경우 있음 | 호출자별 경로 확인. Conv3D P2400은 NORMAL에서 다른 정상 커널 선택 확인 | A(가이드 §9.3·§9.14). fallback을 라이브러리 전체의 보장으로 일반화하지 않음 |
| rocWMMA | 근본적 미지원 | SIMT 커널 | C |
| Triton | 3.9부터 RDNA2 제거 | **3.8 고정** | B, C(PR #12042) |
| rocBLAS FP16 입력→FP32 출력 | 느린 HSS fallback 커널 선택 | FP16→FP16 또는 FP32 유지 | A(가이드 §9.8, FP16→FP32 QK 3.96 ms 대 FP16→FP16 0.44 ms) |
| MIOpen CK grouped conv | gfx1030 라이브러리 파일 없음. 일반 MIOpen fallback은 실행 가능 | 5종×3 shape의 실제 fallback과 NORMAL 선택 비교 | A/B, 가이드 §9.14. CK 부재 경고·solver applicability 확인. CK 설치 시의 반사실 성능은 미측정 |

## 4. 연산 카탈로그

### 4.1 행렬곱 (GEMM)

| 연산 | 상태 | 정확도 | 속도 | 비고·근거 |
|---|---|---|---|---|
| FP32 GEMM | ✅ rocBLAS 유지 / ❔ 일부 NT | 수학적 동일, shape별 비트 여부 기록 | N64/K4096은 NN 우세. N128/K8192 NT는 변환 포함 약 1.03배라 자동 적용 근거 부족 | A, 가이드 §9.10. mm 3개+bmm 3개 × 4 layout 측정 |
| FP16 GEMM | ✅ rocBLAS 유지 | 수학적 동일 | 직접 Triton은 rocBLAS보다 4.24–5.83배 느림. 새 tall-skinny mm에서도 NN 유지 | A, 가이드 §9.3·§9.10 |
| PyTorch TunableOp로 rocBLAS 솔루션 재선택 (`PYTORCH_TUNABLEOP_ENABLED=1`, hipBLASLt off) | ⚠️ Anima 속도 이득(지각 기준 채택 후보) / ❌ Qwen·MOSS·Idol 이득 없음 | 수학적 동일, 비트 일치 아님. Anima 최종 RGB RMSE 5.60/255(직접 VT와 같은 수준). **같은 튜닝 파일이라도 프로세스마다 출력이 다르다**(Anima 새 프로세스끼리 max 214·RMSE 6.92/255, 같은 프로세스 안에서는 바이트 일치). Qwen은 같은 프로세스 안에서도 실행마다 다르다 | Anima 30단계: SDPA math 81.0 → 60.85 s(1.33배), 직접 VT와 함께 45.18 s(native 대비 1.79배). 튜닝 파일을 새 프로세스에서 `TUNING=0`으로 재사용해도 60.87 / 45.20 s. 첫 튜닝 1회 262 s. Qwen 960×640 25단계: 57.06 → 56.4 s(1.01배), split_alpha 위에서 51.65 → 50.7 s(1.02배). MOSS 4단계: 0.99배, INT8 융합 위 1.02배, 첫 튜닝에서 GPU 메모리 오류(`PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE=0`으로 회피, 1회) | A, 가이드 §9.20, [딥리서치 통합 §5·§6](DEEP_RESEARCH_MERGED_20261107_KO.md). 외부 근거: RX 6900 XT FLUX 7.3 → 3.85 s/it(독립 사용자). 튜닝 파일은 shape별, ROCm/PyTorch 버전이 바뀌면 다시 튜닝. 같은 seed 재현이 필요한 용도에는 쓰지 않는다 **v1.0 원인**: 선택된 FP32 GSU4 커널의 atomic 누산(`buffer_atomic_cmpswap`)이 고정 입력에서도 비결정. 해당 행만 Default로 바꾼 연구 필터로 Anima 새 2프로세스 6장 동일·1.789배 유지, Qwen 같은 프로세스 3장 동일(+1.32%). Half GSU4는 atomic 없이 결정적. 운영 CSV 미변경, 가이드 §9.20 |
| FP16/FP32 bmm·mm 레이아웃 선택 | ⚠️ 속도 이득·strict 실패 / ❔ 일부 mm 모델 미검증 | 48 조합 native/FP64 허용 오차 통과, layout 변경의 비트 일치는 별도 | FP16 B32/M4096/N128/K4096은 NT 준비 22.632→6.570 ms, 복사 포함 27.654→7.091 ms. 같은 FP32 및 Qwen FP32는 NT 퇴보 | A, 가이드 §9.10·§9.11, [48 조합·실제 kernel 표](rdna2-ops/results/LAYOUT_BLAS_V06_KO.md). Anima NN→VT만 바꿔도 최종 RGB strict 실패. TN은 A 전치 포함 최대 300 ms 대로 거절 |
| BF16 GEMM → FP32 계산 → BF16 | ⚠️ 실제 모델 가속 미채택 | 3 shape 허용 오차 통과; MOSS VAE strict 통과, 실험 Anima BF16 Linear strict 실패 | 단위 변환 포함 1.022–1.618배. VAE 일반 1.009/분할 0.997배; Anima 2 step denoise 1.663배·품질 실패 | A, 가이드 §9.12·[보고서](rdna2-ops/BF16_GEMM_V06_KO.md). native BBS/ISA000/WS64 → FP32 SB/ISA1030/WS32. bias·flatten 변화도 포함한 실패 |
| BF16 GEMM → FP16 재표현 | ❔ 조건부 연산 후보, 모델 채택 없음 | 입력 tiny 값 손실·출력 이중 반올림. exact/finite guard 구분 | FP16 출력 1.215–2.022배는 검사 제외. 입력 finite 검사만 0.157–0.478 ms. FP32 출력 0.914–0.961배 | A, 가이드 §9.12. 입력 finite만으로 GEMM의 FP16 출력 overflow를 배제할 수 없음 |
| BF16 직접 Triton dot2 / FP32 SIMT | ❔ dot2 tall-skinny만 / ⚠️ 나머지 거절 | 3 shape·독립 FP64 표본 허용 오차 통과 | dot2 1.426배(tall)·0.838–0.949배(나머지); SIMT 0.296–0.442배 | A, 가이드 §9.12. dot2는 FP16 재표현 guard 제외 시간, exact 조건 별도. ISA·spill 0 확인, 미측정 모델로 자동 적용하지 않음 |
| FP8 E4M3fn/E5M2 weight → FP16+rocBLAS | ⚠️ 새 cache 미채택 / ❔ E5M2 모델 | decoder 256 pattern 정확; 6 GEMM 값 일치; E4M3fn 최종 byte 일치 | prepared 단위 1.015–1.196배. 실제 30 step denoise 79.534→80.681 s(0.986배), cache+4.116 GB. staging 생략 후속 2 step도 0.997배 | A, 가이드 §9.13·[보고서](rdna2-ops/FP8_WEIGHT_GEMM_V06_KO.md). plain unscaled만; 같은 FP8 값의 계산 최적화와 원본→FP8 손실 분리 |
| FP8 tile decode+FP16 dot2 융합 | ⚠️ 교체 거절 | 6 shape 수치 gate 통과, spill 0 | 0.282–0.432배 | A, 가이드 §9.13. 작은 타일·decode·VGPR 비용 해석, 원인별 시간 분해는 미실행 |
| FP8→INT8 재양자화+dot4 | ⚠️ 모델 속도·품질 실패 | 단위 근사 허용 오차 통과; 실제 RGB strict 실패 | projection만 1.205–1.280배. 실제 2 step denoise 5.39→6.84 s | A, 가이드 §9.13. M8192/N2048/K2048만 연결했어도 실패; MLP/encoder는 단위부터 느림 |
| INT8 GEMM (INT32 출력 + 별도 dequant 백엔드) | ✅ | 비트 일치 | 실제 MOSS 생성 2.28배 | A, 가이드 §9.6. 재현 확인(Claude, 2026-11-07) |
| INT8 GEMM (이미 융합된 patientx) | ⚠️ 교체 거절 | 통과 | 0.86–0.90배 | A, 가이드 §9.6. 기존 커널이 이미 autotune+dequant 융합 |
| INT8 전치 복사 제거(stride-aware 백엔드) | ✅ | 비트 일치 | MOSS 1.49배 | A, 가이드 §6 |
| INT4 가중치 GEMM: W4→INT8+dot4 | ⚠️ 속도·모델 품질 실패 | 기본 324개 계산 gate 통과; 실제 MOSS 같은 W4 참조↔dot4 latent/RGB byte 일치 | g64 9개 shape는 native 대비 0.655–0.994배. g128 sub-K는 이전 W4 커널보다 1.80–2.03배 개선돼도 native 대비 0.646–0.955배 | A/B, 가이드 §9.16. 실제 9개 Linear shape×g32/64/128×sym/aff. MOSS 원본→W4 표현 경로 RGB max 240.37/255·RMSE 45.29/255로 strict 실패. 연산 계산 오차와 표현 변경을 분리 **재현 확인(Claude, 2026-11-07)**: Qwen g128 sub-K 2.02배·native 대비 0.653배, MOSS reference↔dot4 바이트 일치. MOSS의 엄격 실패 크기는 native 재실행 흔들림과 같은 자릿수([검수 v0.7 §2.1](REVIEW_V07_CLAUDE_KO.md)). |
| INT4 가중치 GEMM: A8 보존 dot8 | ⚠️ 교체 거절 | signed nibble 분해·보정으로 A8 값 유지, 정수 oracle 통과 | g64 9개 shape 0.265–0.499배 | A, 가이드 §9.16. dot8 두 번+보정·activation 준비 포함. W4A4 근사와 구분, IQ 코드북에 직접 적용 불가 |
| INT4→FP16 dequant+rocBLAS / dot2 융합 | ⚠️ 교체 거절 | 같은 저장값 계산 gate 통과, 원본→W4 손실은 별도 | 호출당 dequant+rocBLAS 0.390–1.017배, dot2 0.242–0.358배(g64). prepared FP16 초기 승리는 교대 재검증에서 0.931/0.995배 | A/B, 가이드 §9.16. 준비·상주 memory·pack metadata 비용 명시. MOSS/Qwen/Anima 실제 기준선 유지 |
| FP16 입력·FP32 출력 bmm | ⚠️ 성능 교체 거절 | — | 동작하지만 느린 fallback 커널 | A, 가이드 §9.8 |

### 4.2 Attention

| 연산 | 상태 | 정확도 | 속도 | 비고·근거 |
|---|---|---|---|---|
| Dense math (SDPA math, ComfyUI split) | ✅ 기준선 | — | — | |
| FP32 math SDPA 안전 검사 축소 | ✅ | 비트 일치 | MOSS 연산 1.38배, 전체 1.5% | A, 가이드 §7. `[1,24,S,128]`, S 128–4096 |
| 후행 scale을 GEMM alpha로 접기 | ✅ | 비트 일치 | qwen 연산 1.31배, 전체 1.08–1.10배 | A, [리허설](rehearsal_qwen_20261106/REHEARSAL_QWEN_KO.md) |
| q/k/v contiguous 복사 제거(strided bmm) | ⚠️ | 비트 일치 | **0.49배** | A, 리허설. rocBLAS strided 역효과 |
| FP16 tiled(Flash식) | ⚠️ | 수학적 동일 | 연산 1.15–3.58배(H2 합성), **Anima 실제 생성 0.85배(느림)** | A, 가이드 §9.1, `anima-rdna2-lab01/REPORT_ANIMA_FP16_KO.md`. B2·H16·S4096·D128에서 rocBLAS 기반 math보다 느림 |
| FP16 FlashAttention Triton(ROCm/flash-attention 2.8.0 + Aceeton RDNA2 블록 설정) | ⚠️ 단독 이득 작음 / TunableOp와 겹치면 이득 없음 | 수학적 동일. 연산 FP64 대비 RMSE 7.3e-6(native 5.4e-6), Anima 최종 RGB RMSE 6.53/255 | 연산: D128·4096·H24 43.7 → 23.7 ms(저자 23.8 ms 재현), Anima self [2,16,4096,128] BSHD view 64.6 → 33.5 ms, cross K=512는 모델 안 native 3.8 ms보다 느린 5.45 ms. Anima 30단계 80.6 → 68.9 s(1.17배). TunableOp를 켜면 native self가 34.2 ms로 같아져 flash_self+TunableOp 60.0 s(TunableOp 단독 60.95 s) | A, 가이드 §9.20, [딥리서치 통합 §6](DEEP_RESEARCH_MERGED_20261107_KO.md). 설치 없이 소스 `PYTHONPATH`, `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, 최초 컴파일은 prewarm 3 s. FP32 미지원 |
| 긴 시퀀스 FP16 fused attention 교체(Idol joint 12294 토큰, B1·H16·D64) | ✅ Flash(수학적 동일) / ✅ Sage(근사, 지각 후보) | Flash는 FP32 참조 대비 Idol 자체 커널과 같은 RMSE 4.3e-6. 8방향 미리보기 평균 차 0.38(Flash)·0.44(Sage)/255 | 연산 86.3 → 69.2(Flash) / 54.7 ms(Sage). 50단계 전체 288.0 → 230.5 s(1.25배) / 193.0 s(1.49배) | A, 가이드 §9.20, [딥리서치 통합 §6.5](DEEP_RESEARCH_MERGED_20261107_KO.md). 기존 Idol `triton_rdna2`(기본 `sdpa_chunked` 대비 5.20배) 위의 추가 이득. Idol 소스 무수정, 실행기에서 attention 속성만 교체 |
| FP16 SDPA math → 직접 GEMM+softmax, P·V는 V 전치 레이아웃 | ⚠️ 엄격 실패; 기존 보고서 육안 동등 | 수학적 동일, 중간 반올림 다름 | Anima 기존 1.43배, 새 동일서버 재현 **1.417배(80.773→56.988 s)** | A, 가이드 §9.11. RGB max 0.8247/RMSE 0.02431, strict 실패. 독립 지각 평가는 미실행 |
| FP16 Attention sqrt-prescale+VT | ⚠️ 기존 직접 VT 교체 거절 | 중간 오차 감소, 최종 RMSE 증가 | self 연산 1.032배/cross 1.280배, 최종 57.163 s로 추가 이득 없음 | A, 가이드 §9.11, 3 shape·실제 30 step A/B |
| FP16 4D native matmul+작은 softmax row repair | ✅ 검증한 warm 추론 범위 | **3 shape 값 일치·실제 latent/RGB 비트 일치** | 추가 warm 2회 67.775/67.727 s, 같은 서버 native 80.607/80.720 s. 중앙값 1.191배. 최초 83.019 s도 보존 | A, 가이드 §9.11. **재현 확인(Claude, 2026-11-07)**: warm 1.197배, 첫 요청 83.097 s, RGB SHA 동일. 첫 요청 지연은 서버 시작 때 같은 shape로 5회 prewarm(1.07 s)하면 사라짐: 첫 요청 67.87 s([검수 §2.2](REVIEW_V06_CLAUDE_KO.md)). allocator·DynamicVRAM 가설은 기각. reduced math flag=True, unmasked FP16 D128 실측. 초회 비용 별도; 57 s VT 속도와 strict 동시 달성은 아님 |
| P·V bmm 레이아웃(rocBLAS) | ⚠️ 속도 이득·strict 실패 | 수학적 동일, 비트 일치 아님 | FP16 NN→NT 준비 22.632→6.570 ms; 실제 직접 NN 63.724→VT56.988 s | A, 가이드 §9.10·§9.11. layout만의 RGB max 0.6914/RMSE 0.01465. FP32/Qwen/MOSS에 일괄 전파 금지 |
| FP32 tiled(Flash식) | ⚠️ | 수학적 동일 | 0.42–0.50배(느림), 추가 메모리 최대 1/41 | A, 가이드 §9.7. 메모리 부족 시 대안 |
| INT8 QK + FP16 PV (Sage식) | ⚠️ | 근사 | qwen 연산 1.13배 | A, 가이드 §9.9. qwen RGB 오차 0.59로 엄격 기준 실패 |
| 혼합 정밀도(FP16 operand, FP32 경계) | ⚠️ | 근사 | — | A, 가이드 §9.8. INT8 ConvRot 경계에서 오차 약 13배 증폭 |
| Sparge-inspired FP32 block sparse | ⚠️ 계산·성능·품질 조건부 실패 | 근사 | 실제 early target 22.666→54.434 ms로 퇴보; middle 22.704→18.251 ms. selector 포함 | A/B, 가이드 §9.17. compressed CDF .95가 실제 mass .95를 보장하지 않음. middle 평균 mass 82.47%, 하위 1% 21.10%. late same-mask 계산 관문 실패. 공식 Sparge의 INT8/PV 최적화를 그대로 포팅한 것은 아님 |
| 2D image radial r4 FP32 | ⚠️ 모델 적용 제외 | 근사; 같은 mask 계산도 late native tolerance 실패 | early 연산 1.219배, 추가 allocator peak 3102.594→38.000 MiB. Qwen warm denoise 1.09657배(n=1 후보) | A/B, 가이드 §9.17. 실제 5-case·25-step 3 seed. 최종 RGB 최대 240.140–243.441/255, RMSE 48.222–54.020/255, 엄격 3/3 실패. 익명 12/12 native 선호·후보 severity 3, 좌우 양방향 지각 실패. 전체 block map까지 O(S)라고 부르지 않음 **재현 확인(Claude)**: 연산 1.216배·38.0 MiB, Qwen warm 1.096배, RGB·이미지 SHA가 GPT과 동일. 전 단계에 희소화를 걸어 인물이 복제됨([검수 v0.7 §2.2](REVIEW_V07_CLAUDE_KO.md)). |
| 원래 temporal Radial의 f1/f2 조건 | ⚠️ 희소 이득 없음 | 해당 geometry에서 dense | f1/f2에서는 mask가 100% dense | B/C, 가이드 §9.17. 영상에서의 이득 미검증. 위 2D image mask의 성과와 구분 |
| 고정 50% block sparse | ⚠️ 퇴보·품질 근거 부족 | 근사 | early target 22.666→28.454 ms, 실제 density 53.37% | A, 가이드 §9.17. 평균 보존 mass 57.89–89.07%, final 모델 치환 미채택 |
| VSA 등 그 밖의 Sparse | ⬜ | 근사 | 미측정 | 실제 비용·희소성부터 확인한 뒤 §7 형식으로 채움 |
| Veda | ❌ 공식 미지원 | 근사 | | 공식은 RDNA3/4만, 모델별 학습 predictor 필요. B, [ATTENTION_PORTING_MAP](ATTENTION_PORTING_MAP.md) |
| Causal/decode attention(LLM) | ⬜ | | | 외부 llama.cpp RDNA2 자료 있음(snapshots/guide-*, llama-*). 이 PC 미측정 |

### 4.3 합성곱 (Conv)

| 연산 | 상태 | 정확도 | 속도 | 비고 |
|---|---|---|---|---|
| BF16 Conv2D → FP32 계산 → BF16, bias 경계 보존 | ✅ 검증한 VAE 범위 | native bias 재구성 비트 일치; 최종 RGB strict 통과, 후보는 비트 동일 아님 | 3 shape 3.10–3.59배. 실제 VAE normal 일반/분할 2.449/2.430배, lowvram 2.626/2.299배 | A, 가이드 §9.14. groups 1·inference, weight cache없음. 실제 40/920 conv, lowvram=True 확인. [결과](rdna2-ops/results/conv_catalog_v06/bf16_conv2d_vae/results.json). **재현 확인(Claude, 2026-11-07)**: normal 2.464/2.451배, 오차·후보 SHA 동일 |
| Conv2D FP16, Applio period discriminator | ⚠️ NORMAL 교체 거절 | 3 shape 중 2개 forward NRMSE 0.77–0.78로 실패; dX/dW 별도 검증 | fwd+dX+dW는 1.19–1.44배여도 정확도 실패로 미채택 | A, 가이드 §9.14·[15 shape 표](rdna2-ops/results/conv_catalog_v06/CONV_CATALOG_V06_KO.md). FAST는 3 shape 통과. **재현 확인(Claude)**: 독립 입력으로 NRMSE 0.7727/0.7747, ROCm 7.2·10.0·10.2 모두 같음 |
| MIOpen `ConvHipImplicitGemmV4R1Fwd` 끄기 (`MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0`) | ✅ Applio d1 범위 / ❔ 다른 family | period Conv2D 3 shape·Conv1D k7 오답이 NRMSE 0.0002–0.0005로 복구. d1 12 shape y/dX 통과 | d1 12 shape 합 FAST 31.740 → 17.860 ms. 실제 Applio 4 epoch 9.261 → 8.480 s(1.092배, 같은 세션 FAST 대조) | A, [검수 §2.1](REVIEW_V06_CLAUDE_KO.md). ROCm 7.2·10.0·10.2 모두 같은 오답. GPT 필터 FindDb(1.106배)와 같은 원인을 변수로 막음. NORMAL로 DB를 시드한 뒤 FAST 재생. 청감·장기 학습 미확인 |
| MIOpen `ConvOclBwdWrW53` 끄기 (`MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53=0`) | ✅ 오답 차단 | FP32 NORMAL Conv1D C128/T3200/k3 dW NaN → 정상 | 미측정 | A, [검수 §2.3](REVIEW_V06_CLAUDE_KO.md). 이 MIOpen은 `OCL_WRW53` 이름이 무효, `HIP_WRW53`만 동작. FP32 참조·FP32 학습에서 NORMAL을 쓸 때 V4R1 변수와 함께 지정 |
| 일반 FP16 Conv3D | ❔ 연산 검증·모델 미연결 | 합성 3 shape y/dX/dW 허용 오차 통과 | NORMAL은FAST 대비 0.997–1.058배 | A, 가이드 §9.14. 일반 Conv3D를 exact-tile Linear로 바꾸는 규칙 아님 |
| FP32 exact-tile Conv3D → Linear | ❔ 연산 성공·모델 판정 불가 | P128/512/2400 FP64 참조 통과. 실제 Qwen native conditioning부터 불일치해 후보 실행 전 중단 | 정상 NORMAL 기준 2.04/1.98/3.34배. 모델 배속 미측정 | A/B, 가이드 §9.14. P2400 입력[2400,3,2,16,16], W[1152,3,2,16,16]. FAST 빈 DB의 Lt fallback 실패도 보존; 운영 flag0 유지 |
| Grouped Conv1D FP16 | ⚠️ NORMAL 교체 이득 없음 | 실제 Applio 3 shape y/dX/dW 통과 | FAST 대비 0.959–1.002배 | A, 가이드 §9.14. CK 부재에도 fallback 정상 실행 |
| Depthwise Conv1D FP16 | ❔ 연산 성공·모델 미연결 | 합성 3 shape y/dX/dW 통과 | FAST→NORMAL 2.03–5.61배 | A, 가이드 §9.14. 이 PC 모델의 해당 경로 연결 미확인, 전역 NORMAL 권고 아님 |
| ConvTranspose1D FP16 | ⚠️ 연산 이득·실제 epoch 무이득 | 3 shape y/dX/dW 통과. 학습 후 음성 동등성은 native 변동으로 미확정 | 연산 FAST replay 1.09–1.30배. 실제 4 epoch 8.41758→8.43412 s =0.9980배 | A, 가이드 §9.14. 검증한 9개DB key만 추가·3 shape 각 64 hit. [재현 레시피](rdna2-ops/results/conv_catalog_v06/CONVTRANSPOSE_DB_RECIPE_KO.md) |
| ConvTranspose2D/3D | ⬜ | | | 1D 결과를 다른 차원에 전파하지 않음 |
| Conv1d 긴 길이·적은 채널(HiFi-GAN, 오디오) fwd+bwd | ✅ 설정으로 해결 | — | MIOpen 기본 find 366 ms → `MIOPEN_FIND_MODE=FAST` 11.4 ms | A, `applio-lab/lab/RESULTS_2026-09-27.md`. 첫 호출에서 느린 solver 선택 문제 |
| Conv1d fwd+bwd, ROCm 7.2 / 10.0 / 10.2 비교(fp16, Applio shape 18종) | ⚠️ 기존 속도 기록, 정확도 승인 아님 | 새 v0.6 NORMAL seed+FAST replay에서 4 shape forward 큰 오차 | 기존 FAST 합계 70.6/71.5/75.3 ms, NORMAL 308/314/193 ms. 기존 기록의 d1 NORMAL 1.5–2.6배·dilated FAST 3–8배 이득은 정확도와 별도 | A, `conv_miopen_20261107/` 및 가이드 §9.14. 속도만 보고 NORMAL 전역 적용 금지 |
| Conv1D shape별 User FindDb(NORMAL d1 seed→FAST) | ⚠️ 무필터 거절 / ❔ 필터의 장기 학습 품질 | 무필터 4 shape forward 실패. 문제 Fwd entry 제외 후 12 shape output/dX/dW 통과 | d1 합 31.740→17.530 ms. 실제 4 epoch 9.310→8.418 s =1.106배 | A, 가이드 §9.14. native 반복 finalwave RMSE 0.1344가 후보차이 0.1066보다 커 품질 동등성 미확정. 운영 DB 불변 |
| Dilated Conv1d | ⚠️ | — | 비-dilated보다 2~3배 느림(FAST). dilation fold는 fp16 1.3배, 학습 전체는 노이즈 수준 | A, 같은 기록 |
| Dilated Conv1D shape별 phase fold | ❔ 제한 범위·추가 학습 이득 작음 | 18 shape output/dX/dW gate 통과 | blanket합 1.004배. C64/T6400/k7·11/d3·5만 1.59–1.80배; 선택 fold 실제 추가 1.013배 | A, 가이드 §9.14. 나머지 0.53–0.79배인 shape로 전파 금지. 필터DB와 합성한 단일 4 epoch·품질 미확정 |
| MIOpen Winograd solver84 직접 호출(K3 bwd-data) | ⚠️ | 오차 증가 | 미측정(정확도 기준 실패로 중단) | A, `applio-lab/docs/t11_miopen_solver84_20261023` |
| Applio v0.8 shape·방향별 MIOpen DB(운영 FAST DB 기준) | ✅ 지각 통과(묶음 B) | 144 geometry × 3방향 = 432/432 FP32 참조 통과. 실제 solver ID 432/432 확인 | 다시 고른 6방향 1.075–1.530배. 나머지는 기존 선택이 최선 | A, 가이드 §9.21. 빈 DB가 아닌 운영 DB 대비. DB SHA `fa2af4e8…` |
| Applio 직접 Conv1d(Triton, implicit GEMM fwd/dX, FP32 partial dW) | ✅ 지각 통과(묶음 B) | 189/189 후보, 실제 dtype·weight_norm 경계 105/105 통과. bit 동일은 아님 | 21 shape·56방향: fwd 1.70–12.07배, dX 1.31–9.44배, dW 1.37–9.66배. C64/K7/d3 dW 3.98 → 0.68 ms | A, 가이드 §9.21. dilated 소채널에서 큰 이득. C128/K11 dW·C192→384/K5 dW는 native가 빨라 유지. shape는 batch 8·40 kHz 기준 |
| MIOpen dilated GEMM workspace 질의 복구(NORMAL 탐색 전용) | ✅ 탐색 누락 해결 | 같은 solver가 계산. 72/72방향 통과 | 속도 이득으로 세지 않음(신규 채택 0) | A, 가이드 §9.21. `8d1ae90e`의 legacy 질의가 dilation>1에서 0을 반환해 후보가 조용히 탈락 |
| Applio ResBlock LeakyReLU·Conv·잔차 융합(Triton, fwd+bwd) | ⚠️ 연산·운영 100 epoch 통과 / 묶음 B′ 청취 보류, 속도 순이득 거의 0이라 운영에서 끔 | 18쌍 FP32 참조 108/108, 실제 AMP·weight_norm 계약 144/144. 검증 입력의 호출 경계 y/dX/gradient는 B와 바이트 일치(전체 학습 아님) | C32/C64 12쌍 event 1.039–1.135배. 대표 쌍 kernel 28→22, GL2C/EA 요청량 −37.5%(DRAM bytes 아님) | A, 가이드 §9.22. C64/K11은 관문 미달, C128/K11은 더 느림 |
| Applio 남은 Conv 방향별 직접 커널(period Conv2d·ConvTranspose1d·Conv1d) | ✅ 보컬 익명 청취 통과(B+native, 2026-11-10), 운영 적용 | 60 shape·169방향, 후보 531/531 FP32 참조 통과, AMP 계약 195/195 | 31 shape·새 68방향(Conv2d 59, ConvT 7, Conv1d 2). 대표 dW 27.3배·21.7배(작은 batch별 GEMM 17 kernel → 2 kernel) | A, 가이드 §9.22 |
| C128/K11 dW·d1 dX, C192→384/K5 dW, ConvTranspose1d fwd, 큰 채널 period Conv2d | ✅ native 유지 | — | 직접 커널이 관문(event 1.10배·wall 1.05배)을 못 넘음. 하드웨어 최적의 증명은 아님 | A, 가이드 §9.22 |

### 4.4 정규화·활성화·원소별 연산

| 연산 | 상태 | 정확도 | 속도 | 비고 |
|---|---|---|---|---|
| GroupNorm+SiLU 융합 | ✅ eager만 | 수학적 동일(허용 오차로 판정) | 연산 1.07–1.70배, VAE 전체 1–5% | A, 가이드 §4·§5. graph/compile에서는 Triton이 더 빠름 |
| LayerNorm | ⚠️ 모델 strict 실패·교체 거절 | 3 shape·FP64 표본 tolerance 통과. 실제 Anima RGB max 0.8247/RMSE 0.02369 실패 | FP32 D2048/4096 단위 1.122/1.508배; FP16 D768은 0.705배 | A/B, 가이드 §9.15. 실제 2520회 치환. 첫 후보 94.312 s에 초기 준비가 포함될 수 있어 warm 성능으로 단정하지 않음 |
| RMSNorm | ⚠️ 모델 strict 실패·교체 거절 | 3 shape tolerance 통과. 실제 Anima RGB max 0.8203/RMSE 0.02516 실패 | FP16 D128 단위 2.348배, FP32 D4096은 1.490배; 작은[1,37,3072]는 0.338배 | A/B, 가이드 §9.15. 실제 1680회 치환. 이미 융합된 RMS+RoPE 경로는 대체하지 않음 |
| native LayerNorm + modulation·cast 융합 | ✅ 비트 보존·연산 융합 / ⚠️ 모델 이득 작음 | 3 shape·실제 Anima 30 step 2회 latent/RGB 실제 byte 일치 | 단위 1.398–2.649배, 대표 추가 peak 192→96 MiB. 모델 denoise 80.663→80.348 s =1.0039배(n=2) | A/B, 가이드 §9.15. FP16 1+scale·FP32 곱/합 경계 보존. 큰 전체 가속으로 권하지 않음 |
| 통계 포함 adaLN 전체 융합 | ⚠️ 모델 strict 실패·교체 거절 | 3 shape tolerance 통과. 실제 Anima RGB max 0.8091/RMSE 0.02491 실패 | 단위 1.514–8.494배·대표 추가 peak 192→32 MiB, 실제 denoise 79.375 s여도 품질 실패 | A/B, 가이드 §9.15. 실제 2520회 치환. 통계 계산 경로 변경을 비트 보존이라고 부르지 않음 |
| 기존 Comfy Kitchen HIP adaLN/RMS-adaLN 교체 | ❔ 연산 검증·해당 모델 RGB 미검증 | MOSS 1 shape·Qwen prefix/target 2 shape tolerance 통과, 비트 불일치 | 기존 융합 대비 1.188/1.654/2.077배 | A/B, 가이드 §9.15. 여기의 Kitchen은 AMD CK와 다른 패키지. 모델 채택 없음 |
| gated residual 추가 융합 | ⬜ | | | 기존 addcmul/addcmul_·FP32 residual 계약 유지, 새 교체는 미시험 |
| Attention score in-place softmax | ⚠️ 속도 교체 거절, 메모리 후보 | 수학적 동일, 반올림 다름 | self 추가 peak 2.125→1.125 GiB, 시간 22.561→23.087 ms로 퇴보 | A, 가이드 §9.11. 3 shape; 성능 관문 실패로 모델 자동 적용 제외 |
| Softmax 단독(Attention 밖) | ⬜ | | | 별도 모델 호출은 미시험 |
| torch.compile(Triton 3.8) 일반 융합 | ⚠️ 부분 | | GN+SiLU graph에서 HIP보다 2배 빠름 | A, 가이드 §4. 모델 단위 compile 시험 ⬜ |

### 4.5 양자화·메모리 이동

| 연산 | 상태 | 비고 |
|---|---|---|
| 활성화 per-row INT8 양자화 | ⬜ | INT8 모델 공통 전처리 |
| ConvRot(Hadamard) 회전 | ⬜ | MOSS·qwen 공통 |
| 원본 BF16 가중치 → raw unscaled FP8 저장 | ⚠️ Anima strict 실패 | 기존 모델로만 재표현 시험. E4M3fn RGB max 0.94775/RMSE 0.17677. 실제 작은 weight의 subnormal 양자화 손실 관측. scaled FP8 checkpoint로 일반화 금지. A, 가이드 §9.13 |
| Host→Device 가중치 이동(lowvram/DynamicVRAM) | ⬜ | qwen 프로파일 886 ms/8단계 |
| KV prefix cache 복사 | ⬜ | qwen 790 ms/8단계 |

### 4.6 LLM 전용

이 PC의 LLM 주력은 Strata 엔진(v0.1.42 + 패치, Qwen 3.8-Flash-Next MoE, 256K 컨텍스트)이다. IQ3_XXS라는 pack 이름과 실제 layer 타입을 구분한다. v0.7에서 읽은 48개 층의 gate/up 중 IQ3_XXS는 6개이며, down도 IQ4_NL 18개/Q2_0 30개가 섞여 있다. 이번 프리필 결과는 아래 첫 세 행과 가이드 §9.18에 있다. 디코드는 이번 범위에서 제외했다.

나머지 행은 `strata-rdna2-gfx1030-recipe` (`../strata-rdna2-gfx1030-recipe/README.ko.md`, 연구 폴더에만 있음)의 역사적 측정(A, 2026-10, 대부분 2–4회)을 옮긴 것이다. 이번에 다시 잰 수치가 아니다. llama.cpp를 실제 Strata 경로의 기준선으로 대신 쓰지 않는다.

| 연산·설정 | 상태 | 비고 |
|---|---|---|
| IQ3_XXS/IQ4_NL MMQ J 타일 선택(프리필) | ⚠️ 연산 이득·모델 무이득 | 실제 layer 35, 16 expert, 새 3개 분포 × 2개 역할 × 9개 선택의 54행 byte exact. 작은 치우친 분포의 평균 J 힌트 GU 1.090배/down 1.080배; 탐색 최저 J40/J32 1.148배/1.224배는 모델 정책 아님. medium/large는 같은 J64라 이득 없음. A/B, 가이드 §9.18 **재현 확인(Claude, 2026-11-07)**: 54행 바이트 일치, skew GU 1.051배/down 1.098배, J 불변 분포 1.003–1.027배 모델 warm A/B(Claude): 9.7K/19.8K/119K 0.991/0.996/0.991배, 생성 32토큰 ID 일치([검수 v0.7 §2.3](REVIEW_V07_CLAUDE_KO.md)) |
| Strata warm 프리필 병목 분해(8,192 MiB swap 관문 조건) | A | GPU timeline ≈ wall, PCIe 대기 0.6%, 스토리지 읽기 ≈ 0 → GPU 계산 병목. 전문가 GEMM 약 25%, QSA attention 약 15%, GDN 약 14%, HC read 약 12%로 분산. MMQ 커널 단독 개선의 상한은 약 12%(추론). [검수 v0.7 §2.3](REVIEW_V07_CLAUDE_KO.md) |
| Strata warm 프리필 GPU 계산 대 PCIe 병목 | ❔ 자원 차단·미판정 | 첫 fresh 9,716 tokens는 48.774 s, 첫 요청·I/O 오염. 가장 큰 9,711-token chunk의 event timeline에서 dequant 라벨 74.92%는 host copy 발행 대기 포함. 이후 priming 2회 swap 보호 종료로 9.7K/19.8K/119K J off/on과 warm 병목 미측정. A/B, [보고서](rdna2-ops/results/strata_v07/STRATA_PREFILL_V07_KO.md) |
| Strata warm 병목, v1.0 256 MiB swap 관문 | ❔ 자원 조건으로 미판정·서버 미실행 | 2026-11-08 11:47:58 UTC CPU 점검: MemAvailable 35.91 GiB, swap 사용 15.50 GiB, expert cache 0%. 이전 92.78% cache에서도 중단됐으므로 같은/나쁜 조건에서 반복하지 않음. 위 8,192 MiB 조건 행과 합치지 않는다. [STRATA_WARM_KO.md](rdna2-ops/results/v10/STRATA_WARM_KO.md) |
| IQ 코드북 인덱스 → 일반 INT4 dot8 직접 이식 | ❌ 같은 계산 아님 | IQ3_XXS/IQ4_NL의 코드는 비선형 lookup이며 signed nibble 값이 아님. 실제 IQ3 커널은 dot4 사용. §4.1 W4 결과를 그대로 전파하지 않음. B, 가이드 §9.18 |
| MoE 전문가 양자화 IQ3_S → IQ3_XXS | ✅ | 당시 64 GB RAM 조건에서 전문가 약 40 GiB를 page cache에 올린 결과. 현재 다른 앱과의 메모리 경합 여부는 별도 확인. 프리필 1.1–3.5배, 디코드 22–41 → 40–45 tok/s. 답변 품질은 끝까지 측정하지 않음 |
| 프롬프트 FP16 GEMM + rocBLAS 튜닝표 (`STRATA_HIP_PROMPT_F16`, `STRATA_ROCBLAS_TUNING`) | ✅ | gfx1030 프롬프트 경로. recipe §효과 순서 2 |
| 프롬프트 attention #3381 백포트 (`STRATA_PROMPT_ATTN_RDNA2`) | ⚠️ | int8 KV에서 9.7K +8.4%. 최종 k8v4 KV에서는 split 경로로 되돌아가 효과 없음. k8v4용 확장은 attention 단계 24–28% 빨랐으나 요청 전체 무이득에 GPU 메모리 fault 1회로 철회 |
| KV cache 양자화 int8 → k8v4 | ✅ | 같은 속도, 전문가 슬롯 증가로 256K 가능. 256K 디코드는 약 −10% |
| GDN 융합 스위치 (`STRATA_GDN_HEAD/PP/CONVL2/NOY`, `STRATA_CVEC_FUSE`) | ✅ | 프롬프트 길이에 따라 +5–12%. 연산 단독은 64토큰 이상 약 2배, 32토큰 미만은 느림 |
| Decode GPU 커널 묶음 (`MMVF_ROWS`, `Q8_PACKED`, `Q6_PACKED`, `ATTN_LANECELL` 등) | ⚠️ 거절 | 디코드 무이득, warm 프리필 −4~−7%. gfx1151 이득이 옮겨지지 않음. 스위치별이 아니라 묶음으로 측정 |
| 지속형 HC 커널 (`STRATA_HC_PERSIST`) | ⚠️ 보류 | 연산 1.35–2.1배·비트 일치였으나 모델 디코드 이득 반복 안 됨. 상한 약 3% |
| 전문가 상주 라우팅 (`STRATA_ROUTE_RESIDENT=1.0`) | ⚠️ 출력 변경 | 디코드 +8–18%. 모델 출력이 달라지므로 품질 확인 필요 |
| 램 고정·상주 KV 확대 | ❌ | 페이지 캐시를 빼앗아 전부 느려짐. 이 기계의 남은 큰 지렛대는 램·VRAM·새 엔진 |
| 일반 INT4 dot8 GEMV | ⬜ | IQ 형식은 코드북 lookup이라 §4.1 INT4 결과를 그대로 옮기지 않음 |

### 4.7 학습(역전파)

| 항목 | 상태 | 비고 |
|---|---|---|
| Conv backward 비중(v0.8 이전 하네스 프로파일) | A | Applio RVC fp16 b8(voice_model_A 31 step): operator GPU 시간의 55.7%가 conv backward, 24.6%가 conv forward. kernel 기준 im2col+col2im 14.1%, AdamW 5.4% |
| fp16 AMP 학습 | ✅ | fp32 대비 epoch 1.5배 빠름, VRAM 1.8 GiB 감소. GradScaler skip 관찰. 100 epoch 3회에서 nonfinite loss 0 |
| torch.compile(학습) | ⚠️ | 9월 기록 5–10% 이득·첫 epoch +15–30초. v0.8 decoder compile은 AMP 경계 오류 복구 뒤에도 eager보다 이득 없음(가이드 §9.21) |
| BF16 학습 | ⬜ | BF16 dot 없음. 효과 미측정 |
| AdamW fused | ⚠️ 수치 관문 실패 | G/D 2.5–3.0배지만 tensor별 update NRMSE 0.003–0.0037 > 0.002. 허용치 완화 없음 |
| AdamW single-tensor(D만) | ✅ 지각 통과(묶음 B) | D 1.42배, moment·step bit 동일. G는 0.45배라 native 유지 |
| segment gather·기록용 grad norm foreach·epoch 사이 `empty_cache` 생략 | ✅ 지각 통과(묶음 B) | gather y·dX bit 동일, norm 5.19배·gradient 불변 |
| D weight_norm cache | ⚠️ 판정 불가 | 유효한 gradient 참조를 얻지 못해 제외 |
| **Applio v0.8 묶음 B**(위 Conv 두 행 + D single + gather + norm + cache 유지) | ✅ 지각 통과(2026-11-08 사용자, 익명 A/B/C) | 하네스 100 epoch 1540/1545 → 1320 s(14.3–14.5% 단축), steady 14.98 → 12.69 s. 운영 경로 실측은 다음 행과 함께 22 → 12.6 s/epoch. 가이드 §9.21 |
| spawn DataLoader 워커의 GPU 큐 제거 | ✅ 계산 불변 | 운영 train.py를 워커 4개가 다시 import하며 GPU 큐를 잡아(학습 중 6 프로세스·19 큐) 학습 kernel 전체가 1.5–2배 느려졌다. 학습 프로세스가 컨텍스트를 만든 뒤 `HIP_VISIBLE_DEVICES=-1` → 22 → 15 s/epoch. 하네스는 진입점이 달라 이 문제를 피했다. 가이드 §9.21 |
| v0.8 B 적용 뒤 운영 경로의 남은 비용 | A | `core.py train` step 40–44 kernel 1,679.5 ms: native Conv2d 41.7%, native Conv1d 29.4%, 기존 직접 Conv 9.5%, optimizer 4.6%, ConvTranspose 1.8%. operator 표와 분모가 다르다. 가이드 §9.22 |
| **Applio v1.0 B′**(v0.8 B + 위 융합 12쌍 + 남은 Conv 68방향) | ❌ 청취 보류(2026-11-09 사용자: 3/4 문장 기존 B 선호, B′ 쪽 아티팩트) | 운영 경로 교대 8 epoch 12.542 → 11.448 s(1.0956배). 100 epoch 각 1회: 프로세스 1,285.2 → 1,172.2 s(−113.0 s, −8.79%), 3,100 step, 초기 SHA 동일, nonfinite 0. 두 겹의 개별 기여는 미분리. 운영 사본 `rdna2_v10`은 준비됐고 꺼져 있음. 가이드 §9.22 |
| **Applio v1.1 B+native**(v0.8 B + 남은 Conv 68방향, 융합 끔) | ✅ 보컬 익명 청취 구분 불가(2026-11-10 사용자), 운영 적용 | 운영 경로 100 epoch 프로세스 1,285.2 → 1,178.2 s(−8.33%), steady 12.566 → 11.439 s. 초기 SHA 동일, nonfinite 0. 운영 런처 5 epoch 스모크 11 s/epoch·31/31 hit. 가이드 §9.22 |

### 4.8 그래픽 셰이더 INT8 추론 (Vulkan, FSR 4.1.2)

Proton Experimental → vkd3d-proton 3.1.0 → RADV/ACO(Mesa 26.2.4), timing-kit 합성 Native AA 1080p·4K. 시간은 업스케일러 GPU 시간이며 게임 FPS가 아니다. 가이드 §9.23.

| 연산/경로 | 상태 | 비고 |
|---|---|---|
| 무수정 SDK 2.4 DLL의 provider 선택 | ❌ FSR4 아님 | 파일 버전 4.1.2.4555지만 자동 선택은 **FSR 3.1.6**. Navi2 `IsSupported`의 D3DKMT type 0 조회를 Wine이 처리하지 않음. "FSR 켜짐" 표시만으로 FSR4 판정 금지. A+B |
| 커뮤니티 exact 셰이더 (vs SDK 원본 셰이더) | ✅ 비트 일치 | 전체 시간 6.074~7.424% 감소, 최종 프레임 28쌍 바이트 동일, provider 4.1.2-exact-r2-cyboman. 실제 게임 provider·FPS 미검증. A |
| clamp·shift 표현 변환 `clamp_all` (vs 커뮤니티 exact) | ❔ 3/4 조건 통과 | 1080p 움직임 −1.91%, 4K 정지 −1.37%, 4K 움직임 −1.43%. 1080p 정지는 CI에 0 포함으로 미채택. 96회 비교 바이트 동일. 커뮤니티 4.1.1 변환의 이식. 운영 적용 없음. A |
| 커뮤니티 lossy (재학습) | ⚠️ 근사 | exact 대비 14.9~23.1% 감소, 출력 다름(선형 RGB MAE 0.003~0.006). 합성 입력이라 육안 판정 불가, 미채택. A |
| Vulkan signed dot4 관측 기준 | A | wave32 probe 중앙값 91.68 TOPS. wave64 FSR의 하드웨어 상한으로 일반화 금지 |

## 5. 모델별 적용 예

| 모델 | 적용된 칸 | 전체 효과 | 근거 |
|---|---|---|---|
| MOSS-Image INT8 ConvRot (512², 4단계) | INT8 전치 제거 → Attention 안전 검사 → INT8 융합 GEMM | 7.446 s → 약 2.14 s(누적, 서로 다른 A/B 연결 추정치) | 가이드 §8, §9.6 |
| Qwen-Image-2.1 INT8 ConvRot (960×640 편집) | 후행 scale alpha 접기 | 1.08–1.10배 | 리허설 보고서 |
| Anima base v1.0 FP16 (1024², 30단계) | FP16 4D native 보존 + row repair | 디노이징 1.19배, latent·RGB 비트 일치. 첫 요청은 1초 prewarm이 있어야 이득 | 가이드 §9.11, [검수](REVIEW_V06_CLAUDE_KO.md) |
| 〃 (속도 우선) | FP16 SDPA → 직접 GEMM+softmax + V 전치 | 디노이징 1.42–1.43배, 엄격 기준 실패·Claude 육안 동등 | `anima-rdna2-lab01/REPORT_ANIMA_FP16_KO.md`, 가이드 §9.11 |
| 〃 (속도 최우선) | TunableOp + 직접 GEMM+softmax + V 전치 | 디노이징 1.79배(81.0 → 45.18 s, 튜닝 파일 재사용 45.20 s), 엄격 실패, **지각 통과(2026-11-07 사용자)**. 서버를 다시 띄우면 같은 seed도 그림이 미세하게 달라짐 | 가이드 §9.20, [딥리서치 통합 §5·§6](DEEP_RESEARCH_MERGED_20261107_KO.md) |
| MOSS BF16 VAE (flux2-vae_small, 512²) | BF16 Conv2D → FP32 계산, bias BF16 경계 | VAE 디코딩 2.30–2.63배, 엄격 통과(전체 생성 배속 아님) | 가이드 §9.14 |
| Idol image→3DGS (50단계, CFG3, 262144) | 자체 Triton fused attention(기본 chunked math 대체) | 24분 25초 → 4분 42초(5.20배), 미리보기 거의 동일 | `idol-rdna2-lab01/README.md` |
| 〃 (v0.8) | 위 커널을 Aceeton Flash / Sage로 교체 | 288.0 → 230.5 s(1.25배) / 193.0 s(1.49배), 미리보기 차 평균 0.4/255 | [딥리서치 통합 §6.5](DEEP_RESEARCH_MERGED_20261107_KO.md) |
| Applio RVC HiFi-GAN 학습 | MIOpen FAST, fp16 AMP, TheRock ROCm10 wheel | 정상 epoch 16~17 s → 11 s, 첫 epoch 44 s → 16 s(fp32 기본 대비) | `applio-lab/lab/RESULTS_2026-09-27.md` |
| 〃 (v0.6) | 검증한 NORMAL FindDb, V4R1 Fwd 항목 제외, FAST 재생 | 정상 epoch 9.310 → 8.418 s(1.106배). 학습 품질 동등성 미확정 | 가이드 §9.14 |
| 〃 (v0.9, voice_model_A 100 epoch) | 워커 GPU 큐 제거 + v0.8 묶음 B(solver DB, 직접 Conv1d, D single AdamW 등) | 운영 경로 22 → 12.6 s/epoch(100 epoch 약 37분 → 약 22분 추정). 하네스 100 epoch 25:40 → 22:00, 익명 청취 지각 통과 | 가이드 §9.21 |
| 〃 (v1.1 B+native, 채택) | v0.9 + 남은 Conv 68방향(융합 끔) | 운영 경로 100 epoch 21.42 → 19.64분(−8.33%). 보컬 익명 청취 구분 불가 | 가이드 §9.22 |
| 〃 (v1.0 B′, 청취에서 보류) | 위 + ResBlock 융합 12쌍 + 남은 Conv 68방향 | 운영 경로 100 epoch 실측 21.42 → 19.54분(−8.79%). 익명 청취 3/4 문장 기존 B 선호로 보류 | 가이드 §9.22 |

누적 7.446 → 2.14 s는 서로 다른 A/B의 기준선을 이어 붙인 값이다. 한 번의 A/B로 측정한 값이 아니다.

## 6. v0.8 운영 후보와 남은 칸

v0.8에서 판정 기준을 지각 기준 기본으로 바꿨다(§1). 아래 후보는 모델마다 스위치 하나와 롤백 명령을 갖춰 하나씩 넣는다. **2026-11-07 적용: Idol, MOSS INT8, Qwen alpha, Applio 안전 변수. 2026-11-08 적용: Applio v0.8 묶음 + 워커 GPU 큐 제거. 2026-11-10 적용: Applio v1.1 B+native. 보류: Applio v1.0 B′(청취, 2026-11-09). 보류: Anima(운영 자리 없음), MOSS VAE(해당 없음).**

| 모델 | 후보 | 효과 | 출력 | 근거 |
|---|---|---|---|---|
| MOSS INT8 ConvRot | INT8 dot4 융합 GEMM | 디노이징+VAE 4.898 → 2.149 s(2.28배) | 바이트 일치 | 가이드 §9.6. **적용됨(2026-11-07)**: `MOSS_RDNA2_INT8_LAYOUT=fused`가 실행 스크립트 기본값. 같은 프로세스에서 노드 fused 2.11 s / view 4.84 s, latent·PNG SHA 동일. 롤백 `LAYOUT=view`, 파일 백업 `moss-image-rdna2-lab01/backups/20261107_pre_int8_fused/` |
| MOSS BF16 VAE | BF16 Conv2D → FP32 계산 | VAE 디코딩 2.30–2.63배 | 엄격 통과 | 가이드 §9.14. **운영 해당 없음**: 운영 ComfyUI는 같은 VAE를 FP16으로 올린다(디코딩 47 ms). BF16 VAE를 쓰는 다른 워크플로에서만 의미가 있다 |
| Qwen-Image-2.1 | 후행 scale alpha 접기 | 1.08–1.10배 | 바이트 일치 | 가이드 §7.1. **적용됨(2026-11-07)**: Studio 생성 워커가 `Q21_RDNA2_ATTN_ALPHA=1`. 노드 경로 25단계 57.06 → 51.14 s, PNG SHA가 native와 동일. Studio e2e 통과. 롤백 `Q21_RDNA2_ATTN_ALPHA=0`, 파일 백업 `qwen-image-21-rdna2-lab01/backups/20261107_pre_attn_alpha/` |
| Anima FP16 | TunableOp + 직접 GEMM+softmax + V 전치 | 디노이징 1.79배 | 지각 통과, 같은 seed 재현은 프로세스마다 다름 | 가이드 §9.20 |
| 〃 (재현 우선) | FP16 4D native + row repair, prewarm | 1.19배 | 바이트 일치 | 가이드 §9.11. Anima 두 행은 **운영 자리 없음으로 보류**(2026-11-07 사용자: 실행기를 새로 만들지 않음). 레시피는 문서에만 둔다 |
| Idol | Aceeton Sage(또는 Flash) attention | 1.49배(1.25배) | Sage 지각 통과(2026-11-07 사용자), Flash는 수학적 동일 | 가이드 §9.20. **적용됨(2026-11-07)**: `ATTN_BACKEND=sage_rdna2`, MCP `idol_submit` 기본값, 실행기 `run_rdna2_sage_idol.sh`. 실행기 PLY가 연구 결과와 SHA 동일, MCP stdio GPU 스모크 통과. 롤백 `idol-rdna2-lab01/backups/20261107_pre_sage_rdna2/ROLLBACK.md` |
| Applio 학습 | MIOpen 두 차단 변수 + 검증 NORMAL FindDb | epoch 1.09–1.11배 | 학습 품질 동등성 미확정 | 가이드 §9.14. **두 차단 변수만 적용(2026-11-07)**: `run-applio.sh`, `lab/train_bench.sh`. FAST 4 epoch 대조 9.337 → 9.359 s(속도 변화 없음), NaN 0, loss 차 0.5% 이내. seeded DB(1.09배)는 학습 음성 동등성 확인 전까지 보류. 롤백 `applio-lab/backups/20261107_pre_miopen_safety/` |
| Applio 학습(v1.1 B+native) | v0.9 + 남은 Conv 68방향(융합 끔) | 운영 경로 100 epoch 21.42 → 19.64분 | 보컬 익명 청취 구분 불가(2026-11-10 사용자) | 가이드 §9.22. **적용됨(2026-11-10)**: `APPLIO_RDNA2_V10=1`이 `run-applio.sh`·`lab/train_bench.sh` 기본값, 융합은 `APPLIO_RDNA2_V10_FUSION` 기본 0. 운영 `core.py train` 5 epoch 스모크 통과. 롤백 `APPLIO_RDNA2_V10=0`, 파일 `applio-lab/backups/20261110_pre_v10_native/ROLLBACK.md` |
| Applio 학습(v1.0 B′) | v0.9 + ResBlock 융합 12쌍 + 남은 Conv 68방향 | 운영 경로 100 epoch 21.42 → 19.54분 | **청취 보류**(3/4 문장 기존 B 선호, B′ 쪽 아티팩트 지적, 2026-11-09 사용자) | 가이드 §9.22. **꺼진 채 둠**: `applio-lab/rvc/train/rdna2_v10/`(연구본 바이트 동일 사본), `APPLIO_RDNA2_V10=1`일 때만 동작, 런처 기본값 아님. 정적 검증만, 운영 사본 GPU 학습 미실행. 켜기·원복 `applio-lab/backups/20261108_pre_v10/ROLLBACK.md` |
| Applio 학습(v0.9) | v0.8 묶음 B + 워커 GPU 큐 제거 | 운영 경로 22 → 12.6 s/epoch | 익명 청취 구분 불가(2026-11-08 사용자) | 가이드 §9.21. **적용됨(2026-11-08)**: `APPLIO_RDNA2_V08=1`과 `MIOPEN_FIND_MODE=FAST`가 `run-applio.sh`·`lab/train_bench.sh` 기본값, GUI는 `.venv-rocm10`으로 전환. 실제 `core.py train` 경로에서 21/21 shape hit 확인. 위 행의 seeded DB 보류는 이 묶음의 solver DB로 대체. 롤백 `APPLIO_RDNA2_V08=0`, 파일 `applio-lab/backups/20261108_pre_v08/ROLLBACK.md` |

v0.8에서 기각한 것: Qwen·MOSS·Idol TunableOp(1.01·0.99·1.01배), FlashAttention의 TunableOp 위 중첩(이득 없음). Qwen Sage INT8 QK는 2026-11-08 실제 생성에서 native 대비 1.05배로, 운영 alpha 접기(바이트 일치)보다 느려 기각했다(가이드 §9.20). 남은 칸 네 가지는 [PRO_TASK_V10_KO.md](PRO_TASK_V10_KO.md)로 넘겼고, v1.0 결과는 다음과 같다(2026-11-08, Claude가 원자료 핵심 수치 재현).

- Applio 남은 시간: B′(−8.79%)는 청취에서 보류됐다. 속도가 거의 native 교체에서 나왔으므로 융합을 뺀 B+native(−8.33%)를 다시 시험했고, 보컬 청취를 통과해 운영에 적용했다(v1.1).
- RTX 5070 대비 몫: 조건부 roofline 모델만 계산했다. 같은 알고리즘·최소 traffic을 가정한 두 하한 차이가 0.35분/100 epoch다. 외부 "12분"의 조건이 같다는 증거가 없어 하드웨어·소프트웨어 분할은 **미식별**이다. 이 PC에서 실제로 줄인 것은 B→B′ 1.88분이다. [GAP_5070_KO.md](rdna2-ops/results/v10/GAP_5070_KO.md)
- TunableOp 비결정: FP32 GSU4 atomic 커널이 원인이다. 연구 필터로 재현성을 회복했고 Anima 1.79배를 유지했다. Qwen 필터의 새 프로세스 재현성과 MOSS는 미측정이고, 운영에는 적용하지 않았다. 위 §4.1 행.
- Strata warm: 자원 조건 미달로 서버를 실행하지 않았다(미판정). 위 §4.6 행.

### 6.1 v0.7 판정과 남은 칸 (기록)

이번 릴리스는 INT4·Sparse·IQ 프리필의 실제 기준 커널과 실패 조건을 채웠다. **새 운영 자동 적용은 0개**다. 구현 내부의 개선, 기존 native 교체 성공, 모델 최종 출력 통과를 서로 다른 판정으로 남긴다. [핵심 숫자 5개와 재현 명령](rdna2-ops/results/V07_REPRO_CORE_KO.md)

| 우선순위 | 칸 | 이번 근거와 판정 | 아직 채우지 못한 범위 |
|---|---|---|---|
| 1 | INT4 weight GEMM | 실제 9 shape·6 포맷·324 계산 관문 통과. G128의 BK64 분리는 이전 W4 대비 1.80–2.03배·byte 일치이나 native 대비 0.646–0.955배. MOSS 같은 W4 reference/custom latent·RGB byte 일치, 원본→W4 경로 strict 실패 | Qwen·Anima W4 최종 RGB, 품질 보존형 양자화, 안정적 native 성능 우위. 가이드 §9.16 |
| 2 | Sparse Attention | Qwen 실제 9 map·5-case·3 seed. r4 early 1.219배·메모리 감소, warm 모델 1.09657배. late same-mask 계산 관문과 최종 RGB 3/3 실패. 독립 익명 12/12 native 선호·후보 severity 3, 지각도 실패 | 새로운 모델·고해상도 Anima·영상의 희소성, late 계산 계약 통과, 품질과 가속의 동시 달성. 가이드 §9.17 |
| 3 | Strata IQ 프리필 | 실제 pack·3 라우팅 분포 54행 byte 일치, 좁은 skew 분포 GU/down 1.090/1.080배. 단계 이름의 host wait 범위와 첫 요청의 I/O 오염 확인 | 자원 보호로 9.7K/19.8K/119K warm fresh J off/on·token ID A/B와 warm 병목 판정 미완료. 서비스 조작 없이 조건이 달라질 때 재개. 가이드 §9.18 |
| 보존·수정 | 원본 코드·config·단위·운영 파일 | kernels.hip와 두 FP8 config의 v0.6 SHA 복원. 카탈로그 숫자–단위 띄어쓰기 정리. 운영 31+48개 전후 SHA 동일 | 전체 디렉터리/cache 불변을 뜻하지 않음. 현재 파일은 v0.7 manifest를 사용. 가이드 §9.19 |

v0.6의 layout·FP16 Attention·BF16/FP8 GEMM·Conv·정규화 결과는 §4의 해당 행과 가이드 §9.10–9.15에 유지한다. Claude 검수의 VAE BF16 Conv **2.46배**, Anima 4D **1.197배·RGB SHA 동일**, 두 MIOpen 차단 변수, 실제 Applio 4 epoch **1.092배**, Anima 같은 shape 5회 prewarm **1.07 s**는 기존 검수 근거로 반영했으며 v0.7에서 새로 잰 값으로 재계상하지 않았다.

다음 실험은 모델 이름보다 해당 칸의 미충족 조건을 먼저 고른다. W4에서는 group scale 비용·unpack·activation 계약과 표현 품질을 분리하고, sparse에서는 실제 보존 mass와 native 계산 계약을 먼저 확인한다. IQ 프리필은 실제 pack type·expert 행 분포·event 경계를 확인한 뒤 자원 관문을 통과한 warm fresh 측정으로 돌아간다.

## 7. 칸 하나를 채우는 최소 실험

1. **기준선:** 그 dtype·shape에서 PyTorch 기본 경로가 실제로 어느 커널을 타는지 프로파일로 확인한다(예: rocBLAS 커널 이름, MIOpen 알고리즘).
2. **후보:** 하드웨어 대체(§2) 또는 라이브러리 공백 메우기(§3) 방향으로 구현한다.
3. **연산 단위:** 대표 shape 3개 이상에서 정확도 등급, 속도, 추가 메모리를 잰다. warmup 뒤 순서를 교대한다.
4. **모델 단위:** 연산 단위를 통과하면 이 PC의 실제 모델 하나로 최종 출력 A/B를 한다. 적합한 모델이 없으면 ❔로 둔다.
5. **기록:** 이 표의 칸, 가이드 근거 절, `results/` 원자료를 함께 갱신한다. 실패도 같은 형식으로 남긴다.
