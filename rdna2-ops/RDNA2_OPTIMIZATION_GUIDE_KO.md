# RX 6900 XT / gfx1030 범용 AI 최적화 사용설명서

작성 기준: **2026-11-08 KST, rdna2-ops v1.0**(v0.9 + GPT Pro의 v1.0 연구 §9.22와 §9.18·§9.20 보충). v1.0은 Applio ResBlock 융합과 남은 Conv 교체의 운영 경로 100 epoch 결과(§9.22), TunableOp 비결정의 원인과 연구 필터(§9.20), Strata warm 재개 판단(§9.18)을 추가한다. **Applio B′는 익명 청취에서 4문장 중 3문장 기존 B 선호·B′ 쪽 아티팩트 지적, 1문장 같음(2026-11-09 사용자)으로 보류했다.** 운영은 v0.8 B 그대로다. 이전 판 기준: **2026-11-08 KST, rdna2-ops v0.9**(v0.8 + GPT Pro의 Applio 학습 연구와 Claude 운영 롤아웃 §9.21). v0.9는 Applio RVC 학습의 solver 선택·직접 Conv1d·Conv 외 최적화와 100 epoch 익명 청취 판정, 그리고 운영 학습 경로에서 찾은 DataLoader 워커의 GPU 큐 문제를 §9.21에 추가한다. 이전 판 기준: **2026-11-07 KST, rdna2-ops v0.8**(v0.7 + Claude 후속 실측 §9.20). v0.8은 판정 기준을 지각 기준 기본으로 바꾸고(§1.2), PyTorch TunableOp와 RDNA2용 FlashAttention Triton의 실제 모델 결과를 §9.20에 추가한다. 아래의 v0.7 서술은 그대로 보존한다. 입구는 [연산 카탈로그](../RDNA2_OP_CATALOG_KO.md)다. RX 6900 XT의 기존 환경에서 실행한 결과를 연산·dtype·shape·layout·backend별로 묶고, MOSS·Qwen·Anima·Applio는 최종 출력 검증에 쓰는 기존 사례로 다룬다. v0.3 운영 검증, v0.4 신규 커널, v0.5 Qwen 리허설 수정과 INT8·FP32 Attention·혼합 정밀도·Sage 이력은 그대로 보존했다. v0.6의 Claude 재현·원인 규명은 다시 측정하지 않고 검수 근거로 반영했다. v0.7은 INT4 가중치 GEMM·Sparse Attention·Strata 프리필의 성공/실패/미검증 칸을 §9.16–9.18에 추가한다.

설치·드라이버·가상환경·운영 기본 설정을 바꾸지 않았고 Triton 3.8을 유지했다. **v0.7에서 추적한 운영 파일 79개(기존 15+16개와 Strata 48개)의 SHA-256이 전후 동일**하다. Strata 서비스는 감사 전후 inactive/dead, MainPID 0이며 시작·중지·재시작하지 않았다. 이는 명시한 파일 범위이며 전체 디렉터리·OS page cache·컴파일 cache 감사가 아니다. GPU는 연구 lock과 KFD 감시 하에 순차 실행했고 겹친 측정·불완전 audit의 timing은 성능 판정에서 제외했다. [31개 사전](results/v07_operating_files/before.json) / [사후](results/v07_operating_files/after.json), [Strata 48개 사전](results/strata_v07/operating_before.json) / [사후](results/strata_v07/operating_after.json). Anima의 prewarm 코드 변경은 Claude 검수에서 이미 있었던 v0.7 시작 상태를 보호했다. 이전 [v0.6 감사](results/v06_operating_files_audit.json), [Applio epoch 감사](results/conv_catalog_v06/epoch_filtered_transpose/result.json), [v0.5 감사](results/v05_operating_files_audit.json)는 역사적 기록으로 남긴다.

이 설명서는 새 모델에서 **실제로 비싼 연산을 찾고, 기존 계산 의미와 운영 환경을 보존하면서 필요한 부분만 교체하는 방법**을 다룬다. 일반 PyTorch, 이미지 모델, VAE, 행렬 연산과 Attention에 적용하는 판단 절차다. MOSS/ComfyUI 수치는 RX 6900 XT에서 검증한 사례이며, 다른 모델의 성능 보증으로 사용하지 않는다.

재사용 구현에는 GN+SiLU, Comfy VAE 연결 helper, stride-aware INT8 전치 복사 제거, 선택적 FP32 math Attention, tiled Attention, INT8 QK와 dot4 GEMM이 있다. v0.5 MOSS INT8 GEMM+FP32 dequant 융합은 이미 view+auto를 쓰던 기준에서 실제 denoising+VAE 4.898→2.149초, 2.279배와 출력 비트 일치를 확인했다. v0.6은 rocBLAS layout·FP16 Attention·BF16/FP8 GEMM·Conv·정규화의 칸을 확장했다. v0.7은 INT4의 같은 저장값 계산과 표현 변경을 분리하고, 실제 Qwen 희소성·최종 출력과 Strata 실제 IQ pack·단계 경계를 확인한다. 실패 후보도 구현과 원자료를 남긴다. 신규 코드의 존재는 자동 채택 근거가 아니며, 명시적 API와 연구 프로세스의 context adapter로 제공한다. SpMM 등 미시험 칸은 카탈로그에 빈칸으로 남겨 둔다.

## 1. 근거 등급과 결과 읽는 법

| 등급 | 의미 | 확인할 수 있는 것 | 그대로 일반화할 수 없는 것 |
|---|---|---|---|
| **A — 실제 측정** | 실제 RX 6900 XT에서 실행한 JSON·trace·출력 비교 | 기록된 입력, dtype, layout, 버전, 실행 모드의 정확성과 성능 | 다른 모델·shape·GPU·버전의 성능, 미검증 입력의 정확성 |
| **B — 소스 확인** | 설치된 런타임 및 배포 코드의 dispatch·dtype·수명 규칙 확인 | 어떤 조건에서 어느 경로를 호출하는지, 통합 시 지켜야 할 계약 | 그 경로가 더 빠른지, 실제 요청에서 호출됐는지 |
| **C — 공식 문서/ISA** | AMD, PyTorch, Triton의 공식 설명 | 명령어·API의 의미와 설계 제약 | 설치된 빌드에 커널이 들어 있는지, 특정 모델에서의 이득 |

각 결과에서 **연산 하나의 측정인지, VAE 전체인지, denoising+VAE 구간인지**를 함께 읽는다. 동일한 GPU에서 측정했더라도 서로 다른 A/B의 시간을 합치거나 개선율을 곱하지 않는다. 코드가 끝까지 실행됐다는 사실과 정확성 기준을 통과했다는 사실도 구분한다.

근거 원본: [README](README.md), [운영 통합 기록](production/README.md), [설치 소스와 revision](SOURCES.md), 각 절의 `results/*.json` 링크.

### 1.1 무엇을 적용하고 무엇을 보류하는가

| 항목 | 결과와 적용 결정 | 근거 |
|---|---|---|
| MOSS `rdna2_int8_matmul`의 INT8 전치 view | stride-aware INT32 GEMM + 별도 FP32 dequant 경로에서 채택. 실제 전체 구간과 바이트 일치 확인 | A, B / 6·8절 |
| MOSS dot4 타일 + FP32 dequant 융합 | 기존 view+auto 기준 전체 4.898→2.149초, 2.279배; latent/RGB 비트 일치. 명시적 연구 API로 제공 | A, B / 9.6절 |
| Qwen patientx `int8_fused_kernel` | per-row dequant GEMM을 사용하며 해당 weight 전치 복사가 없음. MOSS의 전치 view 레시피를 그대로 적용하지 않음 | A, B / 6절 |
| Qwen patientx의 새 INT8 GEMM 교체 | 실제 shape 6개 모두 0.862–0.901배, 정확하지만 느림. 기존 backend 유지 | A / 9.6절 |
| Qwen split Attention 후행 scale 접기 | `baddbmm(beta=0, alpha=scale)` 후보가 해당 설치에서 비트 일치, 전체 1.084배(8단계). 연구 하네스만 적용 | A, B / 7.1절 |
| FP32 math 안전 검사 축소 | 검증한 shape/layout 범위만 자동 선택. 계산 순서 보존 | A, B / 7절 |
| GN+SiLU eager | native eager보다 이길 때 선택. 전체 모델 이득은 따로 판단 | A, B / 4절 |
| HIP GN의 graph 적용 | 거절. 과거 graph에서 compiled Triton보다 약 2.07배 느림 | A / 4절 |
| VAE FP16/BF16 연결 | 실제 regular/tiled·normal/low-VRAM 검증. tolerance 기반이며 bitwise 경로가 아님 | A, B / 5절 |
| FP16 attention 강제 | 해당 MOSS INT8 작업에서 거절. 반올림만으로 큰 오차 재현; short/long 분리도 실패 | A / 9.4절 |
| Tiled FP16 Attention | 긴 H2 합성 입력에서는 이득. 실제 Anima B2/H16/S4096/D128 기존 검증은0.85배로 속도 실패; 큰 head 수·다른 기준선을 분리 | A, B, C / 9.1·9.11절, [Anima 원보고서](results/sources/REPORT_ANIMA_FP16_KO_20261107.md) |
| Tiled FP32 Attention | S8192·D64는 조건부 이득, D128은 측정한 tile에서 속도 퇴보. 메모리 이득과 속도 판정을 분리 | A / 9.1절 |
| 실제 Qwen/MOSS FP32 D128 spill 제거 | Qwen spill 0, 추가 allocator peak 3102.6→75.5 MiB. 22.787→54.597 ms로 느려 속도 대체 거절; MOSS도 퇴보 | A, B / 9.7절 |
| INT8 QK + FP16/FP32 PV | v0.4 합성 입력에서 dot4c 및 operator 개선. 실제 Qwen rectangular 확장은 다음 Sage 행과 구분 | A, B, C / 9.2절 |
| ConvRot FP32 경계 + 부분 FP16 | 5개 후보 모두 RGB 실패. 작은 보정 오차도 INT8 코드를 바꿈. FP16→FP32 BMM fallback도 느림 | A, B, C / 9.8절 |
| 실제 Qwen Sage-inspired 포팅 | dot4/dot2 실행, 최고 1.129배/128 MiB. operator gate 및 최종 RGB max 0.59082·RMSE 0.01356 실패, 적용 제외 | A, B / 9.9절 |
| 직접 만든 FP16 GEMM | 정확성 통과, rocBLAS보다 4.24–5.83배 느려 교체용 채택 거절 | A / 9.3절 |
| hipBLASLt | 현재 PyTorch gfx1030 경로에서 요청이 거절되어 hipBLAS로 fallback | A, B / 9.3절 |
| 가중치 사전 전치 상주 캐시 | GEMM 단독 이득은 확인. 추가 메모리·LoRA/offload 수명 문제로 운영 채택 보류 | A, B / 6절 |
| rocBLAS FP16/FP32 layout | 48조합 실측. Anima FP16 PV의 NT 이득을 FP32·다른 shape로 일반화하지 않음. 실제 layout-only RGB strict 실패 | A / 9.10·9.11절 |
| FP16 4D native 계산 순서 보존 | 실제 Anima30step warm67.775/67.727s, native80.607/80.720s 대비 중앙값1.191배·latent/RGB 비트 일치. 최초83.019s는 별도 초회 관측으로 보존 | A, B / 9.11절 |
| BF16 GEMM 대체 | FP32 변환은 단위1.022–1.618배. 실제 BF16 VAE strict 통과하나 전체 무이득, Anima 실험 BF16 Linear strict 실패 | A, B, C / 9.12절 |
| FP8 가중치 계산 경로 | 같은 E4M3fn의 FP16 cache는 최종 byte 일치하나 실제30step 무이득·VRAM 증가. dot2 융합·projection INT8도 교체 거절. 원본→raw FP8 표현 손실은 별도 | A, B / 9.13절 |
| BF16 Conv2D의 FP32 계산 | BF16 bias 경계를 보존해 실제 VAE normal/lowvram의 regular/tiled 네 경로 strict 통과, decode2.30–2.63배. 검증 범위에서 opt-in | A, B / 9.14절 |
| Applio shape별 FindDb | 무필터 NORMAL의 큰 forward 오류를 재현하고 문제 solver entry 제외. 실제 epoch1.106배, 학습 후 음성 동등성은 native 반복 변동으로 미확정 | A, B, C / 9.14절 |
| Qwen exact-tile Conv3D→Linear | 3shape 정상 NORMAL 대비1.98–3.34배. 실제 모델은 후보 전 native conditioning부터 달라 최종 판정 불가. 운영 변경 없음 | A, B / 9.14절 |
| native LN+modulation 융합 | 3shape1.40–2.65배·모델 출력byte 일치. 실제denoise1.0039배(n=2)로 전체 이득 작음. LN/RMS/full adaLN 통계 교체는strict 실패 | A, B / 9.15절 |
| INT4 가중치 dot8/dot4/FP16 dequant | 실제 9 shape·6 포맷의 324 계산 관문 통과. g128 sub-K는 이전 W4 대비 1.80–2.03배이나 native보다 느림. MOSS 같은 W4 모델 출력은 byte 일치, 원본→W4 경로는 strict 실패 | A, B / 9.16절 |
| Sparse FP32 Attention | 실제 Qwen early r4 연산 1.219배·추가 peak 3102.594→38.000 MiB, 모델 warm 1.09657배. late 계산 관문 및 세 seed RGB strict 실패로 자동 적용 제외. 익명 12/12 native 선호·후보 severity 3으로 지각도 실패 | A, B, C / 9.17절 |
| Strata IQ 프리필 typical-J | 실제 IQ pack·합성 3분포 54행 byte 일치. skew GU/down 1.090/1.080배; warm 모델 A/B와 병목 판정은 swap 보호 차단으로 미완료 | A, B / 9.18절 |
| PyTorch TunableOp(rocBLAS 솔루션 재선택) | Anima FP16 SDPA math 1.33배, 직접 VT와 함께 1.79배. 튜닝 파일 재사용으로도 같은 속도. 지각 통과, 단 프로세스마다 같은 seed 출력이 미세하게 달라짐. Qwen 1.01배·MOSS 0.99배로 기각, MOSS 튜닝 중 GPU 메모리 오류 1회 | A / 9.20절 |
| FlashAttention Triton(ROCm 2.8.0 + RDNA2 블록 설정) | 연산 1.7–1.9배(저자 수치 재현). Anima 실제 생성 단독 1.17배, TunableOp와 겹치면 이득 없음. FP32 미지원 | A / 9.20절 |
| Applio v0.8 학습 묶음(shape·방향별 solver DB, 직접 Conv1d 21 shape, D AdamW single 등) | 하네스 100 epoch 25:40 → 22:00(14.3–14.5% 단축), 익명 청취 구분 불가로 지각 통과(2026-11-08 사용자). 운영 `applio-lab` 기본값으로 적용 | A / 9.21절 |
| spawn DataLoader 워커의 GPU 큐 제거 | 운영 train.py를 워커가 다시 import하며 GPU 큐 3개씩을 잡아 학습 커널 전체가 1.5–2배 느려졌다. 워커에서 GPU를 숨기니 22 → 15 s/epoch, 계산은 바뀌지 않음. 적용 | A / 9.21절 |
| Applio v1.0 B′(v0.8 B + ResBlock 융합 12쌍 + 남은 Conv 68방향) | 운영 경로 100 epoch 프로세스 21.42 → 19.54분(−8.79%), 연산·AMP 계약 통과. **익명 청취에서 4문장 중 3문장 기존 B 선호·B′ 쪽 아티팩트 지적, 1문장 같음(2026-11-09 사용자) → 보류.** 운영 사본은 꺼진 채 둠 | A, C / 9.22절 |
| TunableOp FP32 GSU4 atomic 행만 Default로 바꾸는 필터 | 해당 커널의 고정 입력 비결정 재현. Anima 새 2프로세스 6장 동일·1.79배 유지, Qwen 같은 프로세스 3장 동일(+1.32%). 연구 CSV만, 운영 미적용 | A / 9.20절 |

### 1.2 채택 판정 기준 (v0.8)

**기본은 지각 기준**이다(2026-11-07 사용자 결정). 같은 입력·seed로 원본과 후보의 최종 출력을 나란히 놓고 사용자가 차이를 느끼지 못하면 채택할 수 있다. 엄격 기준(latent/RGB 바이트 일치, 또는 정한 허용 오차)은 계속 측정해 참고 수치로 기록하고, 재현·회귀 검사처럼 바이트가 같아야 하는 용도에서만 채택 조건으로 쓴다. 인물 복제·구도 붕괴처럼 눈에 보이는 결함은 속도와 관계없이 실패다. 아래 표와 §9의 "strict 실패"는 v0.7까지의 엄격 기준 판정이며, 그 자체로 기각을 뜻하지 않는다.

같은 seed로 다시 뽑았을 때 같은 출력이 나오는지(재현성)는 지각 기준과 별도로 기록한다. TunableOp처럼 프로세스마다 출력이 조금씩 달라지는 후보는 지각 기준을 통과해도 "같은 seed 재현 불가"로 표시한다(§9.20).

## 2. 새 모델에서 먼저 할 일

### 2.1 실제 입력과 병목을 확인한다

1. 현재 정상 실행되는 환경과 모델을 그대로 사용해 원본 요청을 한 번 완료한다. 최초 모델 로딩·JIT·캐시 준비 시간과 warm 실행 시간을 구분한다.
2. **모델이 실제 호출하는 Attention 함수를 먼저 확인한다.** 실행 인자(예: `--use-split-cross-attention`, `--use-pytorch-cross-attention`)와 모델 모듈이 참조하는 callable의 `__module__`, `__qualname__`, 소스 파일을 기록한다. 공통 모듈의 `optimized_attention` 이름만 보고 추정하지 않는다. 모델이 `from ... import optimized_attention`으로 별도 alias를 보관하면 그 모델 모듈의 symbol을 감싸야 실제 호출을 관측한다. SDPA wrapper에 hit가 없으면 Attention이 없는 것으로 판단하지 말고 이 경로부터 추적한다.
3. 실제 Attention의 Q/K/V shape·dtype·stride, heads, mask, causal, scale과 호출 수를 기록한다. 모델 함수가 SDPA를 부르는 것이 확인된 경우에만 SDPA backend를 조사한다. `attention_split`, `attention_basic` 같은 수동 BMM 경로는 그 함수를 기준선으로 삼는다.
4. 반복 호출되는 연산의 shape, dtype, stride, 호출 횟수, GPU 시간, 복사량을 확인한다. 양자화된 **저장 dtype**과 **activation/compute dtype**, registry가 선택한 **실제 INT8 callable·커널 이름**을 따로 기록한다. `INT8 ConvRot`라는 모델 형식만으로 backend가 같다고 가정하지 않는다.
5. 운영 모드의 CPU wall time도 잰다. GPU kernel이 짧아져도 Python dispatch, weight 이동, 동기화가 지배하면 실제 요청은 거의 빨라지지 않을 수 있다.
6. 비싼 반복 복사나 중간 tensor를 조사한다. **복사 포함 contiguous 기준선과 stride view 후보를 실제 backend에서 함께 측정한다.** stride 지원은 정확한 주소 계산 조건이고 빠른 실행 조건은 별도다. Qwen의 rocBLAS FP32 BMM은 contiguous 준비를 제거했을 때 약 2배 느려졌다(7.1절).
7. 같은 실제 입력으로 원본/후보를 번갈아 실행하고, 연산 출력과 최종 작업 결과를 모두 비교한다. 정확성을 통과한 후보만 전체 워크로드에 적용한다.

**A/B — 다른 모델의 실제 진입점:** Qwen-Image-2.1 리허설은 `comfy.ldm.qwen_image21.model.optimized_attention`을 감쌌고 실제 함수는 `attention_split`이었다. 가장 많은 호출은 FP32 Q `[1,2400,4096]`, K/V `[1,4964,4096]`, heads 32, D128, mask 없음이다. prefix KV cache 때문에 Q와 K의 길이가 다르다. 이 경우 SDPA hook이나 MOSS의 `[1,24,S,128]` 자동 조건으로는 해당 Attention을 검사할 수 없다. [리허설 §2.1](../rehearsal_qwen_20261106/REHEARSAL_QWEN_KO.md), [실제 wrapper](../rehearsal_qwen_20261106/scripts/rehearsal.py)

**B — wrapper를 붙일 때의 추가 계약:** 동일한 callable도 입력이 Tensor가 아닌 container이거나 `transformer_options`의 Attention override를 사용할 수 있다. mask, `skip_reshape`, output reshape, 위치 인자와 keyword 전달도 원래 함수의 의미를 따른다. 새 adapter가 이 계약을 지원하지 않으면 원래 callable을 호출한다. [연구용 context](research_switches_v05.py)는 이 fallback과 종료 시 복원을 명시하며, 한 실험의 shape guard를 모델 전체 지원으로 표현하지 않는다.

**A — Qwen 8단계 요청의 호출 수 정정:** 원래 리허설의 Attention counter는 warm/profile 사이에 초기화되지 않아 704회가 두 요청의 누적으로 기록됐다. v0.5 하네스는 capture context마다 counter를 새로 만들며, 동일 조건의 **요청 1회는 총 352회**였다. 해당 요청의 실제 함수는 네 경우 모두 `attention_split`, FP32, heads 32, D128이었다.

| Q 길이 | K/V 길이 | Mask shape | 요청 1회 호출 수 |
|---:|---:|---|---:|
| 8 | 8 | `[8,8]` | 32 |
| 2400 | 2408 | 없음 | 32 |
| 156 | 2564 | `[156,2564]` | 32 |
| 2400 | 4964 | 없음 | 256 |

[초기화된 실제 callable/shape/count](results/v05_qwen_capture/actual_call_paths.json), [동일 서버 baseline/capture 바이트 일치](results/v05_qwen_capture/runs.json). 원래 리허설 자료는 수정하지 않고 이 정정과 함께 읽는다.


**A 사례:** 기존 MOSS의 cold profile에서는 `aten::copy_`가 GPU activity 합계의 55.86%, INT8 direct-copy kernel만 53.27%였다. 실제 SDPA 177회는 모두 math backend를 사용했다. 이 비율은 cold 진단용 activity 합계이며, 겹치는 activity가 있는 wall time 비율이나 다른 모델의 병목 비율이 아니다. [프로파일 원본](results/actual_comfy_profile.json)

같은 실제 denoising+VAE profile의 상위 10개는 다음과 같다. GPU activity 합계는 8,906.422 ms이며 부모 연산과 자식 연산을 중복 합산한 표가 아니다.

| 순위 | 귀속 연산/활동 | GPU ms | 비중 |
|---:|---|---:|---:|
| 1 | `aten::copy_` | 4,975.509 | 55.864% |
| 2 | `comfy_kitchen::int8_linear` | 1,202.680 | 13.504% |
| 3 | 미귀속 `_dot_i8_i32_kernel.kd` | 343.641 | 3.858% |
| 4 | `aten::mm` | 325.202 | 3.651% |
| 5 | `aten::add_` | 274.926 | 3.087% |
| 6 | 미귀속 Host→Device 복사 | 230.652 | 2.590% |
| 7 | `aten::bmm` | 214.159 | 2.405% |
| 8 | `aten::mul` | 200.466 | 2.251% |
| 9 | `comfy_kitchen::dequantize_int8_convrot_weight_dtype` | 126.612 | 1.422% |
| 10 | `aten::_softmax` | 124.172 | 1.394% |

이 사례에서는 GN kernel부터 계속 다듬는 것보다 반복 INT8 전치 복사를 제거하는 편이 전체 시간에 훨씬 큰 영향을 줬다. **새 모델에서는 이 순위를 다시 측정한다.**

### 2.2 측정 전에 고정할 항목

| 항목 | 기록/선택 방법 |
|---|---|
| 입력 | 실제 tensor 또는 재현 가능한 입력, seed, mask, scale, image/tile geometry |
| 환경 | GPU 이름과 `gcnArchName`, PyTorch/HIP/Triton 버전, 현재 backend와 precision flag |
| 실행 | eager/compile/graph, grad/autocast 상태, weight residency/offload 상태 |
| 기준선 | 실제로 사용하던 native 경로; 가능하면 이미 쓰는 compiled 경로도 별도 비교 |
| 정확성 | finite, shape/dtype, 필요 시 전체 바이트 일치; 허용 오차를 쓰면 실행 전에 max/RMSE/정규화 오차 한계 고정 |
| 시간 | warmup 제외, 양쪽 순서 교대, 반복별 원자료와 중앙값; CPU wall/event 단위 명시 |
| 메모리 | 입력·출력·workspace의 포함 여부; PyTorch allocator 수치가 외부 라이브러리 할당까지 전부 담는 것은 아님 |
| 적용 범위 | 빠르고 정확했던 dtype/shape/layout만 자동 선택; 나머지는 원래 경로 |

같은 seed만으로 모든 입력이 같다고 가정하지 않는다. **A 사례:** MOSS는 원본 서버를 재시작해도 sampling 전 text conditioning hash가 달랐다. 따라서 승인된 전체 이미지 A/B는 한 서버에서 같은 conditioning을 유지했다. 재시작 간 차이의 내부 원인은 미확인이다. [동일 서버 A/B](results/production_server_shared.json), [원인 범위 기록](production/README.md)

GPU 작업은 순차 실행한다. 캐시 준비와 profiler를 포함한 cold run은 병목 진단에 사용하고, 성능 비교는 profiler 없이 별도로 수행한다. parent SDPA의 inclusive 시간과 그 안의 BMM·softmax 시간을 중복 합산하지 않는다. **CPU operator에 귀속된 GPU 시간과 같은 작업의 GPU kernel 행도 서로 다른 집계 층이다. 두 표의 합계·비율은 분리한다.** 2.3.3절의 필터와 분모를 사용한다.

### 2.3 새 프로젝트에서 사용할 진입점과 프로파일 하네스 작성법

새 가상환경을 만들거나 패키지를 다시 설치하지 않는다. **대상 모델을 이미 실행하던 Python·ComfyUI root·실행 인자·환경변수**를 유지한다. `profile_comfy_workflow.py`는 MOSS workflow와 node ID를 아는 전용 재현 하네스다. 다른 모델에는 아래의 서버 하네스 계약을 연결한다.

| 설정 | MOSS 기존 사례 | Qwen 기존 사례 | 새 모델에서 할 일 |
|---|---|---|---|
| Python | `moss-image-rdna2-lab01/.venv-moss/bin/python` | `qwen-image-21-rdna2-lab01/.venv/bin/python` | 원래 실행기의 interpreter를 사용 |
| Runtime | 각 lab의 `runtime/ComfyUI` | 동일 구조 | 실제 `main.py` 위치 확인 |
| Workflow | `sft_fast4_int8_512.json` | `request-gemini-edit-640x960-api.json` | UI 형식이 아닌 API prompt JSON 사용 |
| Denoising node | workflow의 실제 sampler | `KSampler.sample` | `class_type` → `NODE_CLASS_MAPPINGS` → `FUNCTION` 확인 |
| Decode node | `VAEDecode` 또는 실제 분할 경로 | `VAEDecodeTiled.decode` | regular/tiled를 구분해 실제 노드만 측정 |
| Attention 관측점 | 모델이 호출하는 SDPA/모델 alias | `comfy.ldm.qwen_image21.model.optimized_attention` | 모델이 직접 참조하는 symbol을 지정 |

경로의 기준은 `~/DevSpace`다. 기존 wrapper가 필요한 모델은 그 wrapper도 유지한다. 연구 모듈을 사용할 때만 해당 프로세스의 `sys.path`에 연구 폴더를 추가한다.

#### 2.3.1 API workflow와 실제 callable을 찾아 연결한다

workflow의 node 번호를 미리 가정하지 않고 `class_type`에서 찾는다. 같은 종류의 노드가 여러 개면 실제 출력으로 연결되는 node ID를 명시적으로 선택하고 설정에 저장한다.

```python
import json
from collections import defaultdict
from pathlib import Path

prompt = json.loads(Path(workflow_path).read_text())
by_type = defaultdict(list)
for node_id, node in prompt.items():
    by_type[node["class_type"]].append(node_id)
print(dict(by_type))
# /object_info 응답 또는 서버의 nodes.NODE_CLASS_MAPPINGS로 class/FUNCTION 확인.
# 예: sampler_id = by_type["KSampler"][0]은 실제 후보가 하나일 때만 사용한다.
```

하네스 설정에는 다음 값을 분리해 둔다. 운영 파일을 수정하는 대신 연구 폴더의 설정으로 전달한다.

| 설정 계약 | 필요한 값 |
|---|---|
| 실행 | interpreter, ComfyUI root, 원래 server args/env, extra-model-paths |
| 요청 | API workflow 파일, sampler/seed node ID, 출력 prefix node ID, 입력 파일 경로 |
| 측정 단계 | sampler와 decoder의 class 또는 `module:Class.method`, 결과 tensor 경로 |
| Attention | `module:attribute`, Q/K/V·heads·mask 인자 계약 |
| 수명 | 비어 있는 전용 port, 연구용 input/output/temp, 준비/요청 timeout, 소유 서버 PID |
| 비교 | warm/profile/base/candidate 순서, 동일 seed/conditioning, 사전 출력 오차 기준 |

**실제 Attention 함수 확인은 ComfyUI가 운영 인자와 custom node를 로드한 뒤에 한다.** 다음은 Qwen 리허설의 관측점을 사용한 작성 예다. 새 모델에서는 모듈과 symbol만 실제 호출점으로 바꾼다.

```python
import importlib
import inspect

module = importlib.import_module("comfy.ldm.qwen_image21.model")
symbol = "optimized_attention"
original_attention = getattr(module, symbol)
unwrapped = inspect.unwrap(original_attention)
try:
    source_file = inspect.getsourcefile(unwrapped)
except (TypeError, OSError):  # builtin callable은 Python source가 없을 수 있다.
    source_file = None
print({
    "bound_module": module.__name__,
    "symbol": symbol,
    "callable_module": getattr(unwrapped, "__module__", None),
    "callable_name": getattr(unwrapped, "__qualname__", repr(unwrapped)),
    "source": source_file,
})
```

이 symbol을 `functools.wraps(original_attention)` wrapper로 감싸고 호출별 Q/K/V shape·stride·dtype, heads, mask shape/dtype, scale 및 hit 수를 기록한다. SDPA를 직접 부르는 모델은 SDPA 호출점도 관측한다. wrapper는 원본 인자를 그대로 넘기고 `finally`에서 원래 symbol로 복원한다. Python alias가 이미 복사된 뒤 공통 Attention 모듈만 바꾸는 방식은 호출을 놓칠 수 있다.

#### 2.3.2 서버 프로세스 안에 단계별 측정을 넣는다

ComfyUI 서버는 클라이언트의 Python `profile(...)`로 감싸도 서버 GPU 작업이 잡히지 않는다. **기존 ComfyUI를 같은 Python 프로세스에서 `runpy.run_path(main.py, run_name="__main__")`로 실행하고, 서버가 호출하는 노드 메서드를 그 프로세스 안에서 감싼다.** 실제 동작한 골격은 [Qwen rehearsal.py](../rehearsal_qwen_20261106/scripts/rehearsal.py)의 `Finder/Loader`, `instrument`, `client`다. 아래 순서로 대상 모델에 맞게 연결한다.

1. 원래 실행 인자·환경변수를 적용한 뒤 ComfyUI root를 `sys.path`에 추가한다. 바꾸는 것은 연구용 port/input/output/temp뿐이다. 운영 서버나 worker가 GPU를 사용 중이면 먼저 겹치지 않는 실행 시점을 확보한다. 운영 프로세스를 하네스가 임의 종료하지 않는다.
2. `nodes`가 import될 때 `PathFinder.find_spec`으로 원래 loader를 구하고, `exec_module`이 끝난 직후 core node를 감싼다. custom node의 클래스라면 등록 완료 뒤 `NODE_CLASS_MAPPINGS`에서 해석하고 첫 요청 전에 연결한다. 동기 node 메서드와 async node 메서드의 wrapper 형태를 맞춘다.
3. workflow의 `class_type`으로 실제 클래스를 찾고, 클래스의 `FUNCTION`에 적힌 메서드 이름을 확인한다. `KSampler`, `SamplerCustom` 등 다른 sampler를 같은 것으로 취급하지 않는다. 같은 단계를 중첩해서 감싸는 hook은 선택하지 않는다.
4. 각 노드의 시작 전과 종료 직후 `torch.cuda.synchronize()`를 호출해 wall time을 측정한다. profiler는 `profile` 모드에서만 켠다. latent/RGB의 CPU 복사·해시·파일 저장은 측정 구간 뒤에서 수행한다. 프로파일 실행과 성능 A/B를 분리한다.
5. 같은 프로세스의 client thread가 `/system_stats` 준비를 기다리고 `POST /prompt`에 API prompt를 보낸 뒤 `/history/<prompt_id>`의 성공/오류 상태를 기다린다. 요청은 하나씩 보낸다. candidate 전환도 이전 요청 완료 후에만 한다.
6. 동일 입력의 재실행이 캐시 hit로 생략되지 않도록 **선택한 sampler 클래스의 `IS_CHANGED`만 연구 프로세스에서 임시로 NaN을 반환하게 한다.** 기존 속성의 존재 여부와 값을 보관해 복원한다. 출력 filename만 바꾸는 방법으로 denoising 재실행을 보장하지 않는다. 각 요청에서 sampler/decoder 실제 hit 수가 0이면 그 시간은 비교에서 제외하고 오류로 기록한다.
7. warmup 뒤 같은 conditioning을 유지해 base/candidate를 교대로 실행한다. sampler 반환의 latent와 decoder 반환의 RGB tensor를 직접 비교한다. 최종 기준은 `max_abs ≤ 2/255`, `RMSE ≤ 0.25/255`이며, 비트 보존 후보는 equality/hash도 검사한다. 모든 hook·정밀도 flag를 복원하고 자신이 시작한 서버만 종료한다.

아래는 동기 노드용 **작성용 핵심 wrapper**다. `state`는 요청 사이에만 바꾸고, `consume_result`에서 모델별 반환 구조(예: sampler `result[0]["samples"]`, decoder `result[0]`)를 해석한다. profiler 객체는 요청 종료 뒤 2.3.3절의 두 표로 저장한다.

```python
import functools
import time
from contextlib import nullcontext
import torch
from torch.profiler import profile, ProfilerActivity

def wrap_stage(original, stage, state, consume_result):
    @functools.wraps(original)
    def measured(self, *args, **kwargs):
        torch.cuda.synchronize()
        state["active_stage"] = stage
        prof = (profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes=True)
                if state["mode"] == "profile" else None)
        started = time.perf_counter()
        try:
            with prof if prof is not None else nullcontext():
                result = original(self, *args, **kwargs)
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            state["stage_calls"][stage] += 1
            state["stage_seconds"][stage] += elapsed
            if prof is not None:
                state["profiles"].append((stage, prof))
            consume_result(stage, result)  # CPU 복사/오차 검사는 wall 구간 밖
            return result
        finally:
            state["active_stage"] = None
    return measured

# nodes import/등록 후, 실제 workflow에서 확인한 클래스를 사용한다.
# cls = nodes.NODE_CLASS_MAPPINGS[class_type]
# method_name = cls.FUNCTION
# original = getattr(cls, method_name)
# setattr(cls, method_name, wrap_stage(original, stage, state, consume_result))
# cleanup: setattr(cls, method_name, original)
```

프로파일러를 켠 측정의 wall time은 진단 오버헤드를 포함한다. sampler가 여러 번 호출되는 workflow는 누적 시간과 개별 호출 수를 함께 저장한다. VAE decode도 tiled 여부에 따라 정확히 그 메서드를 관측한다.

**구현된 설정형 하네스:** [research_comfy_v05.py](research_comfy_v05.py)는 위 구조를 사용한다. [MOSS spec](research_configs/moss_v05.json)과 [Qwen spec](research_configs/qwen_v05.json)에 runtime/workflow/node/Attention/실행 인자·환경변수를 분리했다. 새 모델에서는 spec을 복사해 실제 값을 연결한다.

현재 구현의 지원 계약은 **`nodes`가 내보내는 동기 KSampler 계열 메서드**, 입력 이름 `seed`/`steps`, sampler 결과 `result[0]["samples"]`, VAE 결과 `result[0]`다. 제공된 두 spec은 `KSampler.sample`과 `VAEDecode.decode`/`VAEDecodeTiled.decode`를 사용한다. `SamplerCustom`·async custom node·다른 반환 구조는 이 계약에 맞는 adapter를 먼저 연결한다. 설정 파일의 class/method만 바꿔 지원됐다고 판정하지 않는다.

**A — 공통 진입점 검증:** MOSS와 Qwen 모두 같은 서버에서 `warm → capture → baseline`을 실행해 latent/RGB 바이트 일치와 image gate 통과를 확인했다. `capture`는 실제 callable/shape/stride/call count를 기록하고, 긴 QKV를 제한된 수만 CPU에 저장한다. 이 구간의 시간은 캡처·복사 비용을 포함하므로 성능 수치로 사용하지 않는다. [MOSS 결과](results/v05_moss_capture_fixed/runs.json), [Qwen 결과](results/v05_qwen_capture/runs.json)

두 환경의 실제 실행 명령은 다음과 같다. 새 `--out` 경로와 비어 있는 port를 사용하며 GPU 작업은 하나씩 실행한다.

```bash
RDNA2_RESEARCH=~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
RDNA2_MOSS=~/DevSpace/moss-image-rdna2-lab01
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
  bash "$RDNA2_MOSS/scripts/with_local_storage.sh" \
  "$RDNA2_MOSS/.venv-moss/bin/python" -u "$RDNA2_RESEARCH/research_comfy_v05.py" \
  --spec "$RDNA2_RESEARCH/research_configs/moss_v05.json" \
  --out "$RDNA2_RESEARCH/results/my_moss_capture" \
  --modes warm,capture,baseline --port 8463

RDNA2_QWEN_PYTHON=~/DevSpace/qwen-image-21-rdna2-lab01/.venv/bin/python
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
  "$RDNA2_QWEN_PYTHON" -u "$RDNA2_RESEARCH/research_comfy_v05.py" \
  --spec "$RDNA2_RESEARCH/research_configs/qwen_v05.json" \
  --out "$RDNA2_RESEARCH/results/my_qwen_capture" \
  --modes warm,capture,baseline --steps 8 --port 8464
```

단계별 프로파일은 별도의 새 출력 경로에서 `--modes warm,profile,baseline`을 지정한다.


```bash
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
  "$RDNA2_QWEN_PYTHON" -u "$RDNA2_RESEARCH/research_comfy_v05.py" \
  --spec "$RDNA2_RESEARCH/research_configs/qwen_v05.json" \
  --out "$RDNA2_RESEARCH/results/my_qwen_profile" \
  --modes warm,profile,baseline --steps 8 --port 8464
```

`profile` 모드는 **denoise와 VAE 각각** 별도 profiler를 사용한다. `profile_<요청번호>_<단계>_<호출번호>_operators.json`, `_kernels.json`, `_trace.json`을 저장하므로 단계나 반복 요청의 파일이 덮어써지지 않는다. 예를 들어 위 명령의 profile 요청은 `profile_001_denoise_000_*`와 `profile_001_vae_000_*`다.

두 단계의 wall clock은 JSON 요약과 trace export **전에** 확정하고, 출력 tensor의 CPU 복사도 그 뒤에 한다. `runtime_metadata.json`에는 실제 로드한 Python/PyTorch/HIP/Triton 버전, GPU 이름/`gcnArchName`, 초기 precision flag, 하네스 SHA와 노드 소스를 저장한다. `runs.json`에는 단계 호출 수와 profile 산출물 경로가 들어간다. operator와 kernel/copy의 별도 분모는 다음 절과 같다.

`success`는 workflow 실행 성공, `image_gate`는 최종 RGB 허용 기준 통과다. 후보가 image gate에 실패해도 다른 후보의 비교를 계속할 수 있으므로 프로세스 exit code만으로 수치 성공을 판정하지 않는다. `conditioning_first_embedding_sha256`는 각 positive/negative의 첫 embedding만 해시한 보조 기록이며 conditioning의 모든 metadata를 포괄하지 않는다. 동일 서버·같은 workflow를 유지하고 최종 latent/RGB도 함께 확인한다.

**A — 수정된 profile 확장도 실제 검증했다.** 기존 MOSS spec으로 `warm → profile → baseline`을 실행했으며 각 요청의 denoise/VAE가 각각 한 번 실행되고 latent/RGB가 모두 비트 일치했다. 두 stage의 실제 trace를 독립 재합산해 category별 count·duration·비중이 요약과 일치함을 확인했다. stage clock은 export/집계 전에 끝난다. [실제 profile 검증](results/v05_moss_profile_verified/profile_verification_summary.json), [요청 원본](results/v05_moss_profile_verified/runs.json). 기존 Qwen 리허설 자체의 재현 명령과 당시 구현은 [리허설 §5](../rehearsal_qwen_20261106/REHEARSAL_QWEN_KO.md)에 보존했다.

#### 2.3.3 Operator 귀속 표와 GPU activity 표를 분리한다

`prof.key_averages()`에는 CPU operator에 귀속된 device 시간과 GPU kernel/memcpy 행이 같이 들어갈 수 있다. **모든 행의 `self_device_time_total`을 합치거나, 그 혼합 합계를 분모로 쓰지 않는다.** Qwen 리허설에서는 혼합 합계가 약 38.4초였지만 profiler를 켠 denoising wall은 약 20.9초였다. `comfy_kitchen::int8_linear`의 6,851ms와 그 구현 kernel의 8,818ms는 같은 GPU 작업을 서로 다른 관점으로 집계한 값이어서 더할 수 없다. [리허설 §2.1·§3](../rehearsal_qwen_20261106/REHEARSAL_QWEN_KO.md)

**`device_type == CUDA` 필터만으로도 부족하다.** 이 설치에서는 `record_function`의 GPU range도 `gpu_user_annotation`이라는 CUDA 이벤트로 나타났다. 이를 kernel처럼 더하면 단계 전체 span을 중복 합산한다. GPU 활동은 Chrome trace의 **`ph == "X"`인 complete event 중 category가 `kernel`, `gpu_memcpy`, `gpu_memset`인 기록만** 선택한다. 이름 접두사나 device 종류만으로 추정하지 않는다.

```python
from pathlib import Path
from research_comfy_v05 import profile_tables

# profiler context 종료 + synchronize + stage elapsed 확정 뒤 수행한다.
trace_path = Path(output_dir) / "profile_001_denoise_000_trace.json"
prof.export_chrome_trace(str(trace_path))  # 이 profiler에서 딱 한 번
tables = profile_tables(prof, trace_path=trace_path)

# CPU event의 self_device_time_total: operator 귀속 관점
operators = tables["operator_attribution"]
# raw trace의 kernel/gpu_memcpy/gpu_memset: 실제 device 활동 관점
activities = tables["device_activities"]
assert set(activities["included_categories"]) == {
    "kernel", "gpu_memcpy", "gpu_memset"
}
print(operators["denominator_ms"], operators["top10"])
print(activities["denominator_ms"], activities["top10"])
# 두 denominator_ms를 합치지 않는다.
```

`profile_tables(prof, trace_path=...)`는 이미 export된 파일을 읽으며 다시 export하지 않는다. `profile_tables(prof)`만 호출하는 작은 벤치는 연구 폴더에 임시 trace를 한 번 만들고 집계 후 제거한다. source helper에는 category/name/device별 합계, 제외 category의 시간·count, trace SHA와 포함 범위를 남긴다. 새 결과는 `profile_schema_version=2`로 표시한다.

연산 표의 분모는 **operator에 귀속된 self GPU 시간의 합**, activity 표의 분모는 **기록된 GPU kernel/copy 활동 시간의 합**이다. 두 표를 각각 이름 붙이고 별도로 읽는다. 누락된 attribution과 stream 중첩 때문에 각 합계는 실제 wall time과 다를 수 있다. `share_of_this_table`을 전체 요청 wall 비중으로 부르지 않는다.

- Operator 표는 “어느 Python/ATen/custom op를 바꿀까”를 고를 때 사용한다. inclusive 부모 시간과 self 자식 시간을 섞지 않는다.
- Activity 표는 실제 kernel 이름·복사 비용을 조사할 때 사용한다. raw trace category로 CPU 및 GPU annotation을 제외한 뒤 kernel/memcpy/memset을 구분한다.
- MOSS의 2.1절 표처럼 **operator 귀속 + 미귀속 activity**를 한 표에 보이려면 correlation을 통해 GPU activity를 단 한 번만 배분해야 한다. 단순히 두 목록을 이어 붙이지 않는다.
- 실제 최적화 속도는 profiler를 끈 별도의 교대 실행에서 구한다. CPU 대기·offload·GPU 중첩을 포함한 전체 wall은 따로 남긴다.

**A — 집계 오류도 실패 기록으로 보존했다.** 초기 v0.5 mixed 진단의 denoise 분모는 annotation 9553.167 ms가 섞여 14188.412 ms가 됐고, 교정 후 4635.245 ms다. VAE도 127.276→29.852 ms로 교정했다. 원본 trace와 CPU 표는 그대로 두고 잘못된 kernel JSON은 `_pre_fix.json`으로 보존했다. canonical kernel JSON에는 교정 이유·원본 SHA·새 집계기 SHA를 기록했다. [독립 category audit](results/v05_moss_mixed_boundaries/profile_category_audit.json), [교정 denoise 표](results/v05_moss_mixed_boundaries/profile_009_denoise_000_kernels.json), [교정 VAE 표](results/v05_moss_mixed_boundaries/profile_009_vae_000_kernels.json)

수정 뒤 새 실제 MOSS 요청에서 denoise GPU 활동 합은 **4741.963 ms**, VAE는 **29.832 ms**, 합계 **4771.795 ms**였다. 별도의 CPU operator 귀속 합은 **3968.412 ms**다. `comfy_kitchen::int8_linear`의 귀속 비중 73.677%와 아래 실제 INT8 kernel 비중 78.713%는 각각 다른 분모이며 더하지 않는다. 다음은 현재 view+auto 기준의 denoise+VAE GPU 활동 상위 10개다. 2.1절의 과거 cold profile과 구분한다.

| 순위 | GPU 활동 이름/요약 | GPU ms | 활동 합계 내 비중 |
|---:|---|---:|---:|
| 1 | `_dot_i8_i32_kernel.kd` | 3756.044 | 78.713% |
| 2 | FP32 vectorized multiply | 123.404 | 2.586% |
| 3 | rocBLAS `Alik_Bljk`, MT128×128×16 | 120.130 | 2.517% |
| 4 | FP32 warp softmax | 100.233 | 2.101% |
| 5 | rocBLAS `Ailk_Bljk`, MT128×64×16 | 78.066 | 1.636% |
| 6 | rocBLAS `Ailk_Bljk`, MT128×16×8 | 70.207 | 1.471% |
| 7 | `__amd_rocclr_copyBuffer.kd` | 63.534 | 1.331% |
| 8 | Elementwise direct copy | 56.755 | 1.189% |
| 9 | FP32 binary multiply | 45.176 | 0.947% |
| 10 | rocBLAS `Ailk_Bljk`, MT128×128×16 | 37.111 | 0.778% |

긴 커널명은 표에서만 줄였다. 정확한 이름·호출 수·단계별 상위 10개·제외 annotation과 raw trace SHA는 [profile 검증 요약](results/v05_moss_profile_verified/profile_verification_summary.json)에 있다. profiler stage wall 5.520701초/0.070211초와 export 비용을 포함한 request wall을 별도로 기록했으며, 이 수치로 새 커널의 성능비를 계산하지 않는다.

#### 2.3.4 병목에 따른 첫 후보와 기존 microbenchmark 실행기

| 실제 병목 | 먼저 검토할 경로 |
|---|---|
| 반복 transpose/cast/복사 | 수명·stride 계약 확인 후 copy 포함 contiguous와 view를 backend별로 실측. rocBLAS는 contiguous가 빠를 수 있음 |
| 수동 Attention의 `QKᵀ * scale` | 7.1절 후행 scale을 GEMM alpha에 접는 후보; native가 선행 scaling이면 그대로 옮기지 않음 |
| 긴 Attention의 score 메모리 | 실제 Q/K 길이와 native dtype를 유지한 tiled 경로 → operator와 최종 결과 검증 |
| 엄격한 FP32 math SDPA 일치 | 7절의 계산 순서 보존 경로; 실제 SDPA인 경우에만 적용 |
| 정밀도 허용 가능한 긴 QK | 9.2절 INT8 후보, 양자화 준비·PV 비용까지 포함 |
| INT8 linear/GEMM | 실제 선택 backend·tile·dequant·bias/residual 순서 확인, 6절 backend 표와 대응 |
| eager GN 바로 뒤 SiLU | 4·5절 fusion, 이미 compiled인 경로도 같이 비교 |
| 독립 FP16 GEMM | rocBLAS를 기준선으로 유지; copy/fusion 이득이 있는지 먼저 확인 |
| CUDA 전용 extension 오류 | CPU reference와 계약 확보 → HIP으로 표현 가능한 부분 이식 → 실제 gfx1030 컴파일·실행 |

CUDA extension은 CUDA runtime API 이름만 HIP으로 치환해도 되는 부분과, PTX·warp 가정·WMMA/MFMA 같은 아키텍처 전용 부분을 구분한다. 후자는 같은 계산을 RDNA2 SIMD dot/reduction으로 다시 구현해야 할 수 있다. 이 설명서의 성공 수치는 새 extension의 포팅 성공이나 Sparse 연산의 지원을 대신 증명하지 않는다.

뒤의 기존 `rdna2_python` 명령은 **MOSS 환경에서 작성된 microbenchmark의 재현용**이다. 새 모델의 서버 interpreter를 이 함수로 교체하지 않는다. 연구 모듈은 기존 Python에서 `sys.path.insert(0, RDNA2_RESEARCH)`로 import할 수 있다.

```bash
RDNA2_RESEARCH=~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
RDNA2_MOSS=~/DevSpace/moss-image-rdna2-lab01
cd "$RDNA2_RESEARCH"
rdna2_python() {
  HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    nice -n 12 bash "$RDNA2_MOSS/scripts/with_local_storage.sh" \
    "$RDNA2_MOSS/.venv-moss/bin/python" "$@"
}
```

결과 파일은 새 이름으로 지정한다. Attention/GEMM benchmark는 기존 결과 파일을 덮어쓰지 않고 거절한다. 기존 환경으로 실행되지 않으면 interpreter와 import 경로를 먼저 확인한다.

#### 2.3.5 AttentionTensorContainer를 소비하지 않는 캡처

**A/B — Anima에서 재현하고 수정했다.** 이 ComfyUI의 `comfy.ldm.cosmos.predict2:optimized_attention`은 Tensor 대신 `AttentionTensorContainer`를 받는다. wrapper 자체에는 `.shape`가 없고 `.take()`는 내부 tensor를 소비한다. 캡처가 `.take()`를 호출하면 실제 Attention에 전달할 입력을 망가뜨릴 수 있다. `research_comfy_v05.py`의 `capture_tensor_view`는 Tensor를 그대로 반환하거나 container의 **비소비 `.peek()`** 결과를 읽고, 원래 callable에는 원본 인수를 그대로 넘긴다. 지원하지 않는 wrapper를 임의로 변환하지 않고 명시적으로 실패한다. [실제 설치 class로 수행한 CPU 회귀 검사](results/capture_container_cpu_v06.json)

```python
def capture_tensor_view(value):
    if isinstance(value, torch.Tensor):
        return value
    peek = getattr(value, "peek", None)
    if callable(peek):
        tensor = peek()
        if isinstance(tensor, torch.Tensor):
            return tensor
    raise TypeError("Capture a Tensor or a supported non-consuming container")
```

spec의 `attention_capture_target`은 `attention_target`과 따로 지정할 수 있다. Anima 검증은 오류가 났던 상위 container alias를 직접 캡처했고, 후보 적용은 실제 하위 `comfy.ops:scaled_dot_product_attention`에 연결했다. 하위 단계만 기록해도 되지만 이 경우 기록 대상이 상위 Comfy attention과 같은 호출 집합인지 확인해야 한다. `input_wrappers`, shape, stride, dtype, heads, mask와 원본 함수 경로를 함께 저장한다.

실제 Anima 1024²·2step에서 warm/capture/baseline 세 요청은 각 denoise/VAE 1회였고 latent와 float RGB가 모두 **비트 일치**했다. self `[2,16,4096,128]` 및 cross K/V `[2,16,512,128]`가 요청당 각각 56회였다. 2step 시간은 캡처 무결성 진단이며 30step 성능으로 쓰지 않는다. [호출 원자료](results/v06_anima_capture/actual_call_paths.json), [최종 출력 비교](results/v06_anima_capture/runs.json), [GPU 단독 실행 확인](results/v06_anima_capture/gpu_audit.json)

추가 옵션 `gpu_audit`는 연구 job 사이의 flock과 KFD client 감시를 함께 사용한다. 관측된 외부 compute client나 감시 오류가 있으면 timing은 무효다. `comparison_reference_modes`는 native 기준 외에 후보끼리의 최종 출력 차이를 같은 프로세스에서 비교하며, `save_reference_tensors`는 첫 native latent/RGB를 연구 디렉터리에 저장한다. 노드·workflow·운영 설정을 수정하지 않는다.

## 3. 지원 여부를 판단하는 기준

검증에 사용한 기존 환경은 PyTorch `2.12.0+rocm10.2.0a20261011`, runtime HIP `7.17.26405`, Triton `3.8.0+git669b31ac.rocm10.2.0a20261011`, RX 6900 XT `gfx1030`이다. GN용 HIP 라이브러리는 해당 호스트에 이미 빌드되어 있다. 이 버전 목록은 재현 정보이며 새 설치 권고가 아니다. [A/B: 환경과 revision](SOURCES.md)

- **C:** RDNA2 ISA는 FP16 두 쌍을 곱해 FP32 누산에 더하는 `V_DOT2_F32_F16`, signed INT8 네 쌍을 INT32 누산에 더하는 `V_DOT4_I32_I8`를 명시한다. gfx1030 명령 표에도 이 명령들이 있다. 이는 SIMD dot 명령 지원 근거다. [`RDNA2 ISA` §12.10](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture), [AMD gfx1030 명령 표](https://rocm.docs.amd.com/projects/llvm-project/en/docs-7.14.1/LLVM/llvm/html/AMDGPU/AMDGPUAsmGFX1030.html)
- **C → A 확인 필요:** `tl.dot`는 block 행렬곱 API다. FP16 입력이라고 해서 설치된 compiler가 반드시 packed dot2를 생성했다고 단정하지 않는다. 실제 생성 assembly와 측정으로 확인한다. [Triton `tl.dot`](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)
- **B/A:** fused backend의 enable flag와 실제 가용 커널은 별개다. 이 설치의 AOTriton GPU 분류에는 gfx1030이 없었고, 실제 프로파일도 math SDPA를 관측했다. 어떤 옵션이 `True`라는 출력만으로 Flash Attention 실행이나 성능 개선을 주장하지 않는다. [설치 소스 근거](SOURCES.md)
- `torch.cuda` 이름은 이 PyTorch ROCm 환경에서도 사용한다. API 이름의 `cuda`나 `cublas` 문자열을 NVIDIA 실행의 증거로 해석하지 않는다.

**버전 경계:** 이번 코드는 설치된 Triton 3.8 계열에서 실제 컴파일했다. Upstream `main`에는 2026-09-30 RDNA1/RDNA2 target 지원 제거가 병합됐다. 따라서 더 최신 버전이라는 이유로 교체하지 않는다. [B: 공식 PR #12042](https://github.com/triton-lang/triton/pull/12042). 설치 버전 prefix `669b31ac`의 AMD blocked-dot lowering은 FP16→FP32에 `llvm.amdgcn.fdot2`, INT8→INT32에 `llvm.amdgcn.sdot4`를 선택한다. 이번 생성 ISA의 dot2c/dot4c 실측은 9절에 별도로 남겼다. [B: 해당 compiler 소스](https://github.com/ROCm/triton/blob/669b31ac/third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/FMA.cpp)

## 4. 레시피 A: GroupNorm 바로 뒤의 SiLU를 합친다

### 언제 적용하는가

eager 추론에서 `F.group_norm(...)` 다음 `F.silu(...)`가 반복되고, 그 사이에 관찰하거나 다른 용도로 사용하는 출력이 없을 때 후보가 된다. 기존 layout을 바꾸지 않고 적용할 수 있어야 이득을 유지하기 쉽다.

**B — HIP 경로 계약:** `gfx1030`, contiguous `NC...` 입력, FP32/FP16/BF16, 유효한 group 수와 finite nonnegative epsilon, 구현의 INT32 크기 제한을 만족해야 한다. affine weight/bias는 선택 사항이며, 있으면 길이 C의 contiguous vector이고 입력과 dtype/device가 같아야 한다. 채널 마지막 layout이나 mixed affine dtype을 임의로 복사·변환해 숨기지 않는다. [구현](rdna2_ops/normalization.py)

통계는 centered FP32 방식으로 누산하고, **GN 결과를 입력 dtype으로 반올림한 뒤 SiLU를 계산하는 경계**를 보존한다. 낮은 정밀도의 GN 출력을 FP32 상태로 그대로 SiLU에 전달하면 두 원래 연산과 다른 연산이 된다. 작은 group은 한 kernel, group 원소가 8192개를 넘으면 부분 통계와 출력 kernel을 사용한다. [HIP 소스](rdna2_ops/csrc/group_norm_silu.hip)

### 적용과 검증

아래 `x`, `weight`, `bias`, `groups`, `eps`는 이미 사용 중인 모델의 실제 값이다.

```python
import torch
import torch.nn.functional as F
from rdna2_ops import group_norm_silu

tolerances = {
    torch.float32: (3e-4, 3e-5),
    torch.float16: (3e-3, 3e-3),
    torch.bfloat16: (3e-2, 3e-2),
}
with torch.inference_mode():
    native = F.silu(F.group_norm(x, groups, weight, bias, eps))
    candidate = group_norm_silu(x, groups, weight, bias, eps, backend="hip")
    assert torch.isfinite(candidate).all()
    rtol, atol = tolerances[x.dtype]
    torch.testing.assert_close(candidate, native, rtol=rtol, atol=atol)
```

`backend="hip"`는 해당 호출이 HIP 후보를 실제로 검증하도록 하고, 미지원 계약에서는 오류를 낸다. 정상 사용의 `backend="auto"`는 미지원 입력·AD·autocast·compile·trace·capture를 PyTorch로 보낸다. `auto`만 호출하고 속도를 재면 실제로는 native fallback을 측정할 수 있으므로, 검증 시 선택된 경로도 확인한다.

위 비교는 native와의 일치 확인이다. 독립 수치 기준은 [validate_bench.py](validate_bench.py)의 FP64 reference를 사용한다. 이 reference는 FP64 통계와 affine 계산 후 입력 dtype으로 반올림하고 SiLU를 적용한다. 기존 검증에는 nullable affine, 상수, 8191/8192/8193 경계, storage offset, 다른 stream, 큰 평균/작은 분산이 포함된다. 이 tolerance는 연산 시험 기준이며, 최종 이미지나 다른 모델의 허용 오차를 대신 결정하지 않는다.

### 측정된 이득과 한계

다음은 **이전 v0.1 ordinary-eager operator 측정**이다. 각 행은 32 groups, 25회 호출 × 5 rounds의 CPU wall 중앙값이며, 전체 모델의 성능이 아니다. 현재 graph 정책과 섞지 않는다. [A: 원자료](results/validation_benchmark_final.json)

| shape / dtype | Native µs | HIP µs | 비율 |
|---|---:|---:|---:|
| `[1,320,64,64]` FP32 | 49.9769 | 29.4254 | 1.6984× |
| `[1,320,64,64]` FP16 | 46.0909 | 31.5930 | 1.4589× |
| `[1,320,64,64]` BF16 | 46.5309 | 31.9678 | 1.4556× |
| `[2,640,32,32]` FP16 | 34.3686 | 31.9950 | 1.0742× |

shape에 따라 이득이 작아진다. 또한 현재 **HIP GN은 eager 전용**이다. graph warmup·capture·replay 전체에서 `backend="torch"`를 선택한다. warmup은 capture 상태가 아니며 replay는 Python dispatcher를 다시 호출하지 않으므로, capture 중 분기만 검사하는 것으로 전체 lifecycle을 보장할 수 없다.

**A — 현재 graph 검증:** FP16 `[1,320,64,64]`에서 native 42.6605 µs, compiled-auto 14.4510 µs였다. compiled-auto는 PyTorch compiler fallback의 성과다. 과거 JSON의 HIP graph 수치는 현재 지원 경로의 성능으로 인용하지 않는다. [현재 runtime 계약](results/runtime_contracts.json)

**폐기 근거를 보존한 과거 비교:** 같은 FP16 shape의 이전 graph 측정에서 HIP 30.0425 µs, compiled Triton 14.4825 µs로 HIP이 약 2.074배 느렸다. 그 결과 HIP graph 자동 적용을 버리고 eager에서 이기는 위치로 제한했다. vector load 재작성으로 이 차이를 없앴다고 주장하지 않는다. [과거 graph A/B](results/compiled_comparison.json)

**롤백:** `backend="torch"`로 바꾸거나 원래 두 연산을 호출한다. `GroupNormSiLU` 모듈은 GN+SiLU의 조합이므로 뒤에 별도 SiLU를 남겨 두지 않는다. 학습은 native 미분 경로를 사용하며, 자체 HIP backward를 제공하지 않는다.

## 5. 레시피 B: 실제 VAE의 cast/offload 계약을 보존한다

### 일반 GN 함수와 Comfy VAE 통합은 다르다

Comfy norm은 weight/bias의 dtype 변환, weight function, DynamicVRAM prefetch, offload stream 수명, interruption polling을 수행할 수 있다. 외부에서 `norm.weight`만 읽어 커널로 넘기는 단순 교체는 이 계약을 생략한다. norm을 다른 모듈의 하위 속성으로 감싸면 `norm1.weight` 같은 parameter 경로도 바뀌어 model_patcher와 충돌할 수 있다. **B:** [Comfy helper](rdna2_ops/comfy.py), [설치 소스 revision](SOURCES.md)

제공된 helper는 모듈/parameter를 유지하고 인스턴스의 forward를 context 동안만 바꾼다. Comfy의 `CastBiasWeightContext(..., offloadable=True)` 안에서 커널을 호출하고 `run_every_op()`도 유지한다. 알 수 없는 forward와 관련 hook을 가진 블록은 제외하며, 호환되지 않는 전역 hook은 오류로 알린다. 이미 서비스 가능한 내장 fused norm/conv 경로와도 중복 적용하지 않는다.

### 적용 코드

```python
import torch
from rdna2_ops.comfy import fuse_comfy_vae_decoder

stats = {}
with torch.inference_mode():
    reference = vae.decode(latent)
    with fuse_comfy_vae_decoder(vae.first_stage_model.decoder, stats=stats):
        candidate = vae.decode(latent)
print(stats)
```

실제 공개 `vae.decode()`를 사용해야 로딩·dtype·offload 경로까지 검증한다. decoder만 직접 호출한 검사는 이 부분을 검증하지 않는다. tiled도 **같은 tile 크기와 overlap을 사용하는 native tiled와 fused tiled**를 비교한다. full decode와 tiled decode는 통계가 적용되는 공간 범위 자체가 달라질 수 있어, 서로의 차이를 fusion 오차로 해석하면 안 된다.

같은 decoder 인스턴스를 다른 스레드에서 동시에 사용하지 않는다. context 중첩은 허용하지 않는다. context 종료 시 예외가 있어도 원래 instance forward가 복원된다. compile/trace/capture는 원래 norm+swish를 사용하며, graph lifecycle에는 context의 `backend="torch"`를 명시한다. 이 연결 규칙과 전체 compiled VAE의 성능 검증은 별개의 범위다.

### A — 실제 VAE 결과

`flux2-vae_small.safetensors`, 실제 denoiser에서 보존한 latent `[1,128,32,32]`, 512×512 RGB를 사용했다. tiled는 tile size 256 / overlap 64다. 각 행은 **같은 dtype·같은 decode·같은 loading mode 안의 native/fused 비교**이며, 5 alternating rounds의 wall 중앙값이다.

| dtype | Loading | Decode | Native ms | Fused ms | 비율 |
|---|---|---|---:|---:|---:|
| FP16 | Normal | Regular | 45.04 | 43.20 | 1.0427× |
| FP16 | Normal | Tiled | 245.03 | 242.17 | 1.0118× |
| FP16 | Low-VRAM | Regular | 38.66 | 37.22 | 1.0386× |
| FP16 | Low-VRAM | Tiled | 296.20 | 293.54 | 1.0091× |
| BF16 | Normal | Regular | 179.30 | 178.41 | 1.0050× |
| BF16 | Normal | Tiled | 908.59 | 900.06 | 1.0095× |
| BF16 | Low-VRAM | Regular | 172.09 | 170.44 | 1.0096× |
| BF16 | Low-VRAM | Tiled | 936.33 | 931.09 | 1.0056× |

| dtype / decode | RGB max absolute | RGB RMSE | 최대 차이 × 255 |
|---|---:|---:|---:|
| FP16 regular | 0.000976563 | 0.000121774 | 0.2490 |
| FP16 tiled | 0.000493288 | 0.000063062 | 0.1258 |
| BF16 regular | 0.010437012 | 0.000839100 | 2.6614 |
| BF16 tiled | 0.003234267 | 0.000428700 | 0.8247 |

일반/low-VRAM은 같은 dtype·path에서 같은 오차 통계를 냈다. 사전에 정한 finite 및 max/RMSE 기준은 FP16 `0.01/0.001`, BF16 `0.04/0.006`이었고 여덟 case가 통과했다. **이 fusion은 허용 오차 기반이며 bitwise 동일 경로가 아니다.** BF16은 여기서 0.5–1% 정도의 VAE 이득과 최대 2.66개의 8-bit channel step 차이를 함께 보였으므로 기본 적용을 자동으로 정당화하지 않는다.

Normal과 low-VRAM의 절대 시간을 서로 비교해 offload가 일반적으로 더 빠르다고 결론내리지 않는다. FP16/BF16의 시간 차이도 해당 VAE의 관측이며, 모델 전체를 FP16으로 강제하라는 근거가 아니다. 필요한 수치 범위와 최종 출력 기준을 먼저 확인한다.

**실제 offload 검증:** low-VRAM 기록은 `dynamic=false`, `model_lowvram=true`, CPU-weight 모듈 124개, `loaded_bytes=0`을 확인했다. regular는 28회, tiled는 644회의 cast/HIP 호출이 관측됐다. 플래그를 붙였다는 사실만으로 low-VRAM 테스트가 끝난 것은 아니다. CLI의 메모리 예산은 남은 VRAM과 현재 로딩 상태에 따라 달라지므로 새 모델에서는 결과 상태를 다시 확인한다.

근거: [FP16 normal](results/vae_fp16_normal.json), [FP16 low-VRAM](results/vae_fp16_lowvram.json), [BF16 normal](results/vae_bf16_normal.json), [BF16 low-VRAM](results/vae_bf16_lowvram.json).

**롤백:** context를 사용하지 않거나 `backend="torch"`로 실행한다. 기존 whole-image 운영 개선은 이 VAE GN context를 자동 활성화하지 않았다. VAE에서의 작은 이득을 denoising이 지배하는 전체 모델의 큰 이득으로 계산하지 않는다.

## 6. 레시피 C: stride-aware INT8 GEMM의 불필요한 전치 복사를 없앤다

### 먼저 INT8 backend를 구분한다

`comfy_kitchen::int8_linear`라는 profiler operator 이름이나 `INT8 ConvRot` 모델 형식만으로 구현을 식별하지 않는다. custom node 초기화 뒤 registry의 실제 callable과 소스 파일을 확인하고 GPU kernel 이름까지 대응시킨다.

| 모델/실제 경로 | 연산과 layout 계약 | 이 전치 view 레시피 |
|---|---|---|
| MOSS: `scripts/rdna2_int8_matmul.py`의 `rdna2_int8_matmul` + Kitchen eager/연구 adapter | stride-aware INT8→INT32 GEMM. adapter는 `weight.T` view를 전달하고 별도 FP32 scale 복원·bias·residual 순서를 유지 | 반복 `weight.T.contiguous()`가 존재하던 경로에서 적용 |
| Qwen: patientx `int8_fused_kernel`의 `_int8_matmul_dequant_per_row_kernel` | `RDNA2-Qwen21-INT8/rocm_int8_kitchen_patch.py` → `triton_int8_linear_per_row`. 원본 `weight`를 contiguous로 준비하고 per-output-channel scale dequant를 GEMM 경로에 포함 | 같은 전치 복사가 없어 해당 없음. 별도의 tile/dequant 튜닝 대상으로 조사 |
| 새 모델의 INT8 backend | 실제 callable·GPU kernel·weight/scale shape·반환 dtype·stride 요구를 먼저 확인 | backend 계약과 반복 복사가 확인된 경우에만 후보로 사용 |

**B:** MOSS adapter의 scale 순서는 `INT32 → FP32`, `x_scale * weight_scale`, `acc * combined_scale`, output dtype 변환, bias, residual이다. Qwen patch는 input activation/ConvRot 후 scalar/per-row weight scale 개수로 각각 다른 함수를 고르고, 기존 residual 처리를 이어간다. backend를 교체하거나 dequant를 융합할 때는 이 계산 순서까지 기준선에 맞춰야 한다. [MOSS adapter](rdna2_ops/comfy_int8.py), [Qwen 리허설 및 backend 근거](../rehearsal_qwen_20261106/REHEARSAL_QWEN_KO.md)

Qwen 설치 소스 위치: `~/DevSpace/qwen-image-21-rdna2-lab01/custom_nodes/RDNA2-Qwen21-INT8/rocm_int8_kitchen_patch.py`. 확인 지점은 `rocm_int8_linear`의 weight 준비와 scalar/per-row dispatch다. 모델 파일명에 같은 `convrot`가 들어간다는 이유로 MOSS adapter를 중복 설치하지 않는다.

### 적용 조건

이미 INT8로 계산하는 linear 경로에서 `weight.T.contiguous()`가 반복되고, **설치된 GEMM이 실제 stride를 읽는 경우**에 사용한다. 이 변경은 모델을 새로 양자화하지 않고, scale·bias·residual·정수 누산 순서도 바꾸지 않는다. 바뀌는 GEMM 입력은 다음 한 가지다.

```python
# Backend가 양쪽 입력 stride를 실제로 지원한다고 확인한 후에만 적용.
old = int8_matmul(x_int8, weight_int8.T.contiguous())
new = int8_matmul(x_int8, weight_int8.T)
assert torch.equal(old, new)  # INT32 output
```

연속 메모리라고 가정한 CUDA/HIP kernel에 전치 view를 바로 넘기면 주소 계산부터 틀린다. `T`가 복사 없는 view라는 사실만으로 아무 backend에서나 제거할 수는 없다. 작은 정수 reference와 실제 큰 shape에서 INT32 equality를 확인하고, quantize/dequantize를 포함한 전체 linear 및 작업 출력도 비교한다.

**B — 현재 adapter 범위:** FP32 입력이면서 FP32 출력을 요구하는 eager INT8 linear만 바꾼다. FP16/BF16 입력·출력 경로는 원래 Kitchen eager로 전달한다. `weight` 자체의 device 이동/필요한 contiguous 준비와 quantization은 유지하고, 그 뒤의 불필요한 transpose materialization을 제거한다. [구현과 Apache-2.0 attribution](rdna2_ops/comfy_int8.py)

### 임시 적용 코드

호출자의 기존 backend 초기화가 끝난 상태에서, 검증한 callable을 전달한다.

```python
import inspect
from pathlib import Path
import torch
import comfy_kitchen.backends.eager.quantization as quant
from rdna2_ops.comfy_int8 import kitchen_int8_without_transpose_copy

backend = quant._int8_matmul_accumulate
source = inspect.getsourcefile(backend)
if source is None or Path(source).resolve() != validated_backend_path.resolve():
    raise RuntimeError("검증한 stride-aware INT8 backend와 다름")

with torch.inference_mode(), kitchen_int8_without_transpose_copy(backend):
    result = run_existing_workload()
```

`validated_backend_path`는 **해당 프로젝트에서 실제 stride 지원을 검증한 파일**의 `Path`다. MOSS 사례는 `scripts/rdna2_int8_matmul.py`였다. 파일 경로 일치는 수치 검증을 대신하지 않는다.

이 context는 Kitchen의 기존 eager module export를 바꾸므로 **프로세스 전체에 영향**을 준다. 작업을 독점한 상태에서만 사용하며 중첩·동시 context는 거절된다. 이미 운영 custom node가 같은 변경을 설치했다면 별도 context를 중복 설치하지 않는다. 기존 registry의 backend 이름·우선순위·capability는 바꾸지 않고, 원본 `quantization.int8_linear`와 직접 import된 alias는 유지한다.

### A — 무엇이 빨라졌는가

| 측정 범위 | Native/copy | View | 결과 |
|---|---:|---:|---|
| INT8 GEMM 경로, M/N/K=`1061/3072/3072` | 2.33768 ms | 1.88126 ms | 1.2426×, INT32 동일 |
| 실제 MOSS denoising+VAE, INT8 변경만 | 7.446393 s | 4.994781 s | 약 1.491×, latent/RGB 전체 바이트 동일 |

첫 행은 operator 비교, 둘째 행은 같은 서버/conditioning에서의 전체 두 단계 비교다. 각각 독립 결과다. [INT8 layout](results/int8_layout.json), [운영 A/B](results/production_server_shared.json), [dtype scope 및 복원](results/production_backend_scope.json)

### 사전 전치 캐시를 바로 추가하지 않는 이유

불변 weight를 한 번 전치해 상주시킨 microbenchmark는 네 큰 행렬에서 view GEMM보다 2.16–2.55× 빨랐다. 준비 비용·추가 상주 메모리는 그 비율에 포함하지 않았다. 해당 MOSS의 네 행렬 유형 × 42 blocks를 별도 상주시킨다는 가정의 추가 저장량 계산은 약 **4.43 GiB**다. 이는 아직 배포한 전체 생성 개선이 아니다. [A: 사전 전치 비교](results/int8_pretranspose_comparison.json)

**B:** DynamicVRAM의 gathered buffer 복원, LoRA 적용, cast buffer 재사용과 재양자화는 weight의 저장 위치·내용·layout을 바꿀 수 있다. 현재 설치는 모든 경로에 전치 layout 보존 계약이 있지 않다. 주소와 shape만으로 캐시를 재사용하면 오래된 weight를 읽을 수 있다. 따라서 운영 경로는 매 호출 최신 weight의 view를 읽는다. 캐시를 추가하려면 offload·weight 변경·재양자화의 수명 계약과 invalidation까지 구현해야 한다. [소스 근거](SOURCES.md)

**롤백:** 임시 context를 벗어나면 원래 eager export를 복원한다. MOSS의 지속 적용 롤백은 8절을 따른다. 기존 gfx1030 INT8 GEMM 보정까지 제거하지 않는다.

## 7. 레시피 D: FP32 math Attention의 안전 검사를 작게 만든다

### 유지하는 계산과 줄이는 작업

현재 경로는 PyTorch와 같은 Q/K 선행 scaling, rocBLAS BMM, native `torch.softmax`를 유지한다. 작은 Triton kernel이 NaN 확률 행을 확인하고 **모든 원래 score가 `-inf`였던 행만** 0으로 보정한다. 일반 행을 다시 쓰지 않으면서, native safety 경로의 행렬 전체 `isneginf`/`where` 중간 메모리 작업을 줄인다. [B: Attention](rdna2_ops/attention.py), [보정 kernel](rdna2_ops/_softmax.py)

임의의 NaN이나 `+inf`까지 `nan_to_num`으로 지우는 최적화가 아니다. scale 적용 위치나 softmax 구현을 함께 바꾸면 수치 순서까지 바뀌어 별도 후보가 된다.

### 적용 조건과 코드

```python
import torch
from rdna2_ops.attention import scaled_dot_product_attention

with torch.inference_mode():
    result = scaled_dot_product_attention(q, k, v, backend="auto")
```

자동 선택은 다음 범위로 제한된다.

| 조건 | 현재 automatic dispatch |
|---|---|
| 장치/dtype | 같은 gfx1030의 FP32 Q/K/V |
| Shape | Q/K/V 동일 `[1,24,S,128]`, `128 ≤ S ≤ 4096` |
| Layout | 마지막 stride 1, head stride 128; Q/K sequence stride 3072; V sequence stride 3072 또는 packed-QKV의 9216 |
| 연산 | mask 없음, dropout 0, non-causal, GQA 비활성 |
| 실행 | inference; grad/forward-AD/autocast/compile/trace/functorch는 native fallback |

`backend="torch"`는 원래 SDPA를 강제한다. `backend="rdna2"`는 보다 넓은 지원 가능한 batch-one FP32 경로를 **명시적으로 시험**하고 미지원 계약에서는 오류를 낸다. 다른 head 수·layout에서 우연히 빨랐다는 이유로 자동 범위를 바로 넓히지 않는다.

이 구현은 score와 probability 행렬을 materialize하는 math Attention이다. 메모리 효율적인 tiled Attention으로 간주하지 않는다. FP32 `[1,24,4096,4096]` score 하나만 약 1.5 GiB이고 probability도 같은 크기이므로, 실제 peak에는 최소한 이 두 행렬과 다른 작업 버퍼가 영향을 준다.

### A — 이득과 제외한 경우

| Shape | Native wall ms | 개선 wall ms | 결정 |
|---|---:|---:|---|
| `[1,24,1061,128]` | 2.54447 | 1.84830 | 1.3767×, 기록된 입력에서 출력 오차 0 |
| `[1,24,37,128]` | 0.058414 | 0.077564 | 더 느려 automatic에서 제외 |

두 V layout에 대해 경계와 홀수 길이를 포함한 20개 length/layout case가 전체 바이트 일치를 통과했다. 모든 `-inf`, NaN, `+inf`, 음의 overflow, 음수 scale과 zero scale도 별도로 검증했다. [operator](results/attention_math_native_softmax.json), [packed V](results/attention_ranges_packed_v.json), [interleaved V](results/attention_ranges_interleaved.json), [특수값](results/runtime_contracts.json)

**실패에서 얻은 적용 규칙:** 앞선 후보는 scaling을 GEMM 뒤로 옮기고 Triton softmax를 사용했다. random microbenchmark에서 약 1.75× 빨랐고 operator 최대 오차가 약 `7e-7`였지만, 실제 같은-seed 이미지의 max/RMSE는 `0.8528/0.09786`이었다. 그 후보는 거절됐다. 미세한 operator 오차만으로 반복적 모델의 최종 결과를 승인하지 않는다. [거절 기록](results/rejected_attention_ab.json)

FP16/BF16으로 SDPA 내부 정밀도를 바꾼 별도 실험도 세 seed에서 정해 둔 RGB max/RMSE 기준 `2/255` / `0.25/255`를 모두 넘었다. 세 후보는 각각 0/3 통과였고 운영 default로 채택하지 않았다. 이 실패는 해당 모델·변경의 결과이며 FP16/BF16 자체를 모든 모델에서 금지하는 근거는 아니다. [precision 실험](results/production_precision.json)

**롤백:** 직접 호출은 `backend="torch"` 또는 기존 `F.scaled_dot_product_attention`으로 되돌린다. MOSS dispatch를 설치한 경우 `MOSS_RDNA2_ATTENTION=native`로 재시작한다.

### 7.1 레시피 E: 원래 후행 scale인 Attention의 GEMM alpha에 스케일을 접는다

**적용 대상은 원래 `score = QKᵀ; score = score * scale` 순서인 구현이다.** ComfyUI `attention_split`/`attention_basic` 계열은 설치 소스를 확인해 이 형태인지 먼저 판단한다. 7절 MOSS math SDPA의 Q/K **선행** scaling을 이 형태로 바꾸면 다른 후보가 된다.

FP32 Q/K의 head별 contiguous buffer와 원래 native softmax·PV를 유지하고, 전체 score 행렬을 다시 읽고 쓰는 별도 scaling pass만 GEMM alpha에 접는다. 호출 형태는 다음과 같다.

```python
import torch

def qk_with_post_scale(qh, kh, scale):
    # qh: [H, L, D], kh: [H, S, D], FP32, 기존 per-head contiguous 준비 유지
    h, length, _ = qh.shape
    scratch = torch.empty((h, length, kh.shape[1]),
                          device=qh.device, dtype=torch.float32)
    with torch.autocast(device_type="cuda", enabled=False):
        return torch.baddbmm(scratch, qh, kh.transpose(1, 2),
                            beta=0, alpha=scale)

# 원래: scores = torch.bmm(qh, kh.transpose(1, 2)) * scale
# 후보: scores = qk_with_post_scale(qh, kh, scale)
# 이후 native softmax와 PV의 dtype/순서는 그대로 유지한다.
```

`beta=0`으로 scratch 내용은 읽지 않는다. bitwise equality는 이 API의 모든 backend에 대한 보증이 아니라 **현재 설치와 해당 입력에서 확인할 검사 항목**이다. 먼저 실제 native의 score와 최종 Attention 출력이 같은지 확인하고, 그다음 같은 서버·conditioning의 최종 latent/RGB를 비교한다. 리허설의 “FP32 GEMM 결과 반올림 뒤 alpha 곱 반올림이라 기존 두 단계와 같을 것”이라는 설명은 관측을 설명하는 추론이며 내부 명령열을 입증한 결과로 격상하지 않는다.

**A — Qwen FP32, heads 32, D128, 실제 layout의 microbenchmark 중앙값**

| 후보 | L=2400, S=4964 | L=2400, S=2408 | 출력/판정 |
|---|---:|---:|---|
| 원래 `attention_split` | 29.85 ms | 16.38 ms | 기준선 |
| q/k/v contiguous 제거, strided einsum | 60.82 ms | 30.57 ms | 비트 일치, **0.49/0.54배로 느려 폐기** |
| score `mul_` in-place | 29.49 ms | 16.10 ms | 비트 일치, 이득 미미 |
| `baddbmm(beta=0, alpha=scale)` | 22.71 ms | 12.98 ms | 비트 일치, **1.31/1.26배** |

Stride 후보 실패 원자료 (`../rehearsal_qwen_20261106/results/micro_split_strided_view_REJECTED.log`, 연구 폴더에만 있음), scale 비교 (`../rehearsal_qwen_20261106/results/micro_scale_variants.log`, 연구 폴더에만 있음), alpha wrapper 비교 (`../rehearsal_qwen_20261106/results/micro_split_alpha_wrapper.log`, 연구 폴더에만 있음)

Stride view는 주소상 정확했지만 rocBLAS 실행이 느려져 복사 절약을 상쇄했다. 따라서 Qwen alpha 후보는 원래 q/k/v의 head별 contiguous 준비를 유지했다. “stride를 받으면 복사를 없앤다” 대신 **copy를 포함한 전체 Attention 시간을 비교한다**는 기준을 사용한다. 내부 kernel 선택/접근 효율 중 어느 요인의 비중이 얼마인지는 리허설에서 분리하지 않았다.

**A — 같은 서버의 실제 960×640 편집, denoising + VAEDecodeTiled**

| Workflow | Native warm | Alpha 후보 | 비율 | Latent/RGB |
|---|---:|---:|---:|---|
| 8단계 | 22.415 s, 3회 중앙값 | 20.676 s, 4회 중앙값 | **1.084배** | 모두 max/RMSE 0 |
| 25단계 | 58.939 s, warm 1회 | 53.516 / 53.525 s, 2회 | 약 **1.101배** | 모두 max/RMSE 0 |

최초 모델 로딩 요청은 제외했다. 25단계 native 표본이 하나인 한계는 유지한다. [8단계 A/B](../rehearsal_qwen_20261106/results/ab_alpha8/runs.json), [25단계 A/B](../rehearsal_qwen_20261106/results/final25/runs.json), [리허설 원문](../rehearsal_qwen_20261106/REHEARSAL_QWEN_KO.md)

실험 wrapper는 **FP32·batch 1·mask 없음·skip-reshape 없음·GQA 없음·충분한 여유 메모리**에서만 후보를 사용하고 다른 호출은 원래 `attention_split`으로 전달했다. 25단계 요청에서 후보 832회, 원래 경로 64회(mask 있음)가 기록됐다. 메모리 부족으로 원래 split이 여러 조각으로 나뉘는 실제 상황, 다른 해상도·LoRA·Turbo·서버 재시작 간 동일성은 이 결과 범위 밖이다.

이 후보는 score와 probability를 계속 materialize한다. `32×2400×4964×4 B` score 하나는 약 **1.42 GiB**이며, alpha 접기는 그 저장 공간 자체를 없애지 않는다. 다음 메모리 최적화의 기준선은 이 alpha 후보까지 포함한다.

**임시 적용과 롤백:** [리허설 candidates.py](../rehearsal_qwen_20261106/scripts/candidates.py)의 `toggle("split_alpha", True/False)`는 연구 서버의 모델 callable을 교체하고 원래 callable을 복원한다. 운영 Studio에는 적용하지 않았다. 새 모델에서는 동일한 인자/fallback 계약과 종료 시 복원 로직을 연결한다.

## 8. MOSS 사례: 전체 요청과 운영 롤백의 정확한 범위

512×512, 4-step, seed `20261105`, 한 서버의 동일 conditioning에서 3 warm rounds를 교대로 실행했다. sampler/decoder cache hit로 연산을 생략하지 않았다. 아래 시간은 **denoising+VAE의 동기화된 wall 합계**이며 startup, text encoding, PNG 저장, API polling을 제외한다. [A: 실제 launcher](results/production_server_shared.json)

| 모드 | 중앙값 s | 출력 비교 |
|---|---:|---|
| `legacy:native` | 7.446393 | 기준 |
| `view:native` | 4.994781 | latent/RGB dtype·shape·전체 바이트 동일 |
| `view:auto` | 4.762543 | latent/RGB dtype·shape·전체 바이트 동일 |

`view:auto`는 legacy 대비 1.56353×, 구간 시간 36.04% 감소다. 이 구성은 **INT8 전치 복사 제거 + 선택적 FP32 Attention**이며 VAE GN fusion은 포함하지 않는다. 768×512 / S=1573에서는 INT8 view 위에 Attention auto를 더해 8.355858 → 7.849930 s, 1.06445×를 관측했다. [768×512 원자료](results/production_range_768x512.json)

최종 FP32-only guard 이후의 추가 한 쌍은 7.431528 → 4.749214 s였고, INT8 672회 / 개선 Attention 168회 / native Attention 8회 및 전체 바이트 일치를 확인했다. 이 추가 1회는 위 3-round 중앙값에 합치지 않았다. [최종 scope 확인](results/production_final_scope.json)

MOSS 전용 launcher의 default는 `MOSS_RDNA2_INT8_PATCH=1`, `MOSS_RDNA2_INT8_LAYOUT=view`, `MOSS_RDNA2_ATTENTION=auto`다. 이 환경변수들을 다른 프로젝트의 범용 실행 옵션으로 취급하지 않는다.

해당 MOSS 서버를 관리하는 작업에서 원래 경로로 되돌릴 때:

```bash
cd ~/DevSpace/moss-image-rdna2-lab01
bash scripts/stop_moss_comfy.sh
MOSS_RDNA2_INT8_PATCH=1 \
MOSS_RDNA2_INT8_LAYOUT=legacy \
MOSS_RDNA2_ATTENTION=native \
bash scripts/start_moss_comfy.sh
```

`PATCH=1`은 기존 gfx1030 INT8 GEMM 보정을 유지한다. `PATCH=0`은 이번 추가 최적화 외에 그 보정도 끈다. 정상 운영에서는 측정 전용 `MOSS_RDNA2_MEASURE_DIR`를 사용하지 않는다. launcher·원본 backup·재현 명령은 [운영 README](production/README.md)에 있다.

## 9. 신규 커널 실측과 정밀도 원인 분석

이 절의 커널은 실제 RX 6900 XT에서 컴파일·실행했다. **operator 기준 통과와 전체 모델의 채택 승인을 분리**한다. 신규 Attention은 합성 입력의 연산 실험이며, 새 이미지/영상 모델의 최종 출력까지 통과했다는 뜻이 아니다.

### 9.1 전체 score를 저장하지 않는 tiled Attention

**적용 상태: RX 6900 XT의 합성 입력에서 FP16 긴 시퀀스의 시간·추가 할당량 개선을 확인했다. 명시적으로 호출하는 실험 API이며 운영 auto dispatch에는 연결하지 않았다.** 실제 1024×1024 이미지 생성과 영상 모델의 최종 출력·속도는 아직 검증하지 않았다. 여기서 시퀀스 길이 16,384를 실행했다는 사실을 특정 이미지 해상도나 영상 프레임 수의 검증으로 바꾸어 읽지 않는다.

#### API와 입력 계약 — B: 배포 소스 확인

```python
import torch
from rdna2_ops.tiled_attention import tiled_attention

# 예: 같은 gfx1030 장치의 FP16 Q/K/V, [B, H, S, 64].
# autocast는 끄고 입력 dtype을 명시적으로 결정한다.
with torch.inference_mode():
    y = tiled_attention(
        q, k, v,
        is_causal=False,
        scale=None,          # 기본값 1 / sqrt(D)
        block_q=32,
        block_k=64,
        compute_dtype=None, # 입력 dtype 유지
    )
```

Q는 `[B,H,L,D]`, K/V는 `[B,H,S,D]`이며 `D=64/128`, 동일 dtype·device의 FP16/FP32를 받는다. 마지막 축 stride는 1이어야 하고, 다른 축의 nonnegative stride와 storage offset은 직접 읽는다. 직사각형 `L≠S`도 지원한다. causal은 `key_index <= query_index`인 **upper-left 정렬**이며 KV-cache의 절대 위치 offset을 받는 API가 아니다. 출력은 Q와 같은 dtype의 contiguous 텐서다.

지원 타일은 `(16,32)`, `(32,32)`, `(32,64)` 세 개이며 기본값은 `(16,32)`다. 자동 튜닝은 없다. eager forward 추론용이고 mask, dropout, GQA, backward, `torch.compile`, graph capture는 지원하지 않는다. 지원하지 않는 메타데이터는 오류로 알리며 내부에서 큰 native score를 할당하는 fallback을 숨기지 않는다. finite 입력과 finite dot product가 전제이며, 입력값 전체를 매 호출 검사하는 것은 아니다. [B: 구현과 정확한 guard](rdna2_ops/tiled_attention.py)

`compute_dtype=None`이면 입력 dtype을 유지한다. FP32 입력을 FP16 계산으로 낮추려면 `compute_dtype=torch.float16`을 **직접** 지정해야 한다. 이 변환은 타일 안에서 이루어지며 출력 dtype은 여전히 입력 dtype이다. 아래 FP32 실측은 그런 downcast를 하지 않은 FP32 연산이다. FP32→FP16 옵션의 실제 모델 품질을 검증했다고 해석하지 않는다.

#### 계산 원리와 정밀도

각 query 타일은 key/value 타일을 순회하며 행별 최대값 $m$, 지수합 $\ell$, 출력 분자 $A$만 유지한다. 점수 타일 $Z_j=sQK_j^T$에 대해 다음처럼 갱신한다. `max`와 `sum`은 각 query 행의 key 축에 적용한다.

$$
\begin{aligned}
m_j &= \max(m_{j-1},\max Z_j), &
\alpha_j &= \exp(m_{j-1}-m_j),\\
P_j &= \exp(Z_j-m_j), &
\ell_j &= \alpha_j\ell_{j-1}+\sum P_j,\\
A_j &= \alpha_j A_{j-1}+\widetilde P_j V_j, &
O &= A_j/\ell_j.
\end{aligned}
$$

초기값은 $m=-\infty,\ell=0,A=0$이다. 구현은 동등한 `exp2` 형태를 쓴다. QK dot 누산, softmax 통계와 출력 누산은 **FP32**다. FP16 계산에서는 Q/K/V 및 PV에 전달하는 $\widetilde P$를 FP16으로 바꾼다. FP32 계산에서는 P와 V도 FP32다. 따라서 FP16 모드의 “FP32 누산”이 모든 곱셈 피연산자까지 FP32라는 뜻은 아니다.

전체 `[B,H,L,S]` score/probability 텐서는 전역 메모리에 만들지 않는다. 타일과 행별 상태를 유지하고 전역 출력 `[B,H,L,D]`만 새로 할당한다. score 저장을 없애도 dense Attention의 연산량 $O(BHLSD)$은 남는다. native와 합산 순서 및 FP16 경계가 달라 **bitwise 동일 경로는 아니다**. 이 recurrence는 FlashAttention 계열의 online softmax 원리를 참고해 독립 구현했다. [C: FlashAttention 원 논문](https://arxiv.org/abs/2205.14135), [C: Triton fused Attention 설명](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html), [C: `tl.dot` 피연산자·누산 dtype](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)

#### FP16 긴 시퀀스 실측 — A

조건은 **B=1, H=2, noncausal**, interleaved Q/K/V와 0이 아닌 storage offset이다. RX 6900 XT/gfx1030, PyTorch `2.12.0+rocm10.2.0a20261011`, HIP `7.17.26405`에서 실행했다. native는 `SDPBackend.MATH`를 강제하고 `allow_fp16_bf16_reduction_math_sdp(False)`로 측정했다. **Comfy가 초기화하면서 사용하는 reduced-precision math 설정과 다르므로 이 비율을 기존 Comfy 생성 속도에 대입하지 않는다.** [B: 설치 소스·dtype 경로](SOURCES.md)

시간은 JIT·입력 생성·warmup을 제외하고 Python guard, dispatch, 출력 할당을 포함한 complete callable의 동기화 wall 중앙값이다. 순서를 정방향/역방향으로 교대한 **3 rounds × round당 3호출**을 사용했다. 아래 표는 첫 실행에서 타일을 **32×64로 고정**한 결과다. peak는 resident 입력·reference를 뺀 **PyTorch allocator 추가 peak**, 단위 MiB이며 driver/private scratch나 모델 전체 VRAM이 아니다. `max`와 RMSE는 native 전체 출력 대비이며, 표의 반올림 전 수치는 JSON에 있다.

| D | S | native ms | 32×64 ms | wall 배속 | 추가 peak MiB: native → tiled | max abs | RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 4097 | 3.158 | 1.381 | 2.287× | 296.15 → 1.00 | 6.104e-05 | 8.095e-06 |
| 64 | 8192 | 14.325 | 5.303 | 2.701× | 1168.02 → 2.00 | 6.104e-05 | 5.776e-06 |
| 64 | 16384 | 64.392 | 17.984 | 3.580× | 4640.03 → 4.00 | 6.104e-05 | 4.006e-06 |
| 128 | 4097 | 3.617 | 3.156 | 1.146× | 304.15 → 2.00 | 6.104e-05 | 7.980e-06 |
| 128 | 8192 | 15.253 | 12.409 | 1.229× | 1184.02 → 4.00 | 6.104e-05 | 5.684e-06 |
| 128 | 16384 | 72.269 | 45.108 | 1.602× | 4672.03 → 8.00 | 3.052e-05 | 4.049e-06 |

첫 실행에서 D64는 세 길이 모두 32×64가 16×32보다 빨랐다. D128의 16×32 시간은 S=4097/8192/16384 순서로 **3.062/11.996/46.953 ms**였다. 따라서 첫 두 길이에서는 16×32가 위의 32×64보다 조금 빨랐고, S=16384에서는 32×64가 빨랐다. head dimension이 커졌다고 타일도 무조건 크게 잡는 규칙은 쓰지 않는다. [A: FP16 전체 타일·raw samples](results/tiled_attention_fp16_v04.json)

D128에는 **32×32를 별도의 paired trial**로 추가했다. 아래 배속은 이 trial 안의 native와 비교한 값이다. 위 실행의 native 시간을 가져와 새 배속을 만들지 않았다.

| S, D=128 | 동일 trial native ms | 32×32 ms | wall 배속 | 추가 peak MiB: native → tiled | max abs | RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| 8192 | 15.260 | 10.023 | 1.522× | 1184.02 → 4.00 | 6.104e-05 | 5.708e-06 |
| 16384 | 73.295 | 39.557 | 1.853× | 4672.03 → 8.00 | 3.052e-05 | 4.063e-06 |

이 범위에서는 D128의 S=8192/16384에 32×32를 명시적 실험 선택지로 삼을 근거가 생겼다. **이 H=2 trial**에서 S=4097의 32×32는 측정하지 않았다. 뒤의 9.2절에는 H=4/S4097을 포함한 별도 직접 비교가 있다. 기본 타일이나 운영 dispatch는 바꾸지 않았다. [A: 별도 D128 32×32 trial](results/tiled_attention_d128_32x32_v04.json)

정확성 기준은 실행 전에 FP16 `rtol=0.01, atol=0.003`, FP32 `rtol=3e-4, atol=3e-5`로 고정했다. 작은 shape는 각 타일마다 **독립 CPU FP64 공식**과 8개 case를 비교했다. 길이 tail, causal/noncausal, 직사각형, 음수·0 scale이 포함되며, 첫 FP16/FP32 파일은 각각 16검사, 32×32 파일은 8검사가 모두 통과했다. 긴 shape는 native **전체 출력**과 비교하고, 별도로 고르게 고른 8개 query 행을 FP64 공식과 비교해 모두 통과했다. FP64 검사가 긴 출력 전체를 검사한 것은 아니다. [B: reference·측정 구현](bench_tiled_attention.py)

#### FP32 결과와 적용 판단 — A

다음은 B=1/H=2/S=8192, FP32 입력·FP32 계산이다. downcast하지 않았다.

| D | 타일 | native ms | tiled ms | wall 배속 | 추가 peak MiB: native → tiled | max abs | RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 16×32 | 11.954 | 14.855 | 0.805× | 1156.02 → 4.00 | 5.141e-07 | 4.234e-08 |
| 64 | 32×64 | 11.954 | 10.529 | 1.135× | 1156.02 → 4.00 | 5.141e-07 | 4.224e-08 |
| 128 | 16×32 | 12.787 | 23.592 | 0.542× | 1160.02 → 8.00 | 5.886e-07 | 4.387e-08 |
| 128 | 32×64 | 12.787 | 26.981 | 0.474× | 1160.02 → 8.00 | 5.588e-07 | 4.376e-08 |

D64의 32×64는 이 shape에서 1.135×였지만, **D128은 두 타일 모두 느려 속도 최적화로는 거절**한다. 추가 할당량을 줄인 사실은 별도의 메모리 선택 근거일 수 있으나 속도 이득으로 포장하지 않는다. FP16 D64 긴 shape와 D128의 검증된 타일은 재사용 가능한 실험 후보로 남기며, 신규 모델은 실제 QKV 분포·호출 layout·최종 출력 기준을 확인한 뒤 적용한다. 이 긴 시퀀스 실측에는 causal workload나 실제 1024px 이미지/영상 실행 결과가 없다. [A: FP32 결과](results/tiled_attention_fp32_v04.json)

재현은 기존 ROCm 가상환경·wrapper를 그대로 사용하며 결과 파일은 새 이름으로 지정한다. 아래 명령은 위 noncausal 조건을 명시한다.

```bash
rdna2_python bench_tiled_attention.py --dtype fp16 --causal no --heads 2 \
  --lengths 4097 8192 16384 --head-dims 64 128 --tiles 16x32 32x64 \
  --rounds 3 --iterations 3 --output results/tiled_fp16_recheck.json
rdna2_python bench_tiled_attention.py --dtype fp16 --causal no --heads 2 \
  --lengths 8192 16384 --head-dims 128 --tiles 32x32 \
  --rounds 3 --iterations 3 --output results/tiled_d128_32x32_recheck.json
rdna2_python bench_tiled_attention.py --dtype fp32 --causal no --heads 2 \
  --lengths 8192 --head-dims 64 128 --tiles 16x32 32x64 \
  --rounds 3 --iterations 3 --output results/tiled_fp32_recheck.json
```

### 9.2 Per-token INT8 QK + streaming softmax + FP16/FP32 PV

**적용 상태: 실제 gfx1030 컴파일·실행과 16개 synthetic operator gate를 통과한 근사 Attention 실험이다. 이미지·영상 품질은 미검증이며 운영 기본값으로 채택하지 않았다.** 기존 INT8 linear의 전치 view가 bitwise였다는 결과와는 다른 연산이다. 이 API는 매 호출 Q/K를 새로 양자화하므로 출력도 원본과 bitwise 같지 않다.

#### API와 계산 경계 — B

```python
import torch
from rdna2_ops.quantized_attention import quantized_attention

# 실측 조건: FP16 Q/K/V, 동일 [1, 4, S, D], D=64/128.
with torch.inference_mode():
    y = quantized_attention(
        q, k, v,
        smooth_k=True,
        pv_dtype="fp16",   # 속도 실험 선택. API 기본값은 "fp32".
        block_q=16,
        block_k=32,
        is_causal=False,
    )
```

Q/K/V는 동일 `[B,H,S,D]` shape, D64/128, 같은 gfx1030 장치의 FP16/FP32 텐서여야 한다. 실측한 입력·출력은 모두 FP16이다. rectangular Attention, arbitrary mask, dropout, GQA, backward, compile, graph capture는 제공하지 않는다. 출력 dtype은 Q를 따른다. 지원 타일 인자는 BQ16/32, BK32/64지만 아래 실측은 전부 **16×32**다. finite 입력과 적절한 값 범위가 전제다. [B: guard·양자화·streaming 구현](rdna2_ops/quantized_attention.py)

선택적으로 head별 K의 시퀀스 평균 μ를 FP32로 계산해 빼고, Q와 중심화한 K를 **token·head별 symmetric INT8**로 양자화한다.

$$
s_x=\max\!\left(\max_d |x_d|/127,\;10^{-20}\right),\qquad
x_8=\operatorname{clip}\!\left(\operatorname{round}_{\rm away}(x/s_x),-127,127\right).
$$

반올림은 가장 가까운 정수이며 정확히 중간이면 0에서 멀어지는 방향이다. Q8×K8은 **INT32로 누산**한 뒤 FP32로 바꾸고 token별 Q/K scale 및 Attention scale을 곱한다. 이 score 타일을 9.1과 같은 online softmax로 누적한다. `pv_dtype="fp16"`은 P/V 피연산자를 FP16으로, `"fp32"`는 FP32로 전달하며 **둘 다 PV와 출력 누산은 FP32**다. FP32 PV를 골라도 QK 양자화 오차는 남고, FP16 Q 입력의 최종 출력은 여전히 FP16이다.

전체 score/probability는 만들지 않지만 Q8/K8, token별 scale, 평균 계산의 임시 텐서는 할당한다. Q/K 양자화나 K 평균을 사전에 캐시한 속도와 구분해야 한다. 이상적인 실수 연산에서는 $Q(K-\mu)^T=QK^T-(Q\mu)\mathbf{1}^T$로 각 softmax 행에 같은 상수만 빠지므로 결과가 유지된다. 실제 INT8에서는 scale과 반올림이 달라져 오차가 달라질 수 있다. K 중심화와 QK 양자화의 아이디어는 SageAttention의 원저·공식 구현을 참고했으며, 이 코드는 per-token 양자화와 FP32 누산을 사용하는 독립 prototype이다. SageAttention의 NVIDIA 전용 커널을 이식하거나 그 논문의 모델 품질을 재현한 결과가 아니다. [C: SageAttention 원 논문](https://arxiv.org/abs/2410.02367), [C: 저자 공식 구현](https://github.com/thu-ml/SageAttention)

**causal에서는 `smooth_k=False`가 필수다.** 시퀀스 전체의 K 평균을 쓰면 미래 key가 현재 key의 양자화 반올림에 영향을 줄 수 있기 때문에 `is_causal=True, smooth_k=True`는 guard가 거절한다. `smooth_k=False`인 causal 분기는 코드에 있으나 이번 긴 시퀀스 결과는 모두 noncausal이다. causal 성능·정확성까지 검증됐다고 확장하지 않는다.

#### 시간·메모리·오차 — A

입력은 **B=1, H=4**, contiguous FP16, noncausal, seed `20261106`이다. native는 앞 절과 동일하게 reduced-precision을 끈 MATH 경로다. 각 방법을 한 번 준비·warmup한 뒤 **3회씩 같은 횟수**로 실행했고, 매 round 전체 방법 순서를 정방향/역방향으로 교대했다. 아래 wall 중앙값에는 매 호출의 K 평균(smoothing을 켠 경우), Q/K 양자화, 임시·출력 할당, dispatch와 GPU 계산이 **모두 포함**된다. JIT, 입력 생성, correctness/ISA 수집은 시간에서 제외했다. raw 순서·wall/event sample은 JSON에 남겼다. peak는 timing 뒤 별도 측정했다. [A: 전체 기록](results/quantized_attention_v04.json), [B: paired timing·고정 gate](bench_quantized_attention.py)

다음 표는 native 시간과 allocator 추가 peak다. smoothing 켬/끔의 peak는 같은 shape에서 PV dtype에 관계없이 같았다. 입력·reference 상주량과 driver scratch를 포함한 총 VRAM 수치가 아니다.

| S / D | native ms | native peak MiB | smooth 켬 peak MiB | smooth 끔 peak MiB |
| --- | --- | --- | --- | --- |
| 4097 / 64 | 6.315 | 592.30 | 12.00 | 4.13 |
| 4097 / 128 | 6.566 | 608.31 | 24.00 | 8.13 |
| 8192 / 64 | 26.401 | 2336.03 | 16.00 | 8.25 |
| 8192 / 128 | 27.751 | 2368.03 | 32.00 | 16.25 |

아래 배속은 바로 위 표의 **동일 H=4 native**에 대한 값이며, max/RMSE는 전체 FP16 출력 대 native 비교다.

| S / D | K smooth | PV | 전체 호출 ms | wall 배속 | max abs | RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| 4097 / 64 | 켬 | FP16 | 2.184 | 2.892× | 2.075e-03 | 2.246e-04 |
| 4097 / 64 | 켬 | FP32 | 2.782 | 2.270× | 2.075e-03 | 2.245e-04 |
| 4097 / 64 | 끔 | FP16 | 2.133 | 2.961× | 4.028e-03 | 2.252e-04 |
| 4097 / 64 | 끔 | FP32 | 2.757 | 2.291× | 4.028e-03 | 2.252e-04 |
| 4097 / 128 | 켬 | FP16 | 3.640 | 1.804× | 1.831e-03 | 2.393e-04 |
| 4097 / 128 | 켬 | FP32 | 4.950 | 1.327× | 1.831e-03 | 2.393e-04 |
| 4097 / 128 | 끔 | FP16 | 3.590 | 1.829× | 3.113e-03 | 2.397e-04 |
| 4097 / 128 | 끔 | FP32 | 4.923 | 1.334× | 3.113e-03 | 2.396e-04 |
| 8192 / 64 | 켬 | FP16 | 8.150 | 3.239× | 1.343e-03 | 1.591e-04 |
| 8192 / 64 | 켬 | FP32 | 10.470 | 2.521× | 1.343e-03 | 1.590e-04 |
| 8192 / 64 | 끔 | FP16 | 8.099 | 3.260× | 1.495e-03 | 1.589e-04 |
| 8192 / 64 | 끔 | FP32 | 10.441 | 2.529× | 1.495e-03 | 1.588e-04 |
| 8192 / 128 | 켬 | FP16 | 14.449 | 1.921× | 1.587e-03 | 1.703e-04 |
| 8192 / 128 | 켬 | FP32 | 18.404 | 1.508× | 1.587e-03 | 1.703e-04 |
| 8192 / 128 | 끔 | FP16 | 14.321 | 1.938× | 2.289e-03 | 1.704e-04 |
| 8192 / 128 | 끔 | FP32 | 18.310 | 1.516× | 2.289e-03 | 1.704e-04 |

사전에 고정한 operator 기준은 **finite, max abs ≤ 0.02, relative RMSE ≤ 0.03, cosine ≥ 0.999**다. relative RMSE는 `RMSE(error) / RMS(reference)`이며 원소마다 3%라는 뜻이 아니다. 16개 후보가 모두 통과했고, relative RMSE는 **0.8471–0.9311%**, 최소 cosine은 **0.999956654**, 전체 최대 절대 오차는 **0.00402832**였다. 이미지용 기준도 finite 및 RGB max **2/255**, RMSE **0.25/255**로 미리 정했지만, **이 실험에서 이미지 기준을 평가하지는 않았다**.

추가로 smoothing 켬/FP32 PV의 처음 16개 query를 같은 양자화 Q/K를 복원한 native math와 비교했다. 네 shape의 최대 오차는 모두 **3.04e-05 이하**였다. 이는 원래 QK 양자화 오차와 streaming 구현 오차를 구분하기 위한 검사이며, 9.1의 독립 FP64 reference와 같은 검사는 아니다.

FP16 PV는 이번 네 shape와 smoothing 양쪽에서 FP32 PV보다 빨랐고, RMSE 차이는 작았다. smoothing은 네 shape 모두 최대 절대 오차를 낮췄지만 **RMSE까지 항상 낮추지는 않았다**. 예를 들어 S=8192/D64의 FP16 PV는 smoothing 켬 `1.5906e-4`, 끔 `1.5887e-4`였다. 평균 계산의 시간과 임시 할당도 추가된다. 따라서 smoothing은 오차 분포를 실제 입력에서 확인해 선택하며, 무조건 켜는 품질 보증 옵션으로 취급하지 않는다.

#### 실제 생성 ISA와 적용 결정

초기 실행 JSON의 생성 assembly에서 **`v_dot4c_i32_i8`가 D64에 64회, D128에 128회 나타났다**. 같은 기록에서 `v_mfma`, `v_wmma`는 0이었다. 이는 해당 커널이 실제 INT8 SIMD dot 명령을 생성했다는 A 근거다. 횟수는 assembly 텍스트의 정적 mnemonic 개수이며 런타임 실행 횟수나 FLOPS가 아니다. AMD의 gfx1030 명령 표에도 이 opcode가 명시되어 있다. [C: gfx1030 ISA 명령 표](https://rocm.docs.amd.com/projects/llvm-project/en/docs-7.14.1/LLVM/llvm/html/AMDGPU/AMDGPUAsmGFX1030.html)

초기 JSON은 `v_dot2_f32_f16`만 세어 dot2c 변형을 누락했다. 이를 FP16 packed dot의 부재로 해석하지 않았다. metadata를 보강한 별도 실행에서 **FP16 PV는 `v_dot2c_f32_f16`를 D64에 128회, D128에 256회 생성**하는 것을 확인했다. FP32 PV에서는 dot2c가 0이었다. QK의 dot4c는 두 PV 경로에 모두 존재하고 MFMA/WMMA는 없다. FP16 PV는 dot2의 FP32 accumulator를 사용하므로 NVIDIA tensor-core FP16 누산 경로가 필요하지 않다. [A: 생성 ISA·SHA256·register·spill 기록](results/quantized_attention_isa_v04.json)

FP16 PV는 D64 register 142–151, D128 187–241이며 spill 0이었다. FP32 PV는 두 D 모두 register 256, D64 spill 0·shared 8,192 bytes, **D128 spill 12–16·shared 16,384 bytes**였다. 높은 register 값만 보고 spill을 추정한 것이 아니라 compiler metadata의 실제 보고값이다. 이 자원 차이는 FP32 PV의 비용을 분석할 단서지만, 전체 느려진 시간 중 spill의 비중을 분리한 실험은 아니다. 보강 실행의 시간은 앞의 초기 paired 표에 섞지 않았다.

**적용 결정:** 연산 단위의 속도·메모리·고정 오차 gate는 성공했다. FP16 PV와 smoothing 선택은 후속 실제 입력 실험 후보로 유지한다. 실제 이미지·영상의 최종 품질 gate가 없는 상태에서 기본 Attention을 교체하지 않는다. 원래 모델이 FP32 Attention에 민감했던 결과를 이 FP16 synthetic 검증으로 덮을 수도 없다.

**9.1의 FP16 tiled는 H=2, 위 INT8 실험은 H=4이며 layout과 시간 샘플 방식도 다르다.** 두 표의 절대 시간이나 각자의 배속을 서로 나눠 INT8의 추가 이득을 계산하지 않는다. 이 선택 문제를 해결하기 위해 아래 동일 조건 비교를 따로 실행했다.

#### FP16 tiled와 동일 입력으로 직접 비교 — A

`--compare-tiled`를 켜고 **같은 H=4·contiguous FP16 Q/K/V**에서 native, FP16 tiled, INT8 네 후보를 같은 교대 round에 넣었다. tiled는 D64에서 32×64, D128에서 32×32다. 아래 INT8은 smoothing을 켰으며, 모든 시간에 각 방법의 준비/양자화·할당 비용을 동일하게 포함했다. 각 배속은 **해당 행의 tiled 시간 / INT8 시간**이다. [A: 직접 비교 원자료](results/quantized_vs_tiled_v04.json)

| S / D | FP16 tiled ms | INT8 QK + FP16 PV ms | tiled 대비 배속 | INT8 QK + FP32 PV ms | tiled 대비 배속 |
|---|---:|---:|---:|---:|---:|
| 4097 / 64 | 2.711449 | 2.164343 | 1.2528× | 2.772568 | 0.9780× |
| 4097 / 128 | 5.026571 | 3.611832 | 1.3917× | 4.924552 | 1.0207× |
| 8192 / 64 | 9.634185 | 8.306895 | 1.1598× | 10.772166 | 0.8944× |
| 8192 / 128 | 19.077662 | 14.554296 | 1.3108× | 19.405219 | 0.9831× |

모든 operator gate를 통과했다. FP16 tiled의 상대 RMSE는 약 **0.0306–0.0310%**였고, INT8 후보는 앞서 본 약 **0.85–0.93%** 수준이었다. 같은 trial의 tiled 추가 peak는 2.00/4.00/4.00/8.00 MiB, smoothing을 켠 INT8은 12.00/24.00/16.00/32.00 MiB였다. INT8은 full-score math보다 메모리를 크게 줄이지만 **양자화하지 않는 tiled보다 임시 메모리는 더 사용**한다.

따라서 FP16 tiled를 먼저 후보로 삼고, 약 1.16–1.39배의 추가 속도가 QK 양자화 오차와 임시 메모리를 감수할 가치가 있는지 실제 모델에서 판단한다. FP32 PV는 이 비교에서 3/4 shape가 느렸고 나머지도 약 2% 차이여서 범용 가속용으로 채택하지 않는다. FP32 PV의 arithmetic 선택지는 연구 API에 남긴다. 직접 비교의 assembly에는 FP32 PV에서 `v_fmac_f32`가 D64 280회, D128 552회 포함돼, FP16 PV의 dot2c와 다른 SIMD 계산 경로도 확인됐다. 수치는 정적 명령 개수다.

이 직접 비교 역시 **실제 1024px 이미지·영상, 긴 causal, FP32 원본 입력의 양자화 품질**을 검증한 결과는 아니다. 앞 절의 H2/H4 native 기준 시간을 여기로 옮겨 배속을 다시 계산하지 않는다.

재현은 기존 환경에서 다음과 같이 수행한다. 현재 benchmark의 metadata 보강 때문에 새 JSON에는 초기 기록보다 추가 필드가 들어갈 수 있다.

```bash
rdna2_python bench_quantized_attention.py --lengths 4097 8192 --dims 64 128 \
  --heads 4 --repeat 3 --output results/quantized_attention_recheck.json
rdna2_python bench_quantized_attention.py --compare-tiled \
  --lengths 4097 8192 --dims 64 128 --heads 4 --repeat 3 \
  --output results/quantized_vs_tiled_recheck.json
```

두 실험 모두 기존 PyTorch/Comfy auto 경로를 바꾸지 않는다. 직접 호출을 제거하고 기존 SDPA를 사용하면 원래 경로로 돌아간다. native가 긴 입력에서 score 메모리를 필요로 한다는 점은 호출자 측 메모리 예산에 포함한다.

### 9.3 FP16 GEMM: rocBLAS 대 Triton

**결정: 일반 FP16 GEMM의 기본값은 rocBLAS로 유지한다.** [구현](rdna2_ops/gemm.py)은 실제로 컴파일되고 정확성 기준을 통과했지만, 측정한 세 shape에서 rocBLAS보다 4.24–5.83배 느렸다. [A: 전체 원자료](results/gemm_v04.json)

입력·출력 FP16, FP32 accumulator, RHS는 contiguous weight의 전치 view다. Triton은 고정 tile `32×64×32`, 4 warps, 1 stage를 사용했다. 3 warmup 뒤 5 rounds × 10 iterations, 양쪽 순서 교대로 측정했고 할당·일반 dispatch를 포함했다. 작은 `37×53×79` tail shape 두 layout 및 각 큰 shape의 모든 K를 사용하는 32×32 출력 표본을 CPU FP64와 비교했다. 전체 출력은 rocBLAS와 비교했다.

| M / N / K | rocBLAS wall ms | Triton wall ms | native/Triton | 전체 출력 상대 L2 오차 | 결과/등급 |
|---|---:|---:|---:|---:|---|
| 1024 / 1024 / 1024 | 0.094643 | 0.401544 | 0.2357× | 1.5454e-5 | 정확성 통과, 속도 거절 / A |
| 2048 / 2048 / 2048 | 0.521285 | 3.038752 | 0.1715× | 1.8509e-5 | 정확성 통과, 속도 거절 / A |
| 1061 / 3072 / 3072 | 0.680357 | 3.595172 | 0.1892× | 2.0457e-5 | 정확성 통과, 속도 거절 / A |

고정 정확성 기준은 `atol=2e-3`, `rtol=2e-3`, 상대 L2 `≤1e-3`이다. 원소별 `abs(error)/(atol+rtol*abs(reference)) ≤1`도 모두 요구했다. 첫 shape의 native/Triton 최대 절대 차이는 0.0625이지만 출력 RMS가 약 31.99인 FP16 GEMM이다. 정규화하지 않은 최대값만 Attention/RGB 오차와 비교하지 않는다.

생성 ISA에는 `v_dot2c_f32_f16`가 정적 256회 포함됐다. register 104–105, spill 0, shared memory 4,096 bytes였고 MFMA/WMMA는 없었다. **실패 이유를 “RDNA2에서 FP16 dot을 못 써서”라고 설명할 수 없다.** 고정 단일-stage tiled 구현의 실제 처리 효율이 이 shape들의 rocBLAS kernel보다 낮았다. blocking·데이터 재사용·pipeline 중 각각의 기여는 이번에 분리 측정하지 않았다. 추가 순수 GEMM 재작성보다 복사 제거 또는 주변 연산 fusion에 이득이 있는지 먼저 확인하는 편이 이 결과에 맞는 선택이다.

비교/연구용 API는 남겨 둔다. 기본 운영 GEMM으로 자동 교체하지 않는다.

```python
from rdna2_ops.gemm import fp16_gemm
with torch.inference_mode():
    candidate = fp16_gemm(a, weight.T)
    reference = a @ weight.T
```

```bash
rdna2_python bench_gemm.py --probe-hipblaslt --output results/my_model_gemm.json
```

별도 benchmark 프로세스에서 `preferred_blas_library("cublas")`와 `ROCBLAS_USE_HIPBLASLT=0`을 사용하고 원래 설정을 기록한다. **C:** rocBLAS 자체도 내부 GEMM backend로 hipBLASLt를 선택할 수 있고 `ROCBLAS_USE_HIPBLASLT=1`은 실패 시 Tensile fallback을 허용하므로, 설정 이름만 보고 비교 라이브러리를 확정하지 않는다. [rocBLAS design notes](https://rocmdocs.amd.com/projects/rocBLAS/en/latest/conceptual/rocblas-design-notes.html), [PyTorch backend API](https://docs.pytorch.org/docs/main/backends.html)

**A — 현재 hipBLASLt 경로:** 별도 child에서 `cublaslt`를 명시 요청했지만 다음 경고가 발생했다.

```text
Attempting to use hipBLASLt on an unsupported architecture!
Overriding blas backend to hipblas
```

요청 직후와 `torch.mm` 직후의 getter 모두 `_BlasBackend.Cublas`였다. 64×64 FP16 곱은 정확히 완료됐지만 **hipBLASLt 지원 확인이 아니라 fallback 성공**이다. 현재 설치에서 PyTorch를 통한 gfx1030 hipBLASLt 사용은 불가로 판정한다. raw C API의 모든 operation을 시험한 결과는 아니다.

**B — 공식 빌드 소스:** AMD TheRock의 `gfx1030` target은 `EXCLUDE_TARGET_PROJECTS`에 `hipBLASLt`를 명시한다. 이 현재 빌드 정책과 실제 PyTorch probe가 같은 방향을 가리킨다. [TheRock target 목록](https://github.com/ROCm/TheRock/blob/main/cmake/therock_amdgpu_targets.cmake), [역사적 지원 범위 설명 #1062](https://github.com/ROCm/TheRock/issues/1062). 다른 GPU로 가장하는 architecture override나 재빌드는 사용하지 않았다.

**롤백:** `fp16_gemm(...)`를 기존 `torch.mm`/`F.linear` 경로로 되돌린다. benchmark의 환경변수는 해당 자식 프로세스에만 적용되며 운영 환경에 저장하지 않는다.

### 9.4 FP16/BF16 원인 분리

#### 왜 원래 FP32 math였는가

**B:** 설치된 Comfy의 MOSS 지원 dtype은 BF16/FP32이며, quantized loader는 `unet_manual_cast(None, ..., supported_dtypes)`를 통해 계산 dtype을 선택한다. gfx1030의 해당 BF16 계산 판정이 False여서 FP32를 선택했다. BF16 weight 저장과 FP32 compute는 동시에 성립한다. **A:** 실제 denoising QKV는 FP32였다. 설치된 Flash/Efficient backend를 FP32/FP16/BF16으로 강제한 검사에서는 사용할 gfx1030 kernel이 없었고, 실제 profile은 math였다. [설치 소스 경로](SOURCES.md), [이전 backend/precision 실험](results/production_precision.json)

#### 0.98의 RGB 오차는 단순 overflow였는가

**이번 실행에서는 정밀도 변환의 반올림만으로 큰 차이를 재현했다.** 같은 프로세스, seed `20261106`, 동일 conditioning에서 native를 두 번 실행하자 176 SDPA 출력 hash, 672 활성화 양자화 호출, denoiser 4회 입출력, 최종 latent/RGB가 모두 같았다. 따라서 이 A/B의 기준 자체가 흔들려 생긴 차이는 아니다.

다음 두 실행을 독립적으로 비교했다.

1. `fp16_math`: QKV를 FP16으로 변환, math reduction은 FP32, 출력을 원래 FP32 dtype으로 복원.
2. `fp16_qkv_output_roundtrip`: QKV를 FP16으로 반올림 후 FP32로 복원하여 **FP32 SDPA를 계산**하고, 그 출력만 FP16 반올림 후 FP32로 복원.

두 실행은 **176개 Attention 출력 hash, denoiser 4개 출력 hash, 최종 latent와 RGB hash까지 전부 같았다.** 낮은 정밀도 GEMM/softmax backend의 별도 결함을 가정하지 않아도, 변환 경계만으로 동일한 최종 결과를 만들 수 있다.

| 변경 | latent max / RMSE | RGB max / RMSE | 판정/등급 |
|---|---:|---:|---|
| Native 재실행 | 0 / 0 | 0 / 0 | 기준선 exact / A |
| FP16 math, FP32 reduction | 5.524097 / 0.639779 | **0.979736 / 0.167309** | 거절 / A |
| FP32 계산 + QKV·출력 FP16 roundtrip | 5.524097 / 0.639779 | **0.979736 / 0.167309** | 위 FP16 math와 전체 hash 동일 / A |
| FP32 계산 + 출력만 FP16 roundtrip | 6.561951 / 0.454110 | **0.990967 / 0.106226** | 거절 / A |
| FP32 계산 + QKV만 FP16 roundtrip | 4.577822 / 0.623685 | **0.970703 / 0.167343** | 거절 / A |
| FP16 reduced math | 4.387133 / 0.552617 | 0.993896 / 0.146454 | 거절 / A |

출력은 모두 finite였다. 사전 기준은 RGB `max ≤2/255`, `RMSE ≤0.25/255`로 유지했다. 결과를 보고 tolerance를 바꾸지 않았다. 이 표는 **정밀도 원인 진단**이며 CPU hash·중간 probe가 포함된 시간을 성능으로 사용하지 않는다. [분석 요약](results/precision_diagnosis_v04_fixed/analysis.json), 전체 진단 원자료 (`results/precision_diagnosis_v04_fixed/precision_diagnosis.json`, 연구 폴더에만 있음)

#### 실제 QKV와 최초 증폭 경계

선별한 8개 실제 QKV probe에서 Q/K/V 최대 절댓값은 각각 **10.5271 / 11.7252 / 24.9495**, FP32 logits 범위는 **−8.015824~13.114758**이었다. FP16/BF16 cast의 finite→nonfinite는 0이었다. FP16에서 K 원소 1개가 0으로 underflow했고, reduced FP16 경로는 작은 probability 일부가 0이 됐다. FP32 reduction 경로의 probability 최솟값은 약 `3.36e-9`였다. 검사한 scaled Q/K, logits, probability는 모두 finite였다.

FP32 clone 및 contiguous layout 대조는 8개 모두 exact였다. QK→softmax→PV의 shadow 분해도 실제 SDPA endpoint와 exact였으며, FP16/BF16 math 결과는 각각의 FP32 입출력 반올림 모형과 일치했다. 이는 **선별 8개 입력의 단계별 검사**이며 모든 모델·모든 layer의 overflow 부재를 보증하지 않는다. shadow 값도 내부 kernel을 직접 캡처한 값으로 부르지 않는다.

첫 S37 Attention의 FP16 math 국소 오차는 max `0.0034704`, RMSE `0.00019377`이었다. 이어지는 최초 INT8 활성화 양자화 입력 `[1061,3072]`에서 다음이 관측됐다.

| 경계 관측 | FP16 math | 출력만 FP16 roundtrip |
|---|---:|---:|
| 양자화 전 입력 차이 RMS | 2.91713e-5 | 1.70021e-5 |
| 변경된 INT8 code | 427개 | 250개 |
| code 최대 차이 | 1 | 1 |
| 양자화·scale로 복원한 값의 차이 RMS | 3.83048e-4 | 2.95176e-4 |
| 이 경계의 RMS 증폭 | **13.131배** | **17.361배** |

FP16 math의 변경 code는 전체의 약 0.01310%였고, 변경 code가 있는 행은 37/1061이었다. 반올림 경계 근처의 작은 차이가 INT8 code 한 단계 차이로 바뀐 것이다. 위 증폭은 **활성화 양자화 후 복원값의 비교**이며 전체 INT8 Linear 출력의 증폭률이 아니다. 또 최종 오차 중 INT8이 차지한 비율을 분리한 결과도 아니다.

첫 denoiser 입력은 exact였지만 첫 출력부터 차이 RMS가 **0.155628**이었다. 이후 네트워크 호출 출력 RMS 차이는 `0.487661 →0.642934 →0.639779`로 이어졌다. 즉, 첫 네트워크 통과에서 이미 차이가 커졌고 이후 달라진 latent가 다음 호출로 들어갔다. VAE만의 오류로 설명할 수 없다.

#### 짧은 Attention만 FP32로 남기면 되는가

추가로 같은 프로세스 안에서 native / S37-only / long-only를 비교했다. 이 추가 프로세스의 native를 기준으로 한 별도 실험이며 앞 표와 절대 오차를 병합하지 않는다.

| 직접 FP16으로 변경한 범위 | 변경 호출 수 | RGB max | RGB RMSE | 최초 양자화 경계 RMS 증폭 | 결과 |
|---|---:|---:|---:|---:|---|
| S37만 | 8회 | 0.972900 | 0.131347 | 13.347배 | 거절 / A |
| S1061만, S37은 native FP32 | 168회 | 0.997070 | 0.165029 | 10.442배 | 거절 / A |

Long-only의 최초 변경 SDPA index 2에 들어간 Q/K/V hash도 native와 같았다. 국소 오차 RMS `1.54029e-4`가 다음 활성화 양자화 경계에서 `1.60836e-3`로 커졌고 code 18,895개가 달라졌다. **짧은 conditioning 경로만 보호하면 긴 경로의 FP16화를 허용할 수 있다는 가정도 이 작업에서는 실패했다.** [scope 분석](results/precision_diagnosis_scopes_v04/analysis.json), [scope 원자료](results/precision_diagnosis_scopes_v04/precision_diagnosis.json)

따라서 이 MOSS INT8 사례에는 기존 FP32 의미 보존 경로를 유지한다. 다른 모델은 기존 native dtype에서 시작해 Q/K/V·출력 반올림, 첫 후속 양자화, denoiser 출력, 최종 이미지의 순서로 차이를 추적한다. VAE에서 FP16이 빨랐다는 사실이나 긴 합성 attention에서 FP16 tiled가 빨랐다는 사실을 denoiser precision 변경의 승인으로 사용하지 않는다.

진단 재현은 [precision_diagnose.py](precision_diagnose.py)와 [실제 Comfy harness](profile_comfy_workflow.py)의 `--precision-modes`를 사용한다. 기본 활성화 행 수 `(37,1061)`는 이 512² workflow의 명시적 진단 계약이다. 새 shape는 `quantizer_rows`를 실제 행 수로 지정한다. 초기 실패에서는 cold run의 LoRA 가중치 양자화 168회와 활성화 양자화 672회를 섞어 정합성 오류가 났고, 수정 후 **가중치 호출은 별도로 집계하면서 활성화 shape/call 정합성 검사는 유지**했다. 이를 수치 kernel 실패로 세지 않는다. 모든 진단 hook과 reduction flag는 context 종료 때 복원된다.

현재 MOSS workflow를 다시 진단하려면 GPU가 비어 있고 해당 port를 다른 서버가 사용하지 않을 때 실행한다. 이 harness는 요청 후 자신이 시작한 서버를 종료한다. 새 모델 진단은 `PrecisionDiagnosis`에 실제 denoiser·quantizer·shape를 연결해야 하며, 아래 MOSS 명령을 모델 공통 명령으로 취급하지 않는다.

```bash
MOSS_RDNA2_INT8_LAYOUT=view MOSS_RDNA2_ATTENTION=native \
MOSS_RDNA2_MEASURE_DIR= MOSS_RDNA2_STATS=0 \
rdna2_python profile_comfy_workflow.py --project "$RDNA2_MOSS" \
  --out results/my_moss_precision --port 8299 --seed 20261106 \
  --precision-modes native native:all fp16_math fp16_math_reduced \
    fp16_qkv_output_roundtrip fp16_output_roundtrip fp16_qkv_roundtrip
```

### 9.5 폐기·보류 후보 목록

| 후보 | 실패/보류의 직접 근거 | 이후 선택 | 등급 |
|---|---|---|---|
| HIP GN graph 자동 적용 | 과거 30.0425 µs 대 compiled 14.4825 µs, 약 2.07배 느림 | eager 승리 구간만; graph/compile은 native/compiler | A |
| MOSS의 선행 scaling을 GEMM 뒤로 옮기고 Triton softmax로 바꾼 Attention | operator 최대 약 7e-7에도 실제 RGB max 0.8528 | native 선행 scaling·softmax 보존. Qwen의 원래 후행 scale을 alpha에 접는 7.1절과 구분 | A, B |
| Qwen rocBLAS q/k/v contiguous 제거 | 비트 일치지만 29.85→60.82 ms, 16.38→30.57 ms로 약 2배 느림 | contiguous 준비 유지, scale alpha 후보 사용 | A |
| Qwen score mul_만 적용 | 29.85→29.49 ms, 16.38→16.10 ms로 이득 미미 | 별도 pass를 없앤 alpha 후보와 비교 | A |
| FP16/BF16 math 자동 강제 | 기존 세 seed 모두 품질 실패; 이번 FP16 반올림 원인 재현 및 short/long 독립 실패 | 모델별 최종 출력 통과 전 precision 변경 금지 | A |
| 고정 S1061만 지원하는 Attention 자동 조건 | 다른 shape 근거가 부족함 | 두 layout·20개 길이 검증 후 S128–4096 범위로 일반화 | A, B |
| S37 math 안전 검사 축소 | 0.058414→0.077564 ms로 더 느림 | 자동 선택에서 제외 | A |
| 전치 weight 별도 상주 캐시 | GEMM은 빨랐으나 계산상 추가 4.43 GiB, LoRA/gather/offload invalidation 미완성 | 최신 weight view 경로 유지 | A, B |
| FP32 D128 tiled의 속도 대체 | S8192에서 12.7868→23.5918 ms(16×32), 26.9806 ms(32×64) | 메모리 후보로만 명시; 속도 기본값은 native | A |
| INT8 QK의 무조건 모델 적용 | 합성 operator gate만 통과, 최종 이미지/영상 품질 미검증 | explicit 실험 API 유지 | A 범위 밖 |
| FP32 PV가 항상 유리하다는 가정 | INT8 FP16 PV보다 느림; 동일 입력의 FP16 tiled에도 3/4 shape에서 퇴보, D128 spill 관측 | 범용 가속 채택 거절; 필요한 arithmetic 연구용 유지 | A, B |
| 순수 FP16 Triton GEMM 대체 | 정확하지만 rocBLAS보다 4.24–5.83배 느림 | rocBLAS 유지 | A |
| hipBLASLt enable로 gfx1030 지원 확보 | 요청 후 getter가 hipBLAS이고 unsupported 경고 발생 | 실제 지원 경로 사용 | A, B |
| Qwen patientx를 새 dot4/dequant tile로 교체 | 6개 실제 shape에서 0.862–0.901배. 기존 spill 49–57 커널이 새 spill 0보다 빠름 | 기존 fused backend 유지; 9.6절 | A |
| 실제 D128 FP32 tiled를 spill만 줄여 속도 대체 | Qwen spill 0이어도 alpha 대비 2.40배 느림, MOSS도 느림 | 메모리 절약 연구 API로만 유지; 9.7절 | A, B |
| FP32 경계만 보호한 부분 FP16와 세-product 보정 | 5개 모두 최종 RGB 기준 실패. 입력 dtype FP32와 quantizer FP32를 확인해도 코드 경계 차이 잔존 | 비트 보존 INT8 융합 사용; 9.8절 | A |
| FP16 operand/FP32 output이 반드시 빠르다는 가정 | 이 설치 QK에서 HSS fallback 3.959 ms 대 FP32 1.540 ms | 실제 출력 dtype별 kernel/time 확인; 9.8절 | A, B, C |
| Qwen Sage-inspired QK INT8/PV FP16 | 연산 최대 오차 0.1497, 실제 RGB max 0.59082·RMSE 0.01356 | 빠르고 메모리가 작아도 적용 제외; 9.9절 | A |

새로운 실패는 “미지원”, “컴파일 실패”, “수치 실패”, “정확하지만 느림”, “전체 workload 미검증” 중 무엇인지 분리해서 기록한다. 통과한 operator를 전체 모델 통과로 승격하거나, 느린 한 tile을 모든 tiled 알고리즘의 실패로 일반화하지 않는다.

### 9.6 INT8 GEMM: dot4 타일과 FP32 dequant 융합

**v0.5의 질문은 같은 INT8이라는 이름의 두 backend를 실제로 함께 개선할 수 있는가였다.** MOSS는 INT32 GEMM 뒤에 eager FP32 dequant를 수행하고, Qwen patientx는 이미 GEMM과 per-row dequant를 융합한다. 따라서 같은 코드 교체의 이득을 두 모델에 미리 가정하지 않았다. 실제 호출 shape와 quantization/ConvRot/bias 계약은 [MOSS capture](results/v05_moss_capture_fixed/actual_call_paths.json), [Qwen capture](results/v05_qwen_capture/actual_call_paths.json)에서 확인했다.

#### 재사용 코드와 계산 계약

- [int8_gemm.py](rdna2_ops/int8_gemm.py): `int8_matmul`은 INT8×INT8→INT32, `int8_gemm_dequant`는 FP32 출력까지 합친 명시적 eager API다. `[M,K] @ [K,N]`의 양쪽 stride를 사용하므로 기존 `[N,K]` weight의 `.T` view를 받을 수 있다.
- 정수 누산 후 **`float(acc) * (row_scale * col_scale)`** 순서를 유지한다. MOSS는 `rounding="separate"`로 FP32 multiply/add contraction을 끄고, patientx 비교는 `rounding="fused"`로 원래 bias epilogue의 FMA 의미를 맞춘다. quantizer·ConvRot·input activation·residual은 기존 backend를 그대로 사용한다.
- FP32 scale은 contiguous device tensor, weight scale은 scalar 또는 output-channel별 값이다. autograd/graph capture/compile 자동 통합은 제공하지 않는다. 유한 scale/bias가 사전 조건이며 K≤131071로 임의 signed INT8 입력의 INT32 overflow를 막는다. 이것은 다른 INT8 형식의 scale/zero-point 계약까지 지원한다는 뜻이 아니다.
- `moss_int8_gemm_candidate`와 `qwen_int8_gemm_candidate`는 **독점 소유한 연구 프로세스 안에서만** callable을 바꾸고 context 종료 시 복구한다. 파일·바이트코드·운영 설정·가중치는 변경하지 않는다. 최종 MOSS 선택값은 tile `(64,128,64)`, warps 4, stages 2, group M 8, `wrapped=False`다.

```python
from rdna2_ops.int8_gemm import int8_gemm_dequant

# a_int8와 a_scale은 원래 모델의 quantizer가 만든 값이다.
# weight는 [N,K], w_scale은 [N] 또는 scalar FP32 tensor다.
y = int8_gemm_dequant(
    a_int8, weight.T, a_scale, w_scale,
    tile=(64, 128, 64), num_warps=4, num_stages=2, group_m=8,
    rounding="separate",  # 실제 backend의 rounding 계약에 맞춘다.
)
```

#### A — MOSS 네 가지 실제 shape에서 성공

아래는 **실제 모델에서 관측한 shape/layout에 합성 입력을 넣은 operator 측정**이다. 기존 view GEMM+eager dequant와 비교했다. JIT/입력 생성을 제외하고, ordinary synchronized wall·output allocation을 포함하여 3개 교대 round×5회 중앙값을 사용했다. 각 행은 INT32 oracle 표본과 native FP32 출력 비트 일치를 통과했다. 추가 peak는 이미 상주한 입력/weight를 제외한 allocator 증가량이다.

| M,N,K | 기존 GEMM+dequant ms | fused 후보 ms | 속도비 | 양자화까지 포함한 속도비 | 추가 peak MiB 기존→후보 | 판정/등급 |
|---|---:|---:|---:|---:|---:|---|
| 1061,3072,3072 | 2.017 | 0.698 | 2.89× | 2.56× | 62.17→12.43 | 성공 / A |
| 1061,9216,3072 | 6.078 | 1.869 | 3.25× | 3.09× | 190.00→38.00 | 성공 / A |
| 1061,16384,3072 | 11.006 | 3.302 | 3.33× | 3.22× | 331.56→66.31 | 성공 / A |
| 1061,3072,8192 | 5.599 | 1.932 | 2.90× | 2.55× | 62.17→12.43 | 성공 / A |

모든 행에서 같은 `(64,128,64)` 설정이 이겼다. ISA에서 `v_dot4` 계열을 확인했고 VGPR 188–203, spill 0이었다. 별도 INT32 결과 및 scale 곱 중간 tensor를 없앤 결과 추가 peak가 약 1/5로 줄었다. 타일 변경과 dequant 융합을 함께 비교했으므로 두 변경의 기여율을 따로 측정했다고 해석하지 않는다. allocator peak는 driver의 private spill scratch까지 포함한 총 VRAM 수치가 아니다. `v_dot4`가 보인다는 사실만으로 성능을 판정하지 않고, 타일 및 stage에 따른 실제 시간도 함께 기록했다. [네 shape 원본](results/int8_gemm_moss_shapes_v05.json), [첫 sweep과 실패 후보](results/int8_gemm_moss_sweep_v05.json), [측정기](bench_int8_gemm.py)

#### A — 실제 MOSS 이미지 생성에서 성공

기준선은 **이미 INT8 transpose view와 `MOSS_RDNA2_ATTENTION=auto`가 켜진 기존 경로**다. 같은 연구 서버·워크플로·seed 20261106·4 denoising steps에서 `warm,baseline,int8,baseline,int8,int8,baseline`을 실행했다. warm과 첫 candidate 호출을 정상상태 비교에서 제외하고 native 3회/candidate 2회의 중앙값을 사용했다. profiler는 껐다.

| 측정 구간 | 기존 기준선 | fused INT8 | 속도비 | 최종 결과 |
|---|---:|---:|---:|---|
| Denoising | 4.8510 s | 2.1012 s | 2.31× | latent byte-exact |
| Denoising + VAE | **4.8977 s** | **2.1487 s** | **2.279×** | RGB byte-exact, max/RMSE 0 |

이 비교는 디노이징+VAE 구간이며 모델 로드·클라이언트·PNG 저장까지의 총 요청 시간과 다르다. 후자는 원본 `request_wall_s`에 별도 저장했다. 모든 7요청이 native reference와 latent/RGB 비트 일치했고, candidate 요청마다 **672 hits / 0 fallbacks**, 위 네 shape 각각 168회였다. native를 다시 호출한 마지막 요청도 일치하여 context 복구를 확인했다. 별도 weight 전치 상주 캐시는 추가하지 않았다. [전체 A/B](results/v05_moss_int8_tuned_ab/runs.json), [실행 명세](results/v05_moss_int8_tuned_ab/run_spec.json), [실제 런타임](results/v05_moss_int8_tuned_ab/runtime_metadata.json)

초기 `(64,64,64)` 후보도 전체 1.587×와 비트 일치에 성공했지만, 최종 선택은 위 넓은 N tile이다. 초기 결과는 [첫 전체 A/B](results/v05_moss_int8_ab/runs.json)에 남겼다. 이 수치에 이전 세션의 1.56×나 Qwen의 1.084×를 곱하지 않는다.

```bash
# 2.3.4의 기존 rdna2_python 함수를 사용한다. 결과 경로는 새 이름으로 지정.
rdna2_python research_comfy_v05.py \
  --spec research_configs/moss_int8_tuned_v05.json \
  --out results/my_moss_int8_ab \
  --modes warm,baseline,int8,baseline,int8,int8,baseline --port 8463
```

운영 launcher나 custom node에 자동 등록하지 않았다. 연구 호출을 끝내면 원래 callable로 돌아간다. 다른 모델로 옮길 때는 6절의 backend 구분과 이 절의 수치 계약을 다시 확인하고, 실제 shape별 측정과 최종 출력 gate를 수행한다.

#### A — Qwen 여섯 shape에서는 실패, 기존 patientx 유지

같은 방법으로 **관측한 shape/layout의 합성 INT8 입력**을 사용했다. 후보 중 최선 설정은 `(64,128,64)`, warps 4, stages 2, group M 8, `wrapped=True`, `partial_dot=False`였다. 큰 행렬의 전체 FP32 출력이 native와 비트 일치했고, INT64 독립 표본 및 finite 검사도 통과했다. 속도비는 기존 시간÷후보 시간이다.

| M,N,K | patientx ms | 새 후보 ms | 속도비 | 결과/등급 |
|---|---:|---:|---:|---|
| 2400,4096,4096 | 2.321 | 2.614 | 0.888× | 성능 실패 / A |
| 2400,24576,4096 | 13.820 | 15.347 | 0.901× | 성능 실패 / A |
| 2400,4096,12288 | 6.932 | 7.885 | 0.879× | 성능 실패 / A |
| 4964,4096,4096 | 4.760 | 5.400 | 0.882× | 성능 실패 / A |
| 4964,24576,4096 | 27.687 | 31.164 | 0.888× | 성능 실패 / A |
| 4964,4096,12288 | 13.692 | 15.881 | 0.862× | 성능 실패 / A |

원래 patientx quantizer를 포함해도 0.899–0.908×였다. 연산 단계에서 모든 shape가 느려져 **실제 Qwen 전체 이미지 A/B는 이 INT8 후보로 수행하지 않았다**. Qwen 모델 활성화로 검증했다거나 최종 이미지까지 일치했다고 확대하지 않는다. [여섯 shape 원본](results/int8_gemm_qwen_shapes_v05.json)

원래 autotuner는 여섯 shape 모두 `(128,256,64) / 8 warps / 3 stages / group M 8`을 선택했다. 기존 커널은 VGPR 256, spills 49–57인데도 VGPR 188, spill 0인 새 후보보다 빨랐다. **spill 감소는 측정할 중간 지표이며 최종 성능의 대리값이 아니다.**

| 폐기·중간 후보 | 실제 관측 | 원인 판단의 범위 | 근거 |
|---|---|---|---|
| Qwen 작은 타일 8개 | 최선 0.493× | MOSS의 작은 타일을 공통 채택할 수 없음 | [A: 첫 sweep](results/int8_gemm_qwen_sweep_v05.json) |
| 큰 타일/stage 10개 | 최선 0.878×; 64×128 stage 2에서 spill 0 | register pressure 완화에 효과가 있어도 native보다 느림 | [A: 큰 타일](results/int8_gemm_qwen_large_tiles_v05.json) |
| M/N modulo load 및 K 정렬 분기 3개 | 최선 0.893× | 해당 load predicate 제거만으로 차이가 해소되지 않음 | [A: wrapped load](results/int8_gemm_qwen_wrapped_v05.json) |
| `dot(a,b,acc)`→`acc += dot(a,b)` 2개 | 최선 약 2.610→2.611 ms, 큰 타일 약 9.892→9.882 ms | 이 표현 변경만으로 명령 배치·자원·성능이 개선되지 않음 | [A: 누산식 비교](results/int8_gemm_qwen_partial_dot_v05.json) |

**B — 확인한 구조적 원인:** Qwen은 이미 GEMM/dequant가 융합되어 MOSS에서 제거한 중간 행렬 비용이 남아 있지 않다. **미확정:** 동일한 큰 타일을 사용해도 독립 후보와 원래 patientx 사이에 남은 내부 성능 차이의 정확한 원인은 이번 범위에서 확정하지 못했다. 위 가설 검사로 compiler scheduling이나 predication 일반이 무관하다고 결론 내리지 않는다.

이 절의 큰 shape 시간은 기존 `[N,K]` weight의 `.T` view를 사용하는 `weight_nk` layout이다. 물리 전치한 `rhs_kn` layout은 작은 정확성 검사만 했고, 큰 shape 성능을 새로 비교한 것은 아니다. scalar/per-channel scale, bias, tail, offset/padding의 작은 검사도 보존했지만 실제 모델 hot path는 모두 bias=False였다. JSON의 `accepted: true`는 정확성 조건 통과를 뜻하며 성능 채택을 뜻하지 않는다.

```bash
rdna2_python bench_int8_gemm.py \
  --cases 'qwen:2400,4096,4096;qwen:2400,24576,4096;qwen:2400,4096,12288;qwen:4964,4096,4096;qwen:4964,24576,4096;qwen:4964,4096,12288' \
  --configs '64,128,64,4,2,8' --wrapped \
  --output results/my_qwen_int8_shapes.json
```

### 9.7 FP32 tiled Attention: D128 spill 제거와 속도 한계

**판정: register spill 제거와 선형 메모리는 성공(A), 현재 최적 native 경로보다 빠르게 만드는 목표는 실패(A).** 실제 Qwen prefix-cache와 MOSS에서 FP32 정밀도를 유지한 채 검사한 모든 후보가 사전 연산 정확성 기준을 통과했다. 최선 pure Triton 후보도 Qwen `split_alpha`보다 약 2.40배, MOSS `exact_math`보다 약 2.08배 느렸다. 운영 자동 적용은 추가하지 않는다. 속도 탈락 때문에 최종 RGB A/B를 진행하지 않았으므로, RGB 허용 기준을 통과한 후보로 읽으면 안 된다.

#### 실제 입력과 비교 경로

| 실제 캡처 | BHLD Q / K,V | dtype와 layout | native 비교 경로 |
|---|---|---|---|
| Qwen prefix cache | `[1,32,2400,128]` / `[1,32,4964,128]` | FP32, 원래 `[B,S,H*D]` interleaved stride 복원 | 실제 ComfyUI `attention_split`, rehearsal `split_view`가 구현하는 `split_alpha` |
| Qwen 짧은 cache | `[1,32,2400,128]` / `[1,32,2408,128]` | FP32, 원래 stride 복원 | 위와 동일 |
| MOSS | `[1,24,1061,128]` / 동일 | FP32, Q/K stride `(3259392,128,3072,1)`, V stride `(9778176,128,9216,1)` | 강제 PyTorch math SDPA, 기존 `rdna2_ops.attention.scaled_dot_product_attention(backend="rdna2")` (벤치 alias `exact_math`) |

캡처 원본: `results/v05_qwen_capture/actual_qkv_2400_4964.pt`, `actual_qkv_2400_2408.pt`, `results/v05_moss_capture_fixed/actual_qkv_1061_1061.pt`. 각 디렉터리의 `actual_call_paths.json`은 함수 위치·dtype·stride 근거다. 원본 캡처는 연구 PC에 있으며 대용량 tensor를 배포 묶음에 중복하지 않았다.

Qwen 비교는 `comfy/ldm/modules/attention.py`의 `attention_split`을 직접 import했다. native와 alpha 후보 모두 head-contiguous rocBLAS 입력을 유지했다. `split_alpha`는 이 실제 입력들에서 native 출력과 비트가 일치했다. Qwen은 QK 이후 scale, MOSS math SDPA는 Q/K 각각에 `sqrt(scale)`을 곱한 뒤 dot하는 순서를 보존했다. softmax는 전 K 범위의 FP32 정규화다.

Comfy의 `@wrap_attn`은 `transformer_options.optimized_attention_override`와 `AttentionTensorContainer`도 처리한다. 따라서 함수 이름만 보고 모든 운영 호출을 이 경로로 판정하거나 wrapper를 무조건 바꾸면 안 된다. 이 benchmark는 캡처한 **mask 없는 실제 tensor**만 비교하며 운영 wrapper를 교체하지 않았다.

#### 측정 계약

- 환경: PyTorch `2.12.0+rocm10.2.0a20261011`, HIP `7.17.26405`, Triton `3.8.0`, 실제 arch `gfx1030`. 각 JSON에 PyTorch git hash와 파일 SHA256을 저장했다.
- 시간: GPU를 단독 사용하고 warmup 1회 후 3 round × 3회 일반 호출, 후보 순서를 round마다 앞/뒤로 바꾸어 측정했다. 표는 round 평균 시간의 중앙값이다. CUDA/HIP event 시간도 JSON에 있다. 컴파일·입력 생성은 제외하고 Python 계약 검사·출력 할당·Qwen 출력 layout 변환은 포함했다. 그래프 재생 결과가 아니다.
- 메모리: 입력과 참조 출력이 상주하는 상태에서 `torch.cuda.reset_peak_memory_stats`로 측정한 **추가 PyTorch allocated peak**다. 드라이버 scratch, 예약 allocator 전체, 모델 상주량은 포함하지 않는다. 표의 MiB는 2^20 byte다.
- 연산 정확성: 실행 전에 `torch.allclose(rtol=3e-4, atol=3e-5)`와 finite를 고정했다. 이는 원소별 `atol + rtol*abs(reference)` 판정이며 max-abs가 반드시 3e-5 이하여야 한다는 뜻은 아니다. 전체 native 출력 외에 두 head × 네 query row의 FP64 참조도 별도로 계산했다.
- 작은 입력 검사: 각 신규 tile에 rectangular tail을 적용하고 causal upper-left, D64, 음수/0 scale, split-K 등의 계약을 검사했다. initial 23건, rolled-D 15건, PV-chunk 12건이 통과했으며 공통 edge case가 중복되므로 50개의 고유 사례라고 세지 않는다. 실제 Qwen/MOSS 비교는 full tensor를 검사했다.
- 최종 영상 품질 gate는 여전히 RGB max `2/255`, RMSE `0.25/255`다. **이번 pure/hybrid 후보의 최종 RGB는 미측정**이다. 연산 allclose를 RGB 통과로 승격하지 않는다.

#### 대표 결과

각 pure 최종 행은 해당 최종 벤치의 native/alpha 기준과 비교한다. Hybrid는 별도 initial sweep의 같은 round 기준이다. native 기준의 run 간 작은 시간 차이를 섞어 속도비를 과장하지 않는다.

| 입력·방법 | 일반 호출 ms | 추가 peak MiB | native 대비 max abs / RMSE | 판정·근거 |
|---|---:|---:|---|---|
| Qwen K4964 `attention_split` (최종 PV run) | 29.753 | 3140.094 | 0 / 0 | 비교 기준 A |
| Qwen K4964 `split_alpha` (동일 run) | **22.787** | 3102.594 | 0 / 0 | 비트 일치 기준 A |
| Qwen K4964 pure `q32k64d16v128w8p16` | **54.597** | **75.500** | 2.754e-5 / 3.778e-7 | 정확성·선형 메모리 성공, 속도 실패 A |
| Qwen K4964 hybrid query128 (initial run) | 28.203 | 309.063 | 2.313e-5 / 3.179e-7 | 같은 run의 native 29.803보다 빠르나 alpha 22.927보다 느림; 비트 불일치 A |
| Qwen K2408 `attention_split` | 16.307 | 1563.500 | 0 / 0 | 비교 기준 A |
| Qwen K2408 `split_alpha` | **12.865** | 1526.000 | 0 / 0 | 비트 일치 기준 A |
| Qwen K2408 pure `q32k64d16v128w8p16` | 25.655 | **76.000** | 4.745e-5 / 8.506e-7 | 정확성·선형 메모리 성공, 속도 실패 A |
| Qwen K2408 hybrid query128 | 20.567 | 189.125 | 4.780e-5 / 8.284e-7 | 비트 불일치, 속도 실패 A |
| MOSS native math (최종 run) | 2.569 | 246.458 | 0 / 0 | 비교 기준 A |
| MOSS `exact_math` (동일 run) | **1.874** | 245.301 | 0 / 0 | 비트 일치 기준 A |
| MOSS pure `q32k64d16v128w8p16` | 3.889 | **12.434** | 1.907e-5 / 3.165e-7 | 정확성·선형 메모리 성공, 속도 실패 A |
| MOSS hybrid query128 (initial run) | 2.905 | 62.168 | 1.526e-5 / 3.284e-7 | 같은 run의 native 2.608와 exact 1.886 모두보다 느림 A |

Qwen K4964의 단일 full-score는 1.420 GiB다. pure 최선은 추가 allocated peak를 alpha의 3102.594 MiB에서 75.5 MiB로 약 41.1배 줄였다. 이 수치는 전체 모델 VRAM 41.1배 절감을 뜻하지 않는다. Qwen pure 출력의 BHLD→BSHD 변환이 동시에 보유하는 두 output 버퍼까지 시간·메모리에 포함되어 있다. MOSS는 BHLD 출력 하나만 필요하다.

Pure kernel은 global 메모리에 전체 score/probability나 query-block score를 만들지 않는다. 고정 tile의 score·online softmax 상태·출력 누산을 유지하고, split-K를 쓸 때만 고정 개수의 선형 partial을 만든다. 반면 **rocBLAS hybrid에는 `[BH, query_block, K]` global score tile이 있다**. query_block을 32/64/128로 고정하여 메모리가 선형인 별도의 방법이지 score-free/flash kernel은 아니다.

#### 전체 bounded 후보와 실패 보존

26개 신규 tile 후보와 v0.4 두 기준을 비교했다. 무제한 autotune은 하지 않았다. 아래 표는 모두 Qwen K4964이며 연산 정확성 기준을 통과했고, 모든 신규 후보가 alpha보다 느려 속도 개선 후보로는 탈락했다. `s4`는 split-K 4, `r`은 D inner loop unroll 금지, `p16`은 PV reduction을 16단위로 분리한 경우다. 전체 파라미터 tuple, 실제 stride, 시간 샘플, 메모리 byte, 오차, ISA resource는 원본 JSON에 있다.

| 후보 | 시간 ms | 추가 peak MiB | registers / spills | 증거 파일 |
|---|---:|---:|---|---|
| `v04_16x32` | 91.736 | 75.500 | 256 / 206 | initial A |
| `v04_32x64` | 77.739 | 75.500 | 256 / 926 | initial A |
| `q16k16v128w4` | 96.557 | 75.500 | 256 / 126 | initial A |
| `q16k32v128w8` | 150.566 | 75.500 | 256 / 184 | initial A |
| `q32k32v128w8` | 71.125 | 75.500 | 256 / 195 | initial A |
| `q32k64v128w8` | 102.645 | 75.500 | 256 / 615 | initial A |
| `q32k32v128w16` | 103.362 | 75.500 | 256 / 175 | initial A |
| `q16k32d64v128w4` | 94.103 | 75.500 | 256 / 214 | initial A |
| `q32k32d64v128w8` | 72.889 | 75.500 | 256 / 194 | initial A |
| `q32k32d64v64w4` | 123.747 | 75.500 | 256 / 462 | initial A |
| `q16k32d64v64w4` | 186.170 | 75.500 | 256 / 186 | initial A |
| `q32k64d64v64w8` | 197.786 | 75.500 | 256 / 583 | initial A |
| `q16k32d64v32w4` | 282.701 | 75.500 | 256 / 106 | initial A |
| `q32k32d64v128w8s4` | 72.978 | 190.344 | 256 / 193 | initial A |
| `q32k32d64v64w4s4` | 124.974 | 190.344 | 256 / 463 | initial A |
| `q64k32d64v128w8` | 78.106 | 75.500 | 256 / 611 | initial A |
| `q64k32d64v128w16` | 84.218 | 75.500 | 256 / 182 | initial A |
| `q16k16d16v128w4r` | 82.814 | 75.500 | 134 / 0 | rolled-D A |
| `q32k16d16v128w8r` | 55.832 | 75.500 | 127 / 0 | rolled-D A |
| `q64k16d16v128w16r` | 61.040 | 75.500 | 126 / 0 | rolled-D A |
| `q16k16d32v128w4r` | 93.968 | 75.500 | 133 / 0 | rolled-D A |
| `q32k16d32v128w8r` | 98.409 | 75.500 | 129 / 0 | rolled-D A |
| `q32k32d16v64w8r` | 168.120 | 75.500 | 184 / 0 | rolled-D A |
| `q32k16d16v64w4r` | 156.290 | 75.500 | 122 / 0 | rolled-D A |
| `q32k32d16v128w8p16` | 83.187 | 75.500 | 127 / 0 | PV16 A |
| `q32k64d16v128w8p16` | 54.597 | 75.500 | 186 / 0 | PV16 A |
| `q64k32d16v128w16p16` | 88.469 | 75.500 | 126 / 0 | PV16 A |
| `q64k64d16v128w16p16` | 61.254 | 75.500 | 186 / 0 | PV16 A |

Hybrid fixed query block 비교:

| query block | Qwen K4964 ms / MiB | Qwen K2408 ms / MiB | MOSS ms / MiB |
|---:|---|---|---|
| 32 | 56.681 / 250.891 | 33.455 / 160.906 | 4.298 / 52.843 |
| 64 | 30.566 / 270.281 | 23.324 / 170.313 | 3.677 / 55.951 |
| 128 | 28.203 / 309.063 | 20.567 / 189.125 | 2.905 / 62.168 |

MOSS initial 15개 신규 tile의 전체 결과도 `fp32_attention_moss_actual_sweep_v05.json`에 보존했다. 최종 두 no-spill 후보의 MOSS 시간은 `q32k16d16v128w8r` 4.358 ms, `q32k64d16v128w8p16` 3.889 ms다. Qwen K2408에서는 각각 27.322 / 25.655 ms였다.

#### D128 spill 원인: 확인한 것과 남은 추정

##### 확인한 변화

1. **v0.4와 초기 신규 구현의 spill은 실제로 컸다(A).** Qwen에서 v0.4 16×32는 registers 256 / spills 206 / private segment 824 byte, 32×64는 256 / 926 / 3704 byte였다. 새 초기 `q32k32d64v128w8`도 256 / 194 / 776 byte였다. 추측으로 붙인 spill 설명이 아니다.
2. **단순 D64 분리는 operand 수명을 줄이지 못했다(B: 생성 IR).** 초기 TTGIR에서 두 `[32,64]` Q operand load가 K/V streaming loop 밖으로 이동되어 모두 살아 있었다. Python `tl.static_range`로 D128을 두 번 나누는 것만으로는 전체 D128 live state를 해제하지 못했다. 해당 `.ttgir`, `.amdgcn`, 초기 소스 snapshot을 저장했다.
3. **진짜 D inner loop + 작은 K tile은 spill을 제거했다(A/B).** `tl.range(0,D,BD,loop_unroll_factor=1)`와 BD16/BK16을 사용한 rolled-D 후보 7개 모두 spill 0, private segment 0이 됐다. 대표 `q32k16d16v128w8r`은 registers 127, Qwen 55.832 ms였다. static D 분리, 더 많은 warps, PV 출력 channel 분할만 했던 초기 후보들은 이 결과를 내지 못했다.
4. **PV reduction도 16씩 처리하면 BK64에서도 spill 0을 유지했다(A/B).** `tl.gather`로 현재 score tile의 P 16열을 가져와 V16×D128을 곱하는 실제 loop를 추가했다. 최선 `q32k64d16v128w8p16`은 Qwen K4964 registers 186 / spills 0, MOSS 187 / 0, K2408 191 / 0이었다. 모두 private segment 0이다. Qwen 최선 ISA에는 `v_fma_f32`가 있고, dot2/dot4/MFMA/WMMA와 scratch load/store는 없다.
5. **spill 0은 속도 성공 조건이 아니었다(A).** 가장 빨라진 pure 후보도 alpha/exact보다 느렸다. PV channel32/64 분리는 QK 반복을 늘렸고, split-K4는 partial write/reduce와 peak 증가가 생겼다. 해당 후보들은 실제로 개선되지 않았다.

##### 아직 원인으로 확정하지 않은 것

작은 FP32 dot, D/PV loop와 layout conversion, P gather, 반복 Q/K/V load, online softmax reduction, partial merge의 비용이 남은 성능 차이에 기여할 가능성이 있다. 생성 코드에는 이 연산들이 존재하지만 하드웨어 counter로 각 비용의 비중을 분해하지 않았다. 따라서 “spill이 전부였다”, “Infinity Cache 때문에 hybrid가 빨랐다”, “gfx1030 FP32 flash는 원리적으로 rocBLAS를 이길 수 없다”는 결론은 근거가 부족하다. 이번 측정이 확인한 것은 **이 bounded 구현군에서는 메모리 절감과 spill 해결을 달성해도 native 최적 경로의 시간을 이기지 못했다**는 범위다.

#### 재사용 API와 적용 범위

```python
from rdna2_ops.fp32_attention import fp32_attention

# Qwen과 같은 FP32 post-scale, mask 없는 명시적 연구 호출
out = fp32_attention(
    q, k, v, scale_mode="post",
    block_q=32, block_k=64, block_d=16,
    block_v=128, num_warps=8, split_k=1,
    unroll_d=False, block_pv=16,
)
# MOSS math 의미론 비교에서는 scale_mode="sdpa".
```

입력은 BHLD FP32, D64/128, Q/K/V 같은 B/H/D, K/V 같은 길이, 마지막 stride 1이고 outer stride는 0 이상이어야 한다. 채널 외의 interleaved layout과 storage offset을 읽으며 외부 복사를 강제하지 않는다. 커널은 gfx1030, 양의 길이, 유한 입력과 overflow 없는 dot을 전제로 한 forward-only 연구 함수다. `is_causal=True`는 upper-left 정렬이며 prefix-cache 위치 offset을 자동 추론하지 않는다. arbitrary mask/dropout/GQA/backward/graph capture/compile은 지원 계약에 없다. overflow·NaN 탐지용 전체 GPU 입력 스캔은 하지 않는다. 배치/주소/출력 인덱스 범위와 dtype·autocast·autograd 등에 대한 host guard는 함수에 있다.

split-K1의 추가 tensor는 출력뿐이다. split-K>1은 고정 `split_k * B*H*L*(D+2)` FP32 partial/stat storage를 더한다. 따라서 고정 D/H/split_k에서 storage는 L과 S에 선형이다. split-K4는 실측 속도 이득이 없어서 기본 추천이 아니다.

`rocblas_query_tiled`는 `bench_fp32_attention.py`에 남긴 비교 helper다. Q/K/V head-contiguous packing, Qwen `baddbmm(alpha=scale)` 또는 MOSS pre-scale, 고정 query score buffer, native full-K softmax, PV `bmm`을 수행한다. 모델 채택용 dispatch를 추가하지 않았다.

#### 재현 명령과 원본

기존 Qwen venv를 그대로 사용한다. 연구 루트에서 실행하며 새 환경 생성·패키지 변경은 필요 없다. GPU를 다른 실험과 공유 실행하지 않는다.

```bash
~/DevSpace/qwen-image-21-rdna2-lab01/.venv/bin/python -u bench_fp32_attention.py \
  --models qwen \
  --captured results/v05_qwen_capture/actual_qkv_2400_4964.pt \
  --tiles q32k16d16v128w8r q32k64d16v128w8p16 \
  --output results/fp32_attention_recheck_qwen.json

~/DevSpace/qwen-image-21-rdna2-lab01/.venv/bin/python -u bench_fp32_attention.py \
  --models moss \
  --captured results/v05_moss_capture_fixed/actual_qkv_1061_1061.pt \
  --tiles q32k16d16v128w8r q32k64d16v128w8p16 \
  --skip-small --output results/fp32_attention_recheck_moss.json
```

K2408 확인은 Qwen capture 파일명만 `actual_qkv_2400_2408.pt`로 바꾼다. 결과 파일은 새 이름으로 남겨 기존 증거를 덮어쓰지 않는다. 모든 tile 이름은 benchmark `TILES`에 정의되어 있다. 초기 unroll 경로도 현재 소스에 남아 있지만 초기 원문 SHA까지 재현하려면 `results/fp32_attention_initial_source.py` snapshot을 참조한다.

원본 결과 파일(A):

- `results/fp32_attention_qwen_actual_sweep_v05.json`: initial 15개, v0.4 두 개, hybrid 3개, 작은 계약 검사.
- `results/fp32_attention_moss_actual_sweep_v05.json`: MOSS initial 15개, v0.4, hybrid.
- `results/fp32_attention_qwen_rolled_d_v05.json`: D 실제 loop 7개, 모든 spill 0.
- `results/fp32_attention_qwen_pv_chunks_v05.json`: PV16 chunk 4개, Qwen K4964 최선.
- `results/fp32_attention_moss_no_spill_v05.json`: MOSS 최종 두 개.
- `results/fp32_attention_qwen_k2408_v05.json`: 실제 짧은 cache, 최종 두 개와 hybrid.

소스/생성 코드(B), resource 측정(A):

- `rdna2_ops/fp32_attention.py` SHA256 `e64bf1223bd3ed91c8e89c4762b40d75e79a3878c5f2f6722d1542218dab1308`.
- `bench_fp32_attention.py` SHA256 `32bddf8aad31ff7c4334edc167aea3fa9f4a1064a30b7d63c35420009c80db74`.
- `results/fp32_attention_initial_source.py` SHA256 `7b845dc4b0d53adff93fdfde8c79c3aab330e3801617dd19c5fe137b0ad05665`.
- `results/fp32_attention_initial_d64.ttgir` / `.amdgcn`: Q operand hoist와 초기 spill이 있는 컴파일 결과.
- `results/fp32_attention_qwen_no_spill.ttgir` / `.amdgcn`: 최선 Qwen K4964 `q32k64d16v128w8p16` 생성 코드. ASM SHA256 `11937be724f1d416176fd7ba0cc776cb63184104fb75aa6fac6113dfcec1ac64`.
- `results/fp32_attention_isa_exports_v05.json`: 저장한 최종 ISA의 cache 원본 위치·SHA256. GPU를 다시 실행하지 않고 cache의 ASM hash를 결과 JSON과 일치시켜 추출했다. MOSS와 K2408의 ISA resource/hash는 해당 결과 JSON에 있으며 이 export는 Qwen K4964 파일을 담는다.

운영 MOSS·Qwen 설정, 시작 스크립트, custom_nodes 소스는 이 실험에서 수정하지 않았다. 코드·실패 기록은 재사용 연구용으로 남기고 자동 적용 정책에는 추가하지 않는다.

실행 코드: [FP32 kernel](rdna2_ops/fp32_attention.py), [벤치](bench_fp32_attention.py). 주요 근거: [Qwen 초기](results/fp32_attention_qwen_actual_sweep_v05.json), [D 실제 loop](results/fp32_attention_qwen_rolled_d_v05.json), [PV16](results/fp32_attention_qwen_pv_chunks_v05.json), [MOSS 최종](results/fp32_attention_moss_no_spill_v05.json), [K2408](results/fp32_attention_qwen_k2408_v05.json), [ISA 원본 대응](results/fp32_attention_isa_exports_v05.json).

### 9.8 INT8 ConvRot의 FP32 경계 보호와 부분 FP16 계산

**결론: 이번 MOSS에서 경계를 FP32로 유지한 다섯 후보 모두 최종 RGB 기준에 실패했다(A).** 출력 dtype을 보존하고 누산 결과를 직접 FP32로 저장해도, FP16 operand 반올림에서 시작한 값의 차이는 양자화 직전에 사라지지 않았다. 또한 이 설치의 FP16 입력→FP32 출력 BMM은 느린 fallback 커널을 선택했다. 정확도 실패와 성능 실패의 근거를 따로 확인했다.

#### 시험한 경계와 control

[mixed_attention.py](rdna2_ops/mixed_attention.py)는 Q/K 선행 scale, softmax, Attention 출력 저장, 주변 정규화·RoPE·ConvRot·양자화·dequant·residual을 FP32로 유지한다. 선택한 QK/PV BMM의 **operand만 FP16으로 변환**하고 `torch.bmm(..., out_dtype=torch.float32)`로 직접 FP32 결과를 얻는다. FP16 출력에 나중에 `.float()`를 적용하는 방법과 다르다. 기존 short S37 호출 8회는 그대로 두고 long S1061 호출 168회만 교체했다.

먼저 동일 분해를 전부 FP32로 수행하는 `mixed_control`을 두었다. 실제 모델에서 latent와 RGB가 byte-exact여야 다음 후보로 넘어가도록 `require_bitwise_modes=["mixed_control"]`을 설정했다. control은 통과했고, 모든 후보 요청에서 quantizer 호출 672회 중 FP32가 아닌 입력은 **0회**였다. 첫 QKV 양자화 경계는 모든 후보에서 reference와 동일했고, 바로 다음 output projection 경계를 별도로 비교했다. 컨텍스트 종료 후 native 요청을 다시 돌려 원본 latent/RGB 비트 일치도 확인했다.

세 번의 GEMM을 사용하는 보정 후보도 두 개 포함했다. FP32 A/B를 FP16 high와 `4096×잔차` FP16 low로 나누고 `AhBh + (AhBl + AlBh)/4096`을 계산한다. low×low 항은 생략하며 FP32 덧셈 순서도 원래 GEMM과 다르므로 **정확한 FP32 재현 알고리즘이 아니다**. 이 후보는 작은 operand 반올림 오차를 더 줄였을 때 실제 이미지가 회복되는지 확인하기 위한 별도 근사 실험이다.

#### A — 실제 QKV의 일반 호출 성능과 오차

이 표는 앞서 저장한 실제 MOSS QKV의 **원래 stride까지 복원한** 연산 벤치다. `.cpu()`가 압축할 수 있는 packed V의 storage gap을 `empty_strided + copy_`로 되살렸다. 3 warmup 후 3개 교대 round×3회, cast와 output allocation 포함 wall 중앙값이다. native reference는 강제 math SDPA다.

| 방식 | 일반 호출 ms | native 대비 속도비 | native 출력 max abs / RMSE | 판정 |
|---|---:|---:|---|---|
| Native FP32 math | 2.555 | 1.000× | 0 / 0 | 기준선 |
| FP32 control | 2.744 | 0.931× | 0 / 0 | 비트 일치 control |
| QK FP16 operand, PV FP32 | 5.959 | 0.429× | 0.006478 / 8.530e-5 | 느림, 근사 오차 |
| QK FP32, PV FP16 operand | 3.171 | 0.806× | 0.010916 / 7.444e-5 | 느림, 근사 오차 |
| QK/PV 모두 FP16 operand | 6.393 | 0.400× | 0.011749 / 1.131e-4 | 느림, 근사 오차 |
| QK 세-product 보정, PV FP32 | 16.083 | 0.159× | 1.144e-5 / 2.392e-7 | 더 느림, 작은 오차 잔존 |
| QK FP32, PV 세-product 보정 | 6.334 | 0.403× | 1.431e-5 / 3.115e-7 | 더 느림, 작은 오차 잔존 |

캡처 Q/K/V의 최대 절댓값은 10.406 / 9.109 / 24.949였고 FP16 변환도 finite였다. 이 첫 입력에서 나타난 오차는 FP16 범위 초과를 필요로 하지 않는다. 연산 입력은 이전 capture이고 다음 whole-model 요청은 새 연구 서버의 동일 conditioning을 사용한 별도 A/B다. 두 실행의 시간을 더하거나 출력 tensor가 동일한 세션이라고 해석하지 않는다. [원본 연산 결과](results/mixed_attention_actual_qkv_v05.json), [벤치](bench_mixed_attention.py)

#### A — 실제 MOSS 최종 RGB와 첫 output projection 경계

같은 연구 서버에서 4-step workflow와 seed 20261106을 유지했다. 허용 기준은 처음부터 **RGB max≤2/255=0.007843137, RMSE≤0.25/255=0.000980392**다. 아래 오차는 [0,1] decoded RGB tensor 기준이다. boundary 표본은 첫 output projection의 ConvRot 후, INT8 양자화 직전/직후 `[1061,3072]` activation이다.

| 후보 | 최종 RGB max abs | RGB RMSE | 변한 INT8 코드 수 / 행 수 | 양자화 전→후 RMSE | 그 경계의 RMS 비율 | 판정/등급 |
|---|---:|---:|---:|---|---:|---|
| FP32 control | 0 | 0 | 0 / 0 | 0→0 | 해당 없음 | 성공 control / A |
| QK만 half operand | 0.974365 | 0.148057 | 8,923 / 1,061 | 8.407e-5→1.072e-3 | 12.75× | RGB 실패 / A |
| PV만 half operand | 0.965332 | 0.137429 | 6,358 / 1,056 | 7.470e-5→9.734e-4 | 13.03× | RGB 실패 / A |
| QK/PV half operand | 0.967041 | 0.143073 | 11,143 / 1,061 | 1.117e-4→1.261e-3 | 11.29× | RGB 실패 / A |
| QK 세-product 보정 | 0.968994 | 0.125612 | 34 / 34 | 2.895e-7→6.586e-5 | 227.53× | RGB 실패 / A |
| PV 세-product 보정 | 0.981934 | 0.137567 | 64 / 64 | 3.574e-7→8.248e-5 | 230.80× | RGB 실패 / A |

보정 후보의 비율이 더 크다고 절대 오차가 더 커졌다는 뜻은 아니다. 양자화 전 RMSE가 매우 작아져 분모가 작고, 양자화 후 절대 RMSE도 half 단독 후보보다 작다. 확인한 것은 **작은 FP32 값 차이도 불연속 INT8 코드 경계를 넘을 수 있으며, 첫 경계에서 수십 개의 코드만 달라진 후보도 이번 최종 RGB 기준을 통과하지 못했다**는 사실이다. 이 비율은 한 경계의 측정값이며 모든 레이어·모델에 적용되는 고정 증폭 상수가 아니다.

모든 출력은 finite였고 FP32 control 및 마지막 native 복귀 요청은 비트 일치했다. 따라서 이번 실패를 “경계가 실수로 FP16으로 바뀌었다”거나 “control부터 다른 구현이었다”로 설명할 수 없다. FP16 operand 반올림 또는 보정 산술 차이→FP32 activation 차이→INT8 코드 차이→최종 출력 편차를 관측했다. 모든 후속 비선형 연산의 기여율을 각각 분해한 것은 아니다. 이 결과는 MOSS의 해당 INT8 ConvRot workflow에 대한 거절 근거이며, 다른 비양자화 모델에서 부분 FP16이 언제나 실패한다는 주장은 아니다. [전체 요청과 경계 원본](results/v05_moss_mixed_boundaries/runs.json)

이 full-model 실행에는 두 경계의 CPU 복사와 오차 진단이 denoising wall 안에 포함된다. control과 후보의 진단 비용도 같지 않으므로 **그 wall을 정상 inference 속도비로 사용하지 않는다**. 위 일반 호출 표와 다음 BMM profile을 성능 근거로 사용한다.

#### FP16→FP32 출력 BMM이 느렸던 실제 경로

FP16 operand를 미리 준비한 뒤 BMM만 세 번씩 별도 프로파일했다. 표는 세 번 GPU activity 시간 합÷3이며 cast·전체 Attention 시간은 포함하지 않는다. 같은 측정의 operator 귀속 시간과 kernel 시간을 더하지 않았다.

| 실제 MOSS BMM | FP32 입력→FP32 출력 | FP16 입력→FP16 출력 | FP16 입력→FP32 출력 |
|---|---:|---:|---:|
| QK | 1.540091 ms | 0.435558 ms | **3.959237 ms** |
| PV | 0.353439 ms | 0.202705 ms | **0.465972 ms** |
| 실제 선택 kernel family | `SB … ISA1030` | `HHS … ISA1030` | **`HSS … ISA000`** |

**A — 실행 증거:** FP16→FP32 QK는 FP32 BMM보다 2.571배, FP16 출력 경로보다 9.090배 오래 걸렸다. PV에서도 각각 1.318배·2.299배였다. 프로파일에 cast/copy가 없으므로 선택된 GEMM 커널 자체의 비용 차이가 존재한다. [실제 BMM profile](results/mixed_gemm_backends_v05.json)

**B — 설치 파일:** 해당 rocBLAS의 `HS_HPA`에는 네 transpose 조합 모두 `fallback_gfx1030.dat/.hsaco`만 있었고, `HH_HPA`와 `SS`에는 gfx1030 전용 `.dat/.co`도 있었다. 실제 HSS kernel은 64×32×8 / wave64, HHS는 128×256×16 / wave32를 선택했다. 출력 dtype의 조합에 따라 다른 커널군이 선택된 것이 이번 성능 실패의 직접 관측 근거다. 그 타일 내부의 cache/occupancy 비용 비중까지 counter로 분해한 것은 아니다. [설치 파일·header·symbol 기록](results/fp16_fp32_dispatch_source_v05.json)

**B/C — 계산 dtype 의미:** 공식 [PyTorch CUDABlas.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDABlas.cpp)의 ROCm half BMM helper와 [rocBLAS programming guide](https://rocm.docs.amd.com/projects/rocBLAS/en/develop/how-to/Programmers_Guide.html)는 HHS/HSS 모두 compute dtype을 FP32로 구분한다. 바뀐 것은 출력 저장 dtype과 그에 맞는 kernel 선택이며, FP32 누산이 처음 추가되어 느려졌다는 설명은 맞지 않는다. 소스 설명은 확인 시점의 공식 main이고, wheel에 구현 `.cpp`가 없어 설치 git commit과 파일 전체 동일성까지 입증한 것은 아니다. 실제 HHS/HSS 선택과 시간은 설치 버전의 A 실측이다.

#### 새 모델에서의 판단과 재현

FP32 경계 보존은 dtype 계약의 보존이다. 경계에 들어오는 **값**이 달라져도 양자화 코드가 같은지, scale과 후속 레이어까지 영향이 없는지는 별도로 확인해야 한다. 이번 모델에는 부분 FP16이나 보정 후보를 자동 적용하지 않는다. 대신 9.6절의 정확한 INT8 융합처럼 원래 수치 순서를 보존하는 최적화를 사용한다.

```bash
rdna2_python bench_mixed_attention.py \
  --input results/v05_moss_capture_fixed/actual_qkv_1061_1061.pt \
  --out results/my_mixed_operator.json --profile-gemm

rdna2_python research_comfy_v05.py \
  --spec research_configs/moss_mixed_v05.json \
  --out results/my_mixed_boundaries \
  --modes warm,baseline,mixed_control,mixed_qk,mixed_pv,mixed_both,mixed_qk_split,mixed_pv_split,baseline \
  --port 8463
```

`mixed_probes=false`는 진단 copy를 끄는 옵션이지만, 이번에는 품질 gate에서 모두 실패해 그 모드로 정상상태 full-model 성능을 더 측정하지 않았다. 연구 컨텍스트는 기존 callable과 precision flag를 복구하며 운영 파일은 변경하지 않는다.

### 9.9 Sage-inspired rectangular Attention: 가속 가능성과 실제 이미지 실패

**결론: gfx1030에서 INT8 QK dot4와 FP16 PV dot2를 연결해 실제 Qwen의 K4964 입력을 더 적은 메모리로 실행했고, 가장 빠른 tile은 alpha 기준보다 1.129배 빨랐다. 그러나 전체 operator 최대 오차와 실제 최종 RGB가 모두 기준을 넘어서 적용 후보에서는 제외했다(A).** Sparse/Radial/Sage 중 이번에는 기존 dot4·online-softmax 구현을 재사용할 수 있는 Sage 방향을 선택했다. 이는 공식 SageAttention 패키지의 지원/성능 주장이 아니라 독립 구현이다.

#### 구현 계약과 비교 방법

[sage_attention_v05.py](rdna2_ops/sage_attention_v05.py)는 FP32 Q `[B,H,L,D]`, K/V `[B,H,K,D]`를 받는다. Q/K를 token별 symmetric INT8로 양자화하고 INT32 dot4 결과를 FP32로 dequant한다. FP32 online softmax, FP16 P/V operand의 dot2, FP32 output accumulator를 사용한다. K의 sequence 평균 제거를 켠 경우와 끈 경우를 비교했다. K4964와 Q2400을 독립 길이로 처리하며 전역 L×K score/probability 행렬을 만들지 않는다. 고정 D에서 양자화 buffer·scale·출력의 크기는 O(L+K)다.

지원 계약은 finite FP32, gfx1030, D64/128, unmasked/noncausal inference다. dropout/GQA/autocast/gradient/compile/graph 및 모델 override는 이 실험의 적용 범위 밖이다. V가 유한한 FP16으로 표현될 수 있는 범위인지 호출자가 확인한다. 검증한 실제 Qwen 입력의 max abs는 Q 25.574, K 24.036, V 53.719였고 모두 finite였다. 이 사례의 실패는 FP16 overflow를 필요로 하지 않는다.

작은 rectangular Q17/K37/D64·D128에서 smooth on/off 네 경우를 CPU FP64 reference와 비교했다. 성능은 실제 capture의 B1/H32/Q2400/K4964/D128, 원래 stride와 동일한 값으로 측정했다. **mean 계산·QK 양자화·할당·Qwen output layout 복원까지 포함**, 3개 교대 round의 wall 중앙값이다. 메모리는 resident 입력/reference를 제외한 PyTorch allocator의 추가 peak이며 driver/private allocation 전체가 아니다. [벤치](bench_sage_qwen_v05.py), [operator 원본](results/v05_sage_qwen_4964.json)

#### A — 실제 Qwen operator 결과

사전 operator 기준은 finite, max abs≤0.02, relative RMSE≤0.03, cosine≥0.999였다. 다음 오차는 native `attention_split` 대비다. 같은 입력에서 alpha baseline은 native와 비트 일치했다.

| 방식 | wall ms | alpha 대비 속도비 | 추가 peak MiB | max abs / RMSE | 판정 |
|---|---:|---:|---:|---|---|
| Native `attention_split` | 29.662 | 0.769× | 3140.594 | 0 / 0 | 원래 기준 |
| 후행 scale alpha 접기 | **22.814** | 1.000× | 3102.594 | 0 / 0 | 비트 보존 기준 |
| FP32 query-block 128 hybrid | 28.220 | 0.808× | 310.000 | 2.313e-5 / 3.179e-7 | 정확하지만 느림 |
| Sage smooth, Q16/K32 | 22.293 | 1.023× | 128.016 | 0.149712 / 0.001831 | 최대 오차 실패 |
| Sage smooth, Q32/K32 | **20.211** | **1.129×** | 128.016 | 0.149712 / 0.001831 | 최대 오차 실패 |
| Sage no-smooth, Q16/K32 | 22.527 | 1.013× | 76.000 | 0.107472 / 0.001858 | 최대 오차 실패 |
| Sage no-smooth, Q32/K32 | 20.802 | 1.097× | 76.000 | 0.107472 / 0.001858 | 최대 오차 실패 |

네 Sage 후보 모두 relative RMSE 약 0.0091–0.0092와 cosine 기준은 통과했지만 **최대 오차 기준을 실패했다**. K 평균 제거 후보는 두 head의 네 query row를 뽑은 FP64 표본 검사를 통과했어도 전체 tensor의 최대 오차는 통과하지 않았다. 평균 제거가 없는 후보는 이 표본 검사에서도 최대 오차 0.029516으로 실패했다. 적은 row 표본을 실제 전체 품질 승인으로 대체하면 안 되는 사례다. K 평균 제거는 여기서 RMS를 조금 줄이는 대신 최대 오차를 키웠다. QK 양자화와 PV 반올림의 오차 기여를 별도 ablation으로 완전히 분해한 것은 아니다.

#### A/B — 실제 생성 ISA와 spill

Q16 tile은 registers 234, spills 0, private segment 0 byte, Q32 tile은 registers 256, spills 103, private segment 412 byte였다. 둘 다 shared memory 8192 byte였다. 실제 assembly에서 Q16의 정적 dot4c/dot2c 수는 128/256, Q32는 256/512였고 MFMA/WMMA는 없었다. 이 수는 정적 명령어 개수이며 실행 횟수나 처리량 자체가 아니다.

더 빠른 Q32에는 spill이 남았다. 따라서 “spill이 0이면 항상 더 빠르다”는 선택 기준은 이 Sage 결과와 9.6절의 patientx 결과에서 모두 맞지 않는다. 메타데이터의 spill/private size와 실제 wall을 함께 비교한다. 단순 `scratch` 문자열 검색 결과가 0이라는 사실만으로 spill이 없다고 판정하지 않는다. 원본 JSON의 kernel resource와 instruction 요약에 근거를 보존했다.

#### A — 최종 Qwen RGB 진단

operator gate는 이미 실패했지만, 사용자가 지정한 최종 RGB 기준에 대한 직접 판정을 위해 별도의 실제 8단계 Qwen 이미지 요청을 실행했다. **기준을 완화하거나 운영 설정을 바꾸지 않았다.** 같은 연구 서버에서 warm → native → Sage → native를 실행했고, B1/H32/Q2400/K4964/D128인 무마스크 256회만 smooth Q32/K32/warps4 후보를 사용했다. 나머지 96회는 기존 함수였다. masked/container/override 등 계약 밖 입력도 원래 callable로 fallback한다.

| 항목 | 결과 | 사전 기준 | 판정 |
|---|---:|---:|---|
| Latent max abs / RMSE | 2.492601 / 0.127028 | 진단 기록 | 비트 불일치 |
| 최종 decoded RGB max abs | **0.5908203125** | ≤0.007843137 (2/255) | 실패 |
| 최종 decoded RGB RMSE | **0.0135610486** | ≤0.000980392 (0.25/255) | 실패 |
| 마지막 native 복귀 latent/RGB | 모든 byte 일치 | warm/native 기준 일치 | 복원 control 성공 |

RGB 최대 차이는 255배 척도로 약 150.66, RMSE는 약 3.46이다. 모두 finite였고 같은 요청의 conditioning embedding hash도 유지됐다. [전체 RGB 진단 원본](results/v05_qwen_sage_rgb/runs.json), [실제 실행 metadata](results/v05_qwen_sage_rgb/runtime_metadata.json)

이 요청의 denoise+VAE 시간은 native 전/후 **22.526/22.536초**, Sage **19.994초**였다. candidate는 한 번뿐이며 첫 모델 후보 실행이므로 이 수치를 반복 검증된 가속 대표값으로 사용하지 않는다. 또한 이 whole-model 기준은 **원래 attention_split**이고 alpha를 접은 기준은 아니다. operator 표의 1.129배와 전체 요청의 시간을 섞어 alpha 대비 전체 가속을 주장하지 않는다. 품질 기준에 실패했으므로 추가 반복 성능 검증이나 자동 적용 범위 확대는 하지 않았다.

#### 재현과 적용 결정

```bash
RDNA2_QWEN_PYTHON=~/DevSpace/qwen-image-21-rdna2-lab01/.venv/bin/python
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
  "$RDNA2_QWEN_PYTHON" bench_sage_qwen_v05.py \
  --captured results/v05_qwen_capture/actual_qkv_2400_4964.pt \
  --output results/my_sage_qwen.json --repeat 3

HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
  "$RDNA2_QWEN_PYTHON" research_comfy_v05.py \
  --spec research_configs/qwen_sage_v05.json \
  --out results/my_sage_rgb --steps 8 --port 8464 \
  --modes warm,baseline,sage_qwen4964,baseline
```

코드는 기능 포팅과 실패 재현을 위해 남긴다. 이 Qwen에 INT8 QK/PV 근사를 자동 적용하지 않는다. 현재 실측상 선택은 비트 보존 alpha 접기이며, 더 작은 메모리가 반드시 필요한 경우에는 9.7절의 FP32 tiled 비용과 최종 출력 검증을 별도로 판단한다. Sparse/Radial의 모델별 sparsity 근거나 다른 Sage 양자화 정책의 품질은 이번에 측정한 범위가 아니다.

### 9.10 rocBLAS의 dtype·shape·layout별 선택

**질문:** V 전치가 Anima에서는 빠르고 Qwen에서는 느렸다면 새 연산에서 무엇을 기준으로 고를 것인가? 기준선은 이 설치본의 `torch.mm`/`torch.bmm`이 실제 선택한 rocBLAS 커널이다. 라이브러리 버전·dtype·논리 shape뿐 아니라 **실제 stride와 변환 비용**을 한 묶음으로 판단한다. 코드: [`rdna2_ops/layout_blas_v06.py`](rdna2_ops/layout_blas_v06.py), 실행기: [`bench_layout_blas_v06.py`](bench_layout_blas_v06.py).

#### A — 여섯 shape × 두 dtype × 네 layout

Anima/Qwen/MOSS에서 관측한 P·V shape 세 개와 tall-skinny/projection mm 세 개를 사용했다. 입력은 동일 값의 seed 고정 합성 데이터(`std=0.125`)이고 실제 softmax 확률은 아니다. FP16/FP32 각각 NN·NT·TN·STRIDED를 세 warmup 후 세 교대 round×세 반복으로 측정했다. prepared는 변환된 입력이 있을 때, copy-inclusive는 NN 원본에서 변환·할당·GEMM을 모두 포함한 시간이다. **NN/NT/TN은 PyTorch 논리 tensor의 저장 형태를 뜻하며 rocBLAS 내부 transA/transB flag와 동일한 이름 체계가 아니다.**

| (batch,M,N,K) | dtype | NN 준비/복사 포함 wall ms | NT 준비/복사 포함 wall ms | TN 준비/복사 포함 wall ms | STRIDED 준비/복사 포함 wall ms |
|---|---|---:|---:|---:|---:|
| (32,4096,128,4096) | FP16 | 22.632 / 27.654 | 6.570 / 7.091 | 5.391 / 266.912 | 10.890 / 13.589 |
| (32,4096,128,4096) | FP32 | 10.145 / 10.142 | 10.409 / 11.758 | 6.387 / 300.626 | 10.442 / 11.173 |
| (32,2400,128,4964) | FP16 | 2.979 / 3.034 | 3.064 / 3.896 | 2.669 / 16.757 | 2.730 / 2.735 |
| (32,2400,128,4964) | FP32 | 5.952 / 5.996 | 5.782 / 6.986 | 5.533 / 17.343 | 5.961 / 5.872 |
| (24,1061,128,1061) | FP16 | 0.218 / 0.218 | 0.224 / 0.336 | 0.211 / 0.824 | 0.218 / 0.244 |
| (24,1061,128,1061) | FP32 | 0.372 / 0.366 | 0.377 / 0.536 | 0.343 / 1.000 | 0.366 / 0.415 |
| (mm,4096,64,4096) | FP16 | 0.172 / 0.175 | 0.309 / 0.318 | 0.124 / 8.396 | 0.172 / 0.182 |
| (mm,4096,64,4096) | FP32 | 0.272 / 0.275 | 0.505 / 0.526 | 0.225 / 9.437 | 0.288 / 0.294 |
| (mm,4096,128,8192) | FP16 | 0.389 / 0.389 | 0.593 / 0.630 | 0.282 / 18.111 | 0.401 / 0.412 |
| (mm,4096,128,8192) | FP32 | 0.800 / 0.795 | 0.704 / 0.771 | 0.449 / 22.673 | 0.924 / 0.956 |
| (mm,1061,3072,3072) | FP16 | 0.612 / 0.608 | 0.695 / 2.658 | 0.563 / 0.836 | 0.656 / 0.709 |
| (mm,1061,3072,3072) | FP32 | 1.019 / 1.012 | 1.086 / 4.674 | 0.972 / 1.363 | 1.149 / 1.368 |

[원자료와 모든 round·메모리·오차](results/layout_blas_v06.json), [48개 선택 커널과 전체 이름 사전](results/LAYOUT_BLAS_V06_KO.md). 원자료 SHA-256: `b9d1f28118dbc4016a69ccaf2d106b0b4f8d02a6a94607a4ad1924b01c9a600c`. GPU audit 340 samples, foreign client 0, 오류 0으로 유효하다. KFD 감시는 그래픽 부하나 polling 사이의 짧은 client까지 증명하지 않는다.

Anima형 FP16 NN은 `Cijk_Ailk_Bljk_HHS_BH_MT128x128x16...`을, NT는 `Cijk_Alik_Bljk_HHS_BH_MT128x256x16...`을 선택했다. 전체 이름과 시간은 위 사전에 보존했다. NN의 prepared 20.654–23.066 ms와 copy-inclusive 23.022–30.711 ms는 흔들림이 컸고 NT copy-inclusive는 7.069–7.101 ms였다. 이 결과에서 NT의 큰 이득은 유지되지만 특정 배속을 모든 실행에 보장하지 않는다.

#### 새 연산에서 바로 쓰는 규칙

| 실측 범위 | 적용 판단 |
|---|---|
| FP16 bmm B32/M4096/N128/K4096, A/B row-major | 작은 B만 `B.transpose(-2,-1).contiguous().transpose(-2,-1)`로 바꾸는 후보를 먼저 시험한다. 입력 값은 보존되지만 커널 반올림은 달라지므로 최종 출력 관문은 필수다. |
| 같은 shape FP32 또는 Qwen형 FP32 B32/M2400/N128/K4964 | NN을 유지한다. 이번 NT 변환 포함 시간은 각각 10.142→11.758, 5.996→6.986 ms로 퇴보했다. |
| MOSS형 B24/M1061/N128/K1061 | FP16/FP32 모두 NN 유지. 작은 준비 시간 차이를 위해 새 복사를 추가하지 않는다. |
| mm M4096/N64/K4096, 또는 FP16 M4096/N128/K8192 | NN 유지. tall-skinny라는 이름만으로 NT가 빠르다고 판단하지 않는다. |
| FP32 mm M4096/N128/K8192 | B가 이미 column-major일 때 NT를 비교 후보로 쓴다. 새 변환까지 포함한 이득 약 3%는 자동 적용 근거로 부족하다. 이 shape의 모델 출력은 ❔다. |
| TN 전체 | A가 원래 column-major일 때만 prepared 시간을 사용한다. 매번 큰 A를 전치해 넣는 방식은 이번 범위에서 거절한다. |
| mm column stride 2 | 내부 pack이 생길 수 있다. 이번 세 mm shape에서 direct-copy 커널이 실제 관측됐다. |

STRIDED bmm은 B-only batch/head-interleaved 저장의 한 예다. Anima B2/H16 전체 view 또는 Qwen의 Q/K/V 모두 contiguous 제거를 그대로 재현한 것이 아니다. 따라서 기존 Qwen 리허설의 **0.49배 실패**를 작은 B-only 변형 결과로 취소하지 않는다.

#### 정확도와 실패 기록

48조합 모두 전체 NN 기준 및 CPU FP64 표본 허용 오차를 통과했다. FP16 `rtol=5e-3, atol=5e-3`, FP32 `rtol=3e-4, atol=3e-5`이며 표본은 mm 4행×4열, bmm 두 batch 각각 4행×4열이다. 이것은 최종 RGB 판정이 아니다. prepared와 copy-inclusive는 47조합에서 비트가 같았고 FP32 mm M4096/N128/K8192 TN만 max `9.536743e-7` 차이가 있었다. `GSU4`와 추가 `Cijk_S.kd` 커널이 관측되어 split reduction 순서가 가능한 원인이지만 원인을 확정하지 않았다. 이 후보는 전치 포함 22.673 ms라 성능상 이미 거절한다.

모델 관문은 같은 프로세스·conditioning·seed의 Anima에서 `manual_nn`과 `manual_vt`를 비교하는 §9.11에 연결한다. NN→NT만 바뀐 최종 출력 차이와 native SDPA→직접 경로 전체의 차이를 별도 기록한다.

### 9.11 FP16 Attention: 빠른 경로와 비트 보존 경로를 분리한다

**기준선은 기존 Anima 직접 GEMM+softmax+V 전치 경로(`manual_vt`)다.** native SDPA보다 빠르다는 이유만으로 이 기준선보다 개선됐다고 부르지 않는다. 새 구현 [`rdna2_ops/fp16_attention_v06.py`](rdna2_ops/fp16_attention_v06.py)는 score 출력 재사용, Triton in-place softmax, Q/K sqrt-scale 유지, scale+pack 융합, 4D matmul 보존을 비교한다. adapter는 sampling context 동안 하위 `comfy.ops.scaled_dot_product_attention`만 바꾸고 종료 시 복원한다. 운영 자동 적용은 추가하지 않았다.

#### A — 실제 입력 세 shape, 33개 비교

실제 Anima self `[2,16,4096,128]`, cross Q 동일/KV `[2,16,512,128]`와 self 입력을 S2048로 자른 대표 shape를 사용했다. crop은 모델에서 관측한 별도 호출이라고 주장하지 않는다. 세 warmup, 세 교대 round×두 반복이며 packing·전치·할당을 포함했다. 전체 native 비교와 독립 CPU FP64 표본에 대해 max≤0.02, relative RMSE≤0.005를 연산 관문으로 정했고 33조합 모두 통과했다. 모델 RGB 관문은 별도다. 연산 벤치의 `byte_equal` 필드는 `torch.equal`을 사용하므로 signed-zero 바이트까지 구분하지 않는 dtype·값 일치 검사다. 최종 모델의 비트 일치는 별도의 tensor digest로 확인한다.

| 경로 | self4096 wall ms | cross512 wall ms | crop2048 wall ms | self 추가 peak GiB | native 출력 |
|---|---:|---:|---:|---:|---|
| native math SDPA | 67.690 | 3.843 | 10.415 | 2.5314 | 기준 |
| 기존 직접 VT | 22.561 | 2.951 | 6.194 | 2.1250 | 비트 다름 |
| 출력 buffer 재사용 | 22.431 | 2.959 | 6.168 | 2.1250 | 비트 다름 |
| in-place softmax+VT | 23.087 | 2.924 | 6.156 | 1.1250 | 비트 다름 |
| sqrt-prescale+flatten bmm+VT | 21.860 | 2.306 | 5.990 | 2.1250 | 비트 다름, 중간 RMSE 감소 |
| Triton scale+pack+VT | 22.359 | 2.823 | 6.140 | 2.1250 | 위와 같은 관측 오차 |
| scale+pack+in-place softmax+VT | 22.757 | 2.756 | 6.012 | 1.1250 | 비트 다름 |
| 원래 4D matmul+native softmax+작은 row repair | 54.332 | 2.715 | 7.954 | 2.0625 | **세 shape 값 완전 일치** |

메모리는 입력 위의 PyTorch allocated peak이며 driver/private allocation을 포함하지 않는다. 모든 round·33조합·FP64 오차·실제 커널 이름·Triton resource는 [원자료](results/fp16_attention_v06_expanded.json), 전체 방법/실패 표는 [상세 결과](results/FP16_ATTENTION_V06_KO.md)에 있다. GPU audit207 samples, foreign0/error0으로 유효했다.

**확인된 원인:** scale 순서만 보존하는 3D reshape 경로는 충분하지 않았다. self4096에서 native와 4D 보존 경로의 QK는 `Ailk_Bljk/HHS_BH/MT128x128x16...`이고 flatten bmm은 `Alik_Bljk/HHS_BH/MT128x64x8...`를 선택했다. 이는 self4096의 예시이며 cross/crop native QK는 각각 `MT128x256x16`/`MT128x16x8`로 다르다. 여기서는 rank뿐 아니라 K의 scale·transpose·reshape 순서도 바뀌므로 중간 materialization과 library 선택을 함께 보아야 한다. 중간 stride를 모두 기록하지 않은 상태에서 특정 원인 하나로 확정하지 않는다. V 전치도 PV의 다른 누산·반올림 경로를 고른다. native self profile 한 회에서 `isneginf`/reduce/where가 3.611/1.526/4.497 ms였고, 작은 row repair는 별도 profile에서 약0.73 ms였다. 이는 진단 profile 시간이며 warm wall 중앙값과 합치지 않는다.

캡처된 Q/K/V는 finite였고 unscaled QK의 보수적 절대 상한 `D*max|Q|*max|K|`는 세 입력에서 2180–2367로 FP16 최대65504보다 작았다. **이 캡처에서 생긴 오차의 원인을 QK overflow라고 할 수 없다.** 이후 모든 step을 이 bound로 증명한 것은 아니다. 또한 native에 가까운 출력이 FP64 실수 연산에도 더 가까운 것은 아니다. cross FP64 표본에서는 기존 manual 경로의 RMSE가 native보다 작았다. 최종 strict 기준은 원본 출력 재현성을 측정한다.

#### A — 최종 Anima 1024²·30step·seed20261107

한 서버에서 동일 conditioning으로 10요청을 돌렸다. 첫 warm과 두 native 대조는 latent/RGB가 비트 일치했고, 후보는 매 요청 self840+cross840=1680회 적중/fallback0이었다. VAE는 변경하지 않았다. GPU audit2761 samples, foreign0/error0으로 유효했다. [요청별 시간·float RGB·후보간 비교](results/v06_anima_attention_ab/runs.json)

| 경로 | denoise 초 | native 대비 RGB max / RMSE | 판정 |
|---|---:|---:|---|
| native 대조 | 80.7960 / 80.7508 | 0 / 0 | 기준 |
| 직접 NN | 63.7238 | 0.8203125 / 0.0247988 | 엄격 실패 |
| 기존 직접 VT | 56.9525 / 57.0227 | 0.8247070 / 0.0243068 | 엄격 실패, 기존 속도 재현 |
| 새 prescale+VT | 57.1482 / 57.1782 | 0.8056641 / 0.0248880 | 추가 속도 없음, max만 감소하고 RMSE 증가 |
| 4D native 보존, 최초 실험 | 83.0186 / 67.7261 | **0 / 0** | 출력 비트 일치, 초회와 후속 시간은 아래 추가 검증과 분리 |

기존 직접 VT 중앙값56.9876초는 native80.7734초 대비1.417배다. 새 prescale+VT 중앙값57.1632초는 그 기준선보다 약0.3% 느렸고 strict gate도 실패했다. 연산 한 번의 native RMSE가 작아졌다는 사실을 최종 품질 개선으로 올리지 않는다. **NN→VT만 바뀐 후보끼리도 RGB max0.69140625/RMSE0.01465366**이므로 layout-only 변경도 최종 출력 gate가 필요하다.

4D 보존 경로의 **후속 warm 검증을 완료했다.** 정규화 A/B를 마친 같은 서버에서 norm 후보를 끄고 `attn_prescale4d_native`만 두 번 적용했다. 같은 native controls80.6070/80.7198s에 대해67.7750/67.7271s, 중앙값80.6634→67.7510s=**1.1906배**였다. 두 요청 각각1680hit/fallback0이고 positive/negative conditioning·latent·RGB가 이번 native뿐 아니라 위의 이전 native와도 실제 bytes/digest가 같았다. reduced math flag=True 및 현재 guarded source SHA도 기록했다. GPU audit3178 samples, foreign0/error0, valid였다. [후속 원자료](results/v06_anima_norm_ab/runs.json), [후속 audit](results/v06_anima_norm_ab/gpu_audit.json)

최초83.0186s는 삭제하거나 warm 중앙값에 섞지 않는다. 그 요청은 당시 서버의 첫 Triton row-repair 후보였을 가능성이 있고, 새 서버의 첫 Triton LN 후보도 느렸지만 초기화/JIT/host 대기 비용을 분해하지 않았으므로 원인은 미확정이다. **1.1906배는 이미 Triton 경로를 사용한 서버의 warm 범위**다. 새 서버에서 이미지 한 번만 생성할 때의 이득을 보증하지 않는다. 약57초의 빠른 VT 경로와 strict 출력을 동시에 만족한 새 후보도 아직 없다. strict 자체가 구조적으로 불가능하다는 주장은 하지 않는다.

#### 실패·보류를 새 모델에 적용하는 방법

- score output buffer 재사용과 scale+pack 융합은 이 범위에서 실질적인 추가 가속을 만들지 못했다.
- in-place softmax는 self 추가 peak를2.125→1.125GiB로 줄였지만 self 시간이 퇴보했다. 메모리 목적의 명시적 실험 후보로만 보존한다.
- S2048 crop에서는 직접 NN이 VT보다 빨랐다. S4096 FP16의 성공을 모든 S나 FP32로 확장하지 않는다.
- 새 API는 D64/128을 받을 수 있지만 **이번 세 shape 실측은 D128**이다. `prescale4d_native`는 설치본의 `fp16_bf16_reduction_math_sdp_allowed()==True` 계약을 확인하며, 꺼진 상태에서는 adapter가 원본으로 fallback한다. mask/dropout/GQA/backward/compile/graph는 검증 범위가 아니다. 새 shape·dtype·stride는 입구 카탈로그 §7의 최소 실험을 거친다.

#### A — Claude 독립 재현과 첫 요청 지연의 후속 판정 (v0.7 반영)

Claude는 native 81.094/81.528/80.673 s, 후보 warm 67.739/67.728 s와 latent/RGB SHA 일치를 재현했다. 같은 shape의 self·cross Attention을 서버 시작 때 각각 5회 prewarm하면 준비 1.068 s 후 첫 요청부터 67.87 s였다. 이 조치는 1024²의 두 shape에서 확인됐고, 다른 해상도는 그 shape로 별도 준비·검증한다. allocator 통계가 두 요청에서 같았고 DynamicVRAM을 꺼도 지연이 남았으므로 두 가설은 기각했다. 여러 호출에 걸친 launcher/커널 준비라는 설명은 아직 추론이다. 이번 v0.7에서는 이미 재현된 수치를 반복 측정하지 않았다. [Claude 검수 §2.2](../REVIEW_V06_CLAUDE_KO.md)

#### 지각 기준 판정 제안

최종 RGB strict 실패를 지각 통과로 자동 변경하지 않는다. 익명 A/B의 좌우 순서를 바꾸고, 같은 크기의 전체 이미지와 고정 crop을 나란히 보여 준다. 얼굴·손·텍스트·경계·질감·색을 공통 항목으로 삼아 열화/동등/개선을 기록한다. 후보를 만든 연구자와 다른 판정자 또는 독립 vision model에 양방향 순서로 평가를 맡기고 모델명·버전·프롬프트·원응답을 보존한다. 여러 seed·프롬프트에 대한 합격 비율과 불확실성을 정한 뒤 채택한다. 한 장의 자기 평가만으로 동등을 선언하지 않는다. 기존 Anima 보고서의 육안 동등 평가는 별도 이력으로 유지하고, **이번 후보의 독립 지각 평가는 미실행**이다. 새 평가 모델은 다운로드하지 않았다.

### 9.12 BF16 GEMM: 계산 dtype과 실제 호출 경계를 함께 고른다

BF16 dot 명령이 없다는 사실은 BF16 `mm`가 실행되지 않는다는 뜻이 아니다. 이 설치본은 native BF16에 `Cijk_Alik_Bljk_BBS_BH_MT64x32x8_...ISA000...WS64...`를 선택했다(A). FP32 변환 경로는 `...SB_MT128x128x8_...ISA1030...WS32...`, FP16 입력·FP32 출력은 느린 `HSS/...ISA000...WS64...`, FP16 입출력은 `HHS/...ISA1030...WS32...`였다. 이름의 `ISA000`만으로 내부 명령을 단정하지 않는다. 직접 Triton 후보에서만 생성 ISA를 확인했고, dot2 후보에는 `v_dot2c_f32_f16`이 있었다. [전체 결과·kernel 이름·ISA](BF16_GEMM_V06_KO.md), [AMD dot 지원표](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/docs-1.0.0/compiler-builtins/cross-arch/arithmetic-ref/dot-float-builtins.html)

#### A — 세 shape의 변환 포함 시간

세 입력은 실제 활성값 capture가 아닌 대표 shape의 seeded BF16 난수다. B는 물리적 `[N,K]` contiguous에서 얻은 transpose view다. 두 warmup 후 순서를 회전·역전하며 3round×3call을 쟀다. 아래 wall 중앙값은 호출별 입력·출력 변환을 포함하지만, FP16 안전 검사는 별도다. **직접 dot2도 FP16 재표현 guard를 제외한 시간**이며 API 기본 exact 조건과 구분한다. [원자료](results/v06_bf16_gemm/results.json)

| 경로 | mm M1024/N1024/K1024 ms | mm M4096/N128/K4096 ms | bmm B1/M1024/N384/K1024 ms |
|---|---:|---:|---:|
| native BF16 | 0.272655 | 0.938373 | 0.118016 |
| BF16→FP32→BF16 | 0.177375 (1.537×) | 0.579946 (1.618×) | 0.115449 (1.022×) |
| FP16 입력·FP32 출력→BF16 | 0.283748 (0.961×) | 1.026729 (0.914×) | 0.129102 (0.914×) |
| FP16 입력·FP16 출력→BF16 | 0.134836 (2.022×) | 0.465563 (2.016×) | 0.097092 (1.215×) |
| BF16 load→Triton FP16 dot2 | 0.287385 (0.949×) | 0.657862 (1.426×) | 0.140822 (0.838×) |
| BF16 load→Triton FP32 SIMT | 0.921973 (0.296×) | 2.124725 (0.442×) | 0.376974 (0.313×) |

18조합 모두 사전 단위 gate인 normalized RMSE≤0.004, normalized max≤0.04를 통과했다. 분모는 native tensor RMS다. CPU FP64로 24×24 선택 출력 좌표를 전체 K에 대해 독립 계산했으며, 이는 전체 원소 FP64 검증은 아니다. 추가 allocated peak에는 출력이 포함되며 FP32 변환은 각각12/68/7MiB, 직접 dot2는2/1/0.75MiB였다. event 시간·전체 오차는 상세 보고서에 별도로 남겼다.

**FP16의 빠른 숫자에 검사 비용을 숨기지 않는다.** 입력 finite 검사의 wall만 각각0.161546/0.477856/0.156642ms였다. 별도 측정값의 단순 합을 실제 end-to-end 시간으로 부르지는 않지만, 이 정도 비용이면 단위 이득이 사라질 수 있다. 세 입력 모두 작은 값의 FP16 underflow/반올림이 있어 exact-half 계약을 만족하지 않았다. 기본 API는 exact roundtrip을 요구하며, 느슨한 finite guard는 손실을 허용하는 별도 선택이다. `2^80` overflow와 `2^-120` underflow를 strict guard가 거부하는 것도 확인했다. **입력 finite와 GEMM 출력 finite는 별도 조건**이다. FP16 출력의 overflow·이중 반올림을 BF16 입력 범위 검사만으로 보증하지 않는다.

#### A/B — 연결 실패와 최종 출력 A/B

첫 VAE `TorchDispatchMode` 연결은 native profile에 실제 BBS GEMM이 있는데도 hit0이었다. composite SDPA와 MIOpen 내부 GEMM을 그 위치에서 가로채지 못했다. 첫 Anima `--bf16-unet` 시도도 storage는 BF16이지만 `unet_manual_cast`가 FP16 compute를 골라 1,030 Linear 모두 FP16이었다. 둘 다 출력 동일성을 **후보 성공으로 세지 않는다**. [VAE no-hit](results/v06_bf16_vae_fp16/results.json), [CLI BF16 no-hit](results/v06_anima_bf16_2step/runs.json)

| 실제 모델 경로 | 후보가 실제로 한 일 | 시간·오차 | 판정 |
|---|---|---|---|
| Anima 요청 한정 BF16 compute, 1024²·2step | `manual_cast_dtype`을 요청 중에만 BF16로 바꾸고 F.linear 1,030회를 FP32 flatten-mm+bias로 치환 | 같은 BF16 대조 denoise11.3628→6.8347s, RGB max0.78809/RMSE0.036943 | 엄격 실패. 30step 가속으로 표기하지 않음 |
| MOSS BF16 VAE 일반, 실제 저장 latent→512² RGB | 실제 SDPA1회, QK/PV 두 GEMM만 FP32; 기존 BF16 score·softmax·출력 경계 유지 | 182.286→180.660ms (1.009×), RGB max0.00390625/RMSE0.000721635 | 엄격 통과, 안정적 가속 근거 없음 |
| 같은 VAE tiled256/overlap64 | 실제 SDPA23회, GEMM46개 치환 | 915.351→917.806ms (0.997×), RGB max0.00312752/RMSE0.000440921 | 엄격 통과, 속도 이득 없음 |

Anima의 FP32 후보는 입력 precision만 바꾼 실험이 아니다. flatten·GEMM 선택·bias add 경계도 달라졌다. 따라서 큰 RGB 차이를 “FP32가 덜 정확하다”로 설명하지 않는다. 같은 BF16 대조의 반복 출력은 비트 일치했고, 후보는 두 번 같은 실패를 보였다. [`comparison_errors.bf16_control.image`](results/v06_anima_bf16_compute_2step/runs.json)가 이 표의 기준이며 첫 normal-FP16 warm과의 오차는 다른 비교다.

VAE에서는 먼저 기존 BF16 연산 경계를 재구성한 control이 일반·분할 모두 RGB dtype·값 일치함을 확인했다. 해당 원자료의 `bit_equal`은 `torch.equal`이므로 signed-zero의 바이트까지 구분한 검사는 아니다. 그 뒤 GEMM 내부만 FP32로 바꿨다. native attention GEMM 합2.3421ms→후보1.3211ms에 비해 전체 native device activity는165.7301ms였다. 약134ms의 BBS kernel group에는 convolution 내부 GEMM과 PV1회가 함께 들어 있다. 이름이 같다는 이유로 전부 attention 또는 전부 conv에 귀속하지 않고, operator 표와 kernel 표도 합산하지 않는다. 이 새 후보의 lowvram은 미검증이며 이전 VAE 레시피의 결과를 물려받지 않는다. [실제 VAE 원자료·커널 표](results/v06_bf16_vae_sdpa_fp32/results.json)

#### 재사용과 채택 규칙

[`rdna2_ops/bf16_gemm_v06.py`](rdna2_ops/bf16_gemm_v06.py)는 2D mm/동일 batch 3D bmm 후보를 제공한다. [`rdna2_ops/bf16_sdpa_gemm_v06.py`](rdna2_ops/bf16_sdpa_gemm_v06.py)는 기존 BF16 rounding 경계를 보존해 GEMM만 비교한다. [`bench_dtype_gemm_v06.py`](bench_dtype_gemm_v06.py), [`dtype_vae_v06.py`](dtype_vae_v06.py), [`dtype_switches_v06.py`](dtype_switches_v06.py)가 단위·VAE·요청 한정 연결을 재현한다.

새 모델에서는 checkpoint/CPU mmap/GPU resident/실제 입력 dtype을 따로 확인하고, 원래 연산 재구성 control을 먼저 통과시킨다. 변환·범위 검사·bias·출력 cast·임시 memory를 포함해 잰다. 이 절의 명시적 GEMM/SDPA 칸은 **native를 유지하면서 shape별 후보를 명시적으로 시험**한다. 이 결과를 BF16 연산 전체의 결론으로 확대하지 않는다. 내부 BBS 비용이 컸던 Conv2D는 다른 라이브러리 경계에서 큰 이득을 얻었으며 §9.14에 따로 기록했다.

### 9.13 FP8 가중치 GEMM: 저장 표현과 계산 최적화를 분리한다

gfx1030에서 FP8은 여기서는 **저장 표현**이다. 기준선은 `raw FP8 weight→FP16 cast→rocBLAS F.linear`이며 실제 projection 커널은 `Cijk_Alik_Bljk_HHS_BH_MT128x256x16_...ISA1030...WS32...`와 FP8→FP16 copy였다(A). 이 절은 plain unscaled `float8_e4m3fn`/`float8_e5m2`만 다룬다. FNUZ, tensor/row/block scale, QuantizedTensor, MXFP8를 같은 decoder 계약으로 취급하지 않는다. [전체 보고서](FP8_WEIGHT_GEMM_V06_KO.md), [실제 커널·단위 원자료](results/v06_fp8_gemm/results.json)

검사한 기존10개 diffusion/text/VAE safetensors에는 원래 저장된 FP8 tensor가 없었다. 새 모델을 받지 않고 기존 Anima BF16 checkpoint를 연구 요청에서 raw E4M3fn으로 재표현했다. **같은 FP8 값의 native/candidate A/B**와 **원래 BF16 checkpoint→FP8 표현 변경의 손실**은 아래에서 따로 판정한다. E5M2는 연산 검증까지이며 실제 모델 출력은 미검증이다. [기존 모델 inventory](results/dtype_model_inventory_v06.json)

#### A — 두 format, 세 shape, 여섯 경로

입력은 대표 차원의 seeded synthetic tensor다. A는 FP16 `[M,K]`, weight는 FP8 `[N,K]`, 출력은 FP16 `[M,N]`이다. 두 warmup 후 3round×3call, 순서를 회전·역전했다. 아래는 **동기화 wall 중앙값 ms**다. 호출별 decode·activation quantization·출력 cast는 포함하고 prepared weight의 1회 준비와 finite scan은 별도다. 모델 INT8 adapter의 finite scan은 실제 stage 시간에 포함했다.

| format | M/N/K | native upcast | software decode | prepared FP16 | fused decode+dot2 | INT8 준비 포함 | prepared INT8 |
|---|---|---:|---:|---:|---:|---:|---:|
| E4M3fn | 8192/2048/2048 | 2.766797 | 2.793510 | 2.725654 | 7.672668 | 2.295050 | 2.272633 |
| E4M3fn | 8192/8192/2048 | 8.415002 | 8.608528 | 8.156854 | 29.738531 | 9.547644 | 9.510434 |
| E4M3fn | 512/3072/1024 | 0.164029 | 0.156362 | 0.137486 | 0.384187 | 0.197339 | 0.177185 |
| E5M2 | 8192/2048/2048 | 2.925879 | 2.772773 | 2.865133 | 7.701021 | 2.288427 | 2.285330 |
| E5M2 | 8192/8192/2048 | 8.380413 | 8.654347 | 8.178327 | 29.749238 | 9.568764 | 9.552554 |
| E5M2 | 512/3072/1024 | 0.167645 | 0.153659 | 0.140132 | 0.388024 | 0.200892 | 0.177765 |

1회 FP16 준비 wall은 shape/format별0.0445–0.1561ms였다. prepared 경로의 준비 생략 배속을 호출마다 변환하는 경로의 속도로 쓰지 않는다. decoder는 각 형식256개 byte pattern 전부에서 finite FP16 bits와 NaN/±Inf 분류가 일치했다. NaN payload 동일성은 검증하지 않았다. software decode/prepared FP16의 6GEMM은 native와 값 일치(`torch.equal`, signed-zero byte 미구분)였고, 별도 CPU FP64는24×24 선택 출력 좌표를 전체K로 계산했다.

36조합은 사전 연산 gate를 통과했다. exact계열 normalized RMSE/max≤0.002/0.02, INT8계열≤0.03/0.15이며 분모는 native RMS다. INT8의 실제 normalized RMSE는0.01096–0.01181이다. 이 기준은 이미지 strict gate와 다르다. fused dot2는 실제 `v_dot2c_f32_f16`, INT8은 `v_dot4c_i32_i8`를 실행했고 모두 spill0이었다. **spill이 없어도 fused dot2는0.282–0.432배로 느렸다.** 작은 타일의 반복 decode와 VGPR190–191개가 불리한 원인 후보지만 개별 비용을 분해한 실험은 아니다.

메모리도 INT8이 무조건 작지 않았다. projection의 native 추가 allocated peak는40MiB, INT8 준비 포함은116.04MiB였다. activation 양자화 buffer와 FP32 dequant 후 FP16 cast가 추가된다. prepared FP16의 추가 peak32MiB 외에 weight8MiB가 이미 상주한다. 같은 방식의 MLP 상주 weight는32MiB다. 전체 footprint·event·1회 준비 표는 상세 보고서에 있다.

#### B/A — CPU weight 속성만 보고 연결하지 않는다

첫 연결은 `module.weight.dtype==BF16`만 보아 FP8 hit0이었다. 이 DynamicVRAM은 CPU mmap에 BF16 원본을 두고 `_model_dtype`/TensorGeometry로 GPU resident `_v_weight`를 FP8로 준비한다. adapter를 실제 `CastBiasWeightContext` 경계에 연결한 뒤 반환 weight가 resident와 dtype·shape·stride·data_ptr 모두 같음을 확인했다. 30step의13,742회가 이 계약을 만족했다. **hit0의 출력 동일성은 최적화 검증 성공이 아니다.** [첫 실패](results/v06_anima_fp8_e4m3fn_2step/runs.json), [수정된 실제 경계](results/v06_anima_fp8_e4m3fn_30step/runs.json)

#### A — 기존 Anima 최종 출력과 실패 판정

동일 seed20261107·conditioning·1024², 기존 Euler/simple·CFG4를 썼다. 후보는 요청 안에서만 적용하며 캐시도 요청이 끝나면 해제한다. VAE는 바꾸지 않는다.

| 후보·범위 | 실제 denoise wall | 최종 출력 | 결정 |
|---|---|---|---|
| projection INT8, 측정한8192/2048/2048만·2step | native5.3865/5.4055s, 후보6.8429s; 실제336회 | RGB max0.656982/RMSE0.025658 | 속도·strict 모두 실패, 30step 미진행 |
| FP16 weight cache, staging context 유지·30step | native79.6025/79.4648s, 후보80.5853/80.7777s. 중앙값79.5337→80.6815s=0.9858× | 두 요청 latent/RGB 비트 일치 | 준비 상각 후에도 무이득, 미채택 |
| owned weight+bias cache, hit에서 context 생략·2step | native5.4156/5.3886s, 후보5.3823/5.4532s. 중앙값5.4021→5.4178s=0.9971× | 두 요청 latent/RGB 비트 일치 | 사전5% 이득 gate 미달, 추가30step 미진행 |

30step 캐시는515개 준비/13,227회 재사용, 명시적 cache4,116,201,472B였다. PyTorch 요청 peak는3,045,647,872→7,200,400,896B로 늘었고 external DynamicVRAM allocator는 이 통계 밖이다. 모델 준비 비용을 별도 clock으로 분해하지 않았으며 위 시간은 **준비 포함 실제 denoise**다. 후속 owned-cache는 weight+bias를 독립 소유해 불필요한 context 재진입을 제거했지만 외부 prefetch는 그대로였다. 이 후보의30step 수치는 측정하지 않았다. [30step](results/v06_anima_fp8_e4m3fn_30step/runs.json), [INT8](results/v06_anima_fp8_e4m3fn_geometry_2step/runs.json), [owned cache](results/v06_anima_fp8_owned_2step/runs.json)

**원래 모델을 raw FP8로 바꾸는 것은 별도 실패였다.** 저장된 동일 seed/steps/conditioning의 정상 Anima와 raw E4M3fn native 출력은 RGB max0.94775391/RMSE0.17677346이었다. 동일 FP8 값끼리의 cache 오차0과 섞지 않는다. 실제 작은 cross-attention weight 두 개의 RMS는 약0.0061–0.0069이고 raw E4M3fn 변환 normalized RMSE는0.0824–0.0924였다. E4M3fn의 subnormal 간격0.001953125가 이 작은 값들에 거칠다는 것이 직접 관측한 표현 손실의 근거다. scaling된 checkpoint의 성능/화질을 이 결과로 부정하지 않는다. [원본 모델과 별도 비교](results/v06_anima_fp8_e4m3fn_30step/representation_vs_original.json), [실제 weight 조사](results/anima_fp8_weight_conversion_cpu_v06.json)

#### 재사용과 새 모델의 판단

[`rdna2_ops/fp8_weight_gemm_v06.py`](rdna2_ops/fp8_weight_gemm_v06.py)에 decoder·FP16·dot2·INT8 경로가 있고, [`fp8_owned_switches_v06.py`](fp8_owned_switches_v06.py)는 immutable Anima의 요청 한정 cache 실험이다. scale/LoRA/weight/device가 바뀌는 경로를 이 캐시에 임의로 연결하지 않는다. 새 모델은 먼저 raw/scale/FNUZ 계약과 실제 resident dtype을 확인하고, **같은 저장값의 계산 경로**를 3shape 이상 비교한다. 빠른 단위 후보라도 캐시 준비·추가 VRAM·offload·최종 출력을 포함한 모델 A/B를 통과해야 한다. 이번 측정에서는 기존 upcast 경로를 대체할 실사용 후보를 채택하지 않았다.

### 9.14 Conv: FindDb 선택과 forward·backward 정확도를 함께 검증한다

**BF16 Conv2D는 실제 VAE 네 경로에서 엄격 RGB와 속도 관문을 통과했다.** Applio FindDb 필터는 실제 epoch를 줄였으나 학습 출력 동등성은 미확정이다. 다른 Conv family는 연산 성공·정확도 실패·모델 연결 미완료를 아래에서 구분한다. Applio 운영 설정이나 운영 DB는 바꾸지 않고, 기존 interpreter·실제 데이터·pretrained·학습 루프를 연구 프로세스에서 재사용했다. 전체 커널·DB key·실패 로그·재현 명령은 [Conv 연구 기록](results/conv_catalog_v06/CONV_CATALOG_V06_KO.md)에 있다.

#### B/C — NORMAL/FAST를 호출마다 바꾸지 않는다

검사한 MIOpen 소스의 `GetFindModeValue`는 환경변수를 `static const`로 처음 한 번 읽는다. 설치된 PyTorch의 `ConvolutionDescriptor::set`은 `benchmark=True`에서 descriptor를 Normal로 설정한다. 따라서 `os.environ`을 Conv마다 뒤집는 dispatch를 만들지 않고, 실험은 `benchmark=False`를 고정한다. 소스 revision·파일 SHA는 상세 기록에 있다.

**FindDb는 문제 descriptor별 선택 결과**, PerfDb는 개별 solver의 튜닝 파라미터다. User FindDb는 System FindDb보다 우선하며 FAST는 hit일 때 그 결과를 재사용한다(C). 후보는 연구 폴더의 별도 DB에 dilation1만 NORMAL로 seed한 뒤 새 FAST 프로세스에서 전체 모델을 실행하는 방식이다. 버전·dtype·shape·forward/backward 방향이 바뀌면 같은 DB 레시피의 측정 범위로 간주하지 않는다. [MIOpen FindDb](https://rocm.docs.amd.com/projects/MIOpen/en/latest/conceptual/finddb.html), [Find modes](https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html)

#### A — 무필터 NORMAL seed+FAST replay의 정확도 실패

FAST 기준18shape의 output/dX/dW는 사전 gate를 통과했고 fwd+dX+dW의 shape별 wall 중앙값 합은69.73ms였다. 이는 한 모델 step 시간과 다르다. dilation1 12문제를 NORMAL로 seed한 DB를 FAST에서 재생하자 다음 네 forward가 큰 오차를 보였다. independent FP32 reference와 원래 FAST FP16 control 양쪽에서 확인했다. 같은 요청의 dX/dW가 gate를 통과해도 forward 실패를 취소하지 않는다. [유효한 GPU audit·실제 kernel·원자료](results/conv_catalog_v06/resblock_hybrid/result.json)

| B/C/T/kernel/dilation | forward NRMSE 대 FP32 | forward max abs | 결정 |
|---|---:|---:|---|
| 8/64/6400/7/1 | 0.384542 | 2.246003 | 실패 |
| 8/64/6400/11/1 | 0.297546 | 1.705957 | 실패 |
| 8/128/3200/11/1 | 0.517427 | 2.603542 | 실패 |
| 8/64/6400/3/1 | 0.589294 | 3.561350 | 실패 |

첫 shape는 fwd+dX+dW가 약5.24→1.96ms로 빨랐지만 정확도 때문에 거절한다. trace에는 `miopenSp3AsmConv...`와 `gridwise_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw_lds_double_buffer`가 함께 나온다. kernel 집계만 보고 Winograd를 원인으로 단정하지 않는다. 방향별 DB entry와 isolated 재현을 대조한 뒤 실패 forward entry를 제외하는 후보를 별도로 검증한다. **기존 ROCm 비교의 NORMAL d1 속도 이득은 정확도 승인으로 읽으면 안 된다.** 기존 파일과 무필터 DB는 실패 근거로 보존한다.

#### A — forward 항목을 제외한 FindDb와 실제 Applio epoch

새 프로세스의 isolated 재현에서도 C64/k7의 forward 오차가 반복됐다. 방향별 DB entry와 대조해 `ConvHipImplicitGemmV4R1Fwd` entry만 연구 DB 복사본에서 제외하고 B/W 선택은 유지했다. 이 변경 후 dilation1의12shape 모두 output/dX/dW 관문을 통과했고, FP32 reference 대비 최대 NRMSE는4.61e-4였다. FAST의12shape wall 중앙값 합31.740→17.530ms, 개별1.14–2.60배였다. 이 제외 실험은 잘못된 forward 선택과 오류의 연결을 보여 주지만, 해당 kernel 내부의 구체 버그까지 고친 결과는 아니다.

기존 Applio `bench40k`111조각, f0G40k/f0D40k, 실제 generator/discriminator/optimizer/AMP/DataLoader를 그대로 사용했다. batch8, 16step/epoch,4epoch이며 첫 epoch를 분리했다. 연구 DB·config 복사본·출력만 별도 폴더에 썼다. 이 실행은 기존 **PyTorch2.11.0+rocm10.0.0/HIP7.15.26333**이고 뒤의 VAE는 기존 **2.12.0+rocm10.2.0a20261011/HIP7.17.26405**다. 두 환경 사이에 새 설치나 교체는 하지 않았다.

| 실행 | 정상3epoch wall 중앙값 | 비교·출력 판정 |
|---|---:|---|
| FAST A | 9.30998s | 기준 |
| FAST B 반복 | 9.30841s | 시간차 약0.017%; 초기 state와 초기 음성은 byte 일치 |
| 필터 FindDb, FAST | 8.41758s | FAST A대비1.106×; loss median D+0.31%/G−0.95% |
| 필터 FindDb + 선택 fold | 8.30662s | filtered-only대비1.013×, FAST A대비약1.121×; 단일4epoch 추가 이득은 조건부 |

훈련 후 같은 reference 음성의 native A/B 파형 차이는 max0.72192/RMSE0.13440이었고, 필터 후보 대 A의 RMSE는0.10658이었다. 사전 임시 engineering gate(max0.001/RMSE0.0001)는 **native 반복부터 실패**했다. 따라서 후보의 청각 열화도 동등성도 이 수치만으로 단정하지 않는다. 초기 state/eval은 같고 timing은 재현됐으므로 **성능 개선 A, 장기 학습 품질 미확정**으로 남긴다. loss는 finite였지만 초기 AMP grad norm의 nonfinite와 scale 감소는 별도 기록한다. 원래 로그의 비표준 NaN/Inf JSON은 보존하고 strict JSON에는 null과 원인 표기를 남겼다.

Dilated fold의18shape는 output/dX/dW gate를 모두 통과했지만 blanket 합은69.73→69.46ms로1.004배였다. C64/T6400/k7·11/d3·5의 네 shape만1.59–1.80배였고 다른 C32/C128 일부는0.53–0.79배였다. 선택 predicate를 둔 실제 epoch에서는 각 winning shape64회, 총256회만 fold했다. 기존 운영 FoldConv 코드나 설정은 수정하지 않았다. 선택 후보의 최종 파형도 임시 gate는 실패했으며 native 변동과 함께 읽어야 한다. 전체 원자료와 경로별 hit·초기 state·운영 hash는 [Conv 기록](results/conv_catalog_v06/CONV_CATALOG_V06_KO.md)에 모았다.

#### A/B — BF16 Conv2D의 계산만 FP32로 옮긴다

이 후보는 VAE 전체 dtype을 바꾸지 않는다. 기존 Comfy cast/offload가 반환한 BF16 input·weight를 FP32로 계산한 다음 **convolution 출력을 BF16으로 반올림하고, bias도 원래 BF16 경계에서 더한다.** native의 bias 없는 Conv→BF16 bias add 재구성은 세 shape 모두 실제 바이트가 같았다. 후보가 FP32 bias까지 합쳐 마지막에 한 번만 BF16으로 내리면 native 대비 NRMSE 약0.00284–0.00285로 사전0.002 관문을 넘었다. 이 실패는 원본 반올림 경계 재현 기준이며 더 높은 실수 정밀도가 일반적으로 나쁘다는 뜻이 아니다.

native는 `Cijk_Ailk_Bljk_BBS_BH_MT64x32x8_...ISA000...WS64...`와 `Im2d2Col_v2`를 실행했고, FP32 계산은 `miopenSp3AsmConv_v30_3_1_gfx10_fp32_f2x3_stride1`를 선택했다. dtype 변경으로 **라이브러리 알고리즘과 중간 buffer도 바뀐 것**이다. 직접 새 합성곱 알고리즘을 구현했다고 부르지 않는다. [3shape 원자료·전체 커널 이름](results/conv_catalog_v06/bf16_conv2d_micro/results.json)

| input, weight는 C×C×3×3 | native wall ms | FP32 계산/BF16 bias wall ms | 배속 | native 대비 NRMSE | 추가 peak MiB native→후보 |
|---|---:|---:|---:|---:|---:|
| [1,384,128,128] | 4.335687 | 1.240087 | 3.496× | 5.51e-5 | 120→53.469 |
| [1,96,512,512] | 5.496139 | 1.772719 | 3.100× | 3.76e-5 | 480→192.316 |
| [1,192,256,256] | 4.820717 | 1.342400 | 3.591× | 4.71e-5 | 240→97.266 |

shape는 실제 VAE profile에서 가져왔고 단위 입력은 seed 고정 합성 BF16 tensor다. 호출별 input/weight 변환과 output/bias 계산을 포함하며 weight cache는 없다. 추가 peak는 개별 호출의 PyTorch allocated 증가량이고 출력·임시 buffer를 포함한다. native 대비 max 절대오차는0.015625–0.03125지만 normalized gate를 통과했다. 이것은 아직 RGB 오차가 아니다.

기존 MOSS denoiser의 저장 latent(`seed20261105_32x32.pt`)와 `flux2-vae_small.safetensors`를 그대로 사용해 **512² 최종 float RGB**를 비교했다. native observer control은 byte 일치했고, 후보는 regular40개/tiled920개 Conv2D에 실제 적중했다. 아래 RGB 열은 255단계 단위다. strict 한계는 max2, RMSE0.25다.

| 실제 VAE 경로 | native→후보 wall 중앙값 ms | 배속 | RGB max / RMSE(×255) | strict |
|---|---:|---:|---:|---|
| Normal, regular | 180.950→73.897 | 2.449× | 1.4941 / 0.1865 | 통과 |
| Normal, tiled256/overlap64 | 915.928→376.850 | 2.430× | 0.7790 / 0.1062 | 통과 |
| Lowvram, regular | 172.969→65.872 | 2.626× | 1.4941 / 0.1865 | 통과 |
| Lowvram, tiled256/overlap64 | 940.545→409.179 | 2.299× | 0.7790 / 0.1062 | 통과 |

Lowvram은 flag 이름만 기록한 것이 아니다. 실제 `model_lowvram=True`, `dynamic_patcher=False`를 확인했다. Normal은 dynamic patcher를 사용했다. 각 경로 native/candidate를3회씩 교대했고 변환과 Python hook 비용도 포함했다. **위 배속은 VAE decode 구간**이며 denoising+VAE 전체 요청의 배속이 아니다. native/후보의 서로 다른 로드 경로 시간을 교차 비교하지 않는다. 최종 RGB는 tolerance 통과이며 비트 일치가 아니다. 모델 JSON의 `peak_allocated_bytes`는 A/B 여러 호출을 포함한 구간 peak라 candidate 전용 메모리나 절감량으로 읽지 않는다. [Normal 원자료](results/conv_catalog_v06/bf16_conv2d_vae/results.json), [Lowvram 원자료](results/conv_catalog_v06/bf16_conv2d_vae_lowvram/results.json)

재사용 코드는 [`rdna2_ops/bf16_conv_v06.py`](rdna2_ops/bf16_conv_v06.py), 재현기는 [`conv_vae_v06.py`](conv_vae_v06.py)다. 직접 tensor 함수와 scoped 기능 교체를 제공한다.

```python
import torch
from rdna2_ops.bf16_conv_v06 import BF16Conv2dMode

# 기존 VAE의 dtype·cast·offload 경계는 그대로 두고 decode 호출만 감싼다.
with torch.inference_mode(), BF16Conv2dMode("fp32", bias_rounding="bf16"):
    rgb = vae.decode(latent)
```

현재 context의 측정 범위는 gfx1030, inference, CUDA BF16 input/weight, groups1이다. weight cache를 두지 않으므로 매 호출 기존 cast 결과를 사용한다. 종료·예외 시 원래 functional Conv2D로 복원한다. 다른 모델·batch·shape·group·backward·compile/graph는 해당 카탈로그 최소 실험을 거친다. 이 결과를 모든 BF16 연산의 FP32 강제 정책으로 만들지 않는다.

#### A/B — Conv2D·grouped·depthwise·ConvTranspose·일반 Conv3D

같은 Applio runtime에서 각 family의 세 shape를 FAST/NORMAL로 측정했다. 아래는 **forward+dX+dW 전체 wall 중앙값**의 비교다. Conv2D/grouped/ConvTranspose1D는 실제 Applio shape이고, depthwise/일반 Conv3D는 제한된 합성 shape다. 전체15개 shape·각 방향 커널명·메모리·독립 FP32 reference는 [family 원자료](results/conv_catalog_v06/family_fast/result.json)와 [NORMAL 결과](results/conv_catalog_v06/family_normal/result.json)에 있다.

| 연산·세 shape의 요약 | FAST→NORMAL 배속 | 정확도·모델 판정 |
|---|---:|---|
| period Conv2D, C32→128/H2134/P2; C128→512/H712/P2; C512→1024/H238/P2 | 1.186/1.444/1.317× | 첫째·셋째 forward NRMSE0.780/0.773 실패. 빠른 숫자로 채택하지 않음 |
| grouped Conv1D, C16→64/T12800/G4; C64→256/T3200/G16; C256→1024/T800/G64 | 0.959/1.002/1.002× | y/dX/dW 통과, 이득 없어 native 유지 |
| depthwise Conv1D, C32/T12800/k3; C64/T6400/k7; C128/T3200/k11 | 5.605/2.031/2.867× | y/dX/dW 통과, 기존 실제 모델의 해당 경로는 미연결(❔) |
| ConvTranspose1D, C512→256/T32/k16; C256→128/T320/k16; C128→64/T3200/k4 | 1.087/1.312/1.243× | y/dX/dW 통과. 아래 FAST replay 및 실제 epoch에서 최종 판정 |
| 일반 Conv3D, C16→32/D1/HW32; C32→32/D4/HW32; C64→32/D8/HW16 | 1.058/0.997/1.015× | y/dX/dW 통과, 이득 작음·모델 미연결(❔) |

FAST의 대표 경로는 `Im2d2Col_v2`/`Im3d2Col` + Tensile `Cijk_*_HHS_*` + col2im이었다. NORMAL은 일부 문제에서 `gridwise_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw_lds_double_buffer`를 선택했다. 표의 전체 호출 시간을 이 커널 하나의 시간으로 읽지 않는다. **CK grouped library 누락 경고가 나도 15개 FAST 모두 실행되고 정확도 관문을 통과했다.** 설치 버전의 grouped solver `IsApplicable`에서 CK loader 부재를 거절하는 소스도 확인했다(B). 여기서 측정한 것은 실제 fallback의 비용이며, CK가 설치됐을 때 얻을 배속은 측정하지 않았다. 환경 재구축 없이 확인할 수 없는 부분은 빈칸으로 남긴다. ConvTranspose2D/3D도 이번 1D 결과로 채우지 않는다.

ConvTranspose1D는 검증된 세 shape의9개 방향 key만 filtered FindDb 복사본에 병합했다. transpose forward는 MIOpen B, dX는 F, dW는 W 방향이므로 이름만 보고 모든 F 항목을 잘라내지 않는다. FAST replay의 fwd+dX+dW는0.68346/0.92962/1.03584ms로 재현됐고 세 forward는 기존 FAST와 byte 일치, dX/dW도 참조 관문을 통과했다. 실제 Applio4epoch에서 각 shape64회 적중했으나, filtered-only 정상 epoch8.417579→8.434122s=**0.99804배**로 모델 가속이 없었다. 초기 state/audio는 byte 일치, reference5tensor 전후 SHA와 보호 파일·원본 DB SHA도 같았다. 최종 wave max0.711426/RMSE0.061977은 임시 gate를 실패했으며 native 반복의 큰 차이와 함께 품질 미확정으로 남긴다. **이 후보는 실제 epoch 가속용으로 채택하지 않는다.** [9key 병합·재현 레시피](results/conv_catalog_v06/CONVTRANSPOSE_DB_RECIPE_KO.md), [replay 원자료](results/conv_catalog_v06/transpose_merged_replay/result.json), [실제 epoch](results/conv_catalog_v06/epoch_filtered_transpose/result.json)

#### A/B — exact-tile Conv3D를 Linear로 바꾸는 좁은 조건

Qwen의 실제 callable을 따라 `Qwen3VL → Qwen35VisionPatchEmbed.proj`를 확인했다. 입력 FP32 `[2400,3,2,16,16]`, weight FP32 `[1152,3,2,16,16]`, stride `(2,16,16)`, padding0, dilation1, groups1이다. **입력 공간 전체가 한 kernel이고 출력 공간이1인 조건**에서만 flatten view+Linear와 같은 선형식이 된다. 일반 영상 Conv3D의 대체식이 아니다.

같은 Qwen interpreter에서 정상 NORMAL Conv3D를 기준으로 bias를 포함한 forward inference를 비교했다. 독립 CPU FP64 y/dX/dW도 세 shape 모두 통과했다. reshape는 contiguous view다.

| patch 수 P, N1152/K1536 | NORMAL Conv3D ms | Linear ms | 배속 | native 대비 y max / NRMSE |
|---|---:|---:|---:|---|
| 128 | 0.184955 | 0.090718 | 2.039× | 0 / 0, y/dX/dW bytes 일치 |
| 512 | 0.259092 | 0.130696 | 1.982× | 8.82e-6 / 8.20e-7 |
| 2400 | 1.582226 | 0.473349 | 3.343× | 1.17e-5 / 8.45e-7 |

native는 `gridwise_convolution_implicit_gemm_v4r4_ncdhw_kczyx_nkdhw`, 후보는 rocBLAS FP32 GEMM을 실행했다. [정상 native](results/conv_catalog_v06/qwen_patch_fp32_native_normal/result.json), [Linear와 비교](results/conv_catalog_v06/qwen_patch_fp32_linear_vs_normal/result.json).

실패도 선택 규칙의 일부다. **FAST+빈 연구 FindDb**는 P128/P2400 첫 forward에서 gfx1030 hipBLASLt library/hsaco 부재와 `miopenStatusInternalError`로 실패했다. NORMAL은 실패 후보를 건너뛰고 정상 v4r4 경로를 찾았다. 따라서 “CK 파일 누락이면 Conv3D 전체 미지원”도, “PyTorch의 hipBLAS fallback이 모든 MIOpen 호출을 구제한다”도 이 설치본의 결과와 맞지 않는다. 운영 cache를 쓰는 native와 빈 연구 cache를 쓰는 진단은 다른 조건이다. 매우 느렸던 cudnn-off partial job은 중단·불완전 audit로 유효한 배속에서 제외했다.

실제 모델의 후보 연결은 conditioning node 진입에서만 해당 proj 인스턴스를 감싸고 원래 cast/offload tensor 수명을 유지하는 [`qwen_conv3d_v06.py`](qwen_conv3d_v06.py)로 구현했다. 기존 workflow/image/prompt, seed20261106, 2step에서 warm→native→후보→native를 계획했지만, **후보 전 native 재인코딩부터 conditioning digest가 달라져 gate가 중단됐다.** 두 native의 최종 RGB max0.138173/RMSE0.00375442로 엄격 기준 밖이었다. 따라서 Linear의 모델 품질 실패나 성공, 모델 배속으로 판정할 수 없다. 2step 진단 자체도 원래25step 품질 검증을 대체하지 않는다. [실제 시도](results/conv_catalog_v06/qwen_model_screen2/runs.json)

actual native는 cudnn enabled/benchmarkFalse이고 FindMode·UserDB 환경변수는 unset이었다. 두 patch call의 event 구간 합은 약3.55s였으나 host/find/compile/staging에 의한 GPU 대기까지 포함할 수 있어 단위 kernel1.6ms와 직접 배속을 계산하지 않는다. 실제 모델의 kernel 선택은 profile로 확정하지 못했다. CPU 소스 조사에서 Qwen21 VAE encode는 mean을 쓰며 이번 경로의 global RNG 고정을 해결책으로 넣을 근거는 없었다. 다음 최소 진단은 동일 요청의 X/W/bias/Y byte digest로 최초 분기 지점을 찾는 것이다. [소스 근거·미확정 범위](results/conv_catalog_v06/QWEN_NATIVE_DRIFT_SOURCE_NOTE_KO.md)

이 칸은 **연산 후보 A, 실제 모델 판정 불가(❔)**다. 실패·원자료를 보존하고 운영 Conv3D flag는0으로 유지했다. 연구 process를 종료하면 scoped hook이 복구되며 운영 파일에 적용된 변경은 없다.

#### A — Claude 후속 재현: solver 차단 변수 두 개

문제 forward solver는 `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0`으로 직접 제외할 수 있다. ROCm 7.2/10.0/10.2의 period Conv2D 오답과 Conv1D k7 오답이 복구됐다. dilation 1의 12 shape 합은 17.860 ms, 그 연구 DB를 FAST로 재생한 Applio 실제 4 epoch의 정상 epoch 중앙값은 9.261→8.480 s, **1.092배**였다. 4 epoch 한 번의 근거이며 청감·장기 학습 동등성은 미확인이다.

FP32 NORMAL 정확도 참조에서도 `ConvOclBwdWrW53`이 dW NaN을 냈다. 유효한 변수는 **`MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53=0`**이며, 옛 `..._DIRECT_OCL_WRW53`은 효과가 없다. NORMAL 또는 FP32 Conv 참조를 쓰는 새 연구 프로세스에는 두 변수를 함께 지정한다. 운영 런처와 운영 DB를 수정하는 지침은 아니다. [독립 실측 및 로그 요약](../REVIEW_V06_CLAUDE_KO.md)

**v0.9 후속:** 실제 `voice_model_A`(31 step/epoch)·KLM49 pretrain에서 shape·방향별 FAST solver 선택, dilated Conv workspace 질의 복구, 직접 Conv1d와 100 epoch 청취 판정까지 진행했다. 위 `bench40k` 16-step 수치와 구분한 결과는 §9.21에 있다.

### 9.15 LayerNorm·RMSNorm·adaLN: 통계와 modulation을 분리해 고른다

**통계를 교체한 LN/RMS/full adaLN은 실제 Anima RGB 엄격 기준을 실패했다. native 통계를 유지한 modulation 융합은 모델 바이트까지 보존했지만 전체 시간 이득은 약0.39%로 작았다.** LayerNorm3shape, RMSNorm3shape, FP32 residual/FP16 modulation의 adaLN3shape, 기존 Kitchen 융합3shape를 비교했다. 총12개 기준선·15개 후보의 연산 tolerance가 통과했지만 값이 비슷한 것과 native 바이트 재현은 구분한다. 단위 입력은 seed 고정 합성값이며 실제 모델 값의 검증은 아래와 같다. [전체 실측·소스 계약](results/NORM_FUSIONS_V06_KO.md), [27개 기준선/후보 원자료](results/norm_fusions_v06.json)

#### B/A — 기존 융합을 먼저 확인한다

PyTorch LN/RMS는 `vectorized_layer_norm_kernel<..., false/true>`를 실행했다. MOSS/Qwen의 `ck`는 AMD Composable Kernel이 아니라 **Comfy Kitchen**이며, 실제 선택은 `comfy_kitchen.backends.hip.adaln/rms_adaln`과 `comfy::hip_backend::adaln_kernel<float,true/false>`였다. 이미 한 kernel인 경로를 eager 여러 kernel로 풀어 놓고 비교하지 않는다. Qwen prefix는 target 전체 계산 뒤 prefix를 다시 계산·복사하므로 두 normalization과 copy가 기준에 포함된다. MOSS의 weight-fold는 두 경로에 같은 준비값을 주고 별도 비용으로 분리했다.

Anima형 경계는 residual FP32, scale·shift FP16, norm 출력 FP32, attention/MLP 입력 FP16이다. **`1+scale`은 FP16으로 반올림한 뒤** FP32 activation에 곱하고 shift를 더한다. 전체 계산을 FP32 한 식으로 합치면 원래 결과와 다른 수식 평가가 된다. modulation-only 후보는 native LN 통계를 유지하고, 명시적 dtype 경계와 `enable_fp_fusion=False`로 pointwise 부분만 합친다. 입력 modulation은 chunk view의 실제 stride3D를 처리한다. gated residual은 기존 `torch.addcmul`과 FP32 stream을 유지했다.

#### A — 대표 shape와 warm wall 시간

3warmup, 순서를 회전·역전한3round×10call이며 JIT·프로파일·정확도 검사는 timing 밖이다. 필요한 변환·할당·호출 비용은 포함한다. 전체 출력 native 비교와 독립 CPU FP64 4행/full-D 통계 표본을 함께 확인했다. FP16 rtol/atol0.005/0.005, FP32 0.0001/0.00003이며 modulation-only는 추가로 uint8 byte equality를 요구했다. GPU audit140 samples, 외부 compute PID/오류0이었다.

| 연산과 입력 shape | native→후보 wall µs | 배속 | 출력 판정 |
|---|---:|---:|---|
| LN FP32 [2,4096,2048] | 102.628→91.503 | 1.122× | tolerance 통과·byte 불일치 |
| LN FP32 [1,2400,4096] | 103.947→68.917 | 1.508× | tolerance 통과·byte 불일치 |
| LN FP16 [8,257,768] | 16.935→24.016 | 0.705× | 정확하지만 느려 거절 |
| RMS FP16 [2,512,16,128] | 58.469→24.899 | 2.348× | tolerance 통과·byte 불일치 |
| RMS FP32 [1,37,3072] | 8.382→24.771 | 0.338× | 정확하지만 느려 거절 |
| RMS FP32 [1,2564,4096] | 101.214→67.926 | 1.490× | tolerance 통과·byte 불일치 |
| native LN+modulation, [2,1,64,64,2048] | 742.023→280.127 | 2.649× | **byte 일치** |
| 같은 경계, [1,1,32,32,2048] | 61.316→43.855 | 1.398× | **byte 일치** |
| 같은 경계, [1,1,64,128,2048] | 698.631→273.430 | 2.555× | **byte 일치** |
| 통계 포함 전체 adaLN, 위 세 shape | 87.361 / 40.490 / 114.510 | 8.494 / 1.514 / 6.101× | tolerance 통과·byte 불일치 |
| Kitchen RMS-adaLN [1,1061,3072] | 46.007→38.726 | 1.188× | tolerance 통과·MOSS 모델 미검증 |
| Kitchen adaLN [1,4964,4096], prefix2564 | 603.027→364.579 | 1.654× | tolerance 통과·Qwen 모델 미검증 |
| Kitchen adaLN [1,2400,4096], target-only | 133.255→64.157 | 2.077× | tolerance 통과·Qwen 모델 미검증 |

전체 adaLN 행의 세 숫자는 후보 시간이며 위의 같은 shape별 native와 비교한다. 대표 Anima 추가 peak는 native192MiB→modulation-only96MiB→전체융합32MiB였다. Qwen prefix는118.0625→78MiB였다. actual GPU kernel과 event 시간은 raw에 별도 남겼다. 단일 profile의 kernel duration을 반복 wall/event 중앙값과 섞지 않는다.

#### A — 기존 Anima 최종 RGB와 디노이징 시간

기존1024²·seed20261107·30step workflow, 같은 conditioning/native Attention에서 norm만 치환했다. VAE는 후보 적용 밖에서 최종 float RGB를 계산했다. 전체 표의 시간은 denoising이며 decoder나 요청 전체 시간이 아니다. [실제 모델 원자료](results/v06_anima_norm_ab/runs.json)

| 경로 | denoise 초(실행 순서) | 실제 치환 / fallback | 최종 RGB max / RMSE | 판정 |
|---|---|---|---|---|
| native control | 80.6070 / 80.7198 | 후보 없음 | 0 / 0 | 대조 latent/RGB bytes 일치 |
| LN 통계 교체 | 94.3118 | 2520 / 30 | 0.824707 / 0.0236855 | strict 실패 |
| RMS 통계 교체 | 80.4529 | 1680 / 116 | 0.8203125 / 0.0251574 | strict 실패 |
| native LN + modulation·cast | 80.2457 / 80.4511 | 각2520 / 0 | 0 / 0 | 두 요청 latent/RGB bytes 일치 |
| 전체 adaLN 통계+modulation | 79.3754 | 2520 / 0 | 0.809082 / 0.0249091 | strict 실패 |

modulation-only 중앙값은 native80.6634→80.3484s, 1.0039배다. native 두 값의 span0.113s, 후보 span0.205s와 작은 표본 n=2를 함께 남긴다. **단위의2.65배를 실제 모델 배속으로 승격하지 않는다.** 확인된 장점은 원본 바이트 유지와 연산 단위의 traffic·peak 감소이며, 이 Anima workload에서 큰 전체 가속으로 권할 결과는 아니다. 다른 모델의 normalization 비중이 크다면 새 최종 A/B가 필요하다.

첫 LN 후보는 이 서버의 첫 Triton 후보였으며94.312s였다. 원래 FP16 Attention 실험에서도 첫 Triton row-repair 후보가 느린 패턴이 있었지만 초기화/JIT/host 대기 시간을 분해한 실험은 아니다. 이를 warm kernel 성능의 확정 원인으로 해석하지 않는다. LN/RMS/full adaLN은 이미 strict gate가 실패해 성능만을 위한 반복은 추가하지 않았다. 같은 실수 수식이라도 다른 통계·누산 평가 경로가 최종 native RGB 재현성을 보장하지 않는다는 결과다.

#### 구현·적용 범위

[`rdna2_ops/norm_fusions_v06.py`](rdna2_ops/norm_fusions_v06.py)는 `row_norm`, `adaln`, `modulate` 명시 함수를 제공한다. [`norm_switches_v06.py`](norm_switches_v06.py)는 지원 범위의 cast/offload·LoRA·residual 계약을 보존하는 연구 context다. 설치 Anima source SHA를 확인하며, 다른 구현을 같은 모듈 이름이라는 이유로 덮지 않는다. **현재 Anima Block adapter는 custom `mlp_patch`가 하나라도 있으면 원본 Block 전체로 fallback한다.**

후속 CPU 검토에서 측정 당시 orchestration이 여러 MLP patch의 추가 dict key를 다음 patch로 넘기지 않는 범위 결함을 찾았다. 현재 graph의 모델 byte 검증은 그대로 유효하지만 일반적인 patch 호환성 주장으로 넓히지 않는다. 측정 source2b8fcd43와 이전 filter 전9d243e80 원문을 보존하고, 종료 후 계산 본문은 바꾸지 않은 보수적 guard를 추가했다(현재87a11ff0). 새 guard는 GPU 계산을 다시 측정한 버전으로 소급하지 않는다. CPU8조합은 LoRA/position과 **단일 x-only MLP patch**까지의 당시 범위였으며, CPU JSON이 기록하지 않은 adapter SHA를 현재 값으로 채우지 않았다. broadcast/stride3조합도 GPU 실측과 구분해 보존했다. [측정 source](results/sources/norm_switches_v06_measured_model.py), [filter 전 source](results/sources/norm_switches_v06_before_measured_filter.py)

모델 adapter는 느린 FP16 D768 LN과 작은 FP32 D3072 RMS를 제외했다. FP32 D2048/4096 LN, FP16 D128/FP32 D4096 RMS 같은 dtype/width 필터는 **모든 행 수에서 빠르다는 보장**이 아니다. 실제 모델에서 호출 수와 최종 RGB를 확인해야 한다. 이미 융합된 RMS+RoPE, BF16, training, compile/graph, 미측정 broadcast는 이 결과로 채택하지 않는다. context 종료 시 원래 callable로 돌아가고 운영 설정은 바꾸지 않는다.

### 9.16 INT4 가중치 GEMM: 세 경로를 실제 Linear shape에서 비교한다

**dot8 명령이 있다는 사실만으로 W4A8 GEMM이 빨라지지는 않았다.** 실제 MOSS·Qwen·Anima의 projection/MLP up/down 9개 shape, g32/64/128의 대칭·비대칭 6개 포맷을 비교했다. 같은 저장값의 계산 관문 324개는 통과했지만, 기본 g64 경로에서 기존 fused INT8 또는 FP16 rocBLAS를 안정적으로 대체할 속도 근거는 없었다. 다음의 좁은 추가 실험과 모델 진단까지 구분해 읽는다. [전체 표·실패 이력](results/int4_v07/INT4_GEMM_V07_KO.md), [사전 기준](results/int4_v07/INT4_PLAN_V07_KO.md), [전체 matrix](results/int4_v07/matrix.csv)

#### B/A — 기준선과 저장값 계약

기준은 MOSS의 v0.5 `_int8_dot_dequant.kd` 융합, Qwen patientx의 `_int8_matmul_dequant_per_row_kernel.kd`, Anima FP16 `F.linear`의 rocBLAS HHS/ISA1030/WS32다. 이미 융합된 기준을 별도 dequant의 느린 경로로 바꾸어 비교하지 않았다. 실제 checkpoint의 가중치와 실제 Linear shape를 썼으며, 단위 실험의 activation은 seed 고정 합성값이다. 단위 표는 bias 없는 GEMM이고 bias 계약은 별도 oracle과 모델에서 검사했다.

저장 계약은 `W=(q-zero)*FP32_group_scale`이다. 대칭은 q∈[-7,7], affine은 signed nibble 좌표의 16수준과 zero를 쓴다. u32 한 개에 nibble 8개를 저장하며 low/high 순서·NK/KN layout을 명시한다. high-first는 정확성 검증을 했고, 큰 shape throughput은 low-first에서 측정했다.

W4A8 dot8은 activation을 A4로 바꾸지 않는다. signed A8에 대해 다음 항등식으로 두 signed dot8과 보정을 계산한다.

~~~text
lo4 = (A8 & 15) - 8
hi4 = A8 >> 4
A8 = lo4 + 16*hi4 + 8
dot(A8, q4) = dot8(lo4,q4) + 16*dot8(hi4,q4) + 8*sum(q4)
affine 보정: 위 결과 - zero*sum(A8)
~~~

activation nibble 재표현·합계 준비는 dot8 시간에 포함한다. Anima의 dot4/dot8은 별도 A16→A8 양자화가 필요하므로 그 시간과 표현 손실도 분리했다. MOSS/Qwen의 기존 A8 값은 그대로 보존한다.

가중치 준비는 호출자가 소유하며 원본 weight/LoRA 변경 후 다시 만들어야 한다. 이번 공통 pack에는 scale·zero·sum이 각각 4 B이므로 상주량은 **0.5+12/G B/weight**다. g32/64/128은 0.875/0.6875/0.59375 B로, “INT8의 절반”이라고 단정하면 metadata를 빠뜨린다. prepared FP16은 추가 2NK B, prepared INT8은 NK B가 필요하다. [명시 API](rdna2_ops/int4_weight_gemm_v07.py)

설치 Comfy Kitchen의 HIP W4 GEMM은 gfx1030의 `elementwise_only` 범위 밖이며 WMMA-only다. trap 경로를 강제로 호출하지 않았다. Triton/eager W4 decoder의 INT8 grid 반올림·상대 scale·codebook 계약도 이번 선형 W4와 다르다. 같은 “INT4” 이름만으로 서로 교환하지 않는다. [설치 소스 근거](results/int4_v07/kitchen_existing_paths_source.json)

#### A — 대표 9개 shape의 g64 대칭 KN 결과

원본 가중치→W4 pack의 1회 준비는 별도로 기록했다. 다음 표는 이미 pack된 W4를 입력받는 일반 호출 중앙값이며, 해당 경로의 activation 재표현과 호출당 unpack/dequant 비용을 포함한다. 배속은 **해당 모델의 실제 기준 시간 / 후보 시간**이다. FP16 dequant 열은 매 호출 unpack/dequant+rocBLAS를 포함한다. 단위의 작은 1% 차이를 안정적인 모델 이득으로 채택하지 않는다.

| 사례 | M,N,K | native ms | W4→I8 dot4 | A8 보존 dot8 | W4→FP16+rocBLAS |
|---|---|---:|---:|---:|---:|
| MOSS projection | 1061,3072,3072 | 0.6991 | 0.801배 | 0.499배 | 0.774배 |
| MOSS MLP up | 1061,16384,3072 | 3.3610 | 0.714배 | 0.428배 | 0.715배 |
| MOSS MLP down | 1061,3072,8192 | 1.9391 | 0.799배 | 0.278배 | 0.722배 |
| Qwen projection | 2400,4096,4096 | 2.3744 | 0.716배 | 0.341배 | 0.559배 |
| Qwen MLP up | 2400,24576,4096 | 14.0954 | 0.671배 | 0.368배 | 0.390배 |
| Qwen MLP down | 2400,4096,12288 | 8.0106 | 0.754배 | 0.347배 | 0.481배 |
| Anima projection | 8192,2048,2048 | 2.9631 | 0.994배 | 0.415배 | 1.017배 |
| Anima MLP up | 8192,8192,2048 | 8.6260 | 0.655배 | 0.319배 | 0.987배 |
| Anima MLP down | 8192,2048,8192 | 9.5330 | 0.670배 | 0.265배 | 0.953배 |

FP16 dot2 융합은 같은 9개 g64 행에서 0.242–0.358배였다. unpack를 곱셈 안으로 넣는 것만으로 기존 rocBLAS의 재사용·타일 효율을 넘지 못했다. [sweep 원자료](results/int4_v07/sweep01/results.json)

초기 prepared FP16의 MOSS g32 affine 1.056배와 Anima g64 대칭 1.108배는 7라운드×10회 교대 재검증에서 각각 **0.931배와 0.995배**로 재현되지 않았다. 추가 상주 FP16 weight까지 필요한 경로를 성과로 채택하지 않는다. [MOSS 재검증](results/int4_v07/confirm_moss01/results.json), [Anima 재검증](results/int4_v07/confirm_anima01/results.json)

#### A — 계산 오차와 표현 손실을 따로 읽는다

작은 CPU 정수 oracle은 nibble 조합 131,072개를 검사했고, smoke의 정수 출력 72/72·부동소수점 관문 30/30이 통과했다. 실제 projection 크기까지 포함한 bias 계약 9개는 grouped reference와 byte exact였다. 전체 54행×6경로=324개 계산 관문도 통과했다. FP32 같은 저장값 relative L2의 최댓값은 MOSS/Qwen 약 3.62e-7이었다. 이 수치가 원본 가중치를 INT4로 줄이는 손실까지 작다는 뜻은 아니다.

같은 activation, 독립 CPU FP64의 사전 선택 17 output row×19 column에서 **원래 저장 가중치와 W4의 출력**을 비교했다. 전체 weight tensor 오차나 RGB 오차가 아니라 표본 GEMM 출력의 relative L2다.

| 사례의 3개 shape | g32 affine 표현 손실 | g64 대칭 표현 손실 | g128 대칭 표현 손실 |
|---|---:|---:|---:|
| MOSS | 0.08190–0.08555 | 0.10064–0.10543 | 0.10080–0.11519 |
| Qwen | 0.07549–0.08292 | 0.09505–0.11748 | 0.10934–0.12186 |
| Anima | 0.09195–0.09888 | 0.13607–0.16184 | 0.16068–0.20356 |

MOSS/Qwen의 원래 가중치는 이미 INT8 checkpoint를 복원한 값이다. Anima는 현재 경로의 FP16 재표현 값이다. 학습 당시의 미양자화 원본을 새로 받은 것이 아니다. Anima g64 대칭의 A16→A8 추가 표현 손실은 projection/up/down에서 0.008172/0.008395/0.008477로 따로 측정했다. native→W4 모델 RGB에는 저장값 변경뿐 아니라 그룹별 scale·합산 순서 변경도 들어가므로 순수 weight 손실 하나로 이름 붙이지 않는다.

#### A/B — layout·spill을 줄여도 native 승리는 별개다

초기 NK dot4의 256 VGPR/23 spill을 KN에서 237/0으로 줄였다. projection 세 개는 0.570→0.809배, 0.478→0.725배, 0.676→0.932배로 나아졌지만 native를 넘지 못했다. 32×64 tile은 140 VGPR/0 spill이어도 0.726/0.628/0.857배로 느려졌다. packed-word dot4도 정확하고 spill 0이지만 0.263/0.201/0.275배여서 거절했다.

dot8은 A8 보존에 두 dot8과 보정이 필요하고, 그룹별 FP32 scale·누적과 입력 재표현 비용도 남는다. 그룹 INT32 합과 FP32 누적 tile을 동시에 유지하는 구조는 기존 full-K INT8 GEMM 후 마지막 scale보다 비싸다. 코드 구조와 ISA에 일치하는 해석이며, hardware counter로 각 stall 원인을 분해한 결과는 아니다. 모든 RDNA2 INT4 GEMM이 불가능하다는 결론으로 넓히지 않는다.

#### A — g128의 양자화 그룹과 계산 K 폭을 분리한 추가 실험

기본 g128 dot4는 G=128 전체를 한 load/dot 폭으로 묶어 136–137 spill이 생겼다. 양자화 그룹은 그대로 두고 **BK=64 두 번의 INT32 부분합을 합친 뒤 그룹 scale을 한 번 적용**하는 `dot4_split`을 추가했다. 정수합의 범위 안에서 같은 값을 합치므로 원래 그룹 반올림 계약을 보존한다. 원본 커널은 삭제하지 않았다. [실행 전 계획](results/int4_v07/SUBK_PLAN_KO.md)

| projection | g128 기존 → split, 대칭 ms | 기존 W4 대비 | 실제 native 대비, 대칭/affine | spill |
|---|---:|---:|---:|---:|
| MOSS | 1.7074 → 0.9386 | 1.819배 | 0.761 / 0.737배 | 136 → 19 |
| Qwen | 7.1651 → 3.5254 | 2.032배 | 0.663 / 0.646배 | 137 → 18 |
| Anima | 6.0240 → 3.1051 | 1.940배 | 0.955 / 0.865배 | 137 → 18 |

3개 projection×대칭/affine 6개에서 이전 dot4와 **전체 output byte 일치**했고 독립 계산 관문도 통과했다. 원인에 대응한 변형으로 W4 구현 자체는 나아졌지만 기존 fused INT8/FP16 rocBLAS 대비 1.05배 관문에는 모두 미달했다. 더 큰 타일 탐색으로 범위를 늘리지 않았다. [추가 원자료](results/int4_v07/subk_screen01/results.json)

#### A — MOSS 최종 출력: 같은 W4 계산은 byte exact, 표현 변경 경로는 엄격 실패

기존 512²·4-step SFT workflow, seed 20261105, g32 affine/KN/low-first를 결과 전에 고정했다. warm → INT8 fused → W4 독립 grouped reference → W4 dot4 → INT8 fused 순서다. 앞뒤 native와 conditioning은 바이트가 같았다. 각 후보가 실제 4개 Linear shape에 총 **672회** 적용됐고 fallback은 0이었다. VAE는 같은 native 경로로 최종 float RGB를 계산했다. [실제 runs](results/int4_v07/moss_model01/runs.json), [모델 계획](results/int4_v07/MODEL_DIAGNOSTIC_PLAN_KO.md)

| 비교 | latent | 최종 RGB max / RMSE | 판정 |
|---|---|---|---|
| 첫 INT8 fused ↔ 마지막 INT8 fused | byte exact | 0 / 0 | native 제어 통과 |
| 같은 W4 grouped reference ↔ 직접 dot4 | **byte exact** | **0 / 0, byte exact** | 새 커널의 추가 계산 오차 없음 |
| 원래 INT8 fused → W4 표현 경로 | max 5.88128 / RMSE 0.71916 | 0.94262695 / 0.17760117 | 엄격 실패 |
| 같은 RGB를 1/255 단위로 표시 | — | **240.3699/255 / 45.2883/255** | 기준 2/255 / 0.25/255 밖 |

W4 reference는 그룹별 정확한 정수 곱을 FP32 `torch.mm`로 계산하고 동일한 scale·합산 경계를 명시한다. 이 reference와 custom의 모델 전체 byte 일치가 계산 구현을 분리해 검증한 근거다. 원래 INT8 경로와 W4 reference의 차이에는 가중치 표현뿐 아니라 그룹별 FP32 scale·합산 계약의 변경도 포함되므로, 위 RGB를 **순수 weight 표현 손실 하나의 수치**로 부르지 않는다. 단위 FP64 표본의 표현 손실은 앞 표에 따로 남겼다.

진단 denoise는 INT8 2.0866/2.0958 s, W4 oracle 66.3566 s, 직접 dot4 10.4794 s였다. 후보는 borrowed staging pointer를 장기 cache하지 않고 매 호출 원본 W8을 복원·pack했으며 그 비용도 들어간다. oracle보다 빠르다는 수치를 실제 기준선 대비 가속으로 쓰지 않는다. 준비비를 제외한 단위 실험도 속도 관문에 실패했으므로 이 모델에서 상주 pack 최적화를 추가할 근거가 없었다.

재사용 API는 `pack_int4`, `unpack_int4`, `int4_linear_a8`, `int4_linear_a16`이다. caller가 pack의 수명과 invalidation을 책임진다. 연구 adapter는 context 종료 시 원래 export를 복구한다. 운영 registry·설정·checkpoint를 바꾸지 않았으며, 현재 채택 결정은 **기존 MOSS fused INT8/Qwen patientx/Anima FP16 rocBLAS 유지**다.

### 9.17 Sparse Attention: 실제 질량·같은 mask 계산·최종 지각을 각각 판정한다

**Qwen K=4,964에서 token score 행렬을 저장하지 않는 FP32 block-sparse 커널을 실행했고, 일부 입력의 속도·메모리 이득은 확인했다. 후반 입력의 native 재현 tolerance는 실패했으므로 전 구간 자동 채택 후보는 없다.** 가장 유망한 image-only r4는 별도의 품질 진단으로 고정했다. 아래의 확률 질량·연산 결과와 최종 RGB·지각 판정을 섞지 않는다. [단위 5-case 요약](results/sparse_v07/OPERATOR5_SUMMARY_KO.md), [전체 원자료](results/sparse_v07/OPERATOR5_SUMMARY.json), [map 분석](results/sparse_v07/MAP_ANALYSIS_KO.md)

#### A/B — 실제 callable·prefix·mask부터 확인한다

기존 Qwen 960×640 이미지 편집 workflow, 25-step Euler/simple을 썼다. 대상은 FP32 B1/H32/D128, Q=2,400/K=4,964이다. FP32 token score 한 개만 **1,524,940,800 B = 1.4202 GiB**다. 실제 dense 실행은 이 PC에서 완료했으며, Anima 1536²/2048²의 OOM 경계는 이번에 측정하지 않았다.

K 구간은 text [0,8), reference image [8,2408), text [2408,2564), target image [2564,4964)다. image grid는 40×60이다. 실제 총 896회는 target 800회 + reference-prefix 32회 + masked 64회다. alpha 적용의 832회에 prefix 32회가 이미 포함되므로 이를 fallback에 다시 더하면 안 된다. 초기 계측의 fallback=96에는 이 prefix 32회가 중복으로 들어갔고, 새 계측은 세 범주를 분리한다.

단위 기준은 unmasked에서 `attention_split + alpha 접기`, masked-text에서 원래 `attention_split`이다. masked 입력을 원본으로 넘기는 generic alpha wrapper의 Python overhead를 더 느린 기준으로 삼지 않는다. 모델 adapter도 masked 64회를 원래 callable에 넘긴다.

실제 저장된 mask는 **bool [156,2564]**, true 387,894개/false 12,090개였다. 이 설치 함수는 score에 `+= mask`를 하므로 **0/1 additive bias**로 소비한다. bool이라고 PyTorch SDPA의 keep/drop 의미를 가정하면 다른 계산이 된다. 처음 벤치에서 이를 causal 제외로 읽어 native와 최대 5.48432 차이가 났다. 일부 Sparge mask는 visible key 0인 row까지 만들어 reference softmax가 NaN이었다. streaming의 empty-row 출력 0과도 계약이 달랐다. FP32 overflow로 생긴 NaN이 아니다.

`prefix_bias_offset` 옵션으로 관측한 0/1 bias 계약만 보존하자 corrected reference는 native와 bit 일치했고, all-true streaming은 max 2.277e-5/RMSE 5.357e-7로 사전 계산 관문을 통과했다. 운영 callable은 수정하지 않았다. raw mask SHA는 `6963aa8ff515b246078ad28edb9468a2aa7db6ab0d858b55db4a870381d62d01`, callable source SHA는 `9cafaafaf93ff53e8cbefb5e4a204014019985df2da8c1996bf40f1235fc2960`다. [계약과 capture 근거](results/sparse_v07/OPERATOR5_SUMMARY.json)

#### A — step·layer별로 실제 희소성을 먼저 잰다

native step 0/12/24 × layer 0/15/31, 총 9개 target 호출을 관측했다. 선택기 block은 Q 128×K 64다. 아래 oracle은 실제 block-mean 확률 질량의 99%를 보존하는 데 필요한 밀도이며, 모든 query의 최악 조건을 보장하는 수치가 아니다. 9개 선택 호출을 전체 800회의 빈도 가중 평균으로 쓰지 않는다.

| step / layer | oracle 99% block 밀도 | Sparge식 95% 선택 밀도 | 실제 보존 질량 평균 | 실제 질량 p01 |
|---|---:|---:|---:|---:|
| 0 / 0 | 63.44% | 100.00% | 100.00% | 약 100% |
| 0 / 15 | 65.82% | 37.90% | 83.53% | 20.39% |
| 0 / 31 | 36.39% | 24.28% | 85.02% | 13.78% |
| 12 / 0 | 26.94% | 77.99% | 94.50% | 약 0% |
| 12 / 15 | 58.04% | 31.50% | 82.47% | 21.10% |
| 12 / 31 | 32.43% | 21.40% | 85.47% | 15.24% |
| 24 / 0 | 10.38% | 67.02% | 88.77% | 약 0% |
| 24 / 15 | 53.34% | 30.50% | 82.52% | 10.73% |
| 24 / 31 | 29.01% | 19.45% | 85.42% | 7.46% |

**압축 Q/K의 softmax CDF 95%와 실제 token attention 질량 95%는 같지 않다.** pooling과 softmax를 바꿔 적용하면 같은 값이 되지 않으며, 평균 질량이 높아도 특정 query의 질량은 거의 전부 버릴 수 있었다. Sparge식 95%의 실제 최소 질량은 일부 호출에서 1e-18 수준이었다. 임계값 이름을 품질 보장으로 쓰지 않는다.

#### B/C — 후보의 설계 범위

- **Sparge-inspired:** block mean과 평균 pairwise cosine을 한 번의 읽기로 구하고, 압축 QK의 CDF로 block을 고른다. 낮은 유사도는 dense fallback한다. mass .90/.95/.99와 cosine .1/.5를 비교했다. upstream의 normalized similarity·INT8 Sage·PV warp/Hilbert 최적화를 그대로 이식한 커널이 아니다. [원 논문](https://arxiv.org/html/2502.18137v6), [공식 코드](https://github.com/thu-ml/SpargeAttn)
- **원 temporal Radial:** 같은/인접 frame을 보존하는 원래 규칙은 f1/f2 이미지에 적용하면 밀도 100%라 희소성이 생기지 않는다. CPU 공식 검증을 했으며 영상용 포팅 성공으로 세지 않는다. [원 논문 §4.2](https://arxiv.org/html/2506.19852v1), [CPU 원자료](results/sparse_v07/radial_formula_cpu.json)
- **image-only 2D radial-inspired:** reference/target의 40×60 좌표에서 radius 4/8/16과 text 전체 보존을 시험했다. 동일 grid 좌표가 편집 전후의 의미상 대응을 보장하지 않는다. r4 block 밀도는 32.73%, 9개 호출의 실제 보존 질량 평균은 72.03–97.98%, p01 최솟값 19.86%였다. r8/r16은 밀도 47.77%/72.13%로 품질과 계산량이 함께 달라진다.
- **고정 block:** 약 절반을 유지하는 대조다. 실제 밀도 53.37%, 보존 질량 평균 57.89–89.07%였다. 원래 의미나 움직임을 학습하는 후보가 아니다. Veda와 새 predictor/model 다운로드는 범위에서 제외했다.

#### A — 실제 3개 shape + 같은 shape의 중후반 입력 2개

FP32 online softmax로 선택 block만 순차 처리한다. token score 행렬은 만들지 않는다. 작은 독립 FP64 3문제를 먼저 확인한 뒤 아래 실제 값의 전체 tensor를 같은 mask의 dense 참조와 비교했다. 사전 tolerance는 atol 3e-5 + rtol 3e-4이며 결과 후 완화하지 않았다.

2회 warmup 뒤 순서를 회전/반전한 3라운드×2회 일반 호출 중앙값이다. dynamic Sparge는 매 호출 선택기 비용을 포함한다. static r4의 mask 준비는 단위 시간에서 별도이고 모델 시간에는 실제 준비·계측·dispatch가 포함된다. JIT/정확도 검사와 warm 시간은 분리했다.

| 실제 입력 | Q/K | 기준 ms | Sparge .95 선택기 포함 ms | r4 ms / 배속 | r4 같은-mask 판정 |
|---|---|---:|---:|---:|---|
| target step 0/layer 0 | 2400/4964 | 22.666 | 54.434 | 18.600 / 1.219배 | 통과 |
| reference prefix/layer 0 | 2400/2408 | 12.867 | 24.316 | 8.201 / 1.569배 | 통과 |
| masked text prefix/layer 0 | 156/2564 | 1.892 | 1.746 | 해당 image geometry 아님 | masked 원본 유지 |
| target step 12/layer 15 | 2400/4964 | 22.704 | 18.251 | 18.524 / 1.226배 | 통과 |
| target step 24/layer 31 | 2400/4964 | 22.757 | 시간 후보 제외 | 시간 후보 제외 | native 재현 tolerance 실패 |

Sparge .95는 중간 입력에서 1.244배였지만 step 0/layer 0에서는 dense fallback으로 0.416배였다. 고정 50%는 첫 target에서 28.454 ms로 native보다 느렸다. r8/r16도 26.425/39.436 ms로 r4보다 비싸고 기준을 넘지 못했다. 동일 shape라도 입력의 step/layer가 바뀌면 희소성·오차가 달라진다.

q32/k64/w8에서 q16/k64/w4로 줄이는 변형도 한 번 비교했다. 첫 target all-true는 54.377→108.670 ms로 악화됐다. 실제 unmasked q32는 VGPR 187–189/spill 0, q16은 207/0이었다. masked shape는 각각 208/0, 212/0이었다. 작은 타일이 레지스터나 속도를 자동 개선하지 않았다.

첫 target의 torch allocator incremental peak는 기준 **3,102.594 MiB → r4 38.000 MiB**였다. input과 미리 만든 mask를 제외한 추가 allocation이며 driver scratch나 모델 전체 VRAM이 아니다. output/work tile은 선형 또는 고정 크기지만 block metadata는 **O(BH·ceil(Q/128)·ceil(K/64))**다. 고정 block 크기에서 이 전체 경로를 O(S)라고 부르지 않는다.

#### A — 후반 tolerance 실패의 원인 범위

step 24/layer 31에서 두 tile의 모든 streaming 전략이 같은-mask native 재현 관문에 미달했다. all-true는 max 1.8549e-4/RMSE 5.2414e-6, r4는 max 1.8644e-4/RMSE 5.0890e-6였다. 출력은 finite였다. 서로의 차이가 큰 3개 row를 CPU FP64와 각각 비교했다.

예를 들어 r4 head 23/query 1164는 streaming→FP64 max 2.102e-5, rocBLAS reference→FP64 max 1.478e-4로 후보가 더 가까웠다. query 440에서는 각각 1.010e-4/6.467e-5로 반대였다. native reference도 FP32 반올림 오차를 갖는다. 이 관측만으로 수학식 오류나 overflow라고 단정하지 않으며, 선택 3개 row의 결과를 전체 tensor 통과로 바꾸지도 않는다. 이번 목표의 native 재현 관문은 그대로 실패다. [FP64 row 진단](results/sparse_v07/OPERATOR5_SUMMARY.json)

#### A — all-true 모델 제어와 실패 자료 보존

어떤 block도 버리지 않고 target 800회만 같은 FP32 streaming 경로로 바꾼 25-step 제어에서도 RGB max **0.4283854(109.238/255)**, RMSE **0.00681241(1.737/255)**로 strict 실패했다. latent max 1.86124/RMSE 0.0619614였다. native 반복과 alpha 관측 경로는 byte 일치했다. 따라서 sparse 최종 RGB 차이 전부를 block pruning 탓으로 돌릴 수 없다. reduction·online softmax·P·V 순서 변경과 모델의 비선형 반응이 먼저 존재한다. 두 최종 오차를 단순히 빼서 pruning만의 기여로 해석하지 않는다.

이 all-true 자료는 CPU 빌드가 겹쳐 **품질 제어로만** 썼고 속도 비교에서 제외했다. 첫 operator 실행의 JSON NaN 직렬화 실패로 최종 GPU audit가 저장되지 않은 자료도 유효 timing에서 제외했다. 계약 수정 후 첫 3개 shape를 다시 측정해 완전 audit를 확보했고, 중후반 2개는 별도의 정상 audit 원자료를 썼다. 실행 실패·source·로그는 지우지 않았다.

명시 API는 [`sparse_attention`, `sparge_map`, `fixed_map`, `radial_map`](rdna2_ops/sparse_attention_v07.py)이고, [모델 adapter](sparse_switches_v07.py)는 연구 process context 안에서만 동작한다. 기존 서버 설정이나 production node를 수정하지 않았다.

#### A — 실제 Qwen 25-step 모델: 세 seed, strict 실패

미리 고정한 r4/q32/w8 경로로 기존 Qwen INT8 convrot workflow를 25 Euler/simple step, cfg 1에서 실행했다. seed 20261107은 native → alpha → r4 초회 → r4 warm → alpha의 5요청, seed 08/09는 native → alpha → r4 초회의 3요청씩이다. 최종 RGB를 보기 전에 후보·seed·crop·판정 기준을 고정했고 실패를 본 뒤 다른 mask로 바꾸지 않았다. [모델 계획](results/sparse_v07/MODEL_DIAGNOSTIC_PLAN_KO.md), [모델 원자료 요약](results/sparse_v07/MODEL3_SUMMARY.json)

| seed | alpha denoise | r4 denoise / 구분 | alpha/r4 | RGB 최대 /255 | RGB RMSE /255 | 엄격 결과 |
|---|---:|---|---:|---:|---:|---|
| 20261107 | 전후 중앙값 50.874496 s | 46.394129 s, 같은 process 두 번째 후보(warm) | **1.096572배** | 243.441 | 49.800 | 실패 |
| 20261108 | 50.778714 s | 46.411995 s, 후보 초회 | 1.094086배, 초회 관측 | 240.140 | 48.222 | 실패 |
| 20261109 | 51.017942 s | 46.549846 s, 후보 초회 | 1.095985배, 초회 관측 | 241.362 | 54.020 | 실패 |

첫 native 66.895564/65.699724/65.614926 s는 기동·준비가 섞인 초회이며 warm 비교에 넣지 않는다. seed 07의 r4 초회 46.431769 s와 두 번째 46.394129 s의 latent/RGB는 byte 일치했다. seed 08/09는 후보 두 번째 요청을 실행하지 않았으므로 세 seed의 비율을 warm 평균으로 합치지 않는다. warm 성능의 독립 후보 표본은 seed 07의 한 요청이다. VAE는 seed 07 r4 warm에서 2.614251 s였으며 표의 배속은 denoising 구간만이다. tensor 원자료 저장은 stage clock 밖에서 수행하므로 API 요청 전체 wall과 혼동하지 않는다.

세 seed 모두 native↔alpha의 latent/RGB가 byte 일치했고 positive/negative conditioning SHA도 동일했다. 실제 요청당 호출은 target sparse 800 + reference-prefix dense 32 + masked native 64 = **896회**다. adapter hit 832에는 prefix 32가 이미 들어 있으므로 832+32+64로 합산하지 않는다. 세 모델 audit 합계 2,653개 표본에서 관측된 foreign KFD client와 audit error는 0이었다. 마지막 연구 GPU PID 잔류도 없었다. [GPU 정리 확인](results/sparse_v07/model_cleanup.json)

이번 r4는 late same-mask 계산 관문을 이미 통과하지 못한 **품질 진단용 후보**다. 최종 최대 2/255·RMSE 0.25/255 기준도 세 seed 모두 실패하여 자동 적용에서 제외한다. 전술한 all-true streaming 대조도 strict에 실패했으므로 r4 모델 차이 전부를 희소 pruning 손실이라고 부르지 않는다. 반올림·누산 순서 변화와 mask 누락이 비선형 모델을 통과한 최종 차이는 RGB 오차 두 개의 단순 차감으로 분리할 수 없다.

#### A — 익명 지각 평가도 실제 수행했고 실패했다

**2 판정자 × 3 seed × 2 좌우 순서의 12결정 전부가 native를 선호했고 후보 상대 severity는 모두 3**이었다. 판정자×seed의 6개 양방향 쌍에서 인물·얼굴의 다층 반복과 신체/책/컵 연결 붕괴라는 동일한 큰 결함이 반복됐다. 후보 동등·우세 및 severity≤1은 0/12, 순서에 따른 선호 반전과 판정자 간 후보 채택 결론 충돌은 각각 0이다. 사전 지각 gate도 실패다.

네 fresh context(각 판정자·좌우 순서별)에 연구 이력 없이 익명 full과 고정 crop, 공통 입력·생성 지시만 제공했다. 각 세션은 13개 고유 파일을 열람한 기록을 남겼다. 확인 가능한 모델 표시는 모두 GPT-6 Astra Pro, 내부 revision은 unknown이다. **같은 모델 계열의 독립 문맥 반복이며 서로 다른 아키텍처 검증이라고 부르지 않는다.** 원래 경로의 작은 손/소품 지시 차이에 대한 개별 지적도 원문에 보존했다.

[12개 판정·양방향 검토](results/sparse_v07/blind_judgements/BLIND_EVALUATION_KO.md), [기계 판독 verdict](results/sparse_v07/blind_judgements/UNBLINDED_VERDICT.json), [동결 프롬프트](results/sparse_v07/BLIND_JUDGE_PROMPT_V07.json), [실제 세션 지시문](results/sparse_v07/blind_judgements/SESSION_PROMPTS.json), [공개 이미지 SHA·좌표](results/sparse_v07/blind_panels/PUBLIC_CASES.json)를 남겼다. 네 원응답 JSON은 `blind_judgements/`에서 각각 읽을 수 있다. 판정 뒤 해제한 mapping도 보존했다. 같은 워크플로의 세 seed 표본이며 다른 모델·영상·sparse 방식의 품질로 일반화하지 않는다. strict 실패는 이 평가로 대체하지 않는다.


재현은 [Claude용 핵심 숫자와 명령 3·4번](results/V07_REPRO_CORE_KO.md), 전체 코드·실패 이력은 [Sparse 상세 보고서](results/sparse_v07/SPARSE_ATTENTION_V07_KO.md)를 따른다. source와 실제 mask 계약을 먼저 확인하고, 입력을 캡처한 다른 모델에서는 같은 5-case나 같은 허용 오차가 자동으로 성립한다고 보지 않는다.



### 9.18 Strata MoE 프리필: J 타일은 분포별 후보, warm 병목은 자원 차단으로 미판정

**실제 layer 35의 IQ3_XXS gate/up·IQ4_NL down 연산은 3개 새 분포에서 54행 모두 byte exact였다. 작은 치우친 분포의 평균 J 힌트는 1.090배/1.080배였지만, 모델 전체 채택 근거는 아직 없다.** warm 프리필의 계산·PCIe 병목 판정과 9.7K·19.8K·119K J off/on A/B는 메모리 회수에 따른 보호 중단으로 완료하지 못했다. 디코드는 이번 범위에서 제외했다. [전체 보고서](results/strata_v07/STRATA_PREFILL_V07_KO.md), [사전 기준과 후속 자원 보호 조건](results/strata_v07/PROTOCOL_KO.md), [이전 랩 검토](results/strata_v07/PRIOR_LABS_KO.md)

#### B — 모델 이름보다 실제 pack 타입과 타일을 확인한다

운영 pack 명칭이 IQ3_XXS여도 48개 층의 gate/up은 IQ2_XXS 9개, IQ2_XS 10개, IQ2_S 10개, IQ3_S 13개, **IQ3_XXS 6개**였다. down은 IQ4_NL 18개와 Q2_0 30개다. 타입 42는 해당 빌드의 ggml enum에서 Q2_0으로 확인했다. 이번 A 연산 결과의 범위는 layer 35와 16개 실제 expert이며, 한 층의 배속을 전체 모델에 곱하지 않는다. [타입·offset 원자료](results/strata_v07/source_and_pack_provenance.json)

IQ3_XXS의 코드는 signed INT4 값이 아니라 코드북 인덱스다. IQ4_NL도 비선형 표를 쓴다. 인덱스를 그대로 dot8에 넣으면 같은 곱셈이 되지 않는다. §9.16의 선형 W4 결과를 이 경로로 직접 옮기지 않는다. 현재 IQ3 커널의 생성 ISA에는 dot4가 있고 dot8은 없었다. 유효 J는 8/16/24/32/40/48/64이며, 예전 문서의 J128을 이 빌드에 적용하지 않는다. J48은 VGPR 256/private 16 B, J32는 183/0 B, J64는 233/0 B였다. 정적 ISA와 private segment만으로 동적 활용률을 단정하지 않는다. [ISA·리소스](results/strata_v07/iq3_registers.json)

#### A — 최소 3개 분포, 실제 저장 가중치, 같은 산술

GU는 M=1,280/K=2,560, down은 M=2,560/K=640이며 각 분포에 16 expert를 썼다. expert별 라우팅 분포와 FP32 activation은 seed 고정 합성 입력이다. 실제 모델에서 캡처한 라우팅이라고 표시하지 않는다. 기준은 최대 행 수로 J를 고르고, 후보는 평균 행 수를 힌트로 주되 grid coverage와 Q8_1 산술은 보존한다.

| 분포: 총 행/평균 올림/최대 | 역할 | 기준 → 평균 힌트 | 배속·판정 |
|---|---|---:|---|
| 702 / 44 / 512 | GU | J64 440.676 → J48 404.290 μs | 1.090배, byte exact |
| 같은 분포 | down | J64 197.342 → J48 182.684 μs | 1.080배, byte exact |
| 3,389 / 212 / 512 | GU / down | 두 선택 모두 J64 | 같은 커널, 개선 아님 |
| 6,592 / 412 / 1,024 | GU / down | 두 선택 모두 J64 | 같은 커널, 개선 아님 |

3회 prewarm, 순서를 교대한 7라운드×20회 중앙값이다. 3개 분포×2개 역할×9개 선택의 **54행 전체 출력 바이트 일치**를 확인했다. 같은 저장 가중치와 원래 FP32 activation의 독립 FP64 표본은 relative L2 최대 0.002893, max_abs/ref_rms 최대 0.012302로 기존 Q8_1 관문을 통과했다. 이는 표본 참조이며 전체 출력 byte 비교와 구분한다.

탐색 최저 중앙값은 작은 분포에서 GU J40 383.902 μs(1.148배), down J32 161.284 μs(1.224배)였다. 이를 자동 모델 정책으로 채택하지 않았다. medium/large에서 같은 J64를 두 번 잰 비율이 최대 1.0308까지 흔들렸으므로, 1.03이라는 속도 관문만 보고 성공으로 세면 안 된다. **실제로 선택 커널이 바뀌었는지 먼저 확인한다.** [54행 원자료](results/strata_v07/operator_screen/operator.jsonl), [요약](results/strata_v07/operator_summary.json)

#### A/B — 첫 요청의 dequant 74.92%는 순수 GPU dequant가 아니다

운영 pinned ELF와 바이트 동일인 연구 사본, 별도 포트 18477에서 첫 fresh 요청을 완료했다. actual 9,716 tokens/read 9,716/reused 0, native prompt 48.774 s(199.2 tok/s), API 49.141 s였다. 같은 엔진의 priming 없는 **첫 요청**이며, 사전 SHA 읽기가 파일 cache에 영향을 주었으므로 통제된 cold-disk 수치도 아니다.

가장 큰 9,711-token chunk의 GPU event timeline은 38.989 s였다. dequant라는 이름의 구간 29.209 s(74.92%), GU 1.338 s, down 2.143 s, GDN 입력 projection 1.115 s, QSA attention 1.019 s였다. PfTimer는 연속 event 사이의 시간을 한 phase에만 넣는다. GDN 입력 projection과 별도 conv/gates·recurrence·out projection은 서로 배타 구간이다. host staging 누계 38.758 s와 timeline·wall은 겹치므로 더하지 않는다.

MMQ 경로의 dequant 구간에는 GGUF 블록 gather·copy 발행 대기 `wait_issued`와 host 지연에 따른 GPU idle이 들어간다. 실제 IQ lookup과 dot4는 GU/down 커널 안에 있다. native PID는 D 상태의 `folio_wait_bit_common`을 보였고, engine 시작부터 첫 요청 완료까지 OS read는 47,202.9 MB였다. 이는 기동·prefill·8-token 생성이 포함된 누계이며 prefill 전용 I/O가 아니다. 첫 요청 구간의 global swap-out은 **6.134 GB(5.713 GiB)** 증가했다. workload의 발행 H2D bytes는 이 실행에서 미계측(null)이므로, 시작 때의 PCIe probe 28.5 GB/s만으로 전송 병목을 정량 판정하지 않는다. [단계 원자료](results/strata_v07/pinned_stage_first_summary.json)

이 첫 요청 자료는 사용자가 확인한 warm 1,200–1,300 tok/s를 반증하지 않는다. **“warm 병목이 GPU 계산”이라는 가설은 현재 미판정**이다.

#### A — warm 모델 A/B의 보호 중단과 남은 조건

첫 요청의 swap 관측 후, 연구 프로세스 시작 대비 추가 swap-out이 256 MiB를 넘고 1초 간격 두 표본에서 계속 증가하면 소유한 연구 process group만 종료하는 조건을 추가했다. 정확도·속도 기준은 바꾸지 않았다. 이후 priming 두 번은 각각 580.43 MiB, 275.39 MiB 추가 swap-out에서 중단됐다. 두 번째는 read-only mincore로 전문가 파일 92.78%의 cache residency를 확인한 뒤 한정 재시도했지만 같은 관문에 걸렸다.

native RSS는 약 20.00 GB 중 file 18.71 GB/anon 0.607 GB였고 native 자체 swap은 0 B였다. 큰 연구 anon 누수보다 file cache를 채우는 동안의 시스템 회수 압력과 일관되는 관측이다. global swap 증가 전부의 소유 프로세스를 특정한 것은 아니다. MemAvailable의 대부분이 회수 가능한 file cache일 수 있으므로, 그 숫자만으로 모델 A/B가 안전하다고 판단하지 않는다. 운영 서비스·다른 앱·swap 설정·OS cache는 바꾸지 않았다.

| 모델 검증 | 상태 |
|---|---|
| 9.7K warm fresh J off/on 처리량·token ID | 자원 차단, 미측정 |
| 19.8K / 119K 같은 A/B | 선행 관문 차단으로 미실행 |
| 운영 pinned ELF → 연구 ELF 출력 bridge | 미완료 |
| 실제 모델 라우팅 분포·J hit 비율 | 미측정 |
| warm GPU 계산 대 PCIe 병목 | 미판정 |

연산 성공을 모델 성공으로 승격하지 않는다. 구체적인 재개 조건과 그대로 쓸 명령은 [보고서 §8](results/strata_v07/STRATA_PREFILL_V07_KO.md)에 있다. 운영 서비스가 이미 꺼져 있고 자원 조건이 달라졌을 때만 같은 보호 조건 아래에서 다시 측정한다. 모든 연구 엔진·vision child는 종료됐고, 서비스는 시작·중지·재시작하지 않았다.

**v1.0 재개 판단 (GPT Pro, 2026-11-08).** 원래 관문(원 프로세스 기준 추가 swap-out 256 MiB, 1초 간격 연속 증가 두 표본)을 그대로 두고 CPU 사전점검만 갱신했다. 11:47:58 UTC에 MemAvailable 35.91 GiB, swap 사용 15.50 GiB, expert cache 0%, KFD client 0, 서비스 inactive였다. 이전에 cache 92.78%·약 44.23 GiB에서도 관문 중단이 났으므로 재개 근거가 없다고 보고 **서버를 띄우지 않았다(미판정)**. 서비스 조작, 관문 완화, 대형 SHA 읽기로 cache를 미리 채우는 일은 하지 않았다. 작은 운영 파일 41개의 SHA는 전후가 같았다. Claude의 2026-11-07 warm 병목 측정([검수 v0.7 §2.3](../REVIEW_V07_CLAUDE_KO.md), GPU 계산 병목·PCIe 대기 0.6%)은 별도의 8,192 MiB swap 관문 아래 결과다. 이 256 MiB 조건의 판정으로 옮기지 않고 조건을 붙여 인용한다. 정본: [STRATA_WARM_KO.md](results/v10/STRATA_WARM_KO.md), [strata_preflight_02.json](results/v10/strata_preflight_02.json).

## 9.19 v0.7 보존·배포·재현 범위

운영 파일 31+48개 전후 비교의 changed/missing/added는 0이다. [31개 사후 감사](results/v07_operating_files/after.json)의 SHA는 `49aa39680f78ea12b5c42a8d5d0955b8281a6e7194f39473971cb232eae23b92`, [48개 사후 감사](results/strata_v07/operating_after.json)는 `443be60a4946b745a7d15bbfe5bc0af6b3f3680362166d5e7c6a124850cff64f`다. Strata 사후 감사는 모든 GPU job 종료 뒤 약 60.4 s 동안 CPU I/O로 수행했다. 대형 pack hash 작업을 GPU 시간 측정과 겹치지 않았다.

누락되었던 `kernels.hip`는 기존 v0.6 원본 바이트로 복구했다(SHA `5fcb2e02724764592adee84d95a96a5cc865b9807f5053d93dfe62a4a86dfa20`). FP8 두 config는 의미를 바꾸지 않고 v0.6 manifest의 직렬화 바이트와 일치시켰다: E4M3fn `4318d1a566cd62f386dba09c12a25d99dfc683fa1d7266a1349ab8ff120dc322`, E5M2 `1522a3f908146c32169121f8d8993a4131c0237cf576c6f20ef9b8ba9b3abf40`. [수정 이력](results/v07_package_sync/repairs.json)의 catalog SHA는 당시 단계의 snapshot이며 최종 파일 SHA는 새 manifest를 따른다.

현재 배포 파일의 기준은 [v0.7 manifest](../RDNA2_PACKAGE_V07_MANIFEST.json)와 SHA256SUMS (`SHA256SUMS`, 연구 폴더에만 있음)다. 기존 v0.6 manifest는 그 릴리스의 역사적 snapshot으로 보존하며 현재 편집된 문서에 대한 checksum으로 사용하지 않는다. 코드·설정·compact raw·실패 이력·ISA·블라인드 원응답과 패널은 묶고, 모델 가중치·대용량 QKV/출력 `.pt`·runtime cache·운영 설정에서 유래한 Strata config 복제본과 그 식별 정보를 담은 stage envelope은 PC의 해당 연구 폴더에 남긴다. 이것은 새 모델 다운로드나 환경 배포 묶음이 아니다.

Claude가 우선 확인할 **핵심 숫자 5개와 정확한 명령**은 [V07_REPRO_CORE_KO.md](results/V07_REPRO_CORE_KO.md)에 모았다. Strata 모델 세 길이는 자원 조건이 바뀌기 전까지 미완료로 남긴다. 설치 코드나 운영 설정을 원복할 것은 없으며 연구 후보는 context adapter를 종료하거나 명시적 후보 선택을 사용하지 않으면 원래 경로로 돌아간다. 새 모델에는 이 패키지의 실험 함수를 전역 monkey-patch로 자동 적용하지 않는다.

### 9.20 PyTorch TunableOp와 RDNA2 FlashAttention Triton (v0.8, Claude 후속 실측)

전체 숫자와 원자료 경로는 [딥리서치 통합 §5–6](../DEEP_RESEARCH_MERGED_20261107_KO.md)에 있다. 여기에는 새 모델에 적용할 때 필요한 판단만 남긴다.

**TunableOp.** `PYTORCH_TUNABLEOP_ENABLED=1`, `PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0`(gfx1030은 hipBLASLt 미지원), `PYTORCH_TUNABLEOP_FILENAME=<연구 폴더>/tunableop_results.csv`로 켠다. 실제로 쓰이는 파일명은 장치 번호가 붙은 `tunableop_results0.csv`다. 처음 보는 shape는 `PYTORCH_TUNABLEOP_TUNING=1`일 때 그 자리에서 튜닝하고, 이후에는 `TUNING=0`으로 파일만 읽어도 된다.

| 모델 | 무거운 GEMM의 backend | 결과 |
|---|---|---|
| Anima FP16 | rocBLAS(SDPA math의 bmm, FP16 Linear) | SDPA math 81.0 → 60.85 s(1.33배), 직접 VT와 함께 45.18 s(1.79배). 파일 재사용 새 프로세스에서 60.87 / 45.20 s. 지각 통과 |
| Qwen-Image-2.1 INT8 | patientx INT8 Triton, FP32 split attention | 1.01–1.02배. 기각 |
| Idol 3DGS DiT | Sage attention 위 FP16 Linear | 193.0 → 191.2 s(1.01배), 튜닝 877 s. 기각 |
| MOSS INT8 | MOSS INT8 Triton 융합 | 0.99–1.02배. 첫 튜닝에서 GPU 메모리 접근 오류(14.8 GB GPU core dump), `PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE=0`으로 재시도 시 완료(1회). 기각 |

- 적용 판단: 프로파일에서 **rocBLAS GEMM이 상위**일 때만 시도한다. 무거운 계산이 Triton·HIP 커스텀 커널이면 효과가 없다.
- 재현성: Anima는 같은 프로세스 안에서 바이트 일치지만, 같은 튜닝 파일이라도 프로세스가 바뀌면 같은 seed의 출력이 달라진다(RGB RMSE 4–7/255). Qwen·MOSS는 같은 프로세스 안에서도 실행마다 달라진다. 원인은 미확인이다.
- 안전: 튜닝은 연구 폴더에서 한 번 하고, `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1`을 함께 둔다. 운영에는 검증한 튜닝 파일을 `TUNING=0`으로만 쓴다. ROCm·PyTorch 버전이나 해상도가 바뀌면 다시 튜닝한다.

**FlashAttention Triton.** ROCm/flash-attention `v2.8.0-cktile`에 Aceeton RDNA2 패치(gfx103x prefill 블록을 BLOCK_M 128·BLOCK_N 64·8 warps·2 stages로 고정)를 적용했다. pip 설치 없이 소스를 `sys.path`에 두고 `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`로 쓴다. 최초 Triton 컴파일은 prewarm 약 3 s다.

- 연산: D128·S4096·H24 FP16 43.7 → 23.7 ms(저자 23.8 ms 재현). FP64 대비 RMSE는 native와 같은 등급이다. FP32 입력은 지원하지 않는다.
- 모델: Anima self-attention은 모델 안에서도 33 ms로 빨라지지만, cross-attention(K=512)은 모델 안 native 3.8 ms보다 느린 5.45 ms다. 실제 생성은 단독 1.17배다.
- TunableOp와의 관계: TunableOp가 native SDPA math의 self-attention을 34 ms로 만들기 때문에 둘을 겹치면 이득이 없다(60.95 → 60.0 s). TunableOp를 쓸 수 없거나 쓰기 싫을 때의 대안이다.
- 교훈: 미세 벤치의 native 시간은 모델 안의 native 시간과 다를 수 있다(Anima self 64.6 ms 대 모델 안 47–59 ms, cross 9.3 ms 대 3.8 ms). 후보를 붙이기 전에 모델 안에서 호출별 cuda event 시간을 잰다(`anima_lab.py --time-attn 1`).

**긴 시퀀스 사례(Idol).** joint 12294 토큰·D64에서는 자체 Triton fused 커널(M32·N64) 86.3 ms에 비해 Aceeton Flash 69.2 ms, Sage 54.7 ms였다. 공식 50단계 전체는 288.0 → 230.5 s(Flash, 1.25배) / 193.0 s(Sage, 1.49배)이고, 8방향 미리보기 차이는 평균 0.4/255 수준이다. 토큰 수가 크고 기본 경로가 rocBLAS math가 아닐 때는 fused 커널의 블록 설정과 INT8 QK가 실제 이득으로 이어진다. 자세한 표는 [딥리서치 통합 §6.5](../DEEP_RESEARCH_MERGED_20261107_KO.md).

SageAttention Triton(INT8 QK)은 Anima에서 Flash보다 빠르지 않아 모델에 붙이지 않았다. Qwen FP32 joint shape에서는 연산 1.33배지만 근사이며, v0.5 Sage식 Qwen 실패 이력과 함께 지각 평가가 필요하다.

**Qwen 실제 생성 (v0.9, Claude, 2026-11-08).**
- 방법: Qwen의 FP32 attention(q·k·v)을 FP16으로 바꿔 Aceeton Sage(INT8 QK, FP16 PV)로 계산하고 FP32로 되돌렸다. 960×640 편집, 25단계, 같은 프로세스에서 A/B했다.
  - 하네스: `rehearsal_qwen_20261106/scripts/rehearsal_tunableop.py --candidate sage`
  - 결과: `results/sage25/`(시드 3개), `results/sage25_timing/`(속도)
- 호출: 시드마다 832회를 Sage로 계산했고, mask가 있는 64회는 native로 돌렸다. nonfinite는 0이었다.
- 속도: 워밍업 뒤 native 56.03 s → Sage 53.18 s(1.05배)였다. **운영 기본값인 alpha 접기(§7.1, 같은 하네스 51.14 s, 바이트 일치)보다 느리다.**
- 오차: 최종 RGB는 native 대비 max 0.76–0.96, RMSE 0.015–0.023(0–1 척도)이었다.
- 판정: 속도에서 운영 경로에 지므로 **기각**했다. 지각 평가는 하지 않았다. 연산 1.33배가 모델에서 1.05배로 줄어드는 이유는 FP32↔FP16 변환과 attention 밖의 시간이다.

**비결정의 원인과 필터 (v1.0, GPT Pro, 2026-11-08).** 정본: [TUNABLEOP_DETERMINISM_KO.md](results/v10/TUNABLEOP_DETERMINISM_KO.md).
- 매핑: Qwen 29 GEMM 중 25개, Anima 57개 전부를 설치된 Tensile 커널에 대응시켰다. 이 설치에서 음수 rocBLAS solution ID는 오류가 아니라 `Tensile_index = -id - 11`이다.
- 선택된 GSU4(split-K 4)는 Qwen 3개·Anima 12개이고, 그중 FP32가 2개·5개다. Qwen entry16·Anima entry2의 FP32 커널 disassembly에 `buffer_atomic_cmpswap`이 있다. Half GSU4(Qwen entry19)는 atomic 없이 별도 PostGSU4 합산을 쓰고 32회 바이트 일치했다. **GSU4와 비결정을 동의어로 쓰지 않는다.**
- 고정 입력 probe: 해당 FP32 행은 같은 storage·솔루션으로 32회 모두 다른 결과를 냈고, off와 Default는 한 SHA였다. 새 프로세스를 합친 64회도 같았다. atomic 합산 순서 자체를 추적한 것은 아니다.
- 필터(해당 FP32 GSU4 행만 Default로 바꾼 연구 CSV):

  | 모델 | 재현성 | warm 시간 |
  |---|---|---|
  | Anima(5행) | 새 2프로세스의 conditioning·latent·RGB 일치, cand 6장이 한 SHA | 45.11 s, off 80.72 s 대비 1.789배 유지 |
  | Qwen(2행) | 같은 프로세스 25-step 3장 일치(원본은 3장 모두 달랐음) | 55.72 → 56.45 s(+1.32%), off 대비 1.015배 |

- Anima에서 이 5행은 CLIP(conditioning)에서만 불렸고 sampler hit는 0이었다. 프로세스마다 conditioning을 한 번 만들어 재사용하므로, "같은 프로세스 안에서는 같고 프로세스가 바뀌면 다르다"는 v0.8 관측이 이것으로 설명된다.
- 미측정: Qwen 필터의 새 프로세스 재현성, MOSS. 운영 CSV는 바꾸지 않았다. 지각 우열도 판정하지 않았다.

### 9.21 Applio RVC 학습: solver 선택·직접 Conv1d·Conv 외 최적화와 운영 경로 (v0.9)

v0.8 연구는 GPT Pro, 운영 롤아웃과 운영 경로 실측은 Claude가 했다. 전체 수치·실패 기록·재현 명령은 [Applio v0.8 보고서](results/applio_v08/APPLIO_V08_KO.md)와 [갱신안 원문](results/applio_v08/V08_CATALOG_GUIDE_PROPOSAL_KO.md)에 있다. 여기에는 새 모델에 옮길 판단만 남긴다.

#### 대상과 시간 분해 — A

- 대상: `voice_model_A`, KLM49 HFG pretrain, 40 kHz, batch 8, FP16 AMP, FP32 master weight·AdamW 상태, 31 step/epoch. 런타임은 PyTorch 2.11.0+rocm10.0.0, MIOpen 3.6.0 `8d1ae90e`, Triton 3.8.0(`.venv-rocm10`)이다.
- Linux trainer의 `cudnn.benchmark`는 실행 중 `False`였다. Windows에서만 True로 설정한다.
- step 40–44의 operator self GPU 2,065.8 ms 중 Conv backward 55.7%, forward 24.6%, 합 80.3%였다.
- kernel 기준(분모 2,057.5 ms, 위와 합산하지 않음)으로는 Winograd 두 kernel 26.8%, im2col+col2im 14.1%, AdamW 5.4%다.
- 교정값: register dot2 48.7 TFLOP/s, 512 MiB copy 436.7 GB/s. dense FLOP으로 잰 Winograd 활용률은 엄밀한 효율이 아니다. 예를 들어 C64/K7/d3 dW의 4 ms 대 0.06 ms 차이를 달성 가능한 66배로 읽지 않는다.

#### Solver 선택 — A

- 기준은 운영 FAST DB의 복사본이다. 빈 DB와 비교하지 않는다. 144 geometry × 3방향 = 432방향을 각각 FP32 참조와 비교해 모두 통과했다.
- 다시 고른 것은 6방향(1.075–1.530배)뿐이다. dX 1개, Conv1d·Conv2d·ConvTranspose1d dW 5개다. 대부분의 방향은 운영 DB의 선택이 이미 최선이었다.
- **dilated GEMM workspace 누락**: MIOpen `8d1ae90e`의 legacy GEMM workspace 질의가 dilation>1에서 0을 돌려줘서, NORMAL 탐색이 해당 solver를 조용히 탈락시켰다.
  - 연구용 preload(`APPLIO_RDNA2_FIX_FIND_WORKSPACE=1`, NORMAL seed 전용)로 72/72방향을 복구했다.
  - 복구 뒤 새로 채택된 solver는 0개다. 성과는 탐색 누락을 고친 것이고, 속도 이득으로 세지 않는다.
  - 교훈: NORMAL 결과를 "모든 후보 중 최선"으로 쓰기 전에 실패 경고(workspace 0)를 확인한다.
- 실제 solver ID 432/432를 로그 replay로 확인했다(Winograd 126방향, GEMM 306방향, 9종). 같은 kernel을 내는 서로 다른 solver가 있으므로 kernel 이름만으로 solver를 적지 않는다.

#### 직접 Conv1d — A

- `v08_direct_conv.py`는 Triton FP16 곱·FP32 누산이다.
  - fwd·dX는 im2col 행렬을 만들지 않는 implicit GEMM이다.
  - dW는 FP32 partial 32개와 고정 순서 reduction을 쓴다. FP16 partial과 atomic은 쓰지 않는다.
  - fwd는 native와 같은 "FP16 반올림 뒤 FP16 bias" 경계를 지킨다.
- 21 geometry × 3방향 × 3 tile = 189후보가 모두 오차 관문을 통과했다. 방향마다 가장 빠른 정확한 native보다 event 1.10배·wall 1.05배 이상 빠른 것만 골랐다.
- 최종 manifest는 21 shape·56방향이다.

  | 방향 | 개수 | event 배속 |
  |---|---:|---|
  | fwd | 21 | 1.70–12.07배 |
  | dX | 20 | 1.31–9.44배 |
  | dW | 15 | 1.37–9.66배 |

  대표 C64/T6400/K7/d3 dW는 3.98 → 0.68 ms다.
- native 유지: C128/K11의 dW·d1 dX, C192→384/K5의 dW. 직접 커널이 느렸다.
- **dtype 경계**: C192→384/K5 세 shape의 실제 모듈 입력은 FP32였다(FP16 microbench와 다름). native AMP·weight_norm·FP32 master gradient를 포함한 호출 경계 105/105 비교를 통과한 뒤에만 fwd·dX를 넣었다.
- 큰 이득은 dilated 소채널 Conv에서 나왔다. MIOpen이 배치마다 im2col과 Tensile GEMM을 반복하는 곳이다. 대표 shape의 native dW는 GEMM이 90%였으므로 손해를 im2col 복사만의 탓으로 돌리지 않는다.

#### Conv 외 — A

| 후보 | 결과 | 선택 |
|---|---|---|
| D AdamW single-tensor | 15.85 → 11.14 ms(1.42배), moment·step bit 동일 | 채택 |
| G AdamW single | 0.45배 | native 유지 |
| fused AdamW(G/D) | 2.5–3.0배지만 tensor별 update NRMSE 0.003–0.0037 > 0.002 | 제외 |
| segment gather | y·dX bit 동일, 같은 RNG | 채택 |
| 기록용 grad norm foreach | 5.19배, gradient 불변 | 채택 |
| epoch 사이 `empty_cache` 생략 | 구성 전체 비교에 포함 | 채택 |
| D weight_norm cache | 유효한 gradient 참조 확보 실패 | 제외 |
| `torch.compile(dec)` | AMP 경계 오류 복구 뒤에도 eager보다 이득 없음 | 제외 |

#### 100 epoch와 청취 — A

| 하네스 100 epoch | A(native) | A′(native 반복) | B(후보) |
|---|---:|---:|---:|
| 프로세스 전체 s | 1540.11 | 1544.65 | **1320.00** |
| steady epoch s | 14.98 | 14.98 | **12.69** |
| scale 감소 / nonfinite loss | 12 / 0 | 11 / 0 | 12 / 0 |
| 마지막 25% mel 중앙값 | 16.372 | 16.234 | 16.383 |

- B는 14.3–14.5% 짧았다. native 반복 차이는 4.5 s였다. 초기 G/D가 같고, 각각 3,100 step을 돌았다.
- 비정렬 파형 RMSE는 B 쪽이 A–A′ 반복의 1.6–1.9배였다. 그래서 수치만으로 같은 음질이라고 판정하지 않았다.
- **같은 네 문장을 A/B/C로 가린 익명 청취에서 사용자가 구분하지 못했다(2026-11-08). 지각 기준(§1.2)으로 B를 채택한다.**

#### 운영 경로는 하네스와 달랐다 — A (Claude)

운영 `python core.py train → rvc/train/train.py`의 native는 하네스 A(15 s)가 아니라 **약 22 s/epoch**였다. 9월의 100 epoch 약 38분과 맞는 숫자다.

**원인**
- 운영 경로에서는 train.py가 `__main__`이다. spawn으로 뜬 DataLoader 워커 4개가 train.py를 다시 import하고, 각자 GPU 큐 3개를 만든다.
- 학습 중 KFD 클라이언트 6개, 큐 19개였다. 실제로 계산하는 것은 학습 프로세스 하나다.
- 하네스에서는 워커가 하네스 스크립트를 import하므로 이 일이 없었다.

**확인**
- 같은 kernel, 같은 호출 수인데 kernel 시간이 1.5–2배 길었다. 5 step GPU 합은 3.06 s 대 2.07 s다.
- 학습 중 sclk 2.6 GHz, mclk 1000 MHz, busy 99%였고, 짧은 교정값(dot2 48.4 TFLOP/s, copy 436.7 GB/s)도 같았다.
- zluda 검사를 끈 대조에서는 큐도 시간도 변하지 않았다.

**수정**
- 학습 프로세스가 GPU 컨텍스트를 먼저 만든 뒤 `HIP_VISIBLE_DEVICES=-1`로 바꿔, 이후에 뜨는 워커가 GPU를 보지 못하게 한다.
- 워커는 데이터만 읽고, pin_memory는 학습 프로세스에서 일어나므로 계산은 바뀌지 않는다.

**결과** (실제 `core.py train` 경로, 새 모델 이름, 4–6 epoch)

| 설정 | steady epoch | 학습 중 큐 |
|---|---:|---:|
| 변경 전 운영 | 약 22 s | 19 |
| 워커 GPU 숨김 | 약 15.0 s | 7 |
| 워커 GPU 숨김 + v0.8 B | 약 12.6 s | 7 |

100 epoch 추정은 약 37분 → 약 21–22분이다. 100 epoch 전체 재실행은 하지 않았다.

**일반화**
- 학습 스크립트를 `__main__`으로 실행하고 spawn 워커를 쓰는 모든 PyTorch ROCm 학습에 해당할 수 있다. 학습 중 `/sys/class/kfd/kfd/proc/*/queues`를 세어 확인한다.
- 하네스가 진입점을 바꾸면 운영 경로의 native를 따로 측정한다.
- 다른 랩 확인(2026-11-08):
  - 정적 확인: Idol MCP 서버·워커와 Qwen Studio 엔진은 torch를 import하지 않고 GPU 프로세스를 subprocess로 하나만 띄운다.
  - 실측: Qwen ComfyUI 생성 중 KFD 클라이언트는 1개, 큐는 7개였다.
  - MOSS는 같은 ComfyUI 단일 프로세스 구조라는 정적 확인만 했다.

#### 적용·원복

- 운영 `applio-lab`의 `rvc/train/rdna2_v08/`에 넣었다. 연구본과 바이트가 같은 사본이고, manifest의 source SHA가 맞아야 설치된다.
- `lab_hooks.install_rdna2()`를 `train.py run()`의 `set_device` 뒤에서 부른다.
- `APPLIO_RDNA2_V08=1`(런처 기본값)이 B 구성과 워커 GPU 숨김을 켠다. torch `2.11.0+rocm10.0.0`이 아니면 "disabled"를 출력하고 native로 학습한다.
- DB는 Applio 전용 `rdna2_v08/miopen_db`를 쓴다. 공용 `~/.config/miopen`은 바꾸지 않는다.
- `run-applio.sh`(GUI)는 `.venv`(rocm7.2)에서 `.venv-rocm10`으로 바꾸고 `MIOPEN_FIND_MODE=FAST`를 기본값으로 넣었다. GUI 실시간 탭에 필요한 `libportaudio2`를 설치했다.
- 직접 Conv의 shape는 batch 8·40 kHz·HiFi-GAN·`APPLIO_DILATION_FOLD` 미설정에 맞춰져 있다. 다른 설정은 대부분 native로 돈다. epoch마다 `[rdna2_v08] direct_conv_shapes=N/21`로 확인할 수 있다.
- 끄기는 `APPLIO_RDNA2_V08=0`이다. 워커 숨김만 따로 제어하려면 `APPLIO_WORKER_NO_GPU=0/1`을 쓴다. 파일 원복은 `applio-lab/backups/20261108_pre_v08/ROLLBACK.md`에 있다.
- 미검증: GUI 버튼으로 시작한 학습, 다른 batch·샘플레이트·vocoder, rocm7.2에서의 워커 숨김 효과.
- 그 위에 얹는 v1.0 B′(융합·남은 Conv 교체)는 §9.22에 있다.

### 9.22 Applio v1.0: ResBlock 융합·남은 Conv 교체와 운영 경로 100 epoch (v1.0, 청취에서 보류)

연구는 GPT Pro, 원자료 재현 확인과 운영 사본 준비는 Claude가 했다(2026-11-08). 전체 수치·진단 실패·재현 명령은 [Applio v1.0 보고서](results/v10/APPLIO_V10_KO.md)와 [갱신안 원문](results/v10/V10_CATALOG_GUIDE_PROPOSAL_KO.md)에 있다. **B′는 익명 청취에서 기존 B보다 못하다는 판정을 받아 보류했다(2026-11-09).** 속도·연산 결과는 기록으로 남긴다.

#### 대상과 기준선 — A

- 기준 B는 운영 중인 v0.8 구성이다(§9.21: 워커 GPU 숨김, solver DB, 직접 Conv1d 21 shape, Conv 외 최적화). B′는 그 위에 아래 두 겹을 더한 것이다.
- 대상과 런타임은 §9.21과 같다. `voice_model_A`, batch 8, FP16 AMP, torch 2.11.0+rocm10.0.0.
- 실행 경로: 연구는 운영 `core.py train → spawn → train.py`를 그대로 썼다. 새 모델 이름과 연구 import hook으로 그 프로세스에만 B′를 연결했다.
- B의 운영 프로파일(step 40–44): kernel 합 1,679.453 ms 중 native Conv2d 41.71%, native Conv1d 29.40%, 기존 직접 Conv 9.52%, optimizer 4.58%였다. §9.21의 "Conv backward 55.7%"는 v0.8 이전 하네스 프로파일이므로 지금 병목 표로 다시 쓰지 않는다.
- spawn/runpy에서는 `run.__globals__`와 모듈 dictionary가 다를 수 있다. 연구 연결은 실제 `run.__globals__`에 적용했다. config/filelist만 복사한 새 모델은 metadata가 없으면 109 speaker/contentvec로 바뀌므로 원래 `model_info.json`을 읽게 했다.

#### ResBlock 융합 — A

- `leaky_relu → Conv1 → leaky_relu → Conv2 → + skip`에서 LeakyReLU를 각 Conv의 입력 load에, skip 덧셈을 Conv2 store에 합쳤다. Conv 두 개는 각각 별도 kernel이다.
- backward:
  - Conv2 dX store에 두 번째 LeakyReLU mask를 적용한다.
  - Conv1 dX store에 첫 번째 mask와 skip gradient 합산을 적용한다.
  - dW는 preactivation에서 LeakyReLU를 다시 계산한다.
  - FP32 split partial과 고정 순서 reduction을 쓰고, 새 atomic은 없다.
- AMP 경계를 보존한다. FP16 반올림 뒤 bias 덧셈 순서, weight_norm graph, FP32 master gradient가 B와 같다.
- 검증: 18쌍, FP32 참조 108/108, 실제 ResBlock AMP·weight_norm 계약 144/144를 통과했다. 최악 NRMSE는 0.000439다. 검증한 고정 입력의 호출 경계 y/dX/gradient는 B와 바이트 일치했다(전체 학습의 바이트 동일을 뜻하지 않는다).
- 선택: event·wall 모두 1.03배 이상인 C32/C64 12쌍(event 1.039–1.135배). C64/K11 세 쌍은 관문 미달, C128/K11 세 쌍은 더 느렸다.
- 대표 C32/T12800/K3 쌍:
  - kernel 수 28 → 22, allocator peak 증가량 33.18 → 26.63 MB
  - 하드웨어 카운터 GL2C/EA 요청량 읽기+쓰기 185.77 → 116.17 MB(37.47% 감소)
  - 이 요청량은 물리 DRAM bytes가 아니다. arm당 1회 측정이라 반복 분산은 없고, 12쌍 전체에 이 비율을 적용하지 않는다.
- rocprofv3 `FETCH_SIZE`·`WRITE_SIZE`는 gfx1030에서 실제 값까지 수집됐다. 처음에는 aqlprofile의 버전 없는 라이브러리 이름을 찾지 못해 실패했고, 연구 폴더 symlink로 같은 wheel의 `.so.1`을 가리켜 복구했다.

#### 남은 native Conv의 방향별 교체 — A

- 60 shape·169방향을 조사했다. 후보 531/531이 FP32 참조를 통과했고, 39개 입력 dtype 변형의 AMP 계약 195/195가 통과했다.
- 선택 기준: 가장 빠른 정확한 native 대비 event ≥ 1.10배이고 wall ≥ 1.05배. 31 shape에서 새 68방향을 골랐다.

  | 종류 | 새 방향 |
  |---|---:|
  | discriminator period Conv2d | 59 |
  | ConvTranspose1d | 7 |
  | Conv1d (C192→384/K5 T220·T298 fwd) | 2 |

- 큰 이득은 작은 batch별 GEMM에서 나왔다. 대표 dW에서 native는 batch별 GEMM 8회·im2col 8회·초기화 1회로 17 kernel이고 출력 tile이 하나뿐이었다. 신규는 2 kernel이고 256 workgroup이 reduction을 나눠 맡는다.
  - period3 C1→32/K5/s3 dW: 0.907 → 0.033 ms
  - ConvTranspose C64→32/K4/s2 dW: 2.399 → 0.110 ms
  - micro 배속을 학습 배속으로 확대하지 않는다.
- native 유지: C128/K11 d1 dX·dW, C192→384/K5 dW(최선 1.095배로 관문 미달), ConvTranspose forward 네 shape, period Conv2d 85방향(큰 채널 Winograd 등). 이번 후보가 못 이겼다는 뜻이고, 하드웨어 최적이라는 증명이 아니다.
- 융합 범위와 native 교체 범위는 겹치지 않는다. B′ 실행에서 융합 12쌍·native 31 shape 모두 실제 hit가 있었다. 이때 v0.8 직접 Conv hit는 21 → 11 shape로 줄어드는데, 이것이 정상이다.

#### 운영 경로 측정 — A

- 사전 관문: 8 epoch씩 B/B′/B/B′ 교대, epoch 3–8 중앙값 12.542 → 11.448 s(1.0956배). 인접 두 쌍 1.0949·1.0944배로 1.02배 관문을 통과했다.
- 100 epoch(각 1회, 3,100 step, 같은 초기 G/D SHA, nonfinite loss 0, KFD 큐 최대 7):

  | 운영 경로 100 epoch | B (v0.8) | B′ (v1.0) | 차이 |
  |---|---:|---:|---:|
  | 프로세스 전체 | 1,285.25 s (21.42분) | 1,172.23 s (19.54분) | −113.02 s, −8.79% |
  | epoch 시간 합 | 1,261.70 s | 1,153.37 s | −108.34 s, −8.59% |
  | epoch 3–100 중앙값 | 12.566 s | 11.468 s | 1.0957배 |
  | AMP scale 감소 | 12 | 11 | |

- 5-step kernel 합은 1,679.5 → 1,530.3 ms(−8.88%)였다. 절감은 주로 period Conv2d(700.5 → 565.9 ms)와 ConvTranspose(29.6 → 12.4 ms)에서 나왔다. 융합한 Conv1d 계열 합은 653.7 → 656.5 ms로 거의 같다. standalone activation이 Conv 안으로 들어갔기 때문에 activation 감소를 그대로 융합의 순이익으로 세지 않는다. 두 겹을 따로 뺀 epoch 실험은 하지 않았다.
- Claude 재현(2026-11-08, GPU 미사용): 보고서 §11의 다섯 값과 assert를 그대로 실행해 통과했다. 보호 검사 1,022개 변경 0, 최종 KFD client 0, 청취 HTML 크기·SHA(4,318,552 B, `d11c4807…`)도 일치했다. 운영 Applio 파일의 수정 시각은 v0.8 롤아웃 시점(17:13–17:44)에서 바뀌지 않았다.

#### 청취 — 보류 판정 (C)

- 같은 네 문장, 100 epoch B·B′ 모델, native FP32 추론으로 익명 WAV 8개를 만들었다: LISTEN_BLIND.html (`results/v10/audio_eval/LISTEN_BLIND.html`, 연구 폴더에만 있음). `valid_for_listening=true`.
- 사용자의 판정 파일(human_judgment_20261109.json (`results/v10/audio_eval/human_judgment_20261109.json`, 연구 폴더에만 있음), run_id `20261108T110508Z_5ca14adf`, SHA `016b3da1…`)을 보존한 뒤 대응표를 열었다. A = B′, B = B(v0.8)였다.

  | 문장 | 선호 | 이상음 | 메모 |
  |---|---|---|---|
  | 1 | B(v0.8) | A(B′) | A가 아티팩트가 더 많은 느낌 |
  | 2 | B(v0.8) | A(B′) | 같음 |
  | 3 | 같음 | 미정 | |
  | 4 | B(v0.8) | A(B′) | A가 조금 더 많은 느낌 |

- 판정: 구분 불가라는 채택 조건을 충족하지 못했고, 방향도 B′ 쪽이 나빴다. **B′는 보류하고 운영은 v0.8 B를 유지한다.**
- 한계:
  - 문장이 4개뿐이다.
  - 메모가 "느낌" 수준이다.
  - 대응표가 문장마다 섞이지 않고 전부 A = B′로 고정이었다. 그래서 순서·위치 편향을 배제하지 못한다.
  - 그래도 사용자가 들은 결과가 지각 기준(§1.2)이므로 수치 쪽으로 판정을 뒤집지 않는다.
- 원인은 확인하지 않았다. 융합과 native 교체 중 어느 쪽 탓인지, 아니면 100 epoch 1회 실행의 학습 변동인지는 분리되지 않는다(§9.21에서 native 반복 A/A′의 파형 차이도 작지 않았다). 다시 볼 때는 다음 순서로 한다.
  1. 문장마다 대응을 섞고 같은 쌍을 반복하는 재청취
  2. 두 겹을 나눈 B+융합 / B+native 100 epoch

#### RTX 5070과의 차이 — 조건부 모델, 분할 미식별

- 정본: [GAP_5070_KO.md](results/v10/GAP_5070_KO.md). 5070은 dense FP16 곱·FP32 누산 61.7 TFLOP/s, 672 GB/s(공표 peak)를 썼다. 이 PC는 실측 ceiling 48.7 TFLOP/s, 436.7 GB/s다. sparse·FP4·FP16 누산 TOPS는 쓰지 않는다.
- B′ kernel 시간의 53.4%를 덮는 모델 범위(100 epoch 환산):

  | 범위 | 분 |
  |---|---:|
  | 현재 GPU kernel 시간 | 8.44 |
  | RDNA2 실측 ceiling 조건부 하한 | 1.50 |
  | 5070 공표 peak 조건부 하한 | 1.15 |
  | 두 하한의 차이 | 0.35 |
  | 현재 구현과 RDNA2 ceiling 사이 | 6.94 |

  - 이 하한은 같은 알고리즘, 최소 traffic, 상한 도달을 가정한 값이다. Winograd에는 직접 Conv FLOP을 붙이지 않았다.
  - 6.94분을 전부 회수 가능한 소프트웨어 비용으로 읽지 않는다.
- 외부의 "5070 12분"은 데이터·step·batch·AMP·라이브러리·측정 범위가 같다는 증거가 없다. 그래서 하드웨어 필연분과 소프트웨어 회수분으로 나눌 수 없다(미식별). 이 PC에서 실제로 회수한 몫은 B→B′ 1.88분(프로세스 기준)이다.

#### 운영 사본 (준비됨, 보류로 꺼진 채 둠)

- `applio-lab/rvc/train/rdna2_v10/`: 연구본 4개 파일과 manifest 2개를 바이트 동일하게 복사했다. 각 manifest가 source SHA를 검사한다. `__init__.py`만 새로 썼다.
- `lab_hooks.install_rdna2()`가 v0.8 설치 뒤 `rdna2_v10.install()`을 부른다. `APPLIO_RDNA2_V10=1`이고 v0.8이 실제로 켜졌을 때만 동작한다. 런처는 아직 이 값을 켜지 않으므로 운영 학습은 v0.8 B 그대로다.
- 정적 검증(GPU 미사용)에서 다음을 확인했다.
  - 변수가 없으면 아무것도 하지 않는다.
  - V08이 없으면 "disabled"를 출력한다.
  - 두 context 진입 시 SHA 검사를 통과하고 패치 4개를 교체하며, 종료하면 원상 복구한다.
  - `APPLIO_DILATION_FOLD=1`이면 거부한다.
  - `.venv`(rocm7.2)에서도 import가 정상이다.
- 아직 안 한 것: 운영 사본으로 GPU 학습. 켜기로 하면 새 모델 이름으로 4–6 epoch를 돌려 다음을 확인한다.
  - `[rdna2_v10] enabled`
  - `fused_pairs=12/12 native_shapes=31/31`
  - 약 11.5 s/epoch
  - KFD 큐 7 이하
- 켜기·끄기·원복 절차는 `applio-lab/backups/20261108_pre_v10/ROLLBACK.md`에 있다.

## 10. 새 모델에 적용하기 전 체크리스트

- [ ] 기존 정상 환경에서 실제 요청과 native reference를 확보했다.
- [ ] 가장 비싼 연산의 shape·dtype·stride·호출 횟수와 copy/offload 비용을 확인했다.
- [ ] checkpoint·CPU mmap·GPU resident·실제 연산 입력 dtype을 분리했다. wrapper/container는 소비하지 않고 관측하며, hook의 실제 hit가 0이 아님을 확인했다.
- [ ] 모델이 실제 부르는 Attention callable/소스와 INT8 backend/커널을 확인했다. SDPA·split·patientx·MOSS 경로를 구분했다.
- [ ] 서버형 모델은 그 서버 프로세스 안에서 실제 sampler/decoder를 측정했고, 같은 요청 재실행이 캐시로 생략되지 않았다.
- [ ] Operator 귀속 시간과 GPU kernel/copy/memset 활동을 별도 표·분모로 집계했다. raw trace category로 GPU annotation도 제외했고, 같은 활동을 두 번 더하지 않았다.
- [ ] Strided view는 backend별로 copy 포함 contiguous 기준선과 비교했고, scale의 원래 위치와 반올림 순서를 확인했다.
- [ ] kernel 단독 개선이 전체 작업에서 차지할 비중을 판단했다.
- [ ] 기존 compiler/runtime을 유지하고, unsupported backend의 enable flag를 실행 증거로 사용하지 않았다.
- [ ] 후보에 필요한 입력 layout과 affine/quantization/scale/activation의 계산 순서를 보존했다.
- [ ] INT8 backend가 이미 dequant를 융합하는지 확인했다. 타일, 융합, quantization 포함 시간과 메모리를 실제 shape에서 비교했다.
- [ ] FP16 입력/FP32 출력 GEMM을 쓰면 실제 선택 kernel과 시간을 별도 확인했다. 출력 dtype이나 enable flag만으로 빠른 경로를 가정하지 않았다.
- [ ] BF16→FP16 후보는 exact 재표현·입력 범위·GEMM 출력 finite를 따로 검사했고, 필요한 검사·변환 비용을 실제 시간에 포함했다.
- [ ] INT4는 W4A8/W4A4/W4A16 계약을 분리하고, dot8의 activation nibble 분해·보정·그룹 scale·packing·준비 비용을 포함했다. 같은 저장값 계산 오차와 원본의 weight/activation 표현 손실을 별도 판정했다.
- [ ] FP8는 raw/scale/FNUZ 계약을 확인했다. 같은 FP8 저장값의 계산 A/B와 원본→FP8 표현 변경의 손실을 별도 판정했다.
- [ ] prepared layout/weight는 1회 준비·재사용 횟수·추가 상주 memory·무효화 조건을 기록했다. PyTorch allocated peak를 external allocator까지 포함한 GPU 총량으로 부르지 않았다.
- [ ] 실제 선택 경로를 확인했다. `auto` fallback을 새 kernel의 성공으로 세지 않았다.
- [ ] 작은 독립 reference와 실제 큰 입력을 모두 검사했다. mask/꼬리 길이/특수값은 해당 연산 계약에 필요한 범위에서 확인했다.
- [ ] D128 tiled 후보의 register/spill/private memory와 layout 복원 비용을 확인했다. spill 감소·O(S) 메모리·속도 개선을 각각 판정했다.
- [ ] 허용 오차와 출력 기준을 먼저 정했고, 결과를 본 뒤 완화하지 않았다.
- [ ] NORMAL 또는 FP32 Conv 참조에는 `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0`과 `MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53=0`을 함께 지정했다. OCL 이름을 대신 쓰지 않았다.
- [ ] Conv solver 선택은 forward/dX/dW를 각각 검증했다. FindDb와 PerfDb, FAST/NORMAL과 benchmark flag를 구분하고 운영 DB와 분리된 연구 DB로 재현했다.
- [ ] 정규화 후보는 기존 CK/compiled 융합을 실제 기준선으로 삼고 reduction·epsilon·scale+1·bias·출력 cast의 반올림 경계를 확인했다.
- [ ] 같은 입력·conditioning·seed에서 작업의 최종 출력까지 비교했다.
- [ ] 학습 A/B는 같은 초기 state·데이터·optimizer·AMP 설정과 native 반복을 확보하고, warm epoch·정상 epoch·loss·skip·최종 출력의 변동을 따로 판정했다.
- [ ] INT8 ConvRot 모델에서 근사를 넣었다면 FP32 경계 dtype뿐 아니라 양자화 scale·코드 변화와 최종 RGB를 확인했다. 작은 operator 오차나 표본 row 통과로 승격하지 않았다.
- [ ] Sparse는 실제 step/layer별 확률 질량과 mask 비용을 측정했고, 엄격 RGB와 익명·역순·crop·여러 seed의 독립 지각 판정을 분리해 원응답을 보존했다.
- [ ] 서버 prewarm은 실제 요청 shape로 실행하고 시간·횟수를 기록했다. 첫 요청과 warm 수치를 섞지 않았다.
- [ ] TunableOp는 rocBLAS GEMM이 상위일 때만 시도했고, 튜닝은 연구 폴더에서 `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1`과 함께 했다. 같은 프로세스 안 반복과 새 프로세스 재사용의 출력 재현성을 따로 기록했다.
- [ ] Attention 후보는 미세 벤치 shape가 아니라 모델 안 호출별 시간(self/cross 분리)으로 이득을 예상했다.
- [ ] 학습은 운영 진입점 그대로 측정했다. 하네스가 진입점을 바꾸면 운영 경로의 native를 따로 쟀고, 학습 중 KFD 클라이언트·큐 수(`/sys/class/kfd/kfd/proc/*/queues`)를 확인해 계산하지 않는 프로세스(spawn 워커 등)가 GPU 큐를 잡지 않게 했다.
- [ ] MIOpen NORMAL 탐색 결과를 최선으로 쓰기 전에 workspace 0 같은 후보 탈락 경고를 확인했다.
- [ ] 학습 모델을 새 이름으로 복제했다면 원본 metadata(speaker 수·embedder)와 초기 G/D SHA를 맞췄고, spawn 함수가 실제로 참조하는 globals에 연구 연결이 들어갔는지 확인했다.
- [ ] Conv+activation+residual 융합은 forward뿐 아니라 부호 mask·skip gradient·bias reduction dtype·FP32 master gradient까지 검증했다.
- [ ] 직접 Conv와 융합을 함께 넣을 때 선택 범위가 겹치지 않는지 확인했고, 각 context의 실제 hit를 epoch마다 기록했다.
- [ ] backward kernel 귀속은 공유 External ID만으로 정하지 않고 runtime launch correlation으로 대조했다.
- [ ] roofline에는 FLOP 정의·누산 dtype·실측 ceiling과 공표 peak·논리 bytes와 실제 traffic(GL2C/EA 요청량 ≠ DRAM bytes)·Winograd 제외 범위를 적었다.
- [ ] TunableOp 비결정은 GSU>1 전체가 아니라 실제 커널의 atomic 여부로 좁혔다.
- [ ] 연산 검증·운영 epoch·100 epoch 기술 완료·사용자 청취·운영 적용을 서로 다른 상태로 기록했다.
- [ ] Strata 프리필은 실제 stage 시간·copy bytes·host stall을 구분했고, 서비스가 off인 상태에서 별도 worktree/build/port로만 모델 A/B했다. 운영 파일 전후 SHA와 fresh prompt·token ID를 비교했다.
- [ ] warmup 후 양쪽 순서를 교대해 wall/event 시간과 반복 분포를 남겼다. 느린 sample을 숨기지 않았다.
- [ ] GPU 작업을 직렬 실행하고 겹친 측정을 폐기했다. 감사의 관측 범위와 한계도 결과에 남겼다.
- [ ] offload·LoRA·cast 함수와 임시 weight 수명, module/state_dict 경로를 보존했다.
- [ ] low-VRAM/tiled를 운영에서 쓰면 실제 그 분기가 실행됐고, 같은 경로끼리 비교했음을 확인했다.
- [ ] compile/graph를 쓰면 eager 이득과 따로 검증하고, 현재 허용된 fallback 정책을 지켰다.
- [ ] 빠르고 정확한 범위만 적용했다. 조건 밖 경로와 즉시 가능한 롤백을 남겼다.
- [ ] 최종 기록에 A/B/C 근거, 적용 범위, 실패·미검증 사항을 구분했다.
- [ ] `torch.equal` 값 일치와 dtype/shape/원본 bytes 일치를 구분했다. 입구 카탈로그·가이드 근거 절·results 원자료를 함께 갱신했다.

새 shape나 새 모델은 이 체크리스트에 필요한 차이만 다시 검증한다. 이미 확인한 환경을 재구축하거나 같은 진단을 대량 반복하는 대신, 실제 병목과 남은 불확실성을 해결하는 데 시간을 쓴다.
