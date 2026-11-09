# rocBLAS FP16/FP32 mm·bmm 레이아웃 실측 — v0.6

2026-11-06 UTC(한국 시각 11-07), RX 6900 XT/gfx1030. 카탈로그 §7의 기준 커널→후보→연산 단위→기존 Anima 모델 A/B를 연결한 기록이다. FP16 P·V의 NT 저장은 연산·모델 속도를 개선했지만 엄격 RGB 기준은 실패했다. 다른 shape의 모델 출력 성공으로 확장하지 않는다.

## 1. 기준선과 계약

- 원자료: [layout_blas_v06.json](layout_blas_v06.json), SHA-256 `b9d1f28118dbc4016a69ccaf2d106b0b4f8d02a6a94607a4ad1924b01c9a600c`. 기존 MOSS Python, Torch 2.12.0+rocm10.2.0a20261011, HIP 7.17.26405.
- 논리 계산은 항상 C=A@B다. NN/NT/TN은 같은 값을 담은 PyTorch tensor의 row/column-major 저장 형태다. PyTorch 내부의 rocBLAS transA/transB flag 문자열과 동일시하지 않는다.
- NN: A/B row-major. NT: B를 `B.T.contiguous().T`로 저장. TN: A를 같은 방법으로 column-major 저장. STRIDED: bmm B의 batch/head-interleaved stride `[N, batch*N, 1]`, mm B의 column stride 2.
- bmm STRIDED는 Qwen/MOSS의 batch=1에서 heads를 batch로 둔 계열의 한 예다. Anima B2/H16을 flatten하기 전의 원래 stride를 그대로 재현했다고 해석하지 않는다. Qwen 리허설의 Q/K/V 전체 contiguous 제거와도 적용 범위가 다르다.
- Anima/Qwen/MOSS에서 관측한 PV shape 3개와 tall-skinny·projection mm 3개, 각각 FP16/FP32를 측정했다. 값은 seed 고정 합성 데이터다. `std=0.125`이며 P를 실제 softmax 확률로 가정한 입력은 아니다.
- 각 조합 3 warmup 뒤 3개 교대 round×3회. prepared와 copy-inclusive 순서도 round마다 바꿨다. prepared는 변환된 입력이 이미 있을 때이고, copy-inclusive는 NN 원본에서 변환·할당·GEMM까지 포함한다.
- 준비된 layout은 하나씩 만들고 폐기했다. GPU KFD audit 340회, foreign client 0, 감시 오류 0. 관측 범위에서 겹친 compute job은 없었다. 그래픽 부하와 sampling 사이의 아주 짧은 KFD client까지 탐지하는 장치는 아니다.
- 모든 48개 조합은 입력값 동일성, finite, 전체 native NN tolerance 및 표본 CPU FP64 tolerance를 통과했다. prepared/copy-inclusive 비트 동등성은 47개 통과했고 1개는 실패했다(§5). 전체 RGB 기준은 별도다.
- 사전 연산 tolerance는 FP16 `rtol=5e-3, atol=5e-3`, FP32 `rtol=3e-4, atol=3e-5`다. CPU FP64는 mm의 4행×4열, bmm의 두 batch 각각 4행×4열을 독립 계산한 표본이며 전체 native 비교를 대체하지 않는다. `prepared_peak_increment_bytes`, `copy_inclusive_peak_increment_bytes`, 별도 준비 storage byte와 full-output max/RMSE를 조합별 원자료에 저장했다.

## 2. 시간 표

단위 ms, 각 셀은 **prepared / copy-inclusive** wall 중앙값이다. 정확한 input stride, event 시간, 모든 round, 메모리, 전체 kernel 이름은 원자료에 있다.

| 연산/shape (batch,M,N,K) | dtype | NN | NT | TN | STRIDED |
|---|---|---:|---:|---:|---:|
| (32,4096,128,4096) | float16 | 22.632 / 27.654 | 6.570 / 7.091 | 5.391 / 266.912 | 10.890 / 13.589 |
| (32,4096,128,4096) | float32 | 10.145 / 10.142 | 10.409 / 11.758 | 6.387 / 300.626 | 10.442 / 11.173 |
| (32,2400,128,4964) | float16 | 2.979 / 3.034 | 3.064 / 3.896 | 2.669 / 16.757 | 2.730 / 2.735 |
| (32,2400,128,4964) | float32 | 5.952 / 5.996 | 5.782 / 6.986 | 5.533 / 17.343 | 5.961 / 5.872 |
| (24,1061,128,1061) | float16 | 0.218 / 0.218 | 0.224 / 0.336 | 0.211 / 0.824 | 0.218 / 0.244 |
| (24,1061,128,1061) | float32 | 0.372 / 0.366 | 0.377 / 0.536 | 0.343 / 1.000 | 0.366 / 0.415 |
| (mm,4096,64,4096) | float16 | 0.172 / 0.175 | 0.309 / 0.318 | 0.124 / 8.396 | 0.172 / 0.182 |
| (mm,4096,64,4096) | float32 | 0.272 / 0.275 | 0.505 / 0.526 | 0.225 / 9.437 | 0.288 / 0.294 |
| (mm,4096,128,8192) | float16 | 0.389 / 0.389 | 0.593 / 0.630 | 0.282 / 18.111 | 0.401 / 0.412 |
| (mm,4096,128,8192) | float32 | 0.800 / 0.795 | 0.704 / 0.771 | 0.449 / 22.673 | 0.924 / 0.956 |
| (mm,1061,3072,3072) | float16 | 0.612 / 0.608 | 0.695 / 2.658 | 0.563 / 0.836 | 0.656 / 0.709 |
| (mm,1061,3072,3072) | float32 | 1.019 / 1.012 | 1.086 / 4.674 | 0.972 / 1.363 | 1.149 / 1.368 |

Anima FP16 NN prepared는 20.654–23.066 ms, copy-inclusive는 23.022–30.711 ms로 흔들렸다. 같은 논리 경로인데도 측정 시점에 따라 차이가 있어 하나의 정확한 배속만 일반화하지 않는다. NT copy-inclusive는 7.069–7.101 ms로 이 표에서 큰 우위가 유지됐다. Qwen FP32 NT의 copy-inclusive 6.986 ms는 NN 5.996 ms보다 느렸다.

## 3. 실제 선택 kernel과 시간

48개 선택을 아래에 남긴다. Kxx는 §6의 **정확한 전체 kernel 이름**을 뜻한다. `+pack`은 prepared 호출 내부에도 ATen direct-copy가 들어간 경우다.

| case | dtype | layout | prepared ms | copy 포함 ms | kernel | NN 비트 일치 |
|---|---|---|---:|---:|---|---|
| anima_pv_self | float16 | NN | 22.632 | 27.654 | K01 | 예 |
| anima_pv_self | float16 | NT | 6.570 | 7.091 | K02 | 아니오 |
| anima_pv_self | float16 | TN | 5.391 | 266.912 | K03 | 아니오 |
| anima_pv_self | float16 | STRIDED | 10.890 | 13.589 | K01 | 예 |
| anima_pv_self | float32 | NN | 10.145 | 10.142 | K04 | 예 |
| anima_pv_self | float32 | NT | 10.409 | 11.758 | K05 | 아니오 |
| anima_pv_self | float32 | TN | 6.387 | 300.626 | K06 | 아니오 |
| anima_pv_self | float32 | STRIDED | 10.442 | 11.173 | K04 | 예 |
| qwen_pv_prefix | float16 | NN | 2.979 | 3.034 | K07 | 예 |
| qwen_pv_prefix | float16 | NT | 3.064 | 3.896 | K02 | 아니오 |
| qwen_pv_prefix | float16 | TN | 2.669 | 16.757 | K08 | 예 |
| qwen_pv_prefix | float16 | STRIDED | 2.730 | 2.735 | K07 | 예 |
| qwen_pv_prefix | float32 | NN | 5.952 | 5.996 | K09 | 예 |
| qwen_pv_prefix | float32 | NT | 5.782 | 6.986 | K10 | 아니오 |
| qwen_pv_prefix | float32 | TN | 5.533 | 17.343 | K11 | 아니오 |
| qwen_pv_prefix | float32 | STRIDED | 5.961 | 5.872 | K09 | 예 |
| moss_pv | float16 | NN | 0.218 | 0.218 | K12 | 예 |
| moss_pv | float16 | NT | 0.224 | 0.336 | K02 | 예 |
| moss_pv | float16 | TN | 0.211 | 0.824 | K13 | 아니오 |
| moss_pv | float16 | STRIDED | 0.218 | 0.244 | K12 | 예 |
| moss_pv | float32 | NN | 0.372 | 0.366 | K14 | 예 |
| moss_pv | float32 | NT | 0.377 | 0.536 | K15 | 아니오 |
| moss_pv | float32 | TN | 0.343 | 1.000 | K16 | 아니오 |
| moss_pv | float32 | STRIDED | 0.366 | 0.415 | K14 | 예 |
| mm_n64_k4096 | float16 | NN | 0.172 | 0.175 | K17 | 예 |
| mm_n64_k4096 | float16 | NT | 0.309 | 0.318 | K02 | 아니오 |
| mm_n64_k4096 | float16 | TN | 0.124 | 8.396 | K03 | 아니오 |
| mm_n64_k4096 | float16 | STRIDED | 0.172 | 0.182 | K17 +pack | 예 |
| mm_n64_k4096 | float32 | NN | 0.272 | 0.275 | K18 | 예 |
| mm_n64_k4096 | float32 | NT | 0.505 | 0.526 | K05 | 아니오 |
| mm_n64_k4096 | float32 | TN | 0.225 | 9.437 | K19 | 아니오 |
| mm_n64_k4096 | float32 | STRIDED | 0.288 | 0.294 | K18 +pack | 예 |
| mm_n128_k8192 | float16 | NN | 0.389 | 0.389 | K20 | 예 |
| mm_n128_k8192 | float16 | NT | 0.593 | 0.630 | K21 | 아니오 |
| mm_n128_k8192 | float16 | TN | 0.282 | 18.111 | K22, K23 | 아니오 |
| mm_n128_k8192 | float16 | STRIDED | 0.401 | 0.412 | K20 +pack | 예 |
| mm_n128_k8192 | float32 | NN | 0.800 | 0.795 | K24 | 예 |
| mm_n128_k8192 | float32 | NT | 0.704 | 0.771 | K10 | 예 |
| mm_n128_k8192 | float32 | TN | 0.449 | 22.673 | K25, K26 | 아니오 |
| mm_n128_k8192 | float32 | STRIDED | 0.924 | 0.956 | K24 +pack | 예 |
| mm_projection | float16 | NN | 0.612 | 0.608 | K27 | 예 |
| mm_projection | float16 | NT | 0.695 | 2.658 | K28 | 아니오 |
| mm_projection | float16 | TN | 0.563 | 0.836 | K08 | 아니오 |
| mm_projection | float16 | STRIDED | 0.656 | 0.709 | K27 +pack | 예 |
| mm_projection | float32 | NN | 1.019 | 1.012 | K29 | 예 |
| mm_projection | float32 | NT | 1.086 | 4.674 | K10 | 아니오 |
| mm_projection | float32 | TN | 0.972 | 1.363 | K30 | 예 |
| mm_projection | float32 | STRIDED | 1.149 | 1.368 | K29 +pack | 예 |

## 4. 이 측정에서 사용할 수 있는 선택 규칙

| 확인한 범위 | 규칙 | 판정/근거 |
|---|---|---|
| FP16 bmm, batch32/M4096/N128/K4096, row-major A와 B | B를 column-major(NT)로 준비하는 것을 우선 후보로 쓴다. 작은 V 전치 비용까지 포함해 빠르다. | A, prepared 22.632→6.570 ms, copy 포함 27.654→7.091 ms. Anima A/B에서 엄격 RGB 실패(§8): 품질 변화를 허용하지 않으면 자동 적용 금지 |
| 같은 shape FP32 | NN을 유지한다. | A, NT는 prepared도 비슷하고 copy 포함 10.142→11.758 ms로 느림 |
| Qwen PV batch32/M2400/N128/K4964, FP32 | NT 전치 복사를 새로 넣지 않는다. | A, copy 포함 5.996→6.986 ms. B-only STRIDED의 약 2% 차이는 큰 개선 근거로 보지 않음 |
| MOSS PV batch24/M1061/N128/K1061, FP16/FP32 | NN 유지. 작은 prepared 차이를 위해 copy를 추가하지 않는다. | A, NT/TN/STRIDED의 copy 포함 시간이 모두 NN보다 큼 |
| mm M4096/N64/K4096, FP16/FP32 | NN 유지. | A, NT도 느리고 TN의 A 전치 비용이 GEMM보다 훨씬 큼 |
| mm M4096/N128/K8192, FP16 | NN 유지. | A, NT는 0.389→0.630 ms, TN은 prepared 0.282 ms여도 copy 포함 18.111 ms |
| mm M4096/N128/K8192, FP32 | B가 이미 column-major이면 NT를 비교 후보로 사용한다. 새 전치 포함 이득은 3% 수준으로 작다. | A, prepared 0.800→0.704 ms, copy 포함 0.795→0.771 ms, 이 합성 입력은 NN과 비트 일치. 해당 shape의 실제 모델 출력은 미검증 |
| mm M1061/N3072/K3072, FP16/FP32 | NN 유지. | A, NT/TN/STRIDED copy 포함 모두 퇴보 |
| 모든 TN 준비 경로 | 이미 A가 column-major인 경우에만 GEMM 단독 시간을 사용한다. 매번 A를 전치하는 방식은 거절한다. | A, 큰 score A 복사가 최대 300 ms대 비용을 만듦 |
| mm column-stride2 B | library가 내부 contiguous pack을 할 수 있으므로 준비 비용 0으로 가정하지 않는다. | A, 세 mm shape 모두 profile에 direct-copy kernel이 나타남 |

Anima FP16 NN은 `Ailk_Bljk / HHS_BH / MT128x128x16`, NT는 `Alik_Bljk / HHS_BH / MT128x256x16`을 실제 선택했다. 같은 NN shape에서 FP32는 다른 `SB / MT128x64x8` 커널이다. 여기서 FP16 NN이 FP32 NN보다 느리게 측정됐다는 사실도 dtype만으로 속도를 예상하면 안 되는 사례다.

## 5. 실패와 불확실성

- FP32 mm M4096/N128/K8192의 TN은 native tolerance와 CPU FP64 표본 모두 통과했다. 그러나 동일한 logical input/stride의 prepared 호출과 copy-inclusive 호출 사이에 max 9.536743e-7, RMSE 7.485021e-8의 차이가 있었다. 사전 byte-equivalence 검사는 실패했다. 수치 tolerance 실패라고 바꿔 쓰지 않는다.
- 그 경로의 실제 kernel 이름에는 `GSU4`가 있고 추가 `Cijk_S.kd`가 관측됐다. split reduction의 순서 차이는 가능한 설명이지만 동일 prepared tensor의 반복까지 분리해 원인을 확정하지 않았다. 새 copy-inclusive TN 자체가 크게 느려 자동 적용 후보에서 제외된다.
- Qwen FP16의 B-only STRIDED는 평균 1.11배 빨랐지만 round 분포가 겹쳤다. 실제 Qwen은 FP32이고 앞선 리허설은 Q/K/V 전체 strided 변형이므로 기존 0.49배 실패를 이 결과로 취소하지 않는다.
- 모든 속도는 warm 작은 반복의 연산 시간이다. 모델 전체, backward, 다른 shape/version으로 자동 확장하지 않는다.

## 6. Kernel 이름 사전

### K01

`Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_SN_AMAS3_BL1_BS1_EPS1_GLVWA8_GLVWB8_GRVW8_GSU1_GSUASB_ISA1030_IU2_K1_KLA_LDL2_LRVW8_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU0_SUM0_SUS0_SVW4_TT8_8_USFGROn1_VAW2_VSn1_VW8_VWB8_WS32_WG16_16_1_WGM8.kd`

### K02

`Cijk_Alik_Bljk_HHS_BH_MT128x256x16_SN_AMAS3_BL1_BS1_EPS0_GLVWA8_GLVWB8_GRVW8_GSU1_GSUASB_ISA1030_IU2_K1_KLA_LDL2_LRVW8_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU32_SUM3_SUS128_SVW4_TT8_16_USFGROn1_VAW2_VSn1_VW8_VWB8_WS32_WG16_16_1_WGM1.kd`

### K03

`Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_SN_AMAS3_BL1_BS1_EPS1_GLVWA2_GLVWB2_GRVW2_GSU1_GSUASB_ISA1030_IU2_K1_KLA_LDL2_LRVW2_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU32_SUM3_SUS128_SVW4_TT4_4_USFGROn1_VAW2_VSn1_VW2_VWB2_WS32_WG16_16_1_WGM1.kd`

### K04

`Cijk_Ailk_Bljk_SB_MT128x64x8_SN_AMAS3_BL1_BS1_EPS1_GLVWA4_GLVWB4_GRVW4_GSU1_GSUASB_ISA1030_IU1_K1_KLA_LDL1_LRVW4_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU32_SUM3_SUS128_SVW4_TT8_8_USFGROn1_VAW1_VSn1_VW4_VWB4_WS32_WG16_8_1_WGM8.kd`

### K05

`Cijk_Alik_Bljk_SB_MT128x256x16_SN_AMAS3_BL1_BS1_EPS0_GLVWA4_GLVWB4_GRVW4_GSU1_GSUASB_ISA1030_IU1_K1_KLA_LDL1_LRVW4_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU32_SUM3_SUS128_SVW4_TT8_16_USFGROn1_VAW1_VSn1_VW4_VWB4_WS32_WG16_16_1_WGM4.kd`

### K06

`Cijk_Ailk_Bjlk_SB_MT128x64x8_SN_AMAS3_BL1_BS1_EPS1_GLVWA4_GLVWB4_GRVW4_GSU1_GSUASB_ISA1030_IU1_K1_KLA_LDL1_LRVW4_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU0_SUM0_SUS0_SVW4_TT8_8_USFGROn1_VAW1_VSn1_VW4_VWB4_WS32_WG16_8_1_WGM1.kd`

### K07

`Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_SN_AMAS3_BL1_BS1_EPS0_GLVWA8_GLVWB8_GRVW8_GSU1_GSUASB_ISA1030_IU2_K1_KLA_LDL2_LRVW8_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU32_SUM3_SUS128_SVW4_TT8_16_USFGROn1_VAW2_VSn1_VW8_VWB8_WS32_WG16_8_1_WGM1.kd`

### K08

`Cijk_Ailk_Bjlk_HHS_BH_MT128x128x16_SN_AMAS3_BL1_BS1_EPS0_GLVWA8_GLVWB8_GRVW8_GSU1_GSUASB_ISA1030_IU2_K1_KLA_LDL2_LRVW8_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU32_SUM3_SUS128_SVW4_TT8_16_USFGROn1_VAW2_VSn1_VW8_VWB8_WS32_WG16_8_1_WGM4.kd`

### K09

`Cijk_Ailk_Bljk_SB_MT128x64x8_SN_AMAS3_BL1_BS1_EPS0_GLVWA4_GLVWB4_GRVW4_GSU1_GSUASB_ISA1030_IU1_K1_KLA_LDL1_LRVW4_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU32_SUM3_SUS128_SVW4_TT8_8_USFGROn1_VAW1_VSn1_VW4_VWB4_WS32_WG16_8_1_WGM4.kd`

### K10

`Cijk_Alik_Bljk_SB_MT128x128x16_SN_AMAS3_BL1_BS1_EPS0_GLVWA4_GLVWB4_GRVW4_GSU1_GSUASB_ISA1030_IU1_K1_KLA_LDL1_LRVW4_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU32_SUM3_SUS128_SVW4_TT8_16_USFGROn1_VAW1_VSn1_VW4_VWB4_WS32_WG16_8_1_WGM1.kd`

### K11

`Cijk_Ailk_Bjlk_SB_MT128x128x16_SN_AMAS3_BL1_BS1_EPS0_GLVWA4_GLVWB4_GRVW4_GSU1_GSUASB_ISA1030_IU1_K1_KLA_LDL1_LRVW4_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU0_SUM0_SUS0_SVW4_TT8_8_USFGROn1_VAW1_VSn1_VW4_VWB4_WS32_WG16_16_1_WGM8.kd`

### K12

`Cijk_Ailk_Bljk_HHS_BH_MT128x256x16_SN_AMAS3_BL1_BS1_EPS0_GLVWA8_GLVWB8_GRVW8_GSU1_GSUASB_ISA1030_IU2_K1_KLA_LDL2_LRVW8_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU32_SUM3_SUS128_SVW4_TT8_16_USFGROn1_VAW2_VSn1_VW8_VWB8_WS32_WG16_16_1_WGM4.kd`

### K13

`Cijk_Ailk_Bjlk_HHS_BH_MT128x128x16_SN_AMAS3_BL1_BS1_EPS0_GLVWA8_GLVWB8_GRVW8_GSU1_GSUASB_ISA1030_IU2_K1_KLA_LDL2_LRVW8_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU0_SUM0_SUS0_SVW4_TT8_16_USFGROn1_VAW2_VSn1_VW8_VWB8_WS32_WG16_8_1_WGM1.kd`

### K14

`Cijk_Ailk_Bljk_SB_MT128x64x16_SN_AMAS3_BL1_BS1_EPS0_GLVWA4_GLVWB4_GRVW4_GSU1_GSUASB_ISA1030_IU1_K1_KLA_LDL1_LRVW4_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU32_SUM3_SUS128_SVW4_TT8_8_USFGROn1_VAW1_VSn1_VW4_VWB4_WS32_WG16_8_1_WGM1.kd`

### K15

`Cijk_Alik_Bljk_SB_MT128x256x16_SN_AMAS3_BL1_BS1_EPS0_GLVWA4_GLVWB4_GRVW4_GSU1_GSUASB_ISA1030_IU1_K1_KLA_LDL1_LRVW4_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU32_SUM3_SUS128_SVW4_TT8_16_USFGROn1_VAW1_VSn1_VW4_VWB4_WS32_WG16_16_1_WGM8.kd`

### K16

`Cijk_Ailk_Bjlk_SB_MT128x64x8_SN_AMAS3_BL1_BS1_EPS0_GLVWA4_GLVWB4_GRVW4_GSU1_GSUASB_ISA1030_IU1_K1_KLA_LDL1_LRVW4_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU0_SUM0_SUS0_SVW4_TT8_8_USFGROn1_VAW1_VSn1_VW4_VWB4_WS32_WG16_8_1_WGM8.kd`

### K17

`Cijk_Ailk_Bljk_HHS_BH_MT64x32x8_SN_AMAS3_BL1_BS1_EPS1_GLVWA2_GLVWB2_GRVW2_GSU1_GSUASB_ISA1030_IU2_K1_KLA_LDL2_LRVW2_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU32_SUM3_SUS128_SVW4_TT4_4_USFGROn1_VAW2_VSn1_VW2_VWB2_WS32_WG16_8_1_WGM1.kd`

### K18

`Cijk_Ailk_Bljk_SB_MT64x64x8_SN_AMAS0_BL1_BS1_EPS1_GLVWA1_GLVWB1_GRVW1_GSU1_GSUASB_ISA1030_IU1_K1_KLA_LDL1_LRVW1_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU32_SUM3_SUS128_SVW4_TT4_4_USFGROn1_VAW1_VSn1_VW1_VWB1_WS32_WG16_16_1_WGM8.kd`

### K19

`Cijk_Ailk_Bjlk_SB_MT64x32x32_SN_AMAS0_BL1_BS1_EPS1_GLVWA1_GLVWB1_GRVW1_GSU1_GSUASB_ISA1030_IU1_K1_KLA_LDL1_LRVW1_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU32_SUM3_SUS128_SVW4_TT4_4_USFGROn1_VAW1_VSn1_VW1_VWB1_WS32_WG16_8_1_WGM8.kd`

### K20

`Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_SN_AMAS3_BL1_BS1_EPS0_GLVWA8_GLVWB8_GRVW8_GSU1_GSUASB_ISA1030_IU2_K1_KLA_LDL2_LRVW8_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU32_SUM3_SUS128_SVW4_TT8_16_USFGROn1_VAW2_VSn1_VW8_VWB8_WS32_WG16_8_1_WGM8.kd`

### K21

`Cijk_Alik_Bljk_HHS_BH_MT128x256x16_SN_AMAS3_BL1_BS1_EPS0_GLVWA8_GLVWB8_GRVW8_GSU1_GSUASB_ISA1030_IU2_K1_KLA_LDL2_LRVW8_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU32_SUM3_SUS128_SVW4_TT8_16_USFGROn1_VAW2_VSn1_VW8_VWB8_WS32_WG16_16_1_WGM8.kd`

### K22

`Cijk_Ailk_Bjlk_HHS_BH_MT128x128x16_SN_AMAS3_BL1_BS1_EPS0_GLVWA8_GLVWB8_GRVW8_GSU4_GSUAMB_ISA1030_IU2_K1_KLA_LDL2_LRVW8_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU0_SUM0_SUS0_SVW4_TT8_16_USFGROn1_VAW1_VSn1_VW8_VWB8_WS32_WG16_8_1_WGM8.kd`

### K23

`Cijk_H_PostGSU4_VW2.kd`

### K24

`Cijk_Ailk_Bljk_SB_MT128x128x8_SN_AMAS3_BL1_BS1_EPS0_GLVWA4_GLVWB4_GRVW4_GSU1_GSUASB_ISA1030_IU1_K1_KLA_LDL1_LRVW4_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU32_SUM3_SUS128_SVW4_TT8_16_USFGROn1_VAW1_VSn1_VW4_VWB4_WS32_WG16_8_1_WGM4.kd`

### K25

`Cijk_Ailk_Bjlk_SB_MT128x64x8_SN_AMAS3_BL1_BS1_EPS0_GLVWA4_GLVWB4_GRVW4_GSU4_GSUAMB_ISA1030_IU1_K1_KLA_LDL1_LRVW4_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU0_SUM0_SUS0_SVW4_TT8_8_USFGROn1_VAW1_VSn1_VW4_VWB4_WS32_WG16_8_1_WGM8.kd`

### K26

`Cijk_S.kd`

### K27

`Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_SN_AMAS3_BL1_BS1_EPS0_GLVWA8_GLVWB8_GRVW8_GSU1_GSUASB_ISA1030_IU2_K1_KLA_LDL2_LRVW8_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU0_SUM0_SUS0_SVW4_TT8_16_USFGROn1_VAW2_VSn1_VW8_VWB8_WS32_WG16_8_1_WGM8.kd`

### K28

`Cijk_Alik_Bljk_HHS_BH_MT128x128x16_SN_AMAS3_BL1_BS1_EPS0_GLVWA8_GLVWB8_GRVW8_GSU1_GSUASB_ISA1030_IU2_K1_KLA_LDL2_LRVW8_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU32_SUM3_SUS128_SVW4_TT8_16_USFGROn1_VAW2_VSn1_VW8_VWB8_WS32_WG16_8_1_WGM8.kd`

### K29

`Cijk_Ailk_Bljk_SB_MT128x64x8_SN_AMAS3_BL1_BS1_EPS0_GLVWA4_GLVWB4_GRVW4_GSU1_GSUASB_ISA1030_IU1_K1_KLA_LDL1_LRVW4_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU0_SUM0_SUS0_SVW4_TT8_8_USFGROn1_VAW1_VSn1_VW4_VWB4_WS32_WG16_8_1_WGM4.kd`

### K30

`Cijk_Ailk_Bjlk_SB_MT128x64x8_SN_AMAS3_BL1_BS1_EPS0_GLVWA4_GLVWB4_GRVW4_GSU1_GSUASB_ISA1030_IU1_K1_KLA_LDL1_LRVW4_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU0_SUM0_SUS0_SVW4_TT8_8_USFGROn1_VAW1_VSn1_VW4_VWB4_WS32_WG16_8_1_WGM4.kd`

## 7. 재현 및 다음 관문

```bash
R=~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
M=~/DevSpace/moss-image-rdna2-lab01
cd "$R"
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
  bash "$M/scripts/with_local_storage.sh" "$M/.venv-moss/bin/python" \
  bench_layout_blas_v06.py --output results/my_layout_map.json
```

실제 모델 단계는 아래 Anima Attention A/B와 연결했다. 기존 Anima 보고서의 엄격 RGB 실패를 취소하지 않으며, 이번 실행에서도 최종 오차를 그대로 남긴다. 이 파일은 운영 설정을 바꾸지 않는다.


## 8. 기존 Anima 모델 A/B — 레이아웃 후보의 최종 관문

공유 원자료: `results/v06_anima_attention_ab/runs.json`, `run_spec.json`, `runtime_metadata.json`, `gpu_audit.json`. 동일 서버·기존 Anima·seed 20261107·30 steps·동일 conditioning에서 비교했다. 각 Attention adapter는 1680 hit/0 fallback을 기록했다. 시간은 synchronized denoise wall이며 VAE는 각 0.55초 수준으로 별도 기록되어 있다.

| 경로 | denoise s | native SDPA 대비 RGB max / RMSE | manual NN 대비 RGB max / RMSE | 엄격 RGB |
|---|---:|---:|---:|---|
| native SDPA controls | 80.7960 / 80.7508 | 0 / 0, true byte 일치 | — | 통과 |
| 직접 QK+softmax, V row-major (`attn_manual_nn`) | 63.7238 | 0.8203125 / 0.024798754 | 0 / 0 | 실패 |
| 같은 직접 계산, V column-major (`attn_manual_vt`) | 56.9525 / 57.0227 | 0.824707031 / 0.024306802 | 0.69140625 / 0.014653656 | 실패 |

NN→VT만의 denoise 개선은 약 1.118배다. native SDPA→VT 전체 약 1.417배에는 직접 Attention 경로의 다른 변경도 포함되므로 이를 V 전치만의 효과로 적지 않는다. VT 두 반복은 latent/RGB digest가 서로 같았고, native와는 다르다. NN→VT만 바꿔도 RGB 오차가 최대 0.6914, RMSE 0.01465였으므로 최대 2/255·RMSE 0.25/255의 엄격 기준을 통과하지 않는다.

판정: **연산 속도 및 모델 속도 성공 / 엄격 품질 실패, 자동 적용 보류**. 값의 범위나 overflow가 같아도 다른 rocBLAS FP16 reduction/출력 반올림이 diffusion 반복에서 증폭될 수 있다. 이번 데이터만으로 그 증폭의 각 단계 원인을 모두 확정하지는 않았다. 새 모델의 최종 품질 허용 범위를 먼저 정한 뒤 이 레이아웃 후보를 다시 A/B해야 한다. FP32 Qwen·MOSS에서 NT 복사가 느리다는 §4의 실패 규칙은 그대로다.
