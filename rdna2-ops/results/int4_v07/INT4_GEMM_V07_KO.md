# INT4 가중치 GEMM v0.7 — 정확성 검증, 성능·엄격 RGB 실패

상태: **연산 단위와 MOSS 최종 출력 진단 완료(A)**. 확인된 native 성능 승격 후보 0개, MOSS g32 affine 표현 변경 경로의 엄격 RGB 실패. 같은 W4 독립 참조와 직접 커널은 모델 전체 latent/RGB byte 일치. 운영 기본값은 변경하지 않았다.

## 1. 카탈로그 판정과 적용 범위

| 경로 | 정확성 | 기존 실제 경로 대비 성능 | 현재 판정·근거 |
|---|---|---|---|
| A8 보존 직접 dot8 + 그룹 dequant | 정수 oracle·부동 gate 통과 | MOSS 0.051–0.797 배, Qwen 0.096–0.368 배, Anima 0.094–0.415 배 | 구현 가능 A, 성능 실패 A |
| INT4→INT8 tile decode + dot4 + 그룹 dequant | gate 통과 | MOSS 0.255–0.804 배, Qwen 0.319–0.763 배, Anima 0.342–0.994 배 | 성능 실패 A |
| INT4→FP16 dequant + rocBLAS | gate 통과 | MOSS 0.703–0.855 배, Qwen 0.384–0.577 배, Anima 0.952–1.017 배 | 성능 승격선 1.05 배 미달 A |
| FP16 미리 복원한 weight cache | gate 통과 | 짧은 두 관문 통과 값은 긴 반복에서 재현 실패 | 성능 증거 부족 A, 기본 적용 안 함 |
| FP16 tile decode + dot2 | gate 통과 | 0.242–0.358 배 | 성능 실패 A |
| G128 group/BK 분리 dot4 | 기존 W4 output 전체 byte 일치 | 기존 느린 W4보다 1.8–2.0 배, native 대비 0.646–0.955 배 | 원인 개선 성공 / native 승격 실패 A |
| MOSS g32 affine 모델 | W4 reference↔custom latent/RGB byte 일치 | per-call pack 진단, 가속 주장 없음 | 원본 대비 RGB max 240.370/255, RMSE 45.288/255: 엄격 실패 A |
| A8 word 직접 읽기 + W4 packed-word 확장 dot4c | 24 포맷·3 bias shape 정확 | 대표 3 shape 0.201–0.275 배 | 성능 실패 A |
| 설치된 Kitchen HIP W4 GEMM | gfx1030은 WMMA 경로 등록 제외 | 실행하지 않음 | B 소스: decoder 지원과 GEMM 지원을 구분 |

위 범위는 9 shape × 6 quantization 포맷의 모든 측정값이다. Anima 정수 경로의 실제 호출 시간에는 A16→A8 양자화와 FP16 출력 변환을 포함한다. MOSS/Qwen은 이미 양자화된 A8을 받는 실제 GEMM 기준이며 공통 activation quantizer/ConvRot 비용은 단위 GEMM 표 밖이다. **모든 단위 측정은 실제 Linear weight·shape의 bias 없는 GEMM이고 full Linear 시간은 아니다.**

새 모델 적용 규칙: INT4 저장량 절감이 GEMM 속도 향상을 뜻하지 않는다. gfx1030에서는 우선 실제 INT8 융합 또는 FP16 rocBLAS 기준을 확인한다. 현재 그룹형 선형 W4 포맷과 이 구현을 자동 적용하지 않는다. 메모리 때문에 INT4가 필요하다면 caller-owned 준비·캐시 용량과 원본→W4 최종 출력 손실을 별도 판단한다. 다른 INT4 포맷·향후 다른 커널까지 불가능하다는 판정은 아니다.

## 2. 실제 기준선·가중치·shape

기준선은 MOSS v0.5 INT8+FP32 dequant 융합 `tile=(64,128,64), warps=4, stages=2, group_m=8`, Qwen의 설치된 patientx per-row kernel, Anima FP16 `F.linear`이다. 예전 MOSS INT32 출력+별도 dequant를 다시 기준으로 사용하지 않았다. `sweep01/profile_*_native.json`에 실제 실행 kernel을 보존했다.

| 모델 | 실제 kernel 이름 | 실제 checkpoint |
|---|---|---|
| MOSS | `_int8_dot_dequant.kd` | `moss_image_sft_int8_convrot.safetensors` |
| Qwen | `_int8_matmul_dequant_per_row_kernel.kd` | `qwen_image_2.1_int8_convrot.safetensors` |
| Anima | rocBLAS `Cijk_Alik_Bljk_HHS_..._ISA1030_..._WS32`, projection/down MT128×256×16, up MT128×128×16 | `anima-base-v1.0.safetensors` |

MOSS/Qwen의 원본은 **이미 INT8인 파일을 FP32로 복원한 값**이다. 학습 FP32 원본의 손실을 측정한 것이 아니다. Anima는 기존 BF16 파일을 현재 실행처럼 FP16으로 읽은 weight가 기준이다. 각 weight key·tensor SHA·source SHA는 raw JSON에 있다. 연산 단위 activation은 실제 관측 shape에 고정 seed를 넣은 합성값이며 실제 모델 activation capture로 부르지 않는다.

g64 대칭·KN·low-first, dot4 64×64 / dot8 16×64 / dot2 64×64×32의 표다. 표의 배속은 동일 행 native에 대한 비율이다. 1보다 작으면 느리다.

| 사례 | M × N × K | 실제 기준 일반 호출 | tile decode+dot4 | A8 보존 dot8 | FP16 dequant+rocBLAS | fused dot2 |
|---|---|---:|---:|---:|---:|---:|
| moss_proj | 1061 × 3072 × 3072 | 0.699 ms | 0.801 배 | 0.499 배 | 0.774 배 | 0.285 배 |
| moss_mlp_up | 1061 × 16384 × 3072 | 3.361 ms | 0.714 배 | 0.428 배 | 0.715 배 | 0.254 배 |
| moss_mlp_down | 1061 × 3072 × 8192 | 1.939 ms | 0.799 배 | 0.278 배 | 0.722 배 | 0.287 배 |
| qwen_proj | 2400 × 4096 × 4096 | 2.374 ms | 0.716 배 | 0.341 배 | 0.559 배 | 0.242 배 |
| qwen_mlp_up | 2400 × 24576 × 4096 | 14.095 ms | 0.671 배 | 0.368 배 | 0.390 배 | 0.248 배 |
| qwen_mlp_down | 2400 × 4096 × 12288 | 8.011 ms | 0.754 배 | 0.347 배 | 0.481 배 | 0.281 배 |
| anima_proj | 8192 × 2048 × 2048 | 2.963 ms | 0.994 배 | 0.415 배 | 1.017 배 | 0.358 배 |
| anima_mlp_up | 8192 × 8192 × 2048 | 8.626 ms | 0.655 배 | 0.319 배 | 0.987 배 | 0.270 배 |
| anima_mlp_down | 8192 × 2048 × 8192 | 9.533 ms | 0.670 배 | 0.265 배 | 0.953 배 | 0.297 배 |

각 후보의 GPU event·동기화 포함 일반 호출·최대 추가 allocation·VGPR/spill·오차를 [matrix.csv](matrix.csv) 324행과 [sweep01/results.json](sweep01/results.json)에 저장했다. 3회 교대 round × 각 3회 호출, 2회 prewarm을 사용했다. HIP event 구간에는 일반 Python 호출의 launch gap이 포함될 수 있어 순수 kernel duration으로 부르지 않는다.

기준의 첫 호출/컴파일·autotune은 별도 `native_first_compile_autotune_s`에 있다. 예를 들어 MOSS projection 0.540 s, Qwen projection 0.831 s / MLP up 1.330 s, Anima projection 0.00266 s다. 이번 실행은 기존 disk JIT cache가 있으므로 깨끗한 cold 환경의 비용으로 일반화하지 않는다. weight 첫 pack 1.609–86.041 ms도 JIT를 포함할 수 있는 준비 비용이며 warm 호출 표에 합치지 않았다.

## 3. 저장 계약·정확한 A8 dot8

사전 기준은 [INT4_PLAN_V07_KO.md](INT4_PLAN_V07_KO.md), SHA `151db7c72107459b68572902c77ea5a57b3809f27b7752a73c628e44d2c46f99`에 첫 GPU 실행 전에 고정했다. FP32 계산 relative L2 ≤ 2e-5, FP16 출력 ≤ 1e-3와 별도 최대 절댓값 관문, RGB 최대 2/255·RMSE 0.25/255, 성능 1.05 배를 결과 후 바꾸지 않았다.

선형 INT4 weight는 `W=(q-zero)*scale`, K축 그룹 32/64/128이다. 대칭은 [-7,7], 비대칭은 0을 포함하는 min/max와 16단계 affine quantization을 쓴다. CPU/GPU round는 ties-to-even, 0 그룹은 정확히 0으로 복원한다. 32-bit word 한 개에 signed nibble 8개를 low/high-first로 저장한다. N×K/8(NK), K/8×N(KN) 연속 layout을 비교했다.

RDNA2 `v_dot8_i32_i4`의 입력 둘은 각각 signed nibble 8개다. A8 값을 그대로 유지하려면:

`A8 = lo4 + 16*hi4 + 8`, `lo4=(A8&15)-8`, `hi4=A8>>4`

`dot(A8,q4)=dot8(lo4,q4)+16*dot8(hi4,q4)+8*sum(q4)`

비대칭은 여기에 `-zero*sum(A8)` 보정을 넣는다. 실제 호출 시간에 A8 recoding·두 dot8·보정을 포함했다. A8→A4 근사를 W4A8 성능으로 섞지 않았다. Anima의 A16→A8은 별도 activation 표현 손실이다.

공식 근거는 [AMD RDNA2 ISA](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture), [LLVM gfx1030 syntax](https://rocm.docs.amd.com/projects/llvm-project/en/docs-6.2.4/LLVM/llvm/html/AMDGPU/AMDGPUAsmGFX1030.html), [LLVM intrinsic semantics](https://llvm.org/docs/AMDGPUUsage.html)이다. 지원 판정은 실제 이 GPU의 컴파일·ISA·실행(A)을 우선한다.

공통 packed object는 scale FP32·zero INT32·q 합 INT32를 모든 경로용으로 함께 보유한다. 따라서 weight 하나당 실제 resident byte는 `0.5 + 12/G`다.

| 그룹 | 실제 packed byte/weight | INT8 대비 저장 비율 | FP16 대비 저장 비율 |
|---|---:|---:|---:|
| 32 | 0.875 | 87.5 % | 43.75 % |
| 64 | 0.6875 | 68.75 % | 34.375 % |
| 128 | 0.59375 | 59.375 % | 29.6875 % |

unused metadata를 제거한 이상적인 최소치라고 주장하지 않는다. 준비된 FP16 cache는 여기에 2×N×K byte, INT8 cache는 N×K byte가 추가된다. Anima A8 변환은 activation 중간 메모리도 필요하다. raw의 `additional_peak_bytes`는 이미 상주한 cache를 제외한 호출 추가량이며 전체 모델 VRAM과 다르다.

## 4. 정확성 결과 — 성공 A

- `smoke02`: 24 포맷(그룹 3 × 대칭/비대칭 2 × layout 2 × nibble 순서 2)의 CPU/GPU packed word·scale·zero·sum과 decoder가 일치했다.
- dot4/dot8/dot4_words의 정수 부분은 CPU INT64 oracle 72/72 통과했다. A8 전체 범위와 -128/127을 포함한다.
- 대표 6 포맷 × 5 float 경로의 계산 gate 30/30 통과했다.
- packed-word 확장 자체는 CPU에서 4-nibble 전체 65,536 패턴 × 2 순서=131,072 조합을 검증했다. `cpu_word_unpack.json`.
- 실제 `v_dot8_i32_i4`, `v_dot4c_i32_i8`, `v_dot2c_f32_f16`를 ISA artifact에 보존했다.
- `reference01`: nonzero FP32 bias, shape (37,53,384), (19,3072,3072), (1061,3072,3072)에서 dot4/dot8/word의 9개 출력이 독립 그룹 torch.mm 참조와 **byte 일치**했다.
- `sweep01`: 실제 weight 9개 × 그룹/대칭 6개 × 후보 6개 = **324/324** 같은 저장값 계산 gate 통과.

독립 bias 참조는 그룹별 중심화 INT8과 A8의 정수 dot을 FP32 torch.mm으로 계산한다. G≤128에서 모든 정수 부분합 절댓값은 최대 128×15×128=245,760 < 2^24이므로 FP32로 정확하다. 그 이후 group scale 곱과 그룹 합산의 FP32 반올림은 별도 torch mul/add와 그룹 증가 순서로 정의했다. 이것은 정확도 oracle 비용이며 빠른 실행 기준으로 쓰지 않는다.

MOSS/Qwen의 같은 W4 저장값 FP32 계산 relative L2 최대는 3.618e-7이다. Anima 정수 core는 최대 3.020e-7이고 FP16 출력까지 포함하면 최대 2.382e-4다. FP16 dequant/연산 경로의 같은 저장값 relative L2 최대는 약 3.96e-4다. 아래 표현 손실과 자릿수가 다르므로 둘을 합치지 않는다.

## 5. 가중치 표현 손실과 activation 표현 손실

사전에 고정한 17개 output row × 19개 column의 독립 CPU FP64 표본에서 계산했다. 아래는 각 모델의 3개 actual weight shape에 걸친 output relative L2 범위다. 전체 weight tensor relative L2나 실제 RGB 오차가 아니다.

| 모델 | INT4 포맷 | 원래 저장 weight → W4 표현 손실 |
|---|---|---:|
| moss | g32_sym | 0.08416–0.10279 |
| moss | g32_aff | 0.08190–0.08555 |
| moss | g64_sym | 0.10064–0.10543 |
| moss | g64_aff | 0.09165–0.09286 |
| moss | g128_sym | 0.10080–0.11519 |
| moss | g128_aff | 0.09381–0.10484 |
| qwen | g32_sym | 0.08676–0.10874 |
| qwen | g32_aff | 0.07549–0.08292 |
| qwen | g64_sym | 0.09505–0.11748 |
| qwen | g64_aff | 0.08718–0.10434 |
| qwen | g128_sym | 0.10934–0.12186 |
| qwen | g128_aff | 0.09317–0.11522 |
| anima | g32_sym | 0.11519–0.14658 |
| anima | g32_aff | 0.09195–0.09888 |
| anima | g64_sym | 0.13607–0.16184 |
| anima | g64_aff | 0.09934–0.12526 |
| anima | g128_sym | 0.16068–0.20356 |
| anima | g128_aff | 0.12228–0.16034 |

Anima g64 대칭에서 A16→A8의 추가 output relative L2는 projection 0.008172, MLP up 0.008395, down 0.008477이다. 이 값도 W4 weight 손실과 같은 저장값 계산 오차에서 분리했다. MOSS/Qwen의 A8 저장값은 보존하므로 이 추가 activation 표현 변경이 없다.

## 6. layout·tile 실패 원인

초기 `screen01`의 NK dot4는 256 VGPR / spill 23이었다. KN은 237 VGPR / spill 0으로 줄여 MOSS projection 0.570→0.809 배, Qwen 0.478→0.725 배, Anima 0.676→0.932 배로 나아졌지만 native를 넘지 못했다. 큰 실제 shape의 throughput은 low-first만 측정했으며 high-first는 packing/계산 정확성만 A로 확인했다.

`screen02`에서 tile을 32×64로 줄이자 140 VGPR / spill 0이 됐지만 재사용량과 tile 작업량의 균형이 나빠져 0.726/0.628/0.857 배였다. dot8를 16×64→32×64로 키우는 비교도 악화됐다. register 수가 줄었다고 속도가 자동 개선된다고 해석하지 않는다. 타일 재사용과 병렬 처리 균형에 관한 원인 설명은 코드 구조에 근거한 추론(B)이며, hardware counter로 각 stall 비중을 분해한 측정은 하지 않았다.

직접 packed-word dot4는 A8 원래 4-byte word를 읽고 W4 word를 두 INT8 word로 풀어 broadcast 전에 처리한다. 정확한 구현과 ISA는 확보했지만 대표 3 shape에서 0.263/0.201/0.275 배라 폐기했다. spill 0만으로 충분하지 않았고 dot·load·broadcast 재사용을 함께 봐야 한다.

전체 sweep에서 dot4 g128은 256 VGPR와 136–137 spill을 보였으며, 일부 큰-K g32도 spill이 생겼다. g64 KN의 237 VGPR / spill 0이 이 구현에서는 비교적 나았다. 그룹마다 INT32 부분합과 FP32 누적 tile을 유지하고 scale을 읽고 적용하는 구조가 기존 full-K INT8 dot+마지막 scale보다 비싸다(B 소스와 A의 일치). dot8는 spill이 0이어도 A8 보존을 위한 두 dot8와 보정·recoding, 현재 타일의 데이터 재사용 비용으로 느렸다. 이를 RDNA2의 모든 INT4 GEMM에 대한 불가능성 증명으로 확대하지 않는다.

### 6.1. quant group G와 계산 BK 분리 — 원인 개선 성공, native 승격 실패 A

[SUBK_PLAN_KO.md](SUBK_PLAN_KO.md)에 관측된 G128 spill을 겨냥한 변형 한 개만 사전 고정했다. 새 `dot4_split`은 BK=min(G,64)의 inner K loop에서 INT32 부분합을 그룹 끝까지 합친 뒤, 기존과 같은 순서로 group scale을 한 번 적용한다. 원래 kernel을 보존했고 무작위 tile 탐색은 추가하지 않았다.

G128의 대칭/비대칭 × NK/KN × low/high 8포맷 CPU INT64 oracle과 float gate를 통과했다. 아래 큰 shape에서도 새로운 FP32 core의 **전체 output**이 이전 dot4와 byte 일치했다. Anima의 수치는 A16→A8 및 FP16 출력 변환까지 포함한 실제 호출이다.

| 사례 | 기존 dot4 | BK64 split | 기존 W4 kernel 대비 | 실제 native 대비 | spill | 같은 W4 output |
|---|---:|---:|---:|---:|---:|---|
| moss_proj 대칭 | 1.707405 ms | 0.938620 ms | 1.819 배 | 0.761 배 | 136 → 19 | 일치 |
| moss_proj 비대칭 | 1.695105 ms | 0.939374 ms | 1.805 배 | 0.737 배 | 136 → 19 | 일치 |
| qwen_proj 대칭 | 7.165080 ms | 3.525428 ms | 2.032 배 | 0.663 배 | 137 → 18 | 일치 |
| qwen_proj 비대칭 | 7.296946 ms | 3.722547 ms | 1.960 배 | 0.646 배 | 137 → 18 | 일치 |
| anima_proj 대칭 | 6.023968 ms | 3.105105 ms | 1.940 배 | 0.955 배 | 137 → 18 | 일치 |
| anima_proj 비대칭 | 6.206553 ms | 3.293680 ms | 1.884 배 | 0.865 배 | 137 → 18 | 일치 |

spills 136–137→18–19와 약 1.8–2.0 배의 이전 W4 kernel 개선은 실제 성과지만, **native 대비 최고 0.955 배**라 전체 성능 채택에는 실패했다. 기존 구현이 느렸다는 사실을 가리고 상대 개선만 최종 가속으로 부르지 않는다. VGPR는 256으로 유지됐고 spill이 완전히 사라지지는 않았다. 사전 계획대로 9×6 추가 확장·추가 tile 탐색은 중단했다.

원자료는 [subk_smoke01/results.json](subk_smoke01/results.json), [subk_screen01/results.json](subk_screen01/results.json)이다. 둘 다 exit 0, GPU guard 유효, foreign/error 0이다. subk screen SHA `4c1ff007ed99fbda4255ebabf0402e6bab50a11a6b18299ef7ea18311c4a57e7`, kernel `9876326b9585f81816c2f695c7b1a5c0a01e1a8205d15e35130004833c73e1ab`, bench `6ebea2b993e7ffe5c861a0f04ae260dcca3b941e4a25ce6f9239629cd495bfc5`. 이번 변형의 실제 측정 범위는 G128이며 자동 선택하지 않는다.

## 7. 짧은 승리 수치 재검증

초기 full sweep의 prepared FP16은 MOSS g32 affine 1.056 배, Anima g64 대칭 1.108 배였다. 관문을 넘겼으므로 그 두 행만 교대 7 round × 10회로 재검증했다. MOSS는 0.931 배, Anima는 0.995 배로 재현되지 않았다. 작은 최고값만 성과로 채택하지 않는다. 이 준비 경로는 FP16 dense weight를 상주시켜 쓰므로 별도의 메모리 대가도 있다.

| 모델 | full sweep 준비 FP16 | 7×10 native | 7×10 준비 FP16 | 재검증 배속 | dequant 포함 배속 |
|---|---:|---:|---:|---:|---:|
| MOSS projection g32 affine | 1.056 배 | 0.685799 ms | 0.736684 ms | 0.930927 배 | 0.755754 배 |
| Anima projection g64 대칭 | 1.108 배 | 2.846608 ms | 2.859822 ms | 0.995379 배 | 0.980769 배 |

원시 round별 일반 호출 시간(ms, 각 원소는 10회 평균)도 보존한다. 실제 raw에는 event 값도 있다.

| 모델 | 경로 | round 1–7 |
|---|---|---|
| moss | native | 0.685799, 0.684874, 0.683975, 0.695805, 0.684863, 0.727541, 0.700406 |
| moss | fp16_dequant | 0.895818, 0.875477, 0.876348, 0.944671, 0.907438, 0.988456, 0.922407 |
| moss | fp16_prepared_w | 0.716429, 0.717805, 0.719604, 0.736684, 0.773568, 0.815693, 0.782019 |
| anima | native | 2.754532, 2.839853, 2.818553, 2.919765, 2.846608, 2.873155, 2.955124 |
| anima | fp16_dequant | 2.805679, 2.903244, 2.884798, 2.902424, 2.871044, 2.945183, 2.960140 |
| anima | fp16_prepared_w | 2.801733, 2.864918, 2.933395, 2.803471, 2.859822, 2.881396, 2.811844 |

## 8. 설치된 Kitchen W4 지원을 잘못 읽지 않기

`kitchen_existing_paths_source.json`에 기존 설치 소스 SHA와 판독을 남겼다. gfx1030은 HIP backend의 `elementwise_only`에 있고, `w4a8_int8_linear`는 WMMA-only 등록에서 제외된다. HIP W4 decoder가 노출돼도 HIP W4 GEMM이 된다는 뜻이 아니다. trap 경로는 실행하지 않았다.

설치된 Triton/eager W4A8은 `round(clamp(level(q4)*s_rel,-127,127))`로 INT8 grid에 풀고 channel scale을 쓰는 형식이다. 16-entry codebook·FP8 상대 scale·일부 affine fallback도 있다. 이번 `(q-zero)*FP32_group_scale` 저장값 계약과 다르므로 무심코 같은 INT4 backend라고 교환하지 않는다. IQ3_XXS·IQ4_NL 같은 코드북 index를 이 signed 선형 nibble로 읽어 dot8에 바로 넣을 수도 없다. 이번 unit의 실제 baseline은 여전히 MOSS v0.5 융합 / Qwen patientx / Anima FP16 rocBLAS다.

## 9. MOSS 실제 최종 출력 — 계산 정확성 성공, 표현 변경 경로의 엄격 RGB 실패 A

[MODEL_DIAGNOSTIC_PLAN_KO.md](MODEL_DIAGNOSTIC_PLAN_KO.md)의 사전 기준과 기존 512²·4-step SFT workflow, seed 20261105를 유지했다. g32 affine / KN / low-first, dot4 tile 32×64다. 포트 18671의 owned process에서 `warm,int8_fused,int4_reference,int4_dot4,int8_fused`를 순차 실행했고 5요청 모두 정상 종료했다. 운영 서버를 재시작하거나 기존 파일을 바꾸지 않았다.

첫 요청은 준비·로딩을 포함한다. 아래 W4 경로는 매 호출 원래 W8 복원→pack까지 포함하는 정확성 진단이며 warm 캐시를 적용한 성능 실험으로 부르지 않는다.

| 요청 | denoising | VAE | 요청 전체 | 역할 |
|---|---:|---:|---:|---|
| warm 원본 | 8.755818 s | 0.269627 s | 14.397716 s | 첫 요청·모델 준비 |
| INT8 fused 처음 | 2.086640 s | 0.047804 s | 2.210297 s | 실제 기준 제어 |
| W4 독립 그룹 참조 | 66.356610 s | 0.048187 s | 66.641127 s | 계산 정확성 oracle |
| W4 직접 dot4 | 10.479447 s | 0.046843 s | 10.744363 s | 같은 W4 커널 진단, 매회 pack 포함 |
| INT8 fused 마지막 | 2.095777 s | 0.047116 s | 2.211610 s | 원본 복원·결정성 제어 |

각 INT8/W4 요청에서 672개의 실제 INT8 Linear 호출이 치환됐고 fallback은 0이었다. shape (M,N,K)는 (1061,9216,3072), (1061,3072,3072), (1061,16384,3072), (1061,3072,8192), 각각 168회였다. 현재 이 모델 호출의 bias는 672회 모두 None이었다. 비영 bias API 계약은 §4의 독립 3 shape 검사로 별도 검증했다. input activation·ConvRot·A8 quantizer·residual은 기존 adapter 경로를 보존했다.

| 비교 | latent | 최종 FP32 RGB | 판정 |
|---|---|---|---|
| 첫/끝 INT8 fused ↔ warm 원본 | byte 일치 | byte 일치, max/RMSE 0 | native 제어 성공 |
| W4 독립 참조 ↔ W4 dot4 | byte 일치 | **byte 일치, max/RMSE 0** | 같은 저장값의 모델 전체 계산 정확성 성공 |
| 원본 INT8 ↔ W4 독립 참조 | max 5.881277, RMSE 0.719163 | **max 0.942626953125, RMSE 0.177601174363** | 엄격 기준 실패 |

W4의 최종 RGB 차이는 최대 **240.370/255**, RMSE **45.288/255**로, 사전 기준 최대 2/255·RMSE 0.25/255를 크게 넘는다. 원본→W4 reference의 모델 차이는 W4 표현과 그룹 scale/FP32 합산 순서 변경을 함께 포함하는 **표현 변경 경로의 차이**다. 순수 weight 표현 오차는 §5의 동일 activation·CPU FP64 연산 표본에서 분리했다. W4 reference↔custom 모델 전체 byte 일치는 새 직접 커널의 추가 계산 오류가 이 RGB 차이에 섞이지 않았다는 근거다.

원시 tensor SHA-256은 아래와 같다. PNG 파일 자체는 서로 다른 workflow metadata를 포함하므로 파일 SHA와 pixel SHA를 혼동하지 않는다.

| 경로 | latent tensor SHA-256 | FP32 RGB tensor SHA-256 |
|---|---|---|
| 원본 / 첫·끝 INT8 fused | `a36c9b2b5e8926f4426d4f90e220abdd8b81f007cd378668bcd6be97479a5396` | `62dcd98312d66cc1820b06d3fe016dcd17412d53947c3f588046aabd5996cad7` |
| W4 reference / dot4 | `09f39e0744529ebcec74d21b8cc827476fbc88dc3ab8ebc209d3d85eede236a5` | `a7edb8abb07fc8524adbc15294bfe999a32e0bcfe6ada88f8947bc1c1e5c29b7` |

실제 저장 PNG도 CPU로 다시 읽었다. 원본 처음/끝 RGB8 pixel bytes는 일치했고 W4 참조/직접 커널도 일치했다. 원본↔W4 PNG의 최대 pixel 차이는 240, RMSE는 45.290239였다. [artifact_manifest.json](moss_model01/artifact_manifest.json)에 source snapshot 6개·PNG 5개 SHA와 이 CPU 검사를 남겼다.

이미지 원본: INT8 기준 (`moss_model01/images/v05_1_int8_fused_00001_.png`, 연구 폴더에만 있음), W4 독립 참조 (`moss_model01/images/v05_2_int4_reference_00001_.png`, 연구 폴더에만 있음), W4 직접 커널 (`moss_model01/images/v05_3_int4_dot4_00001_.png`, 연구 폴더에만 있음). 원자료: [runs.json](moss_model01/runs.json), [gpu_audit.json](moss_model01/gpu_audit.json), [제출 workflow](moss_model01/submitted_workflow.json), [실행 환경](moss_model01/runtime_metadata.json).

packed weights는 매 호출 범위 안에서 새로 만들었고 borrowed staging pointer cache를 유지하지 않았다. 최대 단일 packed object는 44,040,192 B였으며 원래 FP32 복원 weight·activation까지 합친 전체 transient peak를 뜻하지 않는다. 연구 export는 context 종료 때 복원됐고 owned server도 정상 종료했다. guard 405 samples에서 foreign/error 0이었다.

**최종 채택 0개.** 같은 W4 계산 커널의 정확성은 검증됐지만 기존 native보다 빠른 후보를 확보하지 못했고, MOSS g32 affine 표현 변경 경로는 엄격 RGB에 실패했다. Qwen/Anima의 최종 모델 INT4 RGB는 실행하지 않았으며 해당 모델의 품질 판정은 미검증으로 남긴다. 실패한 이 단순 group W4 결과를 모든 INT4 학습/보정·모델에 일반화하지 않는다.

## 10. 재사용과 롤백

`rdna2_ops/int4_weight_gemm_v07.py`의 `pack_int4`, `unpack_int4`, `int4_linear_a8`, `int4_linear_a16`는 명시 호출 API다. caller가 packed weights의 수명·갱신 책임을 가진다. LoRA/weight 교체 후 기존 pack을 재사용하지 않는다. 자동 dispatch, 모델 registry, production config는 변경하지 않았다. 그러므로 이번 연구의 롤백은 해당 명시 호출을 제거하거나 owned-process를 종료하는 것이다.

연구 파일을 운영 기본값에 연결하지 않는다. 현재 실패를 반영한 올바른 기본값은 모델별 확인된 기존 fused INT8 / FP16 rocBLAS 경로다.

## 11. 원자료·GPU 직렬성·SHA

모든 완료 GPU job은 exit 0이고 guard에서 foreign KFD PID 0 / error 0이었다. `sweep01`은 490 polling sample이다. guard는 관측한 KFD client를 확인하며 graphics load나 polling보다 짧은 transient를 완전히 검출한다는 뜻은 아니다. 초회/컴파일과 warm을 분리했고 병렬 GPU run은 없었다.

| 원자료 | SHA-256 |
|---|---|
| sweep01/results.json | `9e27a4de90da80c4167ffdef6143db4e821ff9563f059af5852b37cb0a50ea11` |
| screen02/results.json | `69c335a4f53285dfdf173742c9af885c45612e5afe12570413b3f7be4115ce29` |
| reference01/results.json | `66b01e44b03d54380456e164f0b67134199b1351664d80d9e0474991f8476c9f` |
| confirm_moss01/results.json | `0958a5c37798c674c0f886b76ce0c150f47e88cc46f02471b6a29d741824c754` |
| confirm_anima01/results.json | `5d71a53ba02a602860458ed70fad9dcc2fbefd79e953d1972d76db0851774fef` |
| matrix.csv | `d5936ba1ce8edb5542244f48518216fb84edb97f5d3ea158337e1df9060139d5` |
| summary.json | `4ab4209f10315a20f0c7b28b3ad4d3507643797d8d06922292fad5449436cd6a` |

| 초기 full sweep 실행 source / 보존 위치 | SHA-256 |
|---|---|
| `sweep01/source/bench_int4_v07.py` | `b65e21b6abda44405c7e2608b0de06ab5deba5120ea277a32fc75473de207391` |
| `sweep01/source/int4_weight_gemm_v07.py` | `a4fc124619e3a8fa7525411f5abb1f2f1f350cfe4a95d25d34090d220c52dee8` |
| `rdna2_ops/int8_gemm.py` | `0b2d0970af17ddf61aa19d242bfebbb6f42add330e07387442dbe97ab316109d` |
| `rdna2-ops/gpu_guard_v06.py` | `7409f98dbf2c1975f5727338346af248657d98229c4a92732809379207b4b329` |
| `RDNA2-Qwen21-INT8/int8_fused_kernel.py` | `70c014d3c394b69cc907c60dd7086e367861c6bf03031320524c3ba2dcf86603` |

각 run의 `source/`에 bench·kernel·adapter·사전 기준을 보존했다. 초기 smoke01/screen01 source는 `sources/`에 원래 SHA로 남겼다. 보호 대상 운영 파일의 전후 SHA는 상위 v0.7 manifest를 참조한다.


초기 full sweep과 sub-K/current model을 구분한다.

| 단계 | bench | kernel | model adapter |
|---|---|---|---|
| 초기 smoke01 | `f8a6ab5e66fb7b41f61fa3a3748aef993934dc135b8e043566d40466b10ef64d` | `4ce635e68c32d1870da1e4618f449679c2a31d4013db127ef16ee9391cbed896` | 해당 없음 |
| 초기 screen01 | `dfc34fa50c05bd13c4f7e5ea7bc01525015bb51c2a46495113c023686a064128` | `4ce635e68c32d1870da1e4618f449679c2a31d4013db127ef16ee9391cbed896` | 해당 없음 |
| full sweep01 / smoke02 / reference01 | `b65e21b6abda44405c7e2608b0de06ab5deba5120ea277a32fc75473de207391` | `a4fc124619e3a8fa7525411f5abb1f2f1f350cfe4a95d25d34090d220c52dee8` | `79085f271771cd35ebb44479d7d767bc61f82f3640ebdc95f092628ac13fe6e4` |
| sub-K 현재 main | `6ebea2b993e7ffe5c861a0f04ae260dcca3b941e4a25ce6f9239629cd495bfc5` | `9876326b9585f81816c2f695c7b1a5c0a01e1a8205d15e35130004833c73e1ab` | 같은 adapter |
| MOSS 모델 | harness `f9c315caf7200e0ea664f491dc27b807e7b380f7d63d17a9ad9a962801136b30` | `9876326b9585f81816c2f695c7b1a5c0a01e1a8205d15e35130004833c73e1ab` | `79085f271771cd35ebb44479d7d767bc61f82f3640ebdc95f092628ac13fe6e4` |

모델 spec SHA는 `1b92ac1bf0ad071a55c930a743065130a7cb2d5be4856a85b078cb79beba187d`다. 모델 당시 실제 source와 spec의 원문도 `moss_model01/source/`에 복사하고 실행 SHA와 대조했다. 모델 runs SHA `111a0a375789d52a2ef7c70613c82a1546de806686b95a2b92e08a029d08be7a`, guard SHA `daa3bc127fb33f352f6cdb68420c1faf7dff8a111657afb47310b61ef825ebaf`, artifact manifest SHA `346987af2171de5cfa896f5eb1ef6b8bf85300502622b96864af3a0e992e9f54`.

## 12. 재현 명령

아래는 기존 환경만 사용한다. 새 out 경로를 쓰며 이미 존재하는 out에는 덮어쓰지 않고 중단한다. 명령을 하나씩 실행하며 각 run 자체의 `gpu_guard`를 확인한다. 운영 서비스나 다른 GPU 작업이 켜진 동안 실행하지 않는다.

```bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
moss_runtime=~/DevSpace/moss-image-rdna2-lab01
export HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0
export MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53=0
bash "$moss_runtime/scripts/with_local_storage.sh" "$moss_runtime/.venv-moss/bin/python" -u bench_int4_v07.py --mode smoke --out results/int4_v07/repro_smoke
bash "$moss_runtime/scripts/with_local_storage.sh" "$moss_runtime/.venv-moss/bin/python" -u bench_int4_v07.py --mode reference --out results/int4_v07/repro_reference
bash "$moss_runtime/scripts/with_local_storage.sh" "$moss_runtime/.venv-moss/bin/python" -u bench_int4_v07.py --mode sweep --cases all --groups 32,64,128 --quant sym,aff --layouts kn --dot4-tile 64,64 --dot8-tile 16,64 --methods dot4,dot4_prepared,dot8,fp16_dequant,fp16_prepared_w,dot2 --out results/int4_v07/repro_sweep
bash "$moss_runtime/scripts/with_local_storage.sh" "$moss_runtime/.venv-moss/bin/python" -u bench_int4_v07.py --mode sweep --cases moss_proj --groups 32 --quant aff --layouts kn --methods fp16_dequant,fp16_prepared_w --rounds 7 --iters 10 --out results/int4_v07/repro_confirm_moss
bash "$moss_runtime/scripts/with_local_storage.sh" "$moss_runtime/.venv-moss/bin/python" -u bench_int4_v07.py --mode sweep --cases anima_proj --groups 64 --quant sym --layouts kn --methods fp16_dequant,fp16_prepared_w --rounds 7 --iters 10 --out results/int4_v07/repro_confirm_anima
bash "$moss_runtime/scripts/with_local_storage.sh" "$moss_runtime/.venv-moss/bin/python" -u bench_int4_v07.py --mode split_smoke --out results/int4_v07/repro_subk_smoke
bash "$moss_runtime/scripts/with_local_storage.sh" "$moss_runtime/.venv-moss/bin/python" -u bench_int4_v07.py --mode sweep --cases moss_proj,qwen_proj,anima_proj --groups 128 --quant sym,aff --layouts kn --dot4-tile 64,64 --methods dot4,dot4_split --out results/int4_v07/repro_subk_screen
bash "$moss_runtime/scripts/with_local_storage.sh" "$moss_runtime/.venv-moss/bin/python" -u research_comfy_v05.py --spec research_configs/moss_int4_v07.json --out results/int4_v07/repro_moss_model --seed 20261105 --steps 4 --modes warm,int8_fused,int4_reference,int4_dot4,int8_fused --port 18671
```


## 13. 독립 재현에서 우선 확인할 숫자

1. `--mode smoke`: 24 포맷, 정수 72/72·float 30/30 통과. `--mode reference`: nonzero bias 3 shape × 3 method 모두 byte 일치.
2. `--mode sweep --cases all ...`: 54행·324/324 계산 gate 통과. g64 sym KN MOSS projection native 0.699075 ms, dot4 0.873230 ms(0.800562 배). native보다 빠른 stable 후보 0개.
3. sub-K 명령: Qwen g128 sym projection 7.165080→3.525428 ms, spill 137→18, output 전체 byte 일치. native 2.338340 ms 대비 0.663278 배로 성능 승격 실패.
4. MOSS 모델 명령: 같은 W4 reference↔dot4 latent/RGB byte 일치; 원본 대비 RGB max 0.942626953125, RMSE 0.177601174363. 첫/끝 native 제어 byte 일치.

완료된 실패도 다음 후보의 기준선으로 남긴다. g128의 spill을 실제로 줄인 구현은 재사용 가능한 연구 코드로 보존하지만, 이번 최종 배포 판정은 채택 없음이다.
