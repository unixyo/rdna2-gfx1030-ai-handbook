# Applio v1.0 — RTX 5070과 남은 시간 차이

상태: **공식 사양·실제 운영 B/B′ 프로파일·correlation 기반 CPU 귀속·운영 B/B′ 교대 측정·양쪽 100 epoch 성능 검증 완료. 최종 지각 품질은 사용자 청취 판정 전이다.** 2026-11-08. 이 문서는 v0.8 프로파일을 새 실측으로 재사용하지 않는다. 최종 B′ 분해는 gap_Bprime_final/gap_analysis.json (`gap_Bprime_final/gap_analysis.json`, 연구 폴더에만 있음), 비교 B는 gap_B_target_steady/gap_analysis.json (`gap_B_target_steady/gap_analysis.json`, 연구 폴더에만 있음)이다. 이전 `gap_B_target`·`gap_B_target_reviewed`의 방향별 계산은 아래 §6의 귀속 결함 때문에 사용하지 않는다.

실제 운영 100 epoch의 **프로세스 전체는 B 21.4208 → B′ 19.5371분, 1.8836분 / 8.794% 감소**했다. 각 epoch recorder 시간 합은 **21.0284 → 19.2228분, 1.8056분 / 8.587% 감소**했다. 같은 초기 G/D state에서 시작한 양쪽 실행이 100 epoch·3,100 step을 완료한 결과다. 이 실제로 회수한 비용은 6900 XT 자체의 불가피한 한계가 아니었다.

앞선 8 epoch 교대의 steady 중앙값은 12.5421 → 11.4475초(1.09561배)였고, 100 epoch 실행의 steady 중앙값은 **12.5655 → 11.4676초(1.09574배)**로 개선이 유지됐다. 외부 RTX 5070의 12분과 조건이 맞는지는 확인되지 않았다. 공개 사양과 이 PC의 프로파일로 계산한 조건부 범위는 아래에 제시하지만, 그 나머지를 하드웨어/소프트웨어로 유일하게 나누지는 못했다.

## 1. 먼저 확정한 사실

비교할 RTX 5070 수치는 **dense FP16 곱·FP32 누산 61.7 TFLOP/s**, 메모리 대역폭 **672 GB/s**다. [NVIDIA Blackwell 백서](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)의 Table 6(54–56쪽)이 정밀도·누산·희소성을 나누어 적고 있다. [제품 사양](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5070-family/)에서도 대역폭을 확인했다.

| 공표 항목 | RTX 5070 | 이번 비교의 쓰임 |
|---|---:|---|
| FP16 Tensor, FP32 누산, dense | 61.7 TFLOP/s | v0.8 직접 Conv와 같은 산술 계약의 모델 상한 |
| FP16 Tensor, FP32 누산, sparse effective | 123.5 TFLOP/s | 구조적 희소성을 쓰지 않는 학습에는 적용하지 않음 |
| FP16 Tensor, FP16 누산, dense | 123.5 TFLOP/s | 누산 계약이 달라 별도 수치 검증 전에는 적용하지 않음 |
| FP4 Tensor, FP32 누산, sparse effective | 987.8 TFLOP/s | 제품의 약 988 AI TOPS. 이 FP16 학습의 처리량이 아님 |
| FP32 CUDA core | 30.9 TFLOP/s | Tensor Core를 쓰지 않는 FP32 계산의 공표값 |
| GDDR7 대역폭 | 672 GB/s | DRAM에서 같은 bytes를 이동한다고 가정할 때 |

값은 부스트 클럭 기준 이론치다. RX 6900 XT의 48.72519058 TFLOP/s·436.74511510 GB/s는 [기존 교정 실측](../applio_v08/hardware_calibration/result.json)이므로, 한쪽은 실측 ceiling·한쪽은 공표 peak인 비교라는 차이도 남는다. 계산 상한 비는 약 **1.2663배**, 대역폭 상한 비는 약 **1.5387배**다. 두 비율을 곱해서 학습 배속으로 쓰지 않는다.

cuDNN에는 implicit GEMM과 Winograd/FFT 계열이 있다. implicit GEMM은 full im2col 행렬을 저장하지 않고 입력 tensor에서 필요한 곱을 구성한다. NCHW 입력의 layout 변환, 작은 채널/타일의 비효율, 알고리즘 선택 비용은 남는다. 이는 **가능한 구현 계열의 확인**이고 이 외부 사용자에게 그 커널이 선택됐다는 증거는 아니다. [NVIDIA Conv 성능 안내 §3–4](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html), [cuDNN 정밀도 설명](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/core-concepts.html), [cuDNN CNN API](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.21.1/api/cudnn-cnn-library.html).

## 2. 식별할 수 있는 것과 계산 가정

계측된 한 연산의 GPU 시간 `t`, 직접 convolution과 동등한 연산량 `F`, 최소 tensor 읽기/쓰기량 `Q`를 둔다. 초 단위 모델은 다음과 같다.

```
L_RDNA2 = max(F / (48.72519058e12), Q / (436.74511510e9))
L_5070  = max(F / (61.7e12),       Q / (672e9))
dot2 활용 지표 = F / (t × 48.72519058e12)
최소 bytes 대역폭 지표 = Q / (t × 436.74511510e9)
```

`L_RDNA2 − L_5070`은 **같은 직접 알고리즘·같은 DRAM traffic·두 장치가 각각 상한에 도달한다는 조건**에서의 용량 차이다. `t − L_RDNA2`는 낙관적 상한과 현재 구현 사이의 여지다. 여기에 launch 지연·정수 주소 계산·LDS·register·타일 점유율·동기화 같은 비용도 포함되므로, 전부 소프트웨어 변경으로 제거 가능하다고 해석하지 않는다.

- Winograd는 직접 Conv보다 곱셈 수가 적다. 그런 행에는 dense-equivalent `F`를 참고값으로만 두고 실제 dot2 활용률·직접 알고리즘 하한 적용을 제외한다.
- `Q`는 DRAM counter가 아니다. Cache에서 재사용되거나 im2col·partial reduction으로 추가 이동이 생기면 실제 DRAM bytes는 다르다. 단순 tensor 크기로 만든 수치에는 **최소 논리 bytes / cold streaming 가정** 표식을 둔다.
- FP32 연산·special function·shape를 귀속하지 못한 kernel에는 FP16 dot2 상한을 적용하지 않는다. 계측 시간은 표에 남겨 빠진 비용을 숨기지 않는다.
- 프로파일 시간은 kernel·copy·memset 합과 연산자 귀속 시간을 별도로 집계한다. 같은 활동을 두 번 합산하지 않는다. 5 step GPU 시간을 epoch 분으로 바꿀 때는 실제 steady epoch와 31 step을 함께 적고, 프로파일 오버헤드를 100 epoch 실측처럼 쓰지 않는다.

추가로 `t_5070 = t_RDNA2 × L_5070/L_RDNA2`라는 **같은 ceiling 활용 비율** 시나리오도 출력한다. 이는 양쪽 kernel이 각자 roof의 같은 비율을 낸다는 추가 가정이다. 순수 FP16/FP32 계산 1분은 약 0.790분, 순수 DRAM 이동 1분은 약 0.650분으로 대응한다. 이 시나리오는 실제 구현 속도에서 capacity 차이가 가질 크기를 보여 주지만, RTX 5070의 실측 예측이나 보장값은 아니다. 실제 12분과 남는 차이를 전부 RDNA2 소프트웨어의 탓으로 확정하지 않는다. 온칩 traffic·register·주소 계산·타일 점유율처럼 하드웨어와 구현이 함께 정하는 비용도 있기 때문이다.

## 3. 실제 운영 B/B′의 남은 비용 — 확인된 사실

[profile_B_target/result.json](profile_B_target/result.json)과 [profile_Bprime/result.json](profile_Bprime/result.json)은 모두 `completed=true`, `valid_for_timing=true`다. 새 모델 이름으로 운영 `core.py train`을 실행했고, 원래 모델의 1 speaker·KLM-SVM 메타데이터를 읽어 적용한 것을 확인했다. 프로파일은 각각 step 40–44다. 보호 파일 변경 0, foreign KFD client 0, 최대 큐 7이었다. 파일·metadata·학습 시작 상태는 A의 [APPLIO_V10_KO.md](APPLIO_V10_KO.md)에 있다.

**runtime correlation ID**로 B의 77,026개, B′의 63,555개 kernel을 모두 CPU launch와 연결했다. 두 분석 모두 fallback 0·파싱 오류 0이다. B 직접 Conv의 fwd/dX/dW는 279/259/165호출이고, B′에서는 159/139/45개의 기존 직접 호출과 각 120개의 fused 호출로 나뉜다. 실제 융합 호출이 존재하며, dtype와 residual 여부를 나눈 annotation geometry로 귀속했다. [B 연산자 요약](profile_B_target/profile_summary.json), [B′ 연산자 요약](profile_Bprime/profile_summary.json), [B′ kernel 표](gap_Bprime_final/kernels.csv), [B′ shape·방향 표](gap_Bprime_final/work_units.csv).

| GPU kernel 계열 | B, 5 step ms | B′, 5 step ms | B′ kernel 비중 | B′를 3,100 step 반복한 환산 분 |
|---|---:|---:|---:|---:|
| native Conv1d | 493.836 | 488.023 | 31.89% | 5.0429 |
| native Conv2d | 700.494 | 470.662 | 30.76% | 4.8635 |
| 신규 Conv2d | 0 | 95.279 | 6.23% | 0.9845 |
| v0.8 direct Conv1d | 159.852 | 88.344 | 5.77% | 0.9129 |
| optimizer | 76.983 | 76.983 | 5.03% | 0.7955 |
| 융합 Conv1d | 0 | 73.398 | 4.80% | 0.7584 |
| reduction·normalization | 65.609 | 70.237 | 4.59% | 0.7258 |
| elementwise | 44.652 | 42.710 | 2.79% | 0.4413 |
| copy·cast kernel | 39.760 | 40.046 | 2.62% | 0.4138 |
| 그 외 미분류 | 23.631 | 23.841 | 1.56% | 0.2464 |
| 기타 GEMM | 22.179 | 22.277 | 1.46% | 0.2302 |
| activation | 22.855 | 19.374 | 1.27% | 0.2002 |
| 신규 Conv1d | 0 | 6.752 | 0.44% | 0.0698 |
| 신규 ConvTranspose | 0 | 6.394 | 0.42% | 0.0661 |
| native ConvTranspose | 29.603 | 5.983 | 0.39% | 0.0618 |
| **합계** | **1,679.453** | **1,530.302** | **100%** | **15.8131** |

이 표의 환산은 100 epoch 실측이 아니다. GPU kernel 합은 **8.881% 감소**했고, profiler 없는 짧은 교대 steady epoch는 **8.727% 감소**, 실제 100 epoch의 steady 중앙값은 **8.737% 감소**했다. 프로파일 GPU 합 환산은 B 17.3543분·B′ 15.8131분, 짧은 교대 steady 환산은 B 20.9035분·B′ 19.0792분이다. 두 방법의 차이 3.5491분·3.2661분을 전부 제거 가능한 host 비용으로 해석하지 않는다. 5-step shape·AMP skip 분포를 3,100 step 반복한다는 근사도 포함한다. 최종 실측 epoch 합·프로세스 시간은 §4에서 별도로 비교한다.

native와 신규를 합친 Conv2d는 **700.494 → 565.941 ms**, ConvTranspose는 **29.603 → 12.376 ms**로 줄었다. 각각 134.553 ms·17.227 ms 감소이며 다른 계열의 작은 증가가 일부 상쇄한다. LeakyReLU가 들어 있는 activation 계열은 3.480 ms, elementwise 계열은 1.942 ms 줄었다. 융합 Conv 자체에는 원소별 연산이 옮겨 들어가므로 이 감소를 그대로 융합의 순이익으로 세지 않는다.

연산자 self-device 분모는 B **1,685.778 ms**, B′ **1,536.640 ms**다. B′에서 `convolution_backward` 39.85%, `miopen_convolution` 20.63%, `_CandidateConv` 4.18%, `_FusedPairBackward` 4.12%, `_MeasuredConvBackward` 4.08%였다. B′의 kernel 분모 **1,530.302 ms**에서는 Winograd 두 kernel 365.666 ms(23.90%), im2col+col2im 156.162 ms(10.20%)다. B의 해당 합은 369.026 ms·208.471 ms였다. operator 표와 kernel 표는 같은 작업을 서로 다른 관점으로 센 것이므로 더하지 않는다.

프로파일 window는 B 2.18808 s·B′ 1.99909 s, GPU kernel/copy/memset union은 1.68578 s·1.53664 s였다. 기록된 GPU 활동이 없는 구간은 **0.50230 → 0.46245 s**다. `hipMemcpyWithStream`의 CPU 대기 합 0.92280 → 0.66489 s는 진행 중 GPU를 기다린 시간도 포함한다. 이를 GPU 활동 없는 구간과 합치거나 전부 제거 가능한 host 손실로 세지 않는다. profiler 계측/annotation 비용도 있으므로 이 gap을 그대로 100 epoch의 host 시간으로 환산하지 않는다.

## 4. 상한 대비 활용 지표와 분 단위 모델

각 kernel의 계측 시간과 계산 가능한 활용 지표는 [B′ kernels.csv](gap_Bprime_final/kernels.csv)에 있다. 주요 항목은 다음과 같다. `최소 bytes` 지표는 실제 DRAM bandwidth utilization이 아니다.

| B′ kernel | 5 step 합 ms | dense-equivalent GFLOP | dot2 ceiling 대비 | 최소 논리 bytes / streaming BW |
|---|---:|---:|---:|---:|
| 신규 `_forward` | 67.260 | 673.170 | 20.54% | 9.68% |
| 융합 `_dw_act_partial` | 50.306 | 205.521 | 8.38% | 8.10% |
| v08 `_dw_partial` | 38.664 | 164.626 | 8.74% | 4.61% |
| v08 `_fwd` | 25.586 | 544.505 | 43.68% | 10.92% |
| v08 `_dx` | 23.866 | 359.955 | 30.95% | 9.12% |
| 신규 `_backward_weight_flat` | 15.064 | 69.699 | 9.50% | 14.25% |
| 융합 `_dx_mask_res` | 12.642 | 205.521 | 33.37% | 49.91% |
| 융합 `_fwd_act_res` | 10.150 | 205.521 | 41.56% | 44.42% |
| `_dw_reduce` | 0.528 | Conv FLOP 중복 배분 안 함 | 해당 없음 | 174.52% |

소스로 역할이 확인된 dW partial/reduction은 annotation 호출 수와 kernel 횟수가 일치할 때만 분리했다. 융합 activation·잔차·mask의 산술은 Conv dot2 FLOP에 포함하지 않는다. native Winograd는 실제 transform 연산량이 미산정이므로 dense Conv FLOP으로 이용률을 붙이지 않는다. 100%를 넘는 논리 bytes 지표는 cache 재사용 등으로 실제 DRAM 요청량과 다르다는 신호다.

최소 논리 tensor bytes를 436.7 GB/s로 나눈 시간이 실제 kernel보다 큰 행은 B 456개·B′ **462개**다. 이는 물리 DRAM 대역폭을 넘겼다는 측정이 아니다. cache 재사용과 논리 요청량/실제 DRAM traffic의 차이를 드러낸다. 모든 pure elementwise cold-DRAM 가정은 capacity 합계에서 제외했다. B′에는 추가로 C32 융합 fwd/dX **5개 행**에서 cold 모델이 실제보다 1.07–1.39배 길어지는 모순이 있어, 해당 행도 시간·FLOP·논리 bytes 진단은 보존하고 합계에서 제외했다. 이 제외는 성능 게이트 변경이 아니라 관측에 어긋난 traffic 가정을 사용하지 않기 위한 모델 수정이다.

최종 B′에서 쌍으로 계산한 FP16 직접 알고리즘 범위는 **330/1,211 workload 행, GPU 시간의 53.3668%**다. B에서는 312/1,188행·57.3706%였다. Winograd가 포함된 backward pass·FP32 연산·모순된 cold 모델은 보수적으로 제외했다. 모든 행의 계측 시간은 위 표에 남아 있다.

| 최종 B′에서 같은 알고리즘 모델을 적용한 범위 | 현재 GPU 시간 환산 분 | RDNA2 조건부 floor 분 | 5070 조건부 floor 분 | floor 차이 분 | RDNA2 floor까지의 차이 분 |
|---|---:|---:|---:|---:|---:|
| native Conv1d subset | 2.8593 | 0.2523 | 0.1928 | 0.0596 | 2.6070 |
| native Conv2d subset | 2.5189 | 0.5882 | 0.4645 | 0.1237 | 1.9308 |
| 신규 Conv2d | 0.9845 | 0.1981 | 0.1472 | 0.0509 | 0.7865 |
| v0.8 direct Conv1d | 0.9129 | 0.2272 | 0.1790 | 0.0481 | 0.6857 |
| 융합 Conv1d subset | 0.7397 | 0.1536 | 0.1116 | 0.0420 | 0.5861 |
| 기타 FP16 GEMM | 0.2260 | 0.0406 | 0.0265 | 0.0141 | 0.1854 |
| 신규 Conv1d | 0.0698 | 0.0181 | 0.0143 | 0.0038 | 0.0516 |
| 신규 ConvTranspose | 0.0661 | 0.0143 | 0.0105 | 0.0037 | 0.0518 |
| native ConvTranspose | 0.0618 | 0.0085 | 0.0065 | 0.0020 | 0.0534 |
| **모델 범위 합계** | **8.4390** | **1.5008** | **1.1530** | **0.3478** | **6.9382** |

**0.3478분은 이 계산 조건에서의 이상적인 장치 용량 차이**이며, 실제 5070과 줄일 수 없는 시간 차이가 아니다. **6.9382분도 소프트웨어로 전부 회수할 수 있다는 뜻이 아니다.** 주소 계산·tile 낭비·register/LDS·launch·동기화·필요한 추가 traffic이 이 차이에 들어 있다. 실제 B→B′ 100 epoch에서 회수한 시간은 **epoch 합 1.8056분, 프로세스 전체 1.8836분**이다. B의 모델 값 0.3397분·8.4804분은 이전 구현의 조건부 계산으로 보존하며, coverage와 추가 scratch/mask 작업이 달라 B′ 모델과의 단순 차이를 실측 절감으로 쓰지 않는다.

추가 가정인 **같은 ceiling 활용 비율**을 적용하면, B′ 모델로 덮은 비용에서 5070의 capacity 차이에 대응하는 절감은 **2.0739분**이다. 다른 비용은 그대로 둔 시나리오이며, 위 0.3478분과 더하는 항목이 아니다. 5070 실제 kernel/algorithm/DRAM counter를 얻지 못한 현재 자료로는 어느 시나리오가 외부 12분을 설명하는지 확정할 수 없다.

| 최종 비교 항목 | 분 / 100 epoch | 판정 |
|---|---:|---|
| 실제 운영 B, 100 epoch 프로세스 전체 | 21.4208 | 전체 실행 실측 |
| 실제 운영 B′, 100 epoch 프로세스 전체 | 19.5371 | 전체 실행 실측 |
| B → B′ 프로세스 절감 | 1.8836 | 확인된 소프트웨어 변경의 전체 실행 절감, 8.794% |
| B → B′ epoch recorder 합 절감 | 1.8056 | 같은 scope의 100개 epoch 시간 합 차이, 8.587% |
| B′ 모델에 포함한 범위의 이상적 capacity 차이 | 0.3478 | 조건부 모델 |
| B′의 같은 roof 활용 비율 시나리오의 capacity 절감 | 2.0739 | 추가 가정을 둔 대안 시나리오 |
| B′ RDNA2 구현과 낙관적 ceiling 사이 | 6.9382 | 제거 가능량의 보장이 아님 |
| 외부 RTX 5070 보고 | 12 | 데이터·모델·step·batch·AMP·라이브러리·CPU·저장/추론 포함 여부 미검증 |
| 외부 12분과의 불가피/회수 가능 인과적 분할 | 미식별 | 5070의 같은 조건 실측이 필요 |

동일 조건이 확인되지 않은 `T_5070=12분`은 실제 차이를 하드웨어/소프트웨어로 유일하게 나누기에 충분한 관측이 아니다. B′ 프로세스 실측 19.5371분과 단순히 빼면 7.5371분 차이지만, 외부 보고가 초기화·저장 등을 포함하는지도 확인되지 않았다. **이 7.5371분을 불가피한 하드웨어 시간으로 부르거나, 조건부 0.3478분을 뺀 나머지를 제거 가능한 소프트웨어 시간으로 부르지 않는다.**

교대 원자료는 [short_epoch_protocol/result.json](short_epoch_protocol/result.json)이다. B1 → B′1 → B2 → B′2를 각각 8 epoch 실행하고, 각 실행의 첫 2 epoch를 제외한 6개를 합쳐 arm당 12개의 중앙값을 냈다. 네 실행 모두 `completed=true`, `valid_for_timing=true`, 초기 G/D state SHA 동일, finite loss, 최대 KFD queue 7이다. 각 인접 쌍의 배속은 **1.09489배, 1.09437배**로 일관됐다. 프로토콜의 실제 네 result SHA도 CPU 읽기로 다시 맞췄다.

최종 원자료는 [final_epoch_protocol/result.json](final_epoch_protocol/result.json)과 B의 [final_B1/result.json](final_B1/result.json), B′의 [final_Bprime1/result.json](final_Bprime1/result.json)이다. 양쪽 raw result의 SHA를 프로토콜과 대조하고, epoch 합·steady 중앙값을 CPU로 다시 계산했다. 둘 다 `completed=true`, `valid_for_timing=true`, 100 epoch·3,100 step, nonfinite loss 0, 보호 파일 변경 0, 외부 KFD client 0·최대 queue 7이다. 초기 G/D state SHA는 서로 같으며 짧은 교대 실행과도 같다.

| 같은 측정 기준 | B | B′ | 해석 |
|---|---:|---:|---|
| 짧은 교대 steady 중앙값 × 100 epoch | 20.9035분 | 19.0792분 | 환산값, 전체 실행 시간이 아님 |
| 실제 100 epoch 시간 합 | **21.0284분** | **19.2228분** | **1.8056분 절감, 8.587% 감소** |
| 실제 100 epoch 프로세스 벽시계 | **21.4208분** | **19.5371분** | **1.8836분 절감, 8.794% 감소** |
| 실제 100 epoch steady 중앙값 | **12.5655초/epoch** | **11.4676초/epoch** | **1.09574배, 8.737% 감소** |

정확한 epoch 합은 **1,261.701447725 → 1,153.365163803초**, 프로세스 벽시계는 **1,285.245366560 → 1,172.226508294초**다. 프로세스 전체와 epoch 합의 차이 23.5439초·18.8613초를 전부 host 병목의 제거 가능량으로 해석하지 않는다. 100 epoch는 각 arm 한 번씩이며, 앞선 교대 두 쌍의 개선과 같은 방향으로 확인된 지속 실행 결과다. 속도와 finite loss 검증이 최종 지각 품질을 대신하지 않으므로 음질 채택은 사용자 익명 청취 판정 전이다.

## 5. 구현 여지가 실제로 확인된 곳

A의 [native 후보 합본](native_selected/summary.json)은 60 shape·169방향에서 531/531 후보 수치 검사와 39개 AMP/weight_norm 호출 경계(195 metrics)를 통과했다. 31 shape·68방향이 방향별 속도 게이트를 넘었다. 전체 B′는 융합과 native 교체를 함께 넣은 운영 교대 측정에서 시간 8.727% 감소, 실제 100 epoch 프로세스 전체에서 8.794% 감소를 확인했다. 두 변경의 개별 인과적 절감분은 분리 실험하지 않았으므로 이 효과를 각각에 중복 배분하지 않는다.

대표 두 dW는 period 첫 Conv2d에서 **27.35배**, 마지막 ConvTranspose에서 **21.72배** 빨랐다([원자료](native_family_smoke/result.json)). 같은 GPU·같은 산술 검증을 통과한 결과이므로, 그 방향의 기존 비용은 6900 XT 하드웨어의 불가피한 한계가 아니었다는 직접 증거다. native에서 한 batch씩 작은 GEMM을 반복하던 병렬화와 새 split reduction의 차이는 [NATIVE_CONV_NOTES_KO.md](NATIVE_CONV_NOTES_KO.md)에 raw trace와 함께 기록돼 있다.

C128/K11 dW·d1 dX, C192→384/K5 dW, ConvTranspose fwd 네 shape는 이번 후보가 이기지 못했다. 이 음성 결과도 **이번 구현이 못 이겼다는 뜻**이지 하드웨어 최적값을 증명한 것은 아니다. 특히 Winograd의 실제 산술과 온칩 traffic을 계수하지 않고 direct roofline에 붙었다고 판단하지 않는다.

## 6. 계측·분석 수정 이력

- 최초 `profile_B`는 새 모델의 metadata fallthrough가 spk109/contentvec를 만들었고 observer가 step마다 추가 scale 동기화를 넣어 진단용으로 보존했다. 중단된 `profile_B_faithful`도 최종 비교에 사용하지 않는다. 유효한 원자료는 `profile_B_target`이다.
- 첫 CPU 귀속은 External id만 써서 같은 `_MeasuredConvBackward` 안의 dX kernel 일부를 뒤이어 나온 dW scope에 붙였다. 시간·kernel 이름·family 합계는 보존됐지만 방향 FLOP이 틀렸다. `gap_B_target`·`gap_B_target_reviewed`의 방향별 계산은 폐기 상태로 보존한다.
- `gap_B_target_correlated`는 GPU correlation→해당 runtime launch→CPU scope로 수정했다. 77,026/77,026 kernel 연결, fallback 0·파싱 오류 0, 직접 21/20/15 geometry 및 279/259/165 호출 일치가 근거다. GPU 재실행 없이 CPU 귀속을 고쳤다.
- B′ parser는 kind·stride·output_padding·transposed·config/split·FP32 residual 및 mask bytes를 읽는다. 기존 B의 split이 기록되지 않은 것은 실제 v0.8 manifest dW 15개가 모두 32임을 확인해 복원했다. 후보의 split=1과 v0.8 split=1의 scratch 동작도 구분한다.
- `gap_B_target_steady`는 원 trace를 재적재하지 않고, 올바르게 귀속된 JSON에 실제 운영 B 중앙값을 연결했다. 소스로 확인한 split-dW partial/reduction의 kernel별 지표만 추가했으며, 기존 family capacity 합계가 변하지 않는 것을 확인했다.
- B′ trace는 GPU가 쉰 구간에 한 번 적재하여 `v08_analyze_profile`과 v10 gap 분석이 공유했다. 63,555개 launch 귀속을 검증한 뒤 큰 분석 프로세스를 종료했다. `gap_Bprime_steady`에서 발견한 5개 cold-DRAM 모순 행은 작은 JSON만 재검토해 `gap_Bprime_final`에서 제외했다. GPU 측정·시간·수치 게이트는 바꾸지 않았다.

## 7. 원자료와 재현

- 공식 URL·정밀도 구분: [rtx5070_sources.json](rtx5070_sources.json).
- RDNA2 교정: [hardware_calibration/result.json](../applio_v08/hardware_calibration/result.json).
- 신규 운영 B 프로파일: [profile_B_target/result.json](profile_B_target/result.json), [CPU 연산자 요약](profile_B_target/profile_summary.json).
- 신규 운영 B′ 프로파일: [profile_Bprime/result.json](profile_Bprime/result.json), [CPU 연산자 요약](profile_Bprime/profile_summary.json).
- 최종 B′ 분해: gap_Bprime_final/gap_analysis.json (`gap_Bprime_final/gap_analysis.json`, 연구 폴더에만 있음). trace SHA256 `af705086825a77a71932633b4ab3b7213b727f809752a4d29e34563e45d49d81`, run 결과 SHA256 `287276a242c80399c266c2d68264fa7889828ae8ef60af4c07d7e80ad6bc295c`.
- 방향 귀속·steady 연결 정본: gap_B_target_steady/gap_analysis.json (`gap_B_target_steady/gap_analysis.json`, 연구 폴더에만 있음). 원 trace SHA256 `9f216f679f6b035c7ee00614aa2e6e0cae36d51176b517ccc6a2de52ed8b36d7`, run 결과 SHA256 `8565fd9c4897fb1026c7ba84498bb0759d40ca45080ad4ac0a807dab1e086deb`.
- 운영 교대 결과: [short_epoch_protocol/result.json](short_epoch_protocol/result.json), SHA256 `9b6be887e39b32e23e2d1d66d23338806d4a0a39cef2b1cac87dc9e4c372a05a`.
- 실제 운영 B 100 epoch: [final_B1/result.json](final_B1/result.json), SHA256 `35ef68d05dbb3f83afa64aefad9bbc1f2ead348ae75b23f5855553b36b33df69`.
- 실제 운영 B′ 100 epoch: [final_Bprime1/result.json](final_Bprime1/result.json), SHA256 `ea388480fb504714fd301383ba00f44c5420c3d2f7bd9f2aae9fa028f27b3def`.
- 100 epoch 합본: [final_epoch_protocol/result.json](final_epoch_protocol/result.json), SHA256 `33cdc0f310e1813a7f89e45568b29af3a8083971f56386cf74a8153cfd4e735a`.
- CPU 분석기: [v10_gap_analysis.py](../../v10_gap_analysis.py). Python 표준 라이브러리만 사용한다. 출력은 `kernels.csv`, `work_units.csv`, `family_minutes.csv`, `gap_analysis.json`이다. 완료 출력은 덮어쓰지 않는다.

```bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
nice -n 10 python3 -B v10_gap_analysis.py \
  --run results/v10/profile_Bprime \
  --out results/v10/gap_Bprime_reproduction --steps 5 \
  --steady-epoch-s 11.447523355484009
```

새 epoch 중앙값을 붙이거나 CPU 모델만 갱신할 때는 `--from-analysis results/v10/gap_Bprime_final/gap_analysis.json --out <새 연구 경로> --steady-epoch-s 11.467649102210999`을 쓴다. 원 trace를 다시 적재하지 않는다. 기존 profile 분석 JSON의 `scope.operating_steady_epoch_s`는 생성 당시 짧은 교대 중앙값을 보존한다. 최종 실제 시간은 위 100 epoch raw result에서 읽으며, profile 기반 조건부 분해는 3,100 step 환산이므로 steady 중앙값 교체와 무관하게 같다.

100 epoch 실측값은 GPU 없이 다음처럼 다시 읽을 수 있다.

```bash
python3 - <<'PY'
import json
from pathlib import Path
r={}
for arm,folder in [('B','final_B1'),('B_prime','final_Bprime1')]:
    x=json.loads(Path(f'results/v10/{folder}/result.json').read_text())
    assert x['completed'] and x['valid_for_timing']
    r[arm]=(sum(e['seconds'] for e in x['epochs']),x['process_wall_s'],x['steady_epoch_median_s'])
    print(arm,'epoch/process/steady seconds:',r[arm])
print('epoch saving seconds:',r['B'][0]-r['B_prime'][0])
print('process saving seconds:',r['B'][1]-r['B_prime'][1])
PY
```

## 8. Claude가 GPU 없이 확인할 핵심 숫자 5개

1. 실제 운영 B 100 epoch 프로세스 시간: **1,285.245366560초**.
2. 실제 운영 B′ 100 epoch 프로세스 시간: **1,172.226508294초**.
3. 실제 100 epoch recorder 합 절감: **108.336283922초**. 프로세스 절감 113.018858266초와 측정 범위가 다르다.
4. RTX 5070 dense FP16·FP32 누산 공표값: **61.7 TFLOP/s**.
5. B′ 모델 범위의 이상적 capacity 차이: **0.347826285분**. 실제 불가피한 하드웨어 시간의 실측이 아니다.

```bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
python3 - <<'PY'
import json
from pathlib import Path
s=json.loads(Path('results/v10/rtx5070_sources.json').read_text())
g=json.loads(Path('results/v10/gap_Bprime_final/gap_analysis.json').read_text())
b=json.loads(Path('results/v10/final_B1/result.json').read_text())
bp=json.loads(Path('results/v10/final_Bprime1/result.json').read_text())
assert b['completed'] and b['valid_for_timing'] and bp['completed'] and bp['valid_for_timing']
n=s['nvidia_specification']
print('1 B100 process seconds:',b['process_wall_s'])
print('2 Bprime100 process seconds:',bp['process_wall_s'])
print('3 epoch recorder saving seconds:',sum(e['seconds'] for e in b['epochs'])-sum(e['seconds'] for e in bp['epochs']))
print('4 dense FP16/FP32 TFLOP/s:',n['fp16_tensor_fp32_accumulate_dense_tflop_s'])
print('5 conditional ideal-capacity gap minutes:',g['identification']['conditional_capacity_penalty_minutes'])
PY
```

네트워크 자료는 공식 웹 문서로 조회했다. 이 B 분석에서 운영 환경에 추가 다운로드하거나 설치한 파일은 없다. 최종 raw result의 SHA·합계·중앙값·GPU audit와 조건부 모델의 family 합계·coverage를 CPU로 다시 확인했다. RTX 5070의 같은 조건 실측이 없다는 미식별 표식과 사용자 청취 판정 전이라는 품질 상태는 유지한다.
