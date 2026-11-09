# 직접 Conv1d 실험 v0.8 — 중간 연산 근거

2026-11-07, RX 6900 XT / gfx1030, 기존 Applio Torch 2.11.0+rocm10.0.0 / Triton 3.8.0.

**상태(2026-11-08 03:39 KST 기준): 직접 후보 189/189 tile, AMP/weight_norm 호출 경계 21/21 case, 최종 연산 manifest 21 shape·56방향 검증을 완료했다. solver+직접 Conv의 실제 A0→B_conv→A1 4 epoch 교대도 끝났으며 B steady는 13.31436 s, 두 native 대비 1.125–1.134×다. nonconv 통합 후보는 별도 평가 중이고, 100 epoch 전체 시간·음질은 아직 미판정이다.**

## 1. 계산과 호출 계약

- 코드: [v08_direct_conv.py](../../v08_direct_conv.py), [벤치](../../bench_v08_direct_conv.py).
- 입력 NCT, weight OIR, FP16 곱·FP32 누산, stride 1. 그룹도 계산식에 포함하지만 실제 관측한 shape만 채택한다.
- fwd와 dX는 implicit GEMM이다. 펼친 im2col tensor를 만들지 않고 원래 tensor의 좌표를 읽는다.
- dW는 시간·배치 축을 32개 chunk로 나누어 FP32 partial을 기록하고 고정 순서로 합산한다. FP16 partial과 atomic add를 쓰지 않는다.
- forward는 누산을 **먼저 FP16으로 반올림한 후 FP16 bias를 더한다**. 기존 PyTorch MIOpen의 별도 bias add 경계를 맞춘 것이다. FP32 bias까지 한 번에 누산해 반올림 경계를 바꾸지 않는다.
- AMP wrapper는 기존 FP32 weight/Adam 상태를 유지한다. 후보는 계산 중 필요한 FP16 값만 받는다. 승리한 방향만 대체하고 나머지 dX/dW는 한 번의 native aten.convolution_backward 호출로 묶어 유지한다.
- 아직 전역 자동 적용이나 기본 winner는 없다. 정확한 shape·dilation·padding·groups의 검증된 manifest와 APPLIO_RDNA2_DIRECT_CONV=1이 모두 있어야 context 안에서 작동한다. context 종료는 함수 복구다.

## 2. 실측 dot2와 메모리 기준점

[calibration/result.json](hardware_calibration/result.json)과 dot2 ISA (`hardware_calibration/dot2_calibration.s`, 연구 폴더에만 있음).

| 실측 | 값 | 측정 의미 |
|---|---:|---|
| register dot2 FP16 입력·FP32 누산 | 48.72519 TFLOP/s | 8개 독립 accumulator, 총 8,589,934,592 FLOP, 5 round × 20회 중앙값 |
| dot2 GPU event | 0.176294 ms | 컴파일 첫 호출 1.141073 s는 별도 |
| dot2 ISA | v_dot2c_f32_f16 정적 64개 | registers 21, spill 0. 코드 생성 반복과 vector lane 때문에 정적 개수는 8보다 크다 |
| streaming copy | 436.74512 GB/s | 각 256 MiB 버퍼, read+write 512 MiB |
| warm small copy | 801.97019 GB/s | 각 8 MiB 버퍼, read+write 16 MiB. cache에 들어갈 수 있는 범위 |
| GPU audit | valid | 다른 KFD client 관측 0 |

dot2 출력은 finite, 반복 덧셈의 닫힌 식 FP32 비교 최대 상대오차 0.00002477이다. copy는 값 일치였다. 버퍼에 NaN·음의 0을 쓰지 않아 bit 검사와 값 검사의 의미 차이가 없는 입력이다.

**이 수치는 낙관적인 register 연산/복사 기준점이다. Conv가 그대로 달성한다는 예측이 아니다.** 실제 Conv에는 주소 계산, LDS 배치, padding 처리, launch, cache miss, reduction이 있다. streaming 메모리 대역폭만 써서 warm 작은 Conv의 엄밀한 하한이라고 부르지 않는다. 이미 cache에 있는 입력은 DRAM에서 매번 읽지 않아도 된다.

dense stride1 Conv의 표시용 계산량은 다음과 같다.

    F = 2 × B × T_out × C_out × (C_in / groups) × kernel

dX는 T_out 대신 T_in을 쓰며, 같은 길이인 경우 세 방향의 명목 FLOP가 같다. padding 경계의 0 연산 제거와 Winograd 산술 감소를 반영하지 않은 dense 기준이다. Winograd 실행시간에 이 FLOP를 그대로 대입하여 하드웨어 활용도를 확정하면 안 된다.

텐서 1회 읽기·쓰기 가정은 다음과 같다.

    bytes = 2 × (numel(input) + numel(weight) + numel(output))

fwd/dX/dW의 해당 역할 tensor를 각각 넣는다. 이는 최소 이동량 모델이며 실제 MIOpen workspace traffic, 반복 읽기, partial, cast를 따로 봐야 한다.

## 3. 첫 actual shape: B8·C64·T6400·K7·dilation3

기준 DB는 root의 실제 운영 User FindDb 사본 db_fast_native를 다시 direct_fast_db로 복사했다. 현재 runtime의 원본 .ufdb.txt SHA는 a5a7fa17513f541a46184072658ebed7abc808ae0370d30914fd485d8e4286b8이다. 빈 DB를 기준으로 삼지 않았다.

- B=8, CI=CO=64, T=6400, R=7, d=3, pad=9, groups=1, bias 포함.
- 고정 CPU seed 63117로 FP16로 반올림한 동일 x/w/dY를 쓰고 FP32 native 참조를 계산했다.
- 사전 gate: y/dX/dW finite, NRMSE ≤ 0.002, max error / reference RMS ≤ 0.02.
- 3 warmup, 3 round × 10회. 아래는 GPU event 중앙값이다. wall과 재측정 중앙값도 JSON에 있다.
- 두 MIOpen 안전 변수는 모두 0, cudnn.benchmark=False, FAST 새 process.
- 명목 FLOP 2.9360128 GFLOP/방향, 텐서 이동량 13,164,544 bytes(12.55469 MiB, bias 128 bytes 제외).
- register 연산 기준 시간 0.060257 ms. streaming 이동 가정 0.030142 ms, cache copy 가정 0.016415 ms. 이 shape에서는 표시용 compute 기준이 더 크다.

| 방향 | 현재 FAST ms | 직접 후보 ms | 배속 | 후보 NRMSE | 후보 max / reference RMS | 현재 / 연산 기준 시간 | 후보 / 연산 기준 시간 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fwd | 0.817664 | 0.289629 | 2.8231× | 0.00029987 | 0.00374477 | 13.570 | 4.807 |
| dX | 0.491115 | 0.336340 | 1.4602× | 0.00020728 | 0.00153093 | 8.150 | 5.582 |
| dW | 4.082333 | 1.055429 | 3.8679× | 0.00020578 | 0.00087273 | 67.749 | 17.516 |

native도 세 방향 gate를 통과했다. 후보 fwd의 native 대비 max/RMSE는 모두 0이었다. signed-zero까지 별도로 비교한 byte equality 판정은 이 smoke에 넣지 않았으므로 이를 바이트 일치라고 확대하지 않는다. 후보 dX/dW의 native 대비 차이는 각각 NRMSE 0.00032571 / 0.00048198이며 gate 안이다.

후보와 기준 모두 먼저 kernel warmup을 끝낸 후 측정했다. 기준을 후보 뒤에서 다시 측정한 event 중앙값은 fwd 0.818672 ms, dX 0.491769 ms, dW 4.081155 ms로 앞 측정과 가깝다.

### 실제 kernel 증거

[direct_smoke/result.json](direct_smoke/result.json)에 전체 kernel 이름과 trace SHA를 남겼다. event 구간과 profile kernel 합계는 다른 측정이므로 더하지 않는다.

| 방향 | native 실제 kernel 구도 | 직접 후보 구도 |
|---|---|---|
| fwd | Im2d2Col_v2 8회 + Tensile Cijk_Ailk_Bljk_HHS…MT128x128x16… 8회 + bias 1회 | _fwd 1회, trace 0.277959 ms |
| dX | Tensile Cijk_Ailk_Bjlk_HHS… 8회 + Col2Im2dU 8회 | _dx 1회, trace 0.325079 ms |
| dW | Tensile Cijk_Alik_Bljk_HHS…MT128x256x16…SU32… 8회 3.565712 ms + im2col 8회 0.396192 ms + 초기화 | _dw_partial 1회 1.036079 ms + _dw_reduce 1회 0.004159 ms |

dW native trace에서 GEMM이 89.95%였다. 따라서 이 shape의 손해 전부를 im2col 복사 하나 때문이라고 설명하지 않는다. 배치별 8번 GEMM 호출과 작은 출력·긴 reduction의 라이브러리 선택 비용도 함께 줄었다.

모든 계산 kernel은 v_dot2c_f32_f16 정적128개를 포함했고 spill 0이었다. fwd/dX/dW-partial registers는 105/110/96, LDS는 각각4096 bytes다. dW reduction은 dot 연산이 필요 없는 FP32 합산이다. dW scratch는 3,670,016 bytes(3.5 MiB)이며 output을 포함한 PyTorch extra peak는3,727,360 bytes였다.

## 4. 재현

작업 디렉터리는 rdna2-ops다. 새 출력 이름을 사용한다. 운영 .config/miopen을 쓰기 대상으로 넘기지 않는다.

    MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0 \
    MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53=0 \
    ~/DevSpace/applio-lab/.venv-rocm10/bin/python \
      v08_hardware_calibration.py \
      --out results/applio_v08/reproduce_hardware

    MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0 \
    MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53=0 \
    ~/DevSpace/applio-lab/.venv-rocm10/bin/python \
      bench_v08_direct_conv.py \
      --case B8_C64_O64_T6400_k7_d3_g1 \
      --db results/applio_v08/direct_fast_db \
      --out results/applio_v08/reproduce_direct_smoke \
      --tile-count 1 --rounds 3 --reps 10 --profile \
      --label current_FAST_copy

GPU 작업은 위 명령도 하나씩, 다른 연구 job이 없을 때 실행한다. 내장 GpuAudit에서 다른 compute client가 관측되면 timing은 버린다.

## 5. 완료된 두 탐색 — 재실행하지 않고 결과 반영

원자료: [initial9](direct_initial9/result.json), [additional12](direct_additional12/result.json).
두 파일 모두 `completed=true`, `valid_for_timing=true`, foreign KFD client 0이다.
공통 kernel source SHA는 `6ef607af92b0ffd561a633093ff9bd62ae8428deb6f613aa2d38fca3a674b4e3`이며 재개 시 현재 `v08_direct_conv.py`와 일치했다. 선택 도구의 CPU 처리만 보완했고 직접 커널 본체는 바꾸지 않았다.

| 실행 | geometry | native 정확도 | 후보 정확도·dot2 | 현재 FAST 단독 대비 속도 임계 통과 |
|---|---:|---:|---:|---:|
| initial9 | 9 | 27/27방향 | 81/81 tile | 23/27방향 |
| additional12 | 12 | 36/36방향 | 108/108 tile | 33/36방향 |
| 합계 | 21 | 63/63방향 | 189/189 tile | 56/63방향 |

후보의 최댓값은 NRMSE **0.0003002242**, max/reference RMS **0.0038631833**로 사전 gate(0.002/0.02) 안이다.
위 속도 임계는 기존 선택 코드의 event 1.10× 이상, wall 1.05× 이상이다. 추가12의 33방향 중 6방향은 아래 C192 FP32 논리입력 사례여서 첫 잠정 manifest에서는 제외했다. 이후 §6의 별도 AMP/weight_norm 호출 경계 검증을 통과해 최종 연산 manifest에 포함됐다.

### 5.1 실제 FP16 논리입력이 관측된 18 geometry

다음 값은 해당 직접 Conv 실행 안의 **현재 FAST → 가장 빠른 정확한 직접 tile** GPU event 중앙값(ms)이다. fwd는 양쪽 모두 bias를 포함한다. dX/dW는 독립 방향 호출이며 전체 학습 시간과 별개다. 모든 shape는 B8, groups=1, stride=1, same padding이다.

| C=O / T / K / dilation | fwd ms | dX ms | dW ms |
|---|---:|---:|---:|
| 32 / 12800 / 3 / 1 | 0.116010 → 0.038802 | 0.079914 → 0.043308 | 1.927951 → 0.198149 |
| 32 / 12800 / 3 / 3 | 0.490478 → 0.038017 | 0.416890 → 0.044067 | 1.917327 → 0.199876 |
| 32 / 12800 / 3 / 5 | 0.489295 → 0.038410 | 0.415380 → 0.043897 | 1.916253 → 0.197967 |
| 32 / 12800 / 7 / 1 | 0.197432 → 0.073345 | 0.164571 → 0.095471 | 1.366311 → 0.375126 |
| 32 / 12800 / 11 / 1 | 0.243978 → 0.105408 | 0.208987 → 0.140797 | 1.425060 → 0.543205 |
| 32 / 12800 / 11 / 5 | 0.814735 → 0.105757 | 0.751485 → 0.139829 | 1.529987 → 0.542987 |
| 64 / 6400 / 3 / 1 | 0.152933 → 0.063649 | 0.119157 → 0.078933 | 1.446757 → 0.311627 |
| 64 / 6400 / 3 / 3 | 0.617616 → 0.063960 | 0.412476 → 0.078791 | 2.522214 → 0.309450 |
| 64 / 6400 / 3 / 5 | 0.619013 → 0.063156 | 0.416097 → 0.080683 | 2.527489 → 0.310223 |
| 64 / 6400 / 7 / 1 | 0.305067 → 0.132887 | 0.267939 → 0.177421 | 1.445261 → 0.682711 |
| 64 / 6400 / 7 / 3 | 0.820367 → 0.132059 | 0.492928 → 0.179814 | 4.077076 → 0.684107 |
| 64 / 6400 / 7 / 5 | 0.819080 → 0.133009 | 0.575584 → 0.179883 | 4.086877 → 0.687017 |
| 64 / 6400 / 11 / 1 | 0.407388 → 0.194468 | 0.369824 → 0.263431 | 1.445474 → 1.052673 |
| 64 / 6400 / 11 / 3 | 0.995491 → 0.196152 | 0.631596 → 0.267493 | 4.112185 → 1.054052 |
| 64 / 6400 / 11 / 5 | 1.072801 → 0.196083 | 0.775781 → 0.267300 | 4.109290 → 1.053791 |
| 128 / 3200 / 11 / 1 | 0.728750 → 0.376370 | 0.539704 → 0.514531 | 1.051792 → 2.098804 |
| 128 / 3200 / 11 / 3 | 1.006777 → 0.377226 | 0.690119 → 0.522695 | 1.393208 → 2.097970 |
| 128 / 3200 / 11 / 5 | 1.084938 → 0.379166 | 0.833341 → 0.521533 | 1.491381 → 2.093717 |

**C128/K11 dW는 모두 탈락**이다. d1/3/5의 배속은 각각 **0.501/0.664/0.712×**로 native보다 느리다. d1의 dX도 1.049×라 event 선택 임계에 못 미친다. 이 네 방향은 native로 남긴다. 따라서 이 18 shape의 54방향 중 잠정 채택은 **50방향**이다.

대표 개선은 C64/K7/d3의 fwd **6.212×**, dX **2.741×**, dW **5.960×**다. 이는 §3의 1 tile smoke 이후 3 tile 탐색에서 찾은 결과다. C32/K3/d1 dW는 **9.730×**다. 이 절의 수치는 모두 당시 현재 FAST 단독 탐색 이력이며, seeded FAST까지 포함한 최종 기준은 §6에 별도로 적었다.

### 5.2 C192→O384/K5: FP32 모듈 입력과 FP16 벤치 구분

실제 캡처의 T=74/224/326은 **모듈 입력·weight가 FP32**다. 벤치는 `X/W/dY`를 명시적으로 FP16으로 만들었으므로 다음 표는 **FP16 저장 입력에 대한 연산 탐색**이다.

| T (B8, d1, p2) | FP16 fwd ms | FP16 dX ms | FP16 dW ms | dW 배속 |
|---:|---:|---:|---:|---:|
| 74 | 0.486578 → 0.063576 | 0.129491 → 0.077486 | 0.130336 → 0.483374 | 0.270× |
| 224 | 0.155286 → 0.085378 | 0.166391 → 0.103998 | 0.164931 → 0.471379 | 0.350× |
| 326 | 0.534953 → 0.143707 | 0.308456 → 0.162355 | 0.262250 → 0.704028 | 0.372× |

모듈 경계의 FP32를 내부 MIOpen 연산 dtype으로 단정하지 않는다. 기존 [실제 학습 프로파일](native_profile_final/profile_summary.json)의 `forward_shapes`와 `kernel_registry`를 대조하면 T74/326에는 `k0032` HHS GEMM, T224에는 `k0111` gfx10 FP16 dot2 Winograd가 연결된다. 모두 Half bias add(`k0173`)와 FP16 copy(`k0203`)가 관측됐다. 즉 **FP32 논리입력에서 FP16 forward로 넘어가는 실제 AMP 경로의 흔적은 있다.**

이후 [호출 경계 GPU 검증](direct_contract_initial/result.json)에서 실제 FP32 input과 FP32 weight_norm 원본 parameter를 써서 native AMP와 wrapper를 비교했다. 캐스트와 FP32 gradient 귀환을 포함한 y/dX/원본 parameter gradient가 모두 gate를 통과해, 최종 연산 manifest에는 세 shape의 **fwd·dX만 포함**됐다. dW는 FP16 연산 자체에서 느려서 native로 남긴다. 캐스트·dispatch 비용까지 포함한 실제 학습 시간은 §7에 정리했다.

## 6. 최종 연산 선택과 AMP/weight_norm 호출 경계

CPU 선택 도구 `bench_v08_direct_conv.py --select`에 두 검사를 보완했다.

- 외부 solver 결과는 `status=completed`, 유효한 FAST, `dtype=fp16`이어야 비교한다. FP32 solver 시간을 FP16 key에 섞지 않는다.
- manifest의 학습 후보는 캡처된 FP16 입력을 기본으로 삼는다. FP32 입력은 동일 kernel SHA·shape·선택 tile의 AMP/weight_norm 경계 검증을 `--contract-evidence`로 받은 경우에만 연다. 미검증·미관측 입력은 사유와 함께 `decisions`에 남긴다.

[최종 연산 manifest](direct_selected_final/dispatch.json)는 **21 shape·56방향**이다. `contract_probe_only=false`, `status=operator-only; epoch/audio validation still required`이며 SHA256은 `22e41075581e8b2d969388244bf063adba59579942c4959ae892e2aeb4fcfa59`다.

방향별 기준은 두 직접 탐색의 native, current FAST 144 (`solver_baseline144/result.json`, 연구 폴더에만 있음), 합성 DB 최종 FAST replay (`solver_selected_final_replay/result.json`, 연구 폴더에만 있음), 기존 rank 5개, workspace를 보완한 rank 2개의 **총 11개 보고서에서 가장 빠른 정확한 native**다. 후보는 완료된 직접 탐색의 tile만 사용했다. 외부 solver forward 기준은 bias를 제외하지만 직접 forward는 bias를 포함하므로 이 비교는 직접 후보에 보수적이다.

| 선택 방향 | 선택 개수 | 최종 빠른-native 대비 event 배속 범위 | wall 배속 최솟값 |
|---|---:|---:|---:|
| fwd | 21 | 1.698–12.074× | 1.678× |
| dX | 20 | 1.313–9.444× | 1.316× |
| dW | 15 | 1.372–9.663× | 1.374× |

C128/K11의 dW 세 개와 d1 dX는 native로 남는다. C192 세 shape는 fwd/dX만 선택하고 dW는 native로 남는다. 최종 manifest의 모든 shape·선택 방향·tile은 아래 contract가 실제 시험한 설정과 정확히 일치했다.

### 최종 기준으로 다시 표시한 대표 수치

아래 시간도 연산 microbench의 GPU event 중앙값이다. 실제 FP32 입력 호출 경계의 수치 동등성은 별도의 contract로 확인했고, 전체 wrapper를 포함한 실제 epoch 결과는 §7에서 구분한다.

| shape / 방향 | 가장 빠른 정확한 native ms | 직접 후보 ms | 배속 |
|---|---:|---:|---:|
| C64/T6400/K7/d3 fwd | 0.788640 | 0.132059 | 5.972× |
| C64/T6400/K7/d3 dX | 0.492522 | 0.179814 | 2.739× |
| C64/T6400/K7/d3 dW | 3.984700 | 0.684107 | 5.825× |
| C32/T12800/K3/d1 dW | 1.906419 | 0.198149 | 9.621× |
| C192→O384/T74/K5 fwd | 0.438225 | 0.063576 | 6.893× |
| C192→O384/T74/K5 dX | 0.129491 | 0.077486 | 1.671× |
| C192→O384/T224/K5 fwd | 0.144954 | 0.085378 | 1.698× |
| C192→O384/T224/K5 dX | 0.166391 | 0.103998 | 1.600× |
| C192→O384/T326/K5 fwd | 0.479129 | 0.143707 | 3.334× |
| C192→O384/T326/K5 dX | 0.308456 | 0.162355 | 1.900× |

### 호출 경계 GPU 검증 완료

[v08_direct_contract.py](../../v08_direct_contract.py)의 [실행 결과](direct_contract_initial/result.json)는 `completed=true`, `valid_for_contract=true`, **21/21 case·105/105 수치 비교 통과**, foreign KFD client 0이다.

실제 캡처 geometry·입력 dtype을 쓰되 tensor 값은 고정 seed 63117의 합성 입력이다. FP32 weight_norm의 `original0`/`original1`·bias master parameter를 사용하고, native FP16 AMP와 wrapper의 y, dX, 원본 parameter gradient 세 개를 비교했다. 사전 gate는 NRMSE 0.002/max-reference-RMS 0.02다.

- 가장 큰 NRMSE: **0.000561579**, C64/K3/d3의 weight_norm `original0` gradient.
- 가장 큰 max/reference RMS: **0.007820156**, C64/K11/d1의 y.
- 모든 case에서 master parameter와 그 gradient FP32 유지, input/output/gradient dtype 일치, dispatch hit=1, context 종료 후 hook 복구를 확인했다.
- 최종 manifest·계약 검증·현재 직접 kernel source의 SHA가 일치했다.

이 도구의 `valid_for_timing=false`는 정확도 검증이라는 뜻이다. GPU 독점 audit은 별도로 통과했다. optimizer update·실제 학습 step·음질을 시험한 결과로 확대하지 않는다. 기존 189개 FP32-reference 연산 탐색은 재실행하지 않았다.

### 재현 명령과 이전 산출물

아래 CPU 명령은 최종 연산 선택을 재현한다. 완료된 GPU 측정을 다시 실행하지 않으며 새 출력 폴더를 쓴다.

```bash
python3 bench_v08_direct_conv.py --select \
  results/applio_v08/direct_initial9/result.json \
  results/applio_v08/direct_additional12/result.json \
  results/applio_v08/solver_baseline144/result.json \
  results/applio_v08/solver_selected_final_replay/result.json \
  results/applio_v08/solver_rank00/result.json \
  results/applio_v08/solver_rank01/result.json \
  results/applio_v08/solver_rank02/result.json \
  results/applio_v08/solver_rank03/result.json \
  results/applio_v08/solver_rank04/result.json \
  results/applio_v08/solver_workspace_rank00/result.json \
  results/applio_v08/solver_workspace_rank01/result.json \
  --contract-evidence results/applio_v08/direct_contract_initial/result.json \
  --out results/applio_v08/direct_selected_reproduce
```

[첫 잠정 manifest](direct_provisional_current/dispatch.json)의 18 shape·50방향과 [CPU 확인](direct_selection_cpu_check.json)은 C192를 검증 전 제외했던 이력으로 보존한다. [경계 검증용 manifest](direct_provisional_contract_probe/dispatch.json)는 `--contract-probe`로 만든 시험용이며, 실제 epoch에는 `direct_selected_final/dispatch.json`을 사용한다.

계약 검증 자체의 독립 재현이 필요할 때는 새 출력 폴더를 사용한다. 현재 다음 단계로 넘어가기 위해 다시 돌릴 필요는 없다.

```bash
~/DevSpace/applio-lab/.venv-rocm10/bin/python v08_direct_contract.py \
  --manifest results/applio_v08/direct_provisional_contract_probe/dispatch.json \
  --db results/applio_v08/direct_fast_db \
  --out results/applio_v08/direct_contract_reproduce
```

## 7. 실제 4 epoch 교대 — solver+직접 Conv

원자료: [교대 상태](epoch_series_initial_status.json), [A0](epoch_A0/result.json), [B_conv](epoch_B_conv/result.json), [A1](epoch_A1/result.json).
같은 세션에서 A0 → B_conv → A1 순서로 각각 4 epoch, 31 step/epoch, 총 124 step을 실행했다. 세 실행의 초기 G/D state SHA가 동일하다. A는 현재 운영 DB 복사본, B_conv는 선택 DB와 최종 직접 Conv manifest를 함께 사용했다. 이 비교에는 nonconv·compile이 들어 있지 않다.

세 실행 모두 `completed=true`, `valid_for_timing=true`, `valid_for_epoch_timing=true`이며 profiler·음성 변환을 사용하지 않았다.

| 실행 | 첫 epoch s | steady 중앙값 s (epoch 2–4) | 4 epoch 합 s | PyTorch peak allocated GiB | scale 감소 / 최종 scale |
|---|---:|---:|---:|---:|---:|
| A0 현재 운영 | 21.385693 | 15.096908 | 67.079081 | 4.161792 | 11 / 32 |
| B_conv 선택 solver+직접 Conv | 16.575041 | 13.314360 | 56.268797 | 4.165955 | 11 / 32 |
| A1 현재 운영 반복 | 18.631871 | 14.972521 | 63.604978 | 4.165892 | 11 / 32 |

B_conv는 A0 대비 **1.133882×**, A1 대비 **1.124539×**다. 두 native의 steady epoch 6개를 합친 중앙값은 **15.002798 s**이고, 이 기준에서 **1.126813× / 시간 11.254% 감소**다. A0와 A1의 steady 중앙값 차이는 0.824%였다. 첫 epoch에는 초회 준비 비용이 포함되므로 steady와 별도로 유지했다.

### 실제 dispatch·fallback·DB

- B_conv는 선택된 **21/21 shape를 총 5,088회** 실제 dispatch했다. C192/T74·T224·T326도 각각 **112·112·28회** 호출됐다.
- 실제 읽은 manifest SHA는 §6의 최종 manifest와 일치한다. B의 FindDb 시작 SHA는 `fa2af4e846b0b565e23f058ae5d011cba85fe47a757a399a3ffa618563fbfdd5`로 선택 DB seed와 같고, 실행 전후 DB inventory도 같다.
- 선택 계약과 wrapper 소스상 C128/K11/d1의 dX·dW, d3/d5의 dW, C192 세 shape의 dW는 native를 사용한다. manifest에 없는 geometry도 원래 경로를 유지한다. **fallback 호출수를 따로 계측하지 않았으므로 전체 native fallback 횟수는 미측정**이다.
- 이 세 실행 각각에서 KFD foreign client 0, 보호 파일 **16+987개 변경 0**, 연구 사본의 보호 SHA 일치를 확인했다.

### loss·AMP 상태

124 step의 `loss_d`, `loss_g`, mel, fm, kl, adv는 **세 실행 모두 finite**였다. gradient norm의 nonfinite 기록은 세 실행 모두 D 5회·G 11회이고 scale 감소 합도 모두 11회다. 감소 시점은 A0 `[9,1,0,1]`, B `[8,3,0,0]`, A1 `[11,0,0,0]`으로 다르다. 이를 "overflow가 없었다"고 쓰지 않는다.

아래는 전체 124 step의 loss 중앙값이다. 상대 차는 B를 A0/A1 두 중앙값의 평균과 비교한 값이다.

| 지표 | A0 | A1 | B_conv | B 상대 차 |
|---|---:|---:|---:|---:|
| mel | 19.259348 | 19.275813 | 19.293152 | +0.133% |
| fm | 11.589844 | 11.691406 | 11.632813 | −0.067% |
| kl | 1.967313 | 1.986328 | 1.987096 | +0.520% |
| adv | 3.256634 | 3.248070 | 3.222834 | −0.908% |
| loss_d | 4.185529 | 4.177715 | 4.238540 | +1.361% |
| loss_g | 36.802595 | 37.000690 | 37.095711 | +0.526% |

마지막 25%는 이 짧은 실행에서는 마지막 1 epoch뿐이다. 해당 구간 B의 mel/fm/kl/adv 중앙값은 각각 **18.893467 / 11.593750 / 1.523440 / 3.248934**다. A0/A1도 최종 state SHA가 달랐고, B가 두 native의 범위를 넘는 loss 성분도 있다. **이번에 확인한 것은 실제 짧은 학습의 속도 개선과 실행 상태이며, 100 epoch 품질 동등성은 아직 판정하지 않았다.**

## 8. 다음 재개 지점

1. 진행 중인 nonconv 통합 후보의 실제 epoch 결과를 확인해 Conv 단독 이득과 추가 이득을 구분한다.
2. 100 epoch 시작 직전에 실제 측정 기반 예상 시간을 알린 뒤 A/A′/B를 수행한다.
3. 세 모델의 같은 입력에 대한 익명 음성 비교와 사용자 청취 판정을 진행한다.

100 epoch 전체 절감시간과 RTX 5070과의 차이 중 고정 하드웨어 비율은 현재의 4 epoch 결과만으로 확정하지 않는다.
