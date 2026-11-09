# Applio RVC 학습 해체·최적화 v0.8

> 연산·학습·음성 변환·최종 보호 확인 완료. 최종 채택은 사용자 지각 판정 대기다.
> 현재(2026-11-08 KST): 최종 100 epoch A/A′/B의 process 전체는 1540.113550/1544.650001/1320.001536초다. B는 22분 0.002초, 현재 native 대비 3분 40.11초–3분 44.65초(14.29–14.54%) 단축했다. 실제 solver 432/432방향 확인, 익명 12 WAV와 청취 HTML 제작·브라우저 재생 확인, 보호 1003파일 변경 0까지 완료했다.

## 1. 목표와 같은 학습의 정의

대상은 voice_model_A, 40 kHz HiFi-GAN, KLM49 HFG pretrain, FP16 AMP, 배치 8이다. 가중치와 AdamW 상태는 FP32이며 손실·학습률·segment 길이·데이터·epoch 수는 유지한다.
실제 filelist는 233행(231조각 외 추가 항목 포함), 버킷 sampler는 248 samples/epoch를 공급해 31 step/epoch다.
현재 Linux trainer의 `cudnn.benchmark`는 **False**다. 사용자 지시문의 True와 달리 실제 소스는 Windows에서만 True를 설정하며 실행 중 값도 확인했다.

작업 사본은 applio_copy (`applio_copy/`, 연구 폴더에만 있음)다. 소스·학습 입력·KLM G/D pretrain 1,000개 파일, 1,430,099,596 bytes를 SHA 대조해 복사했다.
[보호 manifest](protected_before.json)의 기존 16개와 대상 데이터/기존 체크포인트/소스 987개는 [작업 종료 확인](final_protection_check.json)에서 1003/1003개 SHA 일치, 변경·누락·오류 0개다.
현재 user FindDb는 원본을 읽어서 연구 폴더로 복사했다. 실제 경로와 SHA는 [DB snapshot](baseline_db_snapshot.json)에 있다. 빈 DB FAST를 현재 운영 기준으로 대체하지 않는다.

## 2. 실행 전 고정한 판정

[PREDECLARED_GATES.json](PREDECLARED_GATES.json)을 첫 GPU 실험 전에 저장했다.

- 연산: 고정 seed, 동일 FP16 저장 입력의 FP32 참조, y/dX/dW finite, NRMSE ≤ 0.002, max error/reference RMS ≤ 0.02.
- epoch: 같은 세션에서 기준/후보 교대, steady epoch 중앙값, 초회 비용, scale 감소·NaN·loss 변화·SHA 기록.
- 최종: 100 epoch A(현재), A′(현재 반복), B(후보). 같은 기존 입력 4문장의 익명 A/B/C 청취. B만의 이상이 A/A′ 흔들림을 넘어서는지 사용자 판정.
- 마지막 25% mel/fm/kl/adv 및 출력 포화 지표는 참고 지표다. 짧은 연산 통과를 음질 승인으로 취급하지 않는다.

모든 연구 GPU 프로세스는 `gpu_guard_v06.GpuAudit`와 다음 두 변수를 쓴다.

```bash
MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0
MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53=0
```

## 3. 최신 실제 학습 분해 — A 등급

원자료: [프로파일 요약](native_profile_final/profile_summary.json), [검수](native_profile_final/profile_review.json), trace (`native_profile_final/profile_trace.json`, 연구 폴더에만 있음).
40–44 step, 5 step의 GPU operator self 합은 2,065.797 ms다.

| 순위 | 연산 | GPU self / 5 step | 비중 |
|---:|---|---:|---:|
| 1 | `aten::convolution_backward` | 1150.359 ms | 55.686% |
| 2 | `aten::miopen_convolution` | 508.710 ms | 24.625% |
| 3 | `aten::copy_` | 52.899 ms | 2.561% |
| 4 | `aten::add_` | 40.948 ms | 1.982% |
| 5 | `aten::_foreach_addcdiv_` | 25.659 ms | 1.242% |
| 6 | `aten::linalg_vector_norm` | 24.721 ms | 1.197% |
| 7 | `aten::_weight_norm_interface` | 24.386 ms | 1.180% |
| 8 | `aten::sum` | 23.630 ms | 1.144% |
| 9 | `aten::_weight_norm_interface_backward` | 20.157 ms | 0.976% |
| 10 | `aten::_foreach_lerp_` | 19.501 ms | 0.944% |

Conv forward+backward는 **80.311%**다. 실제 kernel만 모은 합은 2,057.529 ms이며 Winograd 두 커널 26.811%, im2col+col2im 14.067%, AdamW 전체 5.352%다.
Operator와 kernel은 같은 일을 다른 계층에서 센 것이므로 더하지 않는다. GPU annotation 2,065개도 실제 kernel 합계에서 제외했다.

프로파일 벽시간 2.600829 s 중 kernel/copy/memset union은 2.065797 s, 관측 잔여 구간은 0.535032 s다.
이는 profiler overhead를 포함한 관측 상한이며 평상시 idle 비율의 확정치가 아니다.
DataLoader 5회 합은 168.77 µs다. hipMemcpyWithStream의 CPU 대기 1.248975 s는 GPU 실행과 겹치므로 별도의 제거 가능 비용으로 합산하지 않는다.
프로파일 export가 epoch 타이머에 포함되므로 이 실행의 epoch seconds는 성능 기준선으로 사용하지 않는다.

## 4. Shape·방향 기준선과 하드웨어 여지 — A 등급

[전체 432방향 CSV](solver_roofline/shape_direction.csv), [roofline 근거](solver_roofline/ROOFLINE_KO.md), 실제 FAST 원자료 (`solver_baseline144/result.json`, 연구 폴더에만 있음), [하드웨어 교정](hardware_calibration/result.json).

기존 baseline 상세 로그가 꺼져 있던 누락은 [별도 FAST metadata probe](solver_roofline/baseline_metadata_probe144/result.json)로 보완했다. 같은 144개 FP16 descriptor·seed·안전 변수·runtime·baseline DB 사본에서 각 방향을 1회씩 호출했다. 432/432방향의 BEGIN–READY 안 최종 Chosen Algorithm과 실제 API 호출을 대응했고, 모호/누락 0개다. 새 FP32 참조·warmup·timing·profile은 실행하지 않았다. probe는 valid_for_timing=false이며 기존 유효 CSV 시간·정확도 값은 변경 0개다. [행별 solver 근거](solver_roofline/solver_attribution.json), [설명](../../v08_solver_notes_ko.md) §9.

| 관측 solver ID | 방향 수 |
|---|---:|
| ConvBinWinogradRxSf2x3g1 | 69 |
| ConvBinWinogradRxSf3x2 | 37 |
| ConvBinWinogradRxSf2x3 | 20 |
| GemmFwdRest | 73 |
| GemmBwdRest | 75 |
| GemmWrwUniversal | 83 |
| GemmFwd1x1_0_1 | 28 |
| GemmBwd1x1_stride1 | 29 |
| GemmWrw1x1_stride1 | 18 |

합계 432방향이며 Winograd 126/GEMM 306이다. 요청한 singleton DB나 kernel 이름만으로 실제 ID를 추정하지 않았다. 관측은 동일 조건의 별도 metadata 재생에서 얻었으며, 과거 timing 호출 당시의 solver 로그를 소급 복원했다고 주장하지 않는다.

실제 학습에서 관측한 144개 geometry를 FP16으로 정규화한 descriptor × fwd/dX/dW = 432개가 모두 정확도 gate를 통과했다.
모듈 경계의 FP32 입력도 geometry 목록에는 포함된다. 이 표는 FP16 microbench이며 각 모듈의 실제 내부 kernel dtype을 모두 입증한 표는 아니다. 특히 C192→O384/K5 세 shape의 FP16 결과를 실제 FP32 입력 경로의 검증으로 사용하지 않는다.
현재 DB는 d1의 일부에서 이미 Winograd를 선택한다. 예전 빈 DB와의 배속은 이번 개선으로 다시 주장하지 않는다.

register dot2 교정은 **48.725 TFLOP/s**, 512 MiB working-set copy는 **436.745 GB/s**, 16 MiB warm copy는 **801.970 GB/s**였다.
이는 낙관적 비교 기준이다. 직접 Conv의 dense FLOP 가정을 연산량 자체가 다른 Winograd의 엄밀 하한으로 쓰지 않는다.
캐시 재사용, im2col 추가 traffic, LDS, occupancy, launch 비용을 별도로 본다.

현재 병목 상위는 C64/T6400/k7·11/d3·5 dW(약 4.0–4.13 ms)다.
C64/k7/d3의 명목 2.936 GFLOP는 register dot2 기준 0.06026 ms다. 현재 dW 약 4.009 ms와의 66.5배 차이를 그대로 달성 가능한 배속이라고 해석하지 않는다.

## 5. 직접 Conv1d — 21 shape·56방향 선택, 호출 경계 통과

[직접 커널 근거](DIRECT_CONV_V08_KO.md), [초기 스모크](direct_smoke/result.json).

C64/T6400/k7/d3의 초기 1 tile 후보는 current FAST 대비 fwd 2.823배, dX 1.460배, dW 3.868배였고 3방향 strict gate를 통과했다.
v_dot2c_f32_f16 ISA를 확인했다. native dW의 주요 비용은 im2col 이후 Tensile GEMM이며 직접 경로는 FP32 partial과 reduction 두 kernel로 처리한다.
이는 초기 단일-shape 결과다. seeded DB 중 더 빠르고 정확한 경로, 여러 shape, 실제 학습을 통과한 방향만 dispatch에 넣는다.
[direct_initial9/result.json](direct_initial9/result.json)과 [direct_additional12/result.json](direct_additional12/result.json)은 모두 completed=true, valid_for_timing=true다. 2026-11-08 재개 때 재실행하지 않고 원자료를 회수했다.
전체 21 geometry × 3방향 × 3 tile = 189/189 후보가 정확도·dot2 검사를 통과했고 native도 63/63 방향 통과했다.
현재 FAST만 상대하면 56/63 방향이 선택 임계(event ≥1.10×, wall ≥1.05×)를 넘는다. 초기에는 실제 FP16 입력이 관측된 18 geometry의 50방향만 잠정 후보로 두었다. 이후 현재 FAST·7개 rank FAST·합성 DB FAST 중 방향별 가장 빠르고 정확한 기준을 포함해 선택했고 최종은 **21 shape·56방향(fwd 21, dX 20, dW 15)**이다. [최종 manifest](direct_selected_final/dispatch.json).
C128/K11 dW는 d1/3/5 모두 native보다 느려(배속 0.501/0.664/0.712×) 대체하지 않는다.
C192→O384/K5 세 shape는 실제 모듈 입력이 FP32인데 벤치는 명시적으로 FP16을 사용했다. 기존 실제 trace에서는 내부 FP16 Conv가 확인됐다. 이 둘을 구분한 뒤 [호출 경계 검증](direct_contract_initial/result.json)에서 FP32 master weight_norm(original0/original1)·bias·실제 입력 dtype의 native AMP와 wrapper를 비교했다.
21/21 shape에서 y/dX/파라미터 gradient, FP32 master/gradient 유지, 입력 gradient dtype, dispatch hit·복구가 통과했다. 최악 NRMSE 0.000561579, max/reference RMS 0.007820155, KFD foreign 0이다. 이 근거로 C192 fwd/dX를 포함했고 느린 dW는 계속 native로 남겼다. 커널 본체 SHA는 기존 탐색과 동일하다.

## 6. Solver 선택 — 성공·실패를 분리

[방향별 선택 도구](../../v08_solver_select.py), [NORMAL 47 비교](v08_solver_seed47_comparison.json).

안전 변수를 지정한 NORMAL 47 shape × 3방향은 141/141 정확도 통과했다.
하지만 일부 dilated GEMM은 workspace가 필요함에도 ptr 0 / size 0으로 전달돼 탐색에서 제외됐다.
따라서 이 NORMAL 실행을 모든 후보의 공정한 최선 탐색이라고 주장하지 않는다.
FP16 DB는 69개 방향 key만 생성됐고 dilated 24 shape × 3방향은 누락됐다. 실패 경고와 원자료를 보존했다.
2026-11-08 rank_00~04의 기존 DB를 새 FAST 프로세스로 직렬 replay했다. 각 23/18/12/12/9 shape, 69/54/36/36/27 방향이 모두 정확도 gate를 통과했고 5회 모두 valid_for_timing=true, foreign KFD client 0이다.
원자료: [rank00](solver_rank00/result.json), [rank01](solver_rank01/result.json), [rank02](solver_rank02/result.json), [rank03](solver_rank03/result.json), [rank04](solver_rank04/result.json).
변경된 실제 kernel과 event/wall ≥1.05×를 모두 요구한 merge에서 6방향이 후보로 남았다. [선택 기록](solver_selected_rank47_db/selection_manifest.json). C128/K7 dX, C256/K7·11 dW, period Conv2D 한 dW, ConvTranspose1D 두 dW다. 후속 합성 DB FAST replay는 이 절 아래에, 실제 epoch 결과는 §7에 분리했다.
workspace 0은 24개 dilated shape의 fwd/dX/dW에서 필요 2.46–9.01 MB에 비해 ptr/size=0이 전달된 문제였다. 설치 MIOpen과 같은 commit `8d1ae90e`의 `GetWorkSpaceSizeGEMM`이 dilation >1에서 0을 반환하는 원인을 확인했다.
연구 전용 [v08_miopen_workspace.c](../../v08_miopen_workspace.c)는 `APPLIO_RDNA2_FIX_FIND_WORKSPACE=1` + NORMAL에서만 workspace query 세 개를 Range max로 보완한다. 운영 라이브러리와 기존 benchmark 코드는 수정하지 않았다. FAST replay에서는 preload를 제거한다.
[smoke](solver_workspace_smoke1/result.json)에서 FP16 workspace 0→5,734,400 bytes, 실제 Find 전달값 일치, 정확도 3/3을 확인했다. [나머지23](solver_workspace_seed23/result.json)까지 총 72/72 정확도 통과, KFD foreign 0, workspace 부족 경고 0이다. 누락 FP16 72 key가 모두 생겼고 총 90 solver entry를 확보했다. [seed 요약](solver_workspace_seed_summary.json).
공식 소스 7개를 총 621,004 bytes 받았으며 파일명·출처·SHA는 source manifest (`source/miopen_workspace/source_manifest.json`, 연구 폴더에만 있음)에 있다. 추가 패키지 설치는 없다.
새 solver_workspace_rank00/01 FAST replay는 72/72 + 54/54 정확도 통과, audit clean이다. 기존 5개 rank와 합친 최종 선택은 6방향이며 dilated 추가 채택은 0개다. NORMAL seed 시간은 최종 속도 근거로 쓰지 않았다.
최종 합성 FAST replay (`solver_selected_final_replay/result.json`, 연구 폴더에만 있음)는 **144 geometry·432/432 방향 정확도 통과**, valid_for_timing=true다. 선택 6방향의 event 배속은 1.0746/1.1073/1.1913/1.3211/1.2815/1.5295×이며 wall도 모두 1.05×를 넘었다. DB singleton entry와 rank 실행의 실제 kernel 구도도 유지됐다.

## 7. Conv 외 후보 — GPU 판정 완료

기본 AdamW는 이미 foreach 경로다. fused AdamW, device/foreach grad norm, 동일 인덱스 gather, 한 D forward 범위의 weight_norm cache, epoch 간 allocator cache 유지가 opt-in 후보다.
[nonconv 코드](../../v08_nonconv_opt.py)와 [CPU/소스 근거](NONCONV_V08_NOTES_KO.md)를 만들었다.
leaky_relu·residual add를 Conv 입력/출력 단계에 합칠 가능성과 필요한 backward·반올림 경계는 같은 문서 §11에 검토했다. 현재 직접 Conv는 이 주변 연산을 포함하지 않는다. 추가 fusion의 구현·수치·성능은 미검증이며 이번 후보 이득에 포함하지 않는다.
실제 G/D optimizer update 직전의 FP32 parameter·gradient·moment snapshot으로 [nonconv_gpu_actual.json](nonconv_gpu_actual.json)을 실행했다. valid_for_gpu_timing=true, foreign KFD client 0이다.

| 후보 | GPU 판정 | 실제 epoch 후보 |
|---|---|---|
| fused AdamW G/D | 2.527×/2.981× 빠르지만 per-tensor update/moment 오차 gate 실패; overflow skip은 통과 | 제외 |
| single AdamW G | gate 통과, native 대비 0.454×로 느림 | native 유지 |
| single AdamW D | gate 통과, native 대비 1.423×; 3 trial 11.13–11.18 ms | D만 single 후보 |
| gather slicing | 4 shape × FP16/FP32의 y/dX 8/8 바이트 일치, wall 10.20–14.05× | 후보 |
| foreach grad norm | 오차 6.34e-8, 0.214707→0.041388 ms | 로깅 norm 후보 |
| D weight_norm cache | D native 참조부터 일부 gradient nonfinite라 전체 gate 실패, G 경로는 바이트 일치지만 1.008× | 제외 |
| epoch empty_cache 생략 | 산술 변경 없음, epoch당 2회 호출을 보존 계수 | 실제 epoch에서 메모리/시간 판정 |

fused의 대표 실패는 G 작은 tensor update NRMSE 0.0030348, D 0.0036565로 기준 0.002 초과다. 기준을 완화하지 않았다.
D weight_norm 첫 scale 128 실험은 NaN 지표를 JSON으로 저장하지 못했다. 실패 로그를 보존하고 nonfinite 개수와 null 수치를 기록하도록 계측을 수정했다. [scale 1 복구 검사](weight_norm_gpu_actual_scale1.json)는 GPU audit 유효하나 D native/cached 양쪽의 같은 9개 tensor에 nonfinite가 있어 채택하지 않았다. 이 입력에서 WN cache가 원인이라고 단정하지 않는다.
연구 opt-in에 APPLIO_RDNA2_ADAMW_G/D를 추가해 기존 global mode를 기본 상속하면서 G native·D single을 각각 선택할 수 있다. optimizer별 실제 mode/flags를 실행 결과에 기록한다.
torch.compile은 eager 개선과 별도로 초회 비용을 포함해 판정했고, 이번 조합에서는 추가 이득이 없어 최종 후보에서 제외했다.

### 7.1 실제 epoch 교대 비교 — 완료

[실행 계획](epoch_series_initial.json), [완료 상태](epoch_series_initial_status.json). 각 실행은 새 프로세스·새 연구 DB 사본에서 같은 초기 G/D SHA, batch 8, 31 step/epoch, 4 epoch로 수행했다. 프로파일러는 끄고 현재 운영 DB를 A 기준으로 사용했다.

| 실행 | 구성 | 첫 epoch | steady 중앙값 | 4 epoch 합 | scale 감소 | loss NaN/Inf |
|---|---|---:|---:|---:|---:|---:|
| A0 | 현재 native | 21.385693 s | 15.096908 s | 67.079081 s | 11 | 0 |
| B_conv | 최종 solver DB + 직접 Conv | 16.575041 s | 13.314360 s | 56.268797 s | 11 | 0 |
| A1 | 현재 native 반복 | 18.631871 s | 14.972521 s | 63.604978 s | 11 | 0 |
| B_full | Conv + 검증된 nonconv 묶음 | 16.536212 s | 12.570070 s | 55.399439 s | 10 | 0 |
| A2 | 현재 native 반복 | 18.165280 s | 14.922928 s | 62.939109 s | 11 | 0 |

Conv 후보는 앞뒤 A steady 대비 1.1245–1.1339×다. 실제 21/21 dispatch shape가 총 5,088회 호출됐고 C192 세 shape도 포함됐다. 두 초기 G/D state SHA는 전 실행 동일하다. 각 실행의 audit clean·보호 16+987 변경 0·완전한 124 step·finite loss를 확인했다. 짧은 epoch 비교는 100 epoch 음질 판정과 별개다.

[epoch_series_full.json](epoch_series_full.json)의 B_full→A2도 완료했다. B_full은 Conv 후보보다 steady 시간이 5.590% 줄었고, 전후 native A1/A2 대비 약 1.19×(시간 약 16% 단축)다. 첫 epoch를 포함한 4 epoch 합은 B_conv 대비 1.545% 줄었다. 짧은 steady 배속을 시작 비용 포함 전체 시간으로 대신 쓰지 않는다.

B_full의 실제 optimizer G native flags는 foreach/fused=null/null, D single은 false/false다. gather slicing·foreach norm 활성, WN cache OFF, empty_cache 생략 8회를 기록했다. B_conv와 초기 state·manifest·dispatch 5088회가 같고, 124 step의 6개 loss는 전부 finite다. scale 감소 11→10회·최종 scale 32→64로 학습 경로 차이가 있으므로 품질 동등성을 단정하지 않는다. peak allocated는 4,474,342,400 B(4.167 GiB), B_conv 대비 +1,182,720 B다. 이 eager 비교에는 reserved peak가 없으며 compile/100 epoch부터 이를 함께 기록한다. 기존 하네스는 source 보존 (`source/epoch_harness_versions/f7c86386777782a1d76297fb468f72d039b51ae574fa788054fdbbb9ab0d546f.py`, 연구 폴더에만 있음)에 남겼다.

[epoch_series_compile.json](epoch_series_compile.json)은 B_full과 같은 후보에 decoder만 compile한 별도 4 epoch 계획이었다. native RNG를 유지하고, GPU guard에 별도 compiler GPU context가 섞이지 않도록 compile_threads=1을 사용했다. 초회 비용·graph break/recompile·steady·100 epoch 예상 합으로 판정했고, 아래 복구 결과까지 확인한 뒤 최종 후보에서 제외했다.
첫 compile은 학습 epoch 0에서 Dynamo FakeTensor의 Half input/Float bias 오류로 실패했다. C128/K11 native fallback을 추적하는 AMP 경계에서 발생했으며 eager 학습과 직접 커널 수치 검증의 실패로 취급하지 않는다. 실패 result/log와 당시 하네스 SHA 13cc7faf…를 보존했다. [epoch_series_compile_boundary.json](epoch_series_compile_boundary.json)은 하네스의 명시적 `--compile-eager-conv`로 검증된 Conv1d dispatch 전체를 `torch.compiler.disable(recursive=True)` 경계에 두고 다른 경로의 compile을 허용하는 1회 복구였다. 커널 본체/manifest/운영 소스는 변경하지 않았다. WN getter와 ConvTranspose는 이 eager 경계 밖이다.

복구 실행은 4 epoch를 유효하게 완료했지만 decoder/ResBlock loop의 graph break와 frame eager fallback을 관측했다. decoder 전체 compile 성공이나 compile 전무로 단정하지 않는다. 첫 18.608218 s, steady 12.583505 s, 합 56.760980 s로 eager B_full(16.536212/12.570070/55.399439 s)보다 빠르지 않았다. `first+99×steady` 예상 합도 1264.375 s 대 1260.973 s다. 따라서 최종 B는 compile 없는 eager 조합이다.

### 7.2 최종 100 epoch A/A′/B — 학습·export 완료

[고정 계획·선택 근거·source SHA](epoch_series_100.json), [실행 상태](epoch_series_100_status.json). A=`final100_A`, A′=`final100_A_prime`, B=`final100_B`이며 각 새 모델/DB 경로를 사용한다. A/A′는 현재 운영 FindDb 사본, B는 최종 선택 DB와 21 shape/56방향 dispatch + G native/D single + gather + foreach norm + allocator cache 유지다. compile/fused/WN cache는 제외한다.

각 100 epoch·3100 step·batch 8이며 모든 arm은 `--save-every 100 --save-only-latest --export-final --no-reference-audio`를 동일하게 사용한다. native trainer의 100번째 reference inference·checkpoint 저장을 유지하고, 추론용 `model_final.pth`와 FP32 `generator_full.pth`를 추가 보존한다. 각 epoch의 allocated/reserved peak를 함께 기록한다.

시작 직전 사용자에게 A 약 26분/A′ 약 26분/B 약 22분, 합계 약 74분, 저장·변환 포함 75–85분 예상을 안내했다. 짧은 실행으로 계산한 process 포함 예측은 1534.197/1534.197/1294.638 s(총 4363.032 s)였고 최종 저장 비용은 별도 여유로 설명했다. 아래 표는 완료 실측이다. 청취는 기존 4문장 × 3모델의 익명 12 WAV와 LISTEN_BLIND.html로 제공하며 사용자 판정 전에는 채택 승인으로 표시하지 않는다.

세 arm의 실제 완료 기록은 다음과 같다.

| 실행 | 상태 | 첫 epoch | steady 중앙값 | 100 epoch 합 | 외부 process 전체 | scale 감소 | loss NaN/Inf |
|---|---|---:|---:|---:|---:|---:|---:|
| A | 완료·valid | 18.446767 s | 14.983997 s | 1504.043105 s | 1540.113550 s (25분 40초) | 12 | 0 |
| A′ | 완료·valid | 18.913335 s | 14.978252 s | 1509.153059 s | 1544.650001 s (25분 45초) | 11 | 0 |
| B | 완료·valid | 16.962810 s | 12.688541 s | 1284.431671 s | 1320.001536 s (22분 0.002초) | 12 | 0 |

A는 2026-11-07 19:11:54 UTC, A′는 19:37:38 UTC, B는 19:59:38 UTC에 exit 0/valid로 종료했다. coordinator(Core 17913)도 exit 0이며 [최종 집계](final_training_summary.json)는 completed/training_checks_pass=true, issues/pending=[]다. 각 100 epoch·3100 step·마지막 25 epoch 775 step·동일 initial G/D SHA·고정 source SHA·실제 후보 옵션을 확인했다. 각 100번째 epoch 저장과 별도 inference model export를 전체 process 시간에 포함한다.

B는 A/A′ 대비 외부 전체 시간을 220.112–224.648초(3분 40.11초–3분 44.65초), 14.292–14.544% 줄였다. steady 감소는 15.287–15.319%이고 배속은 1.1805–1.1809×다. 외부 전체 배속은 1.1668–1.1702×로 별도 기록한다. A′−A의 외부 전체 차이는 +0.295%, steady 차이는 −0.038%였다. 세 process 합계는 4404.765087초(73분 24.77초)다.

마지막 25 epoch(76–100)의 775 step을 모은 중앙값은 다음과 같다. epoch별 중앙값을 다시 평균하거나 component 중앙값을 더해 total loss로 쓰지 않았다.

| Loss | A | A′ | B | A′−A 상대차 | B−A 상대차 |
|---|---:|---:|---:|---:|---:|
| mel | 16.372444 | 16.234068 | 16.382723 | −0.8452% | +0.0628% |
| fm | 11.625000 | 11.578125 | 11.578125 | −0.4032% | −0.4032% |
| kl | 0.705776 | 0.709181 | 0.702806 | +0.4825% | −0.4208% |
| adv | 3.355348 | 3.358047 | 3.357328 | +0.0805% | +0.0590% |
| generator total | 32.043270 | 31.968176 | 32.111141 | −0.2344% | +0.2118% |
| discriminator total | 3.843299 | 3.828082 | 3.827470 | −0.3959% | −0.4119% |

주요 4 loss의 B−A 절대 상대차는 이 A/A′ 한 쌍에서 관측한 차이 이하였다. 이는 통계적 동등성이나 음질 합격을 증명하지 않는다. 6종 loss는 모든 arm의 각 3100 step에서 finite지만, AMP의 nonfinite gradient step은 D/G가 A=5/12, A′=5/11, B=5/12이며 scale 감소 12/11/12회도 기록했다. loss NaN/Inf 0을 「overflow 0」으로 바꾸어 해석하지 않는다.

최대 allocated/reserved는 A=4.174511/4.335938 GiB, A′=4.177600/4.324219 GiB, B=4.172079/4.341797 GiB다. GPU guard는 6153/6172/5271 sample에서 foreign/error 각 0, 각 실행 종료 시 운영 보호 1003파일 변경 0이다. B의 실제 direct dispatch는 21/21 shape·127452회이며, C192 세 shape도 T74=2800/T224=2800/T326=952회 호출됐다. 실제 옵션은 G native/D single·gather·foreach norm·empty_cache 생략 200회, WN cache/compile OFF로 계획과 같았다.

### 7.3 익명 음성 — 기술 검증 완료, 사용자 청취 통과

청취 페이지 (`audio_eval_final/LISTEN_BLIND.html`, 연구 폴더에만 있음), [공개 manifest](audio_eval_final/blind_manifest.json), [native 변환 결과](audio_eval_final/result.json), [브라우저 확인](audio_eval_final/browser_qa.json). Core 13505 exit 0, valid_for_listening=true이며 quality_judgment는 pending_user다. 공개 A/B/C 라벨은 학습 arm A/A′/B와 별도이며, 같은 글자는 네 문장에서 같은 모델이다. 대응표는 공개하지 않는다.

각 모델은 epoch 100/step 3100, 실제 inference model의 457/457 key를 정확히 1회 load했고 missing/unexpected=[]다. 같은 기존 4문장·pitch 2·RMVPE·index_rate 0.3·volume_envelope 0.3·동일 문장별 seed로 native FP32 추론했다. 학습 후보 패치는 추론에 적용하지 않았다. KLM-SVM/RMVPE/index/입력은 기존 로컬 파일 복사본을 사용했고 새 다운로드 0이다.

12 WAV는 40 kHz mono PCM16, 문장별 4.86/4.30/5.66/5.38초로 세 arm의 길이가 같다. 12개 raw와 12개 저장 WAV 모두 finite다. raw는 decoder tanh 직후·native padding 제거 전 출력이며 각 WAV보다 2초 길다. raw abs ≥ 1.0 비율은 모두 0이다.

| 연구 arm | raw peak 범위 | WAV peak 범위 | peak97 구간 합계 raw/WAV | 최장 구간 raw/WAV |
|---|---:|---:|---:|---:|
| A | 0.779565–0.912154 | 0.467560–0.638611 | 7 / 4 | 0.300 / 0.300 ms |
| A′ | 0.808241–0.880915 | 0.437866–0.615662 | 8 / 4 | 0.275 / 0.275 ms |
| B | 0.791130–0.918669 | 0.420532–0.641846 | 6 / 7 | 0.275 / 0.275 ms |

peak97은 global peak 97% 이상인 값이 6 sample 이상 이어지는 구간이며 clipping이나 음질 합격 gate가 아니다. 비정렬 파형의 B 관련 RMSE는 A/A′ 반복 RMSE의 1.570–1.866배였다. 위 loss 표와 포화 참고값만으로 같은 파형·같은 음질이라고 확정하지 않으며, 원 지시대로 사용자 지각 판정이 남는다.

공개 HTML에 12 WAV가 모두 내장돼 있고 총 6474097 B, WAV 합은 4848528 B다. 공개 manifest·내장 WAV·익명 파일 SHA가 일치하고 실제 arm/model 경로가 공개물에 없다. 브라우저에서 4문장·12 media·readyState 4/오류 0, 무음 재생 진행·다른 음성 재생 시 기존 음성 정지·가로 넘침 없음을 확인했다. 확인 후 모두 정지·처음 위치·음소거 해제, 선택 8개는 undecided·메모는 빈 상태로 유지했다. 사용자 판정은 입력하지 않았다.

### 7.4 작업 종료 보호 확인

[final_protection_check.json](final_protection_check.json): 2026-11-07T20:01:54.341721 UTC에 기존 16+추가 987=1003개, 3238186037 B를 다시 SHA256 대조했다. 변경 0·누락 0·오류 0, passed=true다. 마지막 gpu_guard는 4 sample에서 KFD client 0·오류 0이었다. 모든 학습·metadata·음성 GPU 프로세스가 종료됐으며 운영 파일·기존 모델·운영 venv 설치는 변경하지 않았다.

## 8. 실패·계측 결함 보존

| 항목 | 상태 | 원인과 처리 |
|---|---|---|
| 첫 native_profile | 타이밍 폐기 | 별도 CPU 검증 프로세스의 장치 조회가 KFD에 등록되어 guard가 겹침 검출. PID/시간 보존, gate 유지 |
| native_profile_clean | 최종 비중에 사용 금지 | probe가 parametrized weight getter를 두 번 읽어 weight_norm 추가 실행. 5,355회 → 수정 후 1,785회 |
| native_profile_final | 프로파일 근거 통과 | descriptor metadata만 읽도록 수정. GPU foreign 0, 보호 SHA 변경 0 |
| 첫 NORMAL dilated seed | 누락 원인 수정·복구 완료 | workspace 0으로 GEMM 후보가 탈락했던 원자료·경고를 보존. 복구된 FAST replay에서도 추가 채택은 0방향 |
| 첫 decoder compile | epoch 0 실패 | FakeTensor 추적 중 Half input/Float bias AMP 경계 충돌. 원자료 보존, Conv1d eager 경계를 명시한 별도 실행으로 복구 시도 |
| 100 epoch 음질 | 청취 통과 | 같은 4문장 native 변환과 기술 검사 완료. 2026-11-08 사용자 익명 청취에서 A/B/C 구분 불가 |

## 9. 재현·적용·롤백

재사용 코드는 [v08_direct_conv.py](../../v08_direct_conv.py), [v08_nonconv_opt.py](../../v08_nonconv_opt.py), [applio_epoch_v08.py](../../applio_epoch_v08.py), [직렬 실행기](../../v08_run_epoch_series.py)다. 직접 Conv는 고정한 [dispatch manifest](direct_selected_final/dispatch.json)를 읽고, solver는 선택 DB (`solver_selected_final_db/`, 연구 폴더에만 있음) 사본을 사용한다. 본체 source SHA가 달라지면 기존 manifest로 실행하지 않는다.

최종 후보의 정확한 옵션·환경·고정 source SHA는 [epoch_series_100.json](epoch_series_100.json)에 있다. G는 native, D는 single AdamW이며 gather slicing, foreach 기록용 norm, epoch empty_cache 생략을 켠다. fused·WN cache·compile은 끈다. 이 설정은 연구 프로세스에만 적용되고 기본값은 OFF다.

원복은 새 프로세스에서 `--dispatch`·`--nonconv`와 관련 opt-in 환경을 제거하고, 현재 운영 DB snapshot인 `db_fast_native`를 새 연구 DB 경로로 복사해 실행하는 것이다. 코드의 context/finally도 원래 함수를 복구한다. 운영 launcher·설정·소스·모델에 적용한 변경은 없다.

기존 plan/status/out/DB는 재실행하거나 덮어쓰지 않는다. 독립 재현이 필요하면 [카탈로그·가이드 삽입안](V08_CATALOG_GUIDE_PROPOSAL_KO.md) §7의 새 이름·출력·DB를 만드는 명령을 사용한다. 같은 문서 §6에는 완료 원자료를 읽어 핵심 숫자 5개를 확인하는 CPU 명령이 있다. 최종 100 epoch가 끝나면 `python3 v08_summarize_final.py`로 원자료의 100 epoch·3100 step·시간·loss·SHA·보호 기록을 다시 집계할 수 있다.

## 10. RTX 5070 12분 보고와의 비교

이 PC에서 확정한 소프트웨어 절감은 현재 A/A′의 25분 40.11초/25분 44.65초에서 B의 22분 0.002초로 줄인 3분 40.11초–3분 44.65초(14.29–14.54%)다. 9월의 35–38분 기록을 이번 v0.8 절감률의 분모로 쓰지 않는다. RTX 5070의 12분은 외부 미검증 보고여서 동일 조건 비교가 아니며, 숫자상 B와 약 10분 차이가 남지만 이를 gfx1030의 행렬 유닛 부재 때문에 불가피한 몫으로 확정할 수 없다. 구조적 제약과 별개로 미검증 추가 fusion 등 소프트웨어 여지도 남아 있다. 이번 작업은 줄인 몫을 실측했고, 더 줄일 수 없는 정확한 분·비율은 미식별로 남긴다.

## 11. 완료 상태와 다음 세션의 시작점 — 2026-11-08 KST

- **완료:** 기존 직접 Conv 결과 회수, rank FAST replay, workspace 0 원인 수정·누락 seed, 최종 DB 432/432 정확도 검증, 직접 Conv 21 shape/56방향 manifest·21 shape 호출 경계 검증, nonconv GPU 검사, 실제 짧은 epoch 교대 비교, compile 제외 판정.
- **완료:** Core 17913 exit 0으로 A/A′/B 각 100 epoch·3100 step·checkpoint/export 종료. [고정 계획](epoch_series_100.json), [완료 상태](epoch_series_100_status.json), [최종 집계](final_training_summary.json). 세 arm 모두 training_checks_pass, 같은 initial G/D SHA, 고정 5 source SHA 일치. **학습을 다시 시작하지 않는다.**
- **완료:** Core 35572의 metadata probe는 exit 0, 432방향 실제 solver 확정·모호/누락 0. 기존 유효 시간·정확도 값은 유지했다. [행별 표](solver_roofline/shape_direction.csv)와 [solver 설명](../../v08_solver_notes_ko.md) §9를 읽는다.
- **완료:** Core 13505 exit 0으로 익명 청취 HTML (`audio_eval_final/LISTEN_BLIND.html`, 연구 폴더에만 있음)과 12 WAV를 생성했다. 모두 native 추론이며 3모델 load 완전성·finite·입력/설정 일치·원본 보호와 브라우저 재생을 확인했다. blind_key.json은 대응표이며 청취 전 공개하지 않는다.
- **완료:** [최종 보호 검사](final_protection_check.json), 1003개 변경·누락·오류 0. 운영 Applio·기존 모델·운영 venv 설치는 변경하지 않았고 GPU 실행은 모두 종료됐다. 청취 QA의 임시 탭·로컬 서버도 닫았다.
- **청취 판정(2026-11-08, Claude 기록):** 사용자가 익명 청취 페이지의 4문장 × A/B/C를 듣고 "사람 귀로는 분간이 안 된다"고 답했다. 특정 라벨만 이상하다는 지적은 없었다. 판정 JSON은 저장하지 않았고, 대응표(blind_key.json)는 열지 않았다. 지각 기준(가이드 §1.2)에 따라 B는 같은 음질로 채택 가능하다. 운영 적용은 별도 요청과 백업·ROLLBACK 이후에 한다.
- **검토용 자료:** [카탈로그 §4.3/§4.7·가이드 §9.14 이후 갱신안](V08_CATALOG_GUIDE_PROPOSAL_KO.md), [직접 Conv](DIRECT_CONV_V08_KO.md), [nonconv·음성](NONCONV_V08_NOTES_KO.md). 카탈로그·가이드 원본은 수정하지 않았으며 삽입안에 최종 실측을 반영했다. §6의 핵심 숫자·CPU 명령과 §7의 새 경로 재현 명령을 사용한다.
- **금지 유지:** 운영 파일 변경, 기존 모델명으로 cleanup, 운영 venv 설치, 완료 결과 덮어쓰기, 동시 GPU 작업. 새 연구가 필요하면 완료 경로와 분리된 명시적 범위로 진행한다.

원본 연구 폴더는 `~/DevSpace/rdna2-pro-research-20261102/rdna2-ops`다. 이 보고서의 상대 링크는 그 폴더의 원자료를 기준으로 한다.

## 11. 운영 롤아웃과 운영 경로 실측 (2026-11-08, Claude)

### 11.1 적용

- 운영 `applio-lab`에 `rvc/train/rdna2_v08/`를 추가했다.
  - B 구성 그대로다: 직접 Conv `dispatch.json`, 선택 DB, nonconv 7개 환경.
  - 연구본과 바이트가 같은 사본이다(`SOURCE.md`).
- `lab_hooks.py`와 `train.py`(`run()` 한 줄)에 연결했다.
- 켜는 변수는 `APPLIO_RDNA2_V08=1`이고, torch `2.11.0+rocm10.0.0`에서만 켜진다.
- DB는 Applio 전용 폴더(`rdna2_v08/miopen_db`)를 쓴다. 공용 `~/.config/miopen`(SHA `a5a7fa17…`)은 변하지 않았다.
- 백업과 원복: `applio-lab/backups/20261108_pre_v08/ROLLBACK.md`.

### 11.2 운영 경로는 하네스보다 느렸다 — 원인과 수정

하네스 A(15.0 s/epoch)는 운영 경로와 같지 않았다. 실제 `python core.py train → rvc/train/train.py` 경로는 같은 데이터·pretrain·FAST·안전 변수에서 **약 22 s/epoch**였다. 9월 기록(약 21 s/epoch, 100 epoch 약 38분)과 맞는다.

**원인**
- 운영 train.py는 `__main__`이다. spawn된 DataLoader 워커 4개가 train.py를 다시 import하면서 각자 GPU 컨텍스트와 큐를 만든다.
- 하네스에서는 워커가 하네스 스크립트를 import하므로 이 일이 없었다.
- 학습 중 KFD 클라이언트는 6개, 큐는 19개였다(부모 3, 학습 4, 워커 3×4).

**확인한 것**
- 같은 커널, 같은 호출 수였는데 커널 시간이 1.5~2배 길었다. 5 step GPU 합 3.06 s, 하네스는 2.07 s였다.
- 클럭은 정상이었다. 학습 중 sclk 2.6 GHz, mclk 1000 MHz, busy 99%였다.
- 짧은 교정도 같았다. dot2 48.37 TFLOP/s, copy 436.7 GB/s.
- zluda 검사를 끈 대조에서는 변화가 없었다.

**수정**
- 학습 프로세스가 자기 GPU 컨텍스트를 만든 직후 `HIP_VISIBLE_DEVICES=-1`을 설정한다.
- 이 기능은 `APPLIO_WORKER_NO_GPU=1`로 켜고, V08을 켜면 기본으로 켜진다.
- 워커는 데이터만 읽는다. `data_utils`에 GPU 코드가 없고, pin_memory는 학습 프로세스에서 일어난다.

### 11.3 운영 경로 결과

모두 실제 `core.py train` 경로, `.venv-rocm10`, 새 모델 이름, 6 epoch(마지막 줄은 4 epoch)로 측정했다. epoch당 시간은 tqdm의 초 단위 표시와 epoch 타임스탬프 간격으로 읽었다.

| 설정 | steady epoch | 학습 중 큐 |
|---|---:|---:|
| 기본(변경 전 운영) | 약 22 s | 19 |
| `APPLIO_WORKER_NO_GPU=1`만 | 약 15.0 s | 7 |
| `APPLIO_RDNA2_V08=1`(워커 숨김 포함) | 약 12.6 s | 7 |

- V08 실행에서 직접 Conv는 21/21 shape가 실제로 호출됐다(epoch 5까지 6,353회). 오류는 없었다.
- 추정: 운영 경로 100 epoch는 약 37분에서 약 21~22분으로 줄어든다. 100 epoch 전체 실행은 하지 않았다.
- 워커 숨김은 계산을 바꾸지 않는다. 청취 판정은 B(V08)에 대해 이미 통과했다(§7.3).
- 미검증: `.venv`(rocm7.2) GUI 경로에서의 워커 숨김 효과, 다른 배치·샘플레이트에서의 효과.
