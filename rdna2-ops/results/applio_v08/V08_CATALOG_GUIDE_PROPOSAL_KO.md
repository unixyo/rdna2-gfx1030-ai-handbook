# Applio RVC v0.8 카탈로그·가이드 갱신안

작성 기준: 2026-11-07. 이 파일은 연구 폴더 안의 **삽입 문안과 재현 명령 초안**이다. 카탈로그·가이드 원본에는 반영하지 않았다. **A/A′/B 각 100 epoch 학습, baseline 432방향 실제 solver ID 관측, 12개 음성 변환과 작업 끝 보호 검사까지 완료**했다. 사용자 blind 청취에 따른 지각 품질 판정은 **PENDING**이다.

현재 확정: 최종 **B_full eager**는 외부 process wall **1320.001536 s(22.000026분)**로 A/A′보다 **14.291934–14.543648%** 짧았다. decoder compile은 실제 이득이 없어 제외했다. 아래 사전 예상과 완료 실측을 구분하며, 학습 통과를 지각 품질 통과로 쓰지 않는다.

원 지시문: [PRO_TASK_V08_APPLIO_KO.md](../../../PRO_TASK_V08_APPLIO_KO.md) §3–5. 최신 실행 상태의 정본은 [APPLIO_V08_KO.md](APPLIO_V08_KO.md)다. 이 문서를 별도 실행 일지로 사용하지 않는다.

## 1. 실제 원본과 반영 위치

| 대상 | 확인한 위치 | 제안 |
|---|---|---|
| `RDNA2_OP_CATALOG_KO.md` | §4.3 Conv | 기존 v0.6 행 보존, v0.8 solver·direct Conv·workspace 복구 행 추가 |
| 같은 카탈로그 | §4.7 학습 | 현재 profile·AdamW 판정 갱신, nonconv 및 실제 epoch 근거 추가 |
| `rdna2-ops/RDNA2_OPTIMIZATION_GUIDE_KO.md` | §9.14 끝 | v0.8 후속 근거 연결 문장 추가 |
| 같은 가이드 | 현재 §9.20 다음, §10 직전 | 새 §9.21 추가. §9.15–9.20 보존 |

읽은 원본 SHA-256은 카탈로그 `8bdd7a3184ef23175f319f2cbf47dd9505be35d75e9e7016137adc6a3b2ab755`, 가이드 `aef0b5bb80a79056d4a17b89a5718811c26b427bd0a1fab83758d18ef206bea9`이다.

기존 v0.6의 `bench40k`는 111 chunks / 16 steps/epoch다. 아래 v0.8은 `voice_model_A` filelist 233행 / bucket 248 samples / **31 steps/epoch**다. 이전 epoch 시간·배수를 이번 데이터셋의 성능으로 재표기하지 않는다.

카탈로그 범례를 유지한다. `❔`는 일부 검증이 남은 후보, `⚠️`는 속도·정확도 문제로 제외한 후보다. `✅`는 해당 행에 명시한 검증 범위에서만 쓴다. 짧은 epoch가 빨라도 100 epoch 음질까지 통과한 운영 채택으로 쓰지 않는다.

## 2. 카탈로그 §4.3 삽입 문안

기존 표의 열 순서와 같다. 원래 `Dilated Conv1d` 행은 과거 구현의 근거로 보존하고 v0.8 행을 별도 추가한다.

| 연산 | 상태 | 정확도 | 속도 | 비고 |
|---|---|---|---|---|
| Applio v0.8 shape·방향별 MIOpen DB | ❔ 연산·전체 구성 100 epoch 학습 통과, 청취 PENDING | 최종 FAST 144 geometry × 3방향 = 432/432 FP32 참조 gate 통과. 별도 baseline metadata replay에서 실제 ID 432/432 확인 | 선택 6방향 event **1.075–1.530×**, wall **1.075–1.521×** | 현재 운영 DB 복사본 대비. 합성 DB를 새 FAST 프로세스로 재생. solver 단독 epoch 배수는 측정하지 않음 |
| Applio v0.8 직접 Conv1d 선택 dispatch | ❔ 연산·AMP 경계·전체 구성 100 epoch 학습 통과, 청취 PENDING | 연산 189/189, 실제 dtype·weight_norm 경계 105/105 비교 통과. FP16 곱·FP32 누산 | **21 shape, 56방향**: fwd 21개 1.698–12.074×, dX 20개 1.313–9.444×, dW 15개 1.372–9.663× | 방향별 가장 빠른 정확한 native 기준. C128/K11 dW와 C192→384/K5 dW는 native 유지. ISA `v_dot2c_f32_f16` 확인 |
| Applio v0.8 solver + direct Conv | ❔ 짧은 실제 epoch·최종 전체 구성 학습 통과, 청취 PENDING | 실제 학습 경로·각 run 보호 검사 통과 | Conv만 더한 4-epoch A0/B/A1 steady **15.096908 / 13.314360 / 14.972521 s**, B **1.1245–1.1339×** | nonconv를 포함한 최종 100 epoch 결과와 구분. 31 steps/epoch, 첫 epoch 비용 별도 보고 |
| MIOpen dilated Conv NORMAL workspace 복구 | ✅ seed 탐색 복구 범위 | workspace 질의만 보완, 같은 계산 solver 사용. 24 geometry × 3방향 = 72/72 통과 | 학습 속도 이득으로 집계하지 않음 | legacy 0 → Range.max. FP16 C64/K7/d3 사례 5,734,400 bytes. 복구 rank FAST 통과, 신규 채택 0방향. preload는 NORMAL seed 전용 |

표 아래 추가 문안:

> v0.8은 `voice_model_A` / KLM49 HFG / 40 kHz / batch 8 / FP16 AMP를 유지한다. master weight·AdamW state는 FP32다. C192→384/K5 세 shape의 실제 모듈 입력은 FP32였으므로 FP16 microbenchmark와 구분했다. native autocast 내부 FP16 계산을 확인하고 FP32 입력·weight_norm master gradient를 검증한 뒤 fwd·dX만 후보에 넣었다. 연산·짧은 epoch 통과는 100 epoch 지각 품질 통과를 대신하지 않는다. 근거: `rdna2-ops/results/applio_v08/DIRECT_CONV_V08_KO.md`, `rdna2-ops/v08_solver_notes_ko.md`, 가이드 §9.21.

## 3. 카탈로그 §4.7 갱신 문안

기존 Conv backward·optimizer fusion 행을 아래 결과로 갱신한다. 기존 FP16 AMP·BF16 행은 해당 과거 근거를 유지한다. compile의 이전 16-step 수치와 이번 실제 비교에서의 제외 판정을 구분한다.

| 항목 | 상태 | 비고 |
|---|---|---|
| 실제 Applio v0.8 Conv 비중 | ✅ 분해 확인 | step 40–44 operator self GPU 2,065.797 ms 중 backward **55.686%**, forward **24.625%**, 합 **80.311%**. 별도 kernel 분모 2,057.529 ms에서 AdamW **5.352%**. 두 분모·CPU 대기를 합산하지 않음 |
| AdamW fused | ⚠️ 사전 수치 gate 실패, 제외 | 실제 G/D snapshot의 1·32 update 비교. 단독 warm wall G 2.527× / D 2.981×이나 tensor별 update·state gate 실패. 허용치 완화 없음 |
| G native / D single AdamW | ❔ optimizer·전체 구성 100 epoch 학습 통과, 청취 PENDING | G single은 0.454×로 느려 native 유지. D single **15.853552 → 11.138054 ms, 1.423368×**. D moment·step bit 동일, parameter/update gate 통과. FP32 상태·overflow skip 유지 |
| Segment gather | ❔ 연산·전체 구성 100 epoch 학습 통과, 청취 PENDING | 4 shape × FP16/FP32 = 8/8 y·dX bit 동일. 같은 시작 index·RNG 유지. forward+autograd 구성 warm wall **10.198–14.051×**; backward 속도는 이 배수에 포함하지 않음 |
| 기록용 grad norm foreach | ❔ 연산·전체 구성 100 epoch 학습 통과, 청취 PENDING | 실제 계열 17 shape synthetic gradient에서 **0.214707 → 0.041388 ms, 5.188×**, NRMSE 6.345e-8. gradient 자체는 변경하지 않음 |
| Discriminator weight_norm cache | ⚠️ 유효한 D gradient 참조를 확보하지 못해 제외 | native·cache 양쪽의 같은 9 parameter gradient가 nonfinite. 1.022×를 정확한 후보의 성능 근거로 채택하지 않음 |
| Conv + D single + gather + norm + cache 유지 | ❔ 100 epoch 학습 통과, 청취 PENDING | 짧은 B_full steady **12.570070 s**, A1/A2 대비 **1.1872–1.1911×**. 최종 B100 steady **12.688541 s**, 외부 wall **1320.001536 s**, A/A′ 대비 전체 시간 **14.291934–14.543648%** 감소. `empty_cache` 200회 생략 |
| `torch.compile(dec)` / native RNG | ⚠️ 실제 이득 없음, 최종 B에서 제외 | 첫 trace는 AMP fake tensor 오류로 0 epoch 실패. eager Conv 경계 복구는 4 epochs 완료했지만 loop graph break/fallback, steady **12.583505 s**로 eager **12.570070 s**보다 이득 없음. first도 **18.608218 s**로 더 큼 |
| 최종 100 epoch A / A′ / B | ❔ 세 학습 완료·학습 검사 통과, 청취 PENDING | 외부 wall A/A′/B **1540.113550 / 1544.650001 / 1320.001536 s**. B는 **220.112014–224.648464 s** 단축. 초기 G/D 동일, 각 3,100 steps, nonfinite loss 0. 마지막 25% loss는 native 반복 차이와 비교하며 지각 판정은 별도 |

근거: `rdna2-ops/results/applio_v08/NONCONV_V08_NOTES_KO.md`, `nonconv_gpu_actual.json`, `native_profile_final/profile_summary.json`, `epoch_B_full/result.json`, `epoch_A2/result.json`, `epoch_B_compile_boundary/result.json`, `epoch_series_100.json`, `PREDECLARED_GATES.json`. 위 짧은 실제 epoch는 구성 전체 비교이며 nonconv 개별 변경의 epoch 기여율로 분해하지 않는다.

100 epoch 실측 근거는 `final100_A/result.json`, `final100_A_prime/result.json`, `final100_B/result.json`, `epoch_series_100_status.json`과 검산 집계 `final_training_summary.json`이다. 외부 wall은 coordinator의 `elapsed_wall_s`, steady는 각 result의 `steady_epoch_median_s`로 계측 경계를 구분한다.

## 4. 가이드 §9.14 끝의 연결 문장

> **v0.8 후속 검증:** 실제 `voice_model_A` 31 steps/epoch와 KLM49 pretrain에서 shape·방향별 FAST solver 선택, dilated Conv workspace 질의 복구, 직접 Conv1d의 AMP·weight_norm 경계 검증을 수행했다. 이전 `bench40k` 16-step 결과와 구분한 현재 근거와 학습·음질 판정은 §9.21에 정리한다.

## 5. 가이드 §9.21 삽입 본문

다음 본문의 근거 경로는 가이드가 있는 `rdna2-ops` 기준이다.

### 9.21 Applio RVC v0.8: 실제 학습의 Conv·nonconv 선택과 검증 범위

#### 대상과 시간 분해

대상은 RX 6900 XT `gfx1030`, PyTorch 2.11.0+rocm10.0.0 / HIP 7.15.26333 / Triton 3.8.0의 `voice_model_A`, KLM49 HFG pretrain, 40 kHz, batch 8, FP16 AMP다. FP32 master weight·AdamW 상태와 손실·배치·segment·학습률을 유지한다. filelist 233행은 sampler 248 samples로 구성되어 epoch당 31 step을 실행한다. 실제 Linux trainer의 `cudnn.benchmark`는 `False`여서 원 지시문의 True 가정 대신 확인된 native 설정을 기준으로 삼았다.

`native_profile_final/profile_summary.json`의 step 40–44는 operator self GPU 총 2,065.797 ms 중 Conv backward 1,150.359 ms(55.686%), forward 508.710 ms(24.625%)였다. 별도 kernel 총량 2,057.529 ms에서 Winograd 상위 두 kernel 26.811%, im2col+col2im 14.067%, AdamW 5.352%다. operator와 kernel 표는 같은 작업을 다르게 귀속하므로 합산하지 않는다.

trace wall 2.600829 s와 recorded GPU activity union 2.065797 s 사이 0.535032 s는 profiler 영향도 포함한다. 전부 정상 학습의 GPU idle이나 data loader 병목으로 단정하지 않는다. DataLoader 호출 5회 합은 168.77 µs였다. `hipMemcpyWithStream` CPU 구간도 다른 대기와 겹치므로 절감 시간을 합산하지 않는다.

보조 calibration은 register dot2 **48.725 TFLOP/s**, 512 MiB stream copy **436.745 GB/s**, 16 MiB warm copy **801.970 GB/s**다. 이는 낙관적인 참조이며 shape·재사용·동기화·레지스터 비용을 제거한 실제 Conv의 보장 속도가 아니다. 특히 Winograd를 dense Conv FLOP으로 평가한 비율은 엄밀한 하드웨어 효율이 아니다. C64/K7/d3의 2.936 GFLOP / 48.725 TFLOP/s = 0.06026 ms와 native dW 약 4 ms의 차이를 달성 가능한 66.5× 가속으로 해석하지 않는다.

근거: `results/applio_v08/APPLIO_V08_KO.md`, `native_profile_final/profile_summary.json`.

#### Solver 선택과 workspace 복구

기준 `db_fast_native`는 운영의 현재 FAST DB를 연구 폴더로 복사한 것이다. 빈 DB와 비교하지 않는다. 안전 변수 `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0`, `MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53=0`을 유지하고 NORMAL seed와 FAST replay는 별도 새 프로세스에서 실행한다.

기존 rank 00–04 FAST 222/222방향과 workspace 복구 rank 00–01의 72/72·54/54방향을 정확도·event·wall로 판정했다. 합성 DB 최종 FAST도 144 FP16-normalized geometry의 432/432방향이 고정 gate를 통과했다. 이는 관찰한 모양을 FP16 연산으로 정규화한 목록이며, 실제 모든 모듈 입력이 FP16이었다는 뜻은 아니다.

최종 선택 6방향의 아래 수치는 합성 DB의 **최종 replay event 중앙값**이다. wall 개선도 각 1.05× 이상이었다. solver 열은 선택한 DB ID이며 singleton entry 유지·실제 rank kernel 일치로 검증했다. native 로그에서 ID를 직접 관측한 근거와는 구분한다.

| shape·방향 | 기존 FAST ms | 최종 FAST ms | event 배수 | solver |
|---|---:|---:|---:|---|
| B8 C128/T3200 K7 d1, dX | 0.489507 | 0.455531 | 1.075× | GemmBwdRest |
| B8 C256/T320 K11 d1, dW | 0.387294 | 0.349765 | 1.107× | GemmWrwUniversal |
| B8 C256/T320 K7 d1, dW | 0.313029 | 0.262761 | 1.191× | GemmWrwUniversal |
| Conv2d B8 C128/H712/W2→O512 K5×1 s3×1, dW | 0.598203 | 0.452818 | 1.321× | ConvHipImplicitGemmV4R1WrW |
| ConvTranspose1d B8 C256/T320→O128 K16 s10, dW | 0.397647 | 0.310295 | 1.282× | ConvHipImplicitGemmV4R1WrW |
| ConvTranspose1d B8 C512/T32→O256 K16 s10, dW | 0.191433 | 0.125158 | 1.530× | ConvHipImplicitGemmV4R1WrW |

선택 6방향의 최악 NRMSE 0.000441372, max error/reference RMS 0.00539934였다. 미선택 방향은 기존 안전한 DB 경로를 보존한다. 방향별 단발 시간의 단순 합을 epoch 개선으로 쓰지 않는다.

workspace=0 원인은 설치 MIOpen과 일치하는 commit `8d1ae90e`의 legacy GEMM workspace 질의가 dilation>1에서 0을 반환한 경로였다. PyTorch는 질의한 크기로 할당한 뒤 Find를 호출하고 있었다. `v08_miopen_workspace.c`는 `APPLIO_RDNA2_FIX_FIND_WORKSPACE=1` 및 `MIOPEN_FIND_MODE=NORMAL`일 때만 세 workspace 질의를 `max(original, Range.max)`로 보완한다. 계산은 원래 solver가 한다.

smoke에서 FP16 0 → 5,734,400 bytes, FP32 참조 11,468,800 bytes가 Find에 전달됐다. smoke와 나머지 23개를 합쳐 72/72방향이 통과했다. FP16 seed 72 keys/90 entries를 만들었지만 FAST 검증 뒤 추가 채택된 dilated solver는 0개다. 성과는 탐색 누락 해결이며 학습 속도 이득으로 세지 않는다. preload는 NORMAL seed 전용이고 FAST·학습에는 사용하지 않는다.

근거: `v08_solver_notes_ko.md` §7–8, `results/applio_v08/solver_selected_final_replay/result.json`, `solver_selected_final_db/selection_manifest.json`, `source/miopen_workspace/source_manifest.json`. 최종 solver DB SHA-256: `fa2af4e846b0b565e23f058ae5d011cba85fe47a757a399a3ffa618563fbfdd5`. selection manifest 생성 당시의 상태 문구와 후속 완료 replay를 함께 읽는다.

전체 baseline 432방향 표의 시간·실제 kernel과 실제 solver ID도 구분한다. 기존 timed run은 로그를 껐으므로 432행 모두 native ID 직접 관측이 없었다. kernel 계열과 호환하는 DB 후보는 1개 183행, 복수 126행, 기록 없음 123행이다. 같은 kernel을 낸 singleton rank 141방향 중 7방향은 f2x3/g1 두 요청 ID가 같은 kernel을 내서 kernel명만으로 구별할 수 없다.

`solver_roofline/shape_direction.csv`·`roofline.json`은 요청·후보·관측·증거를 분리하고 기존 시간·정확도를 유지했다. B100 뒤 원본 DB의 새 사본으로 144 shape × 3방향 각 1회의 FAST metadata probe를 완료했다. FP32 reference·timing·profiler는 생략했다. 각 방향 marker 안의 최종 `Chosen Algorithm` 1개와 실제 API 호출 1개를 대응해 **432/432행 native ID 확정, 다중·누락·FP32 혼입 0**이다. 출력은 모두 finite, audit clean, DB 전후 SHA 동일이며 probe timing은 false다. 이름을 보완한 별도 replay의 관측임을 명시하고 과거 timed 호출 자체의 로그로 표현하지 않는다.

실제 ID 9종의 방향 수는 Winograd g1 69, f3x2 37, f2x3 20; GemmFwdRest 73, GemmBwdRest 75, GemmWrwUniversal 83, GemmFwd1x1_0_1 28, GemmBwd1x1_stride1 29, GemmWrw1x1_stride1 18이다. kernel만으로 모호했던 7방향은 f2x3 5개 / g1 2개로 구별했다. 근거: `v08_solver_notes_ko.md` §9, `results/applio_v08/solver_roofline/solver_attribution.json`·`baseline_metadata_probe144/result.json`·`baseline_metadata_probe144/miopen_stderr.log`.

#### 직접 Conv1d

완료한 `direct_initial9`·`direct_additional12`는 재실행하지 않고 저장 결과로 선택했다. 21 geometry × 3방향 × 3 tile = 189후보 모두 FP32 참조 gate를 통과하고 ISA `v_dot2c_f32_f16`를 확인했다. native 63방향도 통과했다.

fwd·dX는 im2col 전체 행렬을 만들지 않는다. dW는 FP32 partial 32개와 고정 reduction을 사용하며 FP16 partial·atomic 누산을 쓰지 않는다. fwd는 native의 FP16 출력 반올림 뒤 FP16 bias 더하기 경계를 유지한다. 수학적으로 같은 계산을 구현하지만 bit 동일을 주장하지 않는다.

native baseline·rank FAST·최종 합성 DB 등 11결과의 방향별 가장 빠른 정확한 FP16 native를 기준으로 event 1.10×, wall 1.05× 이상이며 dtype 경계 근거가 있는 방향만 골랐다. 외부 solver fwd 비교는 native에 bias가 없고 후보에 bias가 있어 보수적이다.

최종 manifest는 **21 shape / 56방향 = fwd 21 + dX 20 + dW 15**다. event 범위는 fwd 1.698–12.074×, dX 1.313–9.444×, dW 1.372–9.663×다. 대표 C64/T6400/K7/d3 dW는 3.984700 → 0.684107 ms, **5.825×**다. C128/K11 d1·d3·d5 dW와 d1 dX, C192→384/K5 dW는 native를 유지한다.

C192→384/K5 세 shape의 실제 입력은 FP32다. FP16 microbenchmark와 구분해 native autocast 내부 FP16을 확인하고, 관찰한 dtype·weight_norm 구성을 fixed-seed synthetic 값으로 재현한 계약 검증을 수행했다. native AMP와 wrapper의 y·dX·FP32 bias 및 weight_norm master parameter gradient **21/21 case, 105/105 비교**가 통과했다. 최악 NRMSE 0.000561579, max error/reference RMS 0.007820156이었다. master parameter·gradient FP32, 입력·출력·gradient dtype 일치, hook 복원도 확인했다. optimizer·음질 검증으로 확대하지 않는다.

최종 manifest SHA-256은 `22e41075581e8b2d969388244bf063adba59579942c4959ae892e2aeb4fcfa59`, direct 소스는 `6ef607af92b0ffd561a633093ff9bd62ae8428deb6f613aa2d38fca3a674b4e3`이다. 근거: `results/applio_v08/DIRECT_CONV_V08_KO.md`, `direct_selected_final/dispatch.json`, `direct_contract_initial/result.json`.

#### Nonconv

실제 optimizer snapshot은 G 560 tensors/36,431,170 values, D 165 tensors/71,410,594 values다. 같은 gradient로 1·32 update의 parameter·moment·step 및 실제 update 차이를 사전 gate와 비교한 뒤 warm timing을 별도 측정했다.

| 후보 | warm wall | 수치 판정 | 선택 |
|---|---|---|---|
| G native | 7.718141 ms | 기준 | 유지 |
| G single | 17.018334 ms, 0.454× | 통과 | 느려서 제외 |
| D native → single | 15.853552 → 11.138054 ms, 1.423368× | 통과 | 실제 epoch 후보에 포함 |
| G fused | 3.053804 ms, 2.527× | 실패 | 제외 |
| D fused | 5.317606 ms, 2.981× | 실패 | 제외 |
| 명시적 foreach | G 1.000×, D 1.005× | native와 1·32 update bit 동일 | 독립적 가속으로 집계하지 않음 |

fused는 finite·overflow skip이 정상이더라도 state·tensor별 update gate를 넘었다. 1 update 최악 tensor update NRMSE는 G 0.003034825, D 0.003656458로 허용 0.002를 초과했다. D single은 moment·step bit 동일, parameter/update gate 통과다. 단독 optimizer 시간 합으로 실제 step 시간을 예측하지 않는다.

gather는 4 shape × FP16/FP32의 y·dX 8/8 bit 동일, 같은 random start index·RNG를 유지한다. forward+autograd 구성 warm wall 10.198–14.051×이며 backward 속도를 포함하지 않는다. foreach norm은 실제 계열 17 shape synthetic gradient에서 0.214707 → 0.041388 ms, 5.188×, NRMSE 6.345e-8이다. 기록용 norm만 바꾸며 gradient를 수정하지 않는다.

D weight_norm cache는 native·cache 양쪽의 같은 9 parameter gradient에 nonfinite가 있어 유효한 참조를 얻지 못했다. 1.022×를 채택 근거로 사용하지 않았다. epoch당 두 `empty_cache` 생략은 B_full 구성에 포함해 실제 짧은 epoch에서 비교했다. 전체 구성의 이득을 이 변경 하나의 효과로 분리하지 않는다.

근거: `results/applio_v08/NONCONV_V08_NOTES_KO.md`, `nonconv_gpu_actual.json`, `weight_norm_gpu_actual_scale1.json`, `PREDECLARED_GATES.json`.

#### 짧은 실제 epoch와 완료한 100 epoch·남은 음질 판정

아래 완료한 6개 짧은 run은 JSON의 timing 유효성·epoch 유효성·보호 무변경을 확인했다. 별도의 첫 compile 실패는 완료 run에 포함하지 않는다. 이어진 A/A′/B 각 100 epoch의 완료 실측은 아래 별도 표에 반영했다. 음질 판정은 PENDING이다.

| run, 각 4 epochs | first s | steady median s | 4 epochs 합 s | 비교 범위 |
|---|---:|---:|---:|---|
| native A0 | 21.385693 | 15.096908 | 67.079081 | 현재 native |
| solver + direct B_conv | 16.575041 | 13.314360 | 56.268797 | A0/A1 대비 steady 1.1245–1.1339× |
| native A1 | 18.631871 | 14.972521 | 63.604978 | native 반복 |
| Conv + D single + gather + norm + cache 유지 B_full | 16.536212 | 12.570070 | 55.399439 | A1/A2 대비 steady 1.1872–1.1911× |
| native A2 | 18.165280 | 14.922928 | 62.939109 | 후속 native 교대 완료 |
| dec compile / native RNG / eager Conv 경계 복구 | 18.608218 | 12.583505 | 56.760980 | valid 완료, eager 이득 없음으로 제외 |

근거는 `results/applio_v08/epoch_A0/result.json`, `epoch_B_conv/result.json`, `epoch_A1/result.json`, `epoch_B_full/result.json`, `epoch_A2/result.json`, `epoch_B_compile_boundary/result.json`이다. scale 감소·nonfinite loss·loss 중앙값 차·초기 state 일치·peak VRAM은 최종 주 보고서의 실제 수치를 함께 붙인다. 구성 전체의 차이를 개별 변경의 epoch 기여로 분리하지 않는다.

첫 `epoch_B_compile_dec`는 완료 epoch 0에서 fake tensor의 FP16 input / FP32 bias 불일치 오류로 실패했다. 한 차례 `--compile-eager-conv` 경계를 추가한 `epoch_B_compile_boundary`는 4 epochs를 정상 완료했으나 로그에 loop graph break와 frame fallback이 남았다. steady 12.583505 s, first 18.608218 s 모두 eager B_full보다 유리하지 않았다. first + 99×steady 예측도 compile 1,264.375238 s, eager 1,260.973176 s여서 최종 B에서 제외했다. 이는 이 조합의 실제 측정 판정이며 모든 compile 구현의 일반적 한계로 확대하지 않는다. 근거: 두 compile result, `epoch_B_compile_boundary.console.log`, `epoch_series_compile.json`, `epoch_series_compile_boundary.json`.

최종 후보를 **B_full eager**로 고정해 A(native), A′(native 반복), B 각 100 epochs를 완료했다. 사전 계획 `results/applio_v08/epoch_series_100.json`에 동일 저장 정책, 후보 args·env, 소스 SHA와 예상 시간을 기록했다. 예측은 A/A′ 각각 **1,534.197278 s(25.570분)**, B **1,294.637667 s(21.577분)**, 합 **4,363.032224 s(72.717분)**였다. process 시작 여유를 포함하며 최종 checkpoint·reference inference·export·음성 변환 여유는 별도였다. 원 지시문의 약 2시간 대신 짧은 실제 run에서 산출한 사전 예상이며 아래 완료 실측과 구분한다.

완료한 A/A′/B는 각각 100 epochs / 3,100 steps이며 모든 epoch가 31 step이다. 세 result의 `completed`·`valid_for_timing`·`valid_for_epoch_timing`와 coordinator의 `valid`가 모두 true이고 exit code 0이다. 초기 G/D state SHA가 세 run에서 일치하며 각 GPU audit의 foreign client·error는 빈 목록이다. `final_training_summary.json`의 학습 검사는 모두 통과했다.

| 최종 100 epoch | A | A′ | B |
|---|---|---|---|
| 고정한 out 이름 | `final100_A` | `final100_A_prime` | `final100_B` |
| 학습 상태 | 100 epochs 완료 | 100 epochs 완료 | 100 epochs 완료 |
| 완료 result | `final100_A/result.json` | `final100_A_prime/result.json` | `final100_B/result.json` |
| 기록된 export | `final100_A/model_final.pth`, SHA 아래 | `final100_A_prime/model_final.pth`, SHA 아래 | `final100_B/model_final.pth`, SHA 아래 |
| 외부 process wall s | **1540.113550** | **1544.650001** | **1320.001536** |
| 첫 epoch s | 18.446767 | 18.913335 | 16.962810 |
| 100 epochs 합 s | 1504.043105 | 1509.153059 | 1284.431671 |
| steady epoch 중앙값 s | **14.983997** | **14.978252** | **12.688541** |
| scale 감소 / nonfinite loss | 12 / 0 | 11 / 0 | 12 / 0 |
| PyTorch peak allocated / reserved GiB | 4.174511 / 4.335938 | 4.177600 / 4.324219 | 4.172079 / 4.341797 |
| 마지막 25% mel 중앙값 | 16.372444 | 16.234068 | 16.382723 |
| 마지막 25% fm 중앙값 | 11.625000 | 11.578125 | 11.578125 |
| 마지막 25% kl 중앙값 | 0.705776 | 0.709181 | 0.702806 |
| 마지막 25% adv 중앙값 | 3.355348 | 3.358047 | 3.357328 |
| 같은 네 문장 변환·포화 지표 | 4 WAV 완료, raw/WAV finite | 4 WAV 완료, raw/WAV finite | 4 WAV 완료, raw/WAV finite |
| blind A/B/C 사용자 지각 판정 | PENDING | PENDING | PENDING |

외부 wall은 `epoch_series_100_status.json`의 subprocess 시작부터 종료까지다. epoch 합·first·steady와 별개다. memory 수치는 result의 각 epoch `peak_allocated_bytes`·`peak_reserved_bytes` 중 최댓값을 2³⁰으로 나눈 PyTorch allocator 기록이며 보드 전체 사용량 측정이 아니다.

B의 외부 wall은 A/A′ 대비 **220.112014–224.648464 s(3.668534–3.744141분)** 감소,
**14.291934–14.543648%** 단축, **1.166751–1.170188×**다.
steady는 **12.688541 s**, native 대비 **15.286904–15.319384%** 단축이다.
세 실행의 외부 wall 합은 **4404.765087 s(73.412751분)**였다.
첫 epoch·마지막 저장·export를 포함한 실제 전체 시간을 주 결과로 사용하며 짧은 epoch에서 외삽하지 않는다.

마지막 25% loss는 epoch 76–100의 775 step을 모은 중앙값이다.
nonfinite loss는 0이지만 AMP가 감지한 nonfinite gradient step은 A/A′/B 각각 D 5/5/5, G 12/11/12였고
scale 감소·skip에 반영됐다. loss와 gradient의 finite 판정을 혼동하지 않는다.

export SHA는 완료 result의 `model_export`에 저장된 값을 인용했다. 이 문서 갱신에서는 큰 모델 파일을 열거나 해시를 다시 계산하지 않았다.

| export | result에 기록된 SHA-256 |
|---|---|
| A `model_final.pth` | `5b95ead005109dec6894499916d3607e0edc0b4b57bfa1a67db103d4d702e0dd` |
| A′ `model_final.pth` | `dacb356af0a38753941c585be205687d51a613fbcf5c5adbc8b5964275724095` |
| B `model_final.pth` | `18fccd849b22bb742463fa24a6067dbf8f8cebc3363d5e19efe67f8b037528c8` |

native 반복 차이는 `100 × (A′ / A − 1)`로 계산했다. 외부 wall은 **+4.536450 s(+0.294553%)**, steady는 **−0.005745 s(−0.038342%)**, epoch 합은 **+5.109954 s(+0.339748%)**다. B 비교에서 외부 wall의 관측 native 범위는 1540.113550–1544.650001 s, steady 범위는 14.978252–14.983997 s다. 이는 두 native 실행에서 관찰한 차이이며 통계적 신뢰구간은 아니다.

| 마지막 25% loss | A 기준 A′ 변화율 `(A′−A)/A` | A 기준 B 변화율 `(B−A)/A` |
|---|---:|---:|
| mel | −0.845178% | +0.062780% |
| fm | −0.403226% | −0.403226% |
| kl | +0.482529% | −0.420828% |
| adv | +0.080457% | +0.059026% |

네 component 모두 A에서의 변화 크기 `|B−A|`가 `|A′−A|`보다 크지 않았다. 이는 관측된 두 native 실행과의 수치 비교이며 지각 품질 통과 판정은 아니다. 각 run의 보호 검사 통과도 아래 작업 끝 최종 보호 검사와 구분한다. 근거: `results/applio_v08/final100_A/result.json`, `final100_A_prime/result.json`, `final100_B/result.json`, `epoch_series_100_status.json`, `final_training_summary.json`.

기존 네 문장을 같은 native inference 설정으로 변환해 **12 WAV를 생성**했다. `audio_eval_final/result.json`은 `valid_for_listening=true`, raw generator·WAV finite, 운영·source model asset 무변경을 기록한다. 세 export 모두 inference model의 **457/457 key**를 로드했고 missing/unexpected key가 0이다. native patch 없이 pitch +2, RMVPE, index 0.3, volume envelope 0.3, protect 0.33, 문장별 동일 seed를 사용했다. raw generator 포화 지표는 tanh 뒤·trim/volume-envelope/peak scaling/PCM encoding 전에 수집했고 WAV 지표와 구분했다. raw의 abs ≥ 1 비율은 모두 0이었다. 같은 문장의 비정렬 파형에서 B 관련 RMSE는 A–A′ 반복 RMSE의 **1.570–1.866배**였다. 이는 파형이 같다는 근거가 아니며 지각 품질의 수치 합격선도 아니다. 수치·정의는 `NONCONV_V08_NOTES_KO.md` §12와 audio result를 함께 본다.

`audio_eval_final/LISTEN_BLIND.html`은 대응을 가린 A/B/C와 네 문장을 제공한다. `browser_qa.json`에서 12개 음원의 decode 완료(readyState 4), 정상 재생 진행과 다른 음원 재생 시 앞 음원의 정지를 확인했다. 확인 뒤 전부 정지·되감기했고 선택·메모는 비워 두었다. 이 동작 검사는 지각 품질 청취를 대신하지 않는다. A–A′의 원래 차이를 기준으로 B만 이상하게 들리는지는 사용자 판정으로 남긴다. loss·포화·WAV 차이 수치는 참고이며 사용자 판정 전에는 100 epoch 품질 통과나 운영 채택을 쓰지 않는다.

`final_protection_check.json`은 **2026-11-07 20:01:54.341721 UTC**에 `protected_before.json`의 기존 16개와 추가 987개, 총 **1,003/1,003개(3,238,186,037 bytes)**를 다시 해시해 `passed=true`를 기록했다. **changed / missing / errors는 모두 0개**이며 최종 idle GPU audit의 관측 KFD client도 0개다. 각 run의 중간 보호 검사와 별도로 수행한 작업 끝 검사다.

#### 적용·원복

연구 복사본과 `applio_epoch_v08.py`에서만 opt-in 적용한다. FAST DB는 각 run에 복사한다. `--dispatch`로 final manifest의 이긴 shape·방향만 사용하고 나머지는 native에 맡긴다. nonconv는 `--nonconv`와 다음 환경을 함께 선언한다.

| 설정 | 최종 100 epoch B에 고정한 값 |
|---|---|
| `APPLIO_RDNA2_ADAMW` | `native` |
| `APPLIO_RDNA2_ADAMW_G` / `APPLIO_RDNA2_ADAMW_D` | `native` / `single` |
| `APPLIO_RDNA2_SLICE_GATHER` | `1` |
| `APPLIO_RDNA2_GRAD_NORM` | `foreach` |
| `APPLIO_RDNA2_SKIP_EMPTY_CACHE` | `1` |
| `APPLIO_RDNA2_D_WN_CACHE` | `0` |

원복은 새 프로세스에서 opt-in을 제거하고 `db_fast_native` 복사본, `--dispatch`·`--nonconv` 없는 native run을 사용하는 것이다. `v08_run_epoch_series.py`는 run마다 `APPLIO_RDNA2_*`·`LD_PRELOAD`·compile cache/log 환경을 정리하고 명시한 값만 넣는다. 운영 launcher·설정·소스·pretrain·venv 설치를 변경하지 않는다. 기존 모델 이름에 `--cleanup`을 붙이지 않는다. workspace preload는 NORMAL seed에서만 사용한다.

## 6. Claude가 확인할 핵심 숫자 5개

이 절의 경로와 명령은 `rdna2-ops` 기준이다. 저장 JSON을 읽는 아래 명령은 stdlib만 사용하며 torch/ROCm import·GPU 실행이 없다.

| 번호 | 핵심 숫자 | 근거 |
|---|---|---|
| 1 | Conv operator self GPU **80.311%** | `native_profile_final/profile_summary.json` |
| 2 | 연산 **432/432**, 실제 ID **432/432·9종**, 선택 6방향 **1.075–1.530×** | baseline·selection·final replay·native metadata |
| 3 | direct **21 shape / 56방향**, C64/K7/d3 dW **5.825×** | `direct_selected_final/dispatch.json` |
| 4 | 실제 dtype·weight_norm **105/105 비교** | `direct_contract_initial/result.json` |
| 5 | 100 epoch B **1320.001536 s**, native 대비 **14.291934–14.543648%** 단축 | `final_training_summary.json`와 A/A′/B 완료 result |

~~~bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
python3 - <<'PY'
import json
from collections import Counter
from pathlib import Path

base = Path("results/applio_v08")
def read(rel):
    return json.loads((base / rel).read_text())

p = read("native_profile_final/profile_summary.json")
assert p["valid_for_catalog_shares"]
ops = {r["name"]: r for r in p["cpu_operator_top10"]}
print("1 Conv %:", sum(ops[n]["share_percent"] for n in
      ("aten::convolution_backward", "aten::miopen_convolution")))

old = read("solver_baseline144/result.json")
final = read("solver_selected_final_replay/result.json")
sel = read("solver_selected_final_db/selection_manifest.json")
assert final["status"] == "completed" and final["valid_for_timing"]
def rows(report):
    return {(c["case"]["name"], d): r for c in report["cases"]
            for d, r in c["directions"].items()}
old_rows, final_rows = rows(old), rows(final)
passed = sum(r["reference_error"]["screen_pass"] for r in final_rows.values())
chosen = [d for d in sel["decisions"] if d["selected"] != "preserve_baseline"]
gains = [old_rows[(d["case"], d["direction"])]["event_median_ms"] /
         final_rows[(d["case"], d["direction"])]["event_median_ms"] for d in chosen]
print("2 gates/total, selected, gains:", passed, len(final_rows), len(chosen),
      min(gains), max(gains))
attribution = read("solver_roofline/solver_attribution.json")
assert attribution["native_attribution_status_counts"] == {"confirmed_native_metadata_replay": 432}
print("2 native IDs:", attribution["native_attribution_status_counts"],
      attribution["observed_solver_counts"])

direct = read("direct_selected_final/dispatch.json")
assert direct["strict_gate_pass"] and not direct["contract_probe_only"]
counts = Counter(d for e in direct["entries"].values() for d in e["directions"])
target = next(e for e in direct["entries"].values()
              if e["case"]["name"] == "conv1d_B8_C64_T6400_O64_K7_S1_D3_G1_P9")
dw = next(e for e in target["evidence"] if e["direction"] == "dw")
print("3 shapes, directions:", len(direct["entries"]), dict(counts))
print("3 native/direct ms, gain:", dw["baseline_event_ms"],
      dw["candidate_event_ms"], dw["speedup"])

contract = read("direct_contract_initial/result.json")
assert contract["completed"] and contract["valid_for_contract"]
metrics = [m for c in contract["cases"] for m in c["metrics"].values()]
print("4 gate/total:", sum(m["screen_pass"] for m in metrics), len(metrics))
print("4 worst nrmse/max:", max(m["nrmse"] for m in metrics),
      max(m["max_over_reference_rms"] for m in metrics))

training = read("final_training_summary.json")
assert training["completed"] and training["training_checks_pass"]
wall = {arm: r["timing"]["external_whole_wall_s"]
        for arm, r in training["runs"].items()}
print("5 100 epoch A/A_prime/B external wall s:", wall)
print("5 B saved percent vs A/A_prime:",
      *[100 * (1 - wall["B"] / wall[arm]) for arm in ("A", "A_prime")])
PY
~~~

선택 처리 자체의 재현도 CPU만 사용한다. 새 출력 이름을 사용하고 완료 bench를 다시 돌리지 않는다.

~~~bash
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
~~~

별도 GPU 재현이 필요할 때의 명령 초안이다. 현재 학습과 겹쳐 실행하지 않으며 결과 경로를 새로 지정한다. 각 harness는 `gpu_guard_v06`를 사용한다.

~~~bash
~/DevSpace/applio-lab/.venv-rocm10/bin/python v08_direct_contract.py \
  --manifest results/applio_v08/direct_provisional_contract_probe/dispatch.json \
  --db results/applio_v08/direct_fast_db \
  --out results/applio_v08/direct_contract_reproduce

~/DevSpace/applio-lab/.venv-rocm10/bin/python v08_nonconv_bench.py \
  --device cuda --only all --profile \
  --snapshots results/applio_v08/optimizer_actual_snapshot/optimizer_0_before_step3.pt \
              results/applio_v08/optimizer_actual_snapshot/optimizer_1_before_step3.pt \
  --out results/applio_v08/reproduce_nonconv_gpu_actual.json
~~~

contract probe manifest는 경계 검증 입력이다. 실제 학습의 `--dispatch`에는 `direct_selected_final/dispatch.json`을 쓴다.

## 7. 실제 epoch·100 epoch 반복 명령 초안

완료 run을 덮어쓰지 않는다. coordinator는 기존 status 또는 out/db가 있으면 중단하므로 기존 `epoch_series_initial.json`을 그대로 재실행하지 않는다. 아래는 **독립 재현용 초안**이며 이번 문서 작업에서는 실행하지 않았다.

~~~bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path("results/applio_v08")
plan = root / "reproduce_epoch_series.json"
dispatch = str((root / "direct_selected_final/dispatch.json").resolve())
candidate_env = {
    "APPLIO_RDNA2_ADAMW": "native",
    "APPLIO_RDNA2_ADAMW_G": "native",
    "APPLIO_RDNA2_ADAMW_D": "single",
    "APPLIO_RDNA2_SLICE_GATHER": "1",
    "APPLIO_RDNA2_GRAD_NORM": "foreach",
    "APPLIO_RDNA2_SKIP_EMPTY_CACHE": "1",
    "APPLIO_RDNA2_D_WN_CACHE": "0",
}
runs = []
for name, kind in [
    ("A0", "native"), ("B_conv", "conv"), ("A1", "native"),
    ("B_full", "full"), ("A2", "native"),
]:
    prefix = root / ("reproduce_" + name)
    candidate = kind != "native"
    args = ["--dispatch", dispatch] if candidate else []
    if kind == "full":
        args.append("--nonconv")
    runs.append({
        "name": "reproduce_" + name,
        "out": str(prefix),
        "db": str(prefix) + "_db",
        "db_seed": str(root / ("solver_selected_final_db" if candidate else "db_fast_native")),
        "epochs": 4,
        "args": args,
        "env": candidate_env if kind == "full" else {},
    })
assert not plan.exists()
assert not plan.with_name(plan.stem + "_status.json").exists()
assert all(not Path(r[k]).exists() for r in runs for k in ("out", "db"))
with plan.open("x") as f:
    json.dump({"runs": runs}, f, ensure_ascii=False, indent=2)
print(plan)
PY
python3 v08_run_epoch_series.py --plan results/applio_v08/reproduce_epoch_series.json
~~~

compile은 최종 후보에서 제외했다. 실패 및 한 차례 복구의 정확한 args·env는 `epoch_series_compile.json`, `epoch_series_compile_boundary.json`에 보존한다. 복구에는 `--compile dec --compile-native-rng --compile-eager-conv`, `TORCHINDUCTOR_COMPILE_THREADS=1`, `TORCH_LOGS=graph_breaks,recompiles`를 사용했다. 완료 결과가 있으므로 인수 과정에서 다시 실행할 필요가 없다.

100 epoch는 아래와 같이 고정해 **A/A′/B 모두 완료**했다. 계획은 `epoch_series_100.json`이며 SHA-256은 `5f82934a952c805c60327785289a23cd798ffebe6e2d07e0efe6da3096bbc34d`다. 음성 변환도 완료했고 **사용자 청취 판정은 PENDING**이다. 완료한 이 plan을 재실행하지 않는다.

| plan 필드 | A | A′ | B |
|---|---|---|---|
| `name` / `out` | `final100_A` | `final100_A_prime` | `final100_B` |
| 독립 DB | `final100_A_db` | `final100_A_prime_db` | `final100_B_db` |
| `db_seed` | `db_fast_native` | `db_fast_native` | `solver_selected_final_db` |
| `epochs` | 100 | 100 | 100 |
| 공통 저장 args | `--save-every 100 --save-only-latest --export-final` | 왼쪽과 동일 | 왼쪽과 동일 |
| 추가 args | 없음 | 없음 | `--dispatch direct_selected_final/dispatch.json --nonconv` |
| `env` | 기본 native | 기본 native | 위 적용 표의 B_full eager 7개 환경 |

out·DB·seed 이름은 모두 `results/applio_v08/` 아래다. B의 dispatch 경로는 plan 안에서 연구 폴더의 절대 경로로 고정했다. fused AdamW·D weight_norm cache·decoder compile은 제외했다.

계획에 고정한 소스 SHA-256은 다음과 같으며 문서 갱신 시 실제 파일과 5개 모두 일치함을 확인했다.

| 파일 | SHA-256 |
|---|---|
| `applio_epoch_v08.py` | `953b67adb43a69b74a6b0d634980543138ef0d3d40846f639dc451e95c64fffd` |
| `v08_run_epoch_series.py` | `d4d7d6b63a5541d00789ad2c7148b20de068c2592efa9fdff13f74c96ac97a81` |
| `v08_direct_conv.py` | `6ef607af92b0ffd561a633093ff9bd62ae8428deb6f613aa2d38fca3a674b4e3` |
| `v08_nonconv_opt.py` | `4912353a4e45e13f18b0c33839ff78af33d872315c2415232fa32888109094a8` |
| `direct_selected_final/dispatch.json` | `22e41075581e8b2d969388244bf063adba59579942c4959ae892e2aeb4fcfa59` |

사전 예상에는 native steady 14.967511 s, native first 18.631871 s, candidate steady 12.570070 s, candidate first 16.536212 s를 사용하고 run마다 process overhead 약 33.7 s를 더했다. 합 72.717분에 최종 저장·export·음성 변환 여유는 별도로 더했다. 이 예상은 보존하며 완료 실측 합 73.412751분과 구분한다. 음성 변환은 완료했고 사용자 청취 판정은 PENDING이다.

새로운 독립 100 epoch 재현이 필요할 때만 다음처럼 확정 plan을 다른 이름으로 복제한다. 아래는 재현용 초안이며 현재 series와 겹쳐 실행하지 않는다. source SHA를 확인하고 세 run의 새 out·DB·status가 모두 없는지 확인한다.

~~~bash
python3 - <<'PY'
import datetime
import hashlib
import json
from pathlib import Path

root = Path("results/applio_v08")
original = root / "epoch_series_100.json"
plan = root / "reproduce_epoch_series_100.json"
data = json.loads(original.read_text())
for name, expected in data["source_sha256"].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
data["reproduction_of"] = str(original)
data["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
for run in data["runs"]:
    run["name"] = run["name"].replace("final100_", "reproduce100_", 1)
    run["out"] = str(root / run["name"])
    run["db"] = str(root / (run["name"] + "_db"))
assert not plan.exists()
assert not plan.with_name(plan.stem + "_status.json").exists()
assert all(not Path(r[k]).exists() for r in data["runs"] for k in ("out", "db"))
with plan.open("x") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(plan)
PY
python3 v08_run_epoch_series.py --plan results/applio_v08/reproduce_epoch_series_100.json
~~~

세 완료 result의 export로 `audio_eval_final` 변환을 완료했다. 아래는 독립 재현 때 runtime Python으로 `v08_audio_eval.py --execute`에 전달할 인자다. 완료한 변환을 다시 실행할 필요는 없으며 재현 시 audio 출력은 새 경로로 지정한다. 사용자 청취 판정은 PENDING이다.

~~~text
--model A=results/applio_v08/final100_A/model_final.pth
--model A_prime=results/applio_v08/final100_A_prime/model_final.pth
--model B=results/applio_v08/final100_B/model_final.pth
--db-seed results/applio_v08/db_fast_native
--out <연구 폴더 안의 새 audio 출력 경로>
~~~

기존 네 문장·준비된 연구 asset을 사용했다. 완성된 `audio_eval_final/LISTEN_BLIND.html`·WAV·`blind_manifest.json`을 사용자에게 전달하고, 대응인 `blind_key.json`은 청취 전에 노출하지 않는다. 사용자 지각 판정 전에는 음질 완료로 표시하지 않는다.

## 8. RTX 5070 비교 문단 틀

> 같은 `voice_model_A` 조건에서 RX 6900 XT native 100 epoch의 외부 process wall은 A **25.668559분**, A′ **25.744167분**, 최종 B_full eager는 **22.000026분**이었다. 초기 조건·학습 의미·저장 정책을 유지한 이번 solver·직접 Conv·nonconv 구성으로 측정된 단축은 **3.668534–3.744141분(14.291934–14.543648%)**이다. native 반복 wall 차이는 +0.294553%였다. 보고된 RTX 5070의 12분까지 남은 약 10분을 GPU 하드웨어의 불가피한 몫으로 수치 배분할 수는 없다. 그 12분을 같은 코드·데이터·계측 경계로 재현하지 않았고, dot2·메모리 calibration도 ROCm/MIOpen 구현·launch 비용까지 분리한 하드웨어 한계 측정이 아니다. 줄인 몫은 위 실측으로 제시하고 줄일 수 없는 몫의 정확한 분·비율은 **현재 미식별**로 남긴다. 학습 완료와 별개로 지각 품질 채택은 청취 결과를 기다린다.

작업 끝 보호 검사까지 완료했다. 사용자 지각 판정이 끝나면 이 문단과 카탈로그의 최종 채택 상태를 함께 갱신한다. 연산 통과·100 epoch 학습 통과·사용자 지각 품질 통과는 별도 판정으로 유지한다.

