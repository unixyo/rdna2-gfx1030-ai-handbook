# Applio v1.0 — 실제 운영 100 epoch 결과

2026-11-08. **B′는 실제 운영 경로의 100 epoch 프로세스 시간을 21분 25.2초에서 19분 32.2초로 줄였다. 절감은 113.019초, 8.794%다.** 수치·AMP 계약, 같은 세션의 교대 비교, 양쪽 100 epoch 학습과 익명 청취 자료 생성을 완료했다. **최종 지각 품질은 사용자 청취 판정 전**이다.

사용자 확인용 자료는 익명 네 문장 청취 페이지 (`audio_eval/LISTEN_BLIND.html`, 연구 폴더에만 있음)다. A의 GPU 실험과 C의 원인·필터 검증은 완료했고, D는 최종 자원 점검에서도 원래 재개 조건을 만족하지 못해 서버를 실행하지 않았다. 연구 전체 종료 보호 검사에서 **1,022개 파일의 변경·누락·읽기 오류가 모두 0개**, 최종 KFD client도 0개였다. 남은 사용자 판정은 새 B′의 익명 청취다. 원지시는 [PRO_TASK_V10_KO.md](../../../PRO_TASK_V10_KO.md)다.

## 1. 최종 성능과 판정

| 같은 측정 범위 | B | B′ | 실제 개선 |
|---|---:|---:|---:|
| 100 epoch recorder 시간 합 | 1,261.70144773초 | 1,153.36516380초 | 108.33628392초, 8.5865% 감소 |
| 준비·종료를 포함한 프로세스 전체 | 1,285.24536656초 | 1,172.22650829초 | 113.01885827초, 8.7936% 감소 |
| 정상 epoch 3–100 중앙값 | 12.56554866초 | 11.46764910초 | 1.09573885배, 시간 8.7374% 감소 |

정본은 [final_epoch_protocol/result.json](final_epoch_protocol/result.json)과 [B raw result](final_B1/result.json), [B′ raw result](final_Bprime1/result.json)다. 양쪽 모두 **100 epoch·3,100 step, 최종 export, timing audit를 통과**했다. 초기 G/D state SHA는 서로 같고 앞선 교대 실행과도 같다. loss 비유한 0, KFD 큐 최대 7, 관측된 외부 client 0, 실행별 운영 추적 파일 변경 0이었다. AMP scale 감소는 B 12회·B′ 11회다.

이는 짧은 실험의 중앙값에 100을 곱한 예상치가 아니다. 실제 완료한 양쪽 실행의 같은 시간 범위를 비교했다. 각 arm의 100 epoch는 한 번씩 실행했으며, 그 전에 교대 두 쌍으로 개선의 일관성을 확인했다. 속도와 finite loss는 음질 채택을 대신하지 않는다.

## 2. 기준 B, 실행 조건, 사전 관문

기준 B는 Claude가 적용한 v0.8 운영 구성이다. DataLoader 워커 GPU 숨김, 선택 direct Conv·nonconv·solver가 이미 포함된다. 기존의 약 22 → 12.6초/epoch 변화는 v0.8 운영 롤아웃 기록이며 이번 B→B′의 새 효과에 포함하지 않는다. [v0.8 정본 §11](../applio_v08/APPLIO_V08_KO.md)을 보존했다.

| 항목 | 이번 실행 조건 |
|---|---|
| GPU·컴파일러 | RX 6900 XT, gfx1030, Triton 3.8.0 |
| 학습 환경 | 운영 `.venv-rocm10`, torch 2.11.0+rocm10.0.0, HIP 7.15.26333 |
| 실제 진입점 | 운영 `core.py train → spawn → rvc/train/train.py` |
| 데이터·정밀도 | voice_model_A의 config·filelist, batch 8, FP16 AMP |
| 원본 metadata | 1 speaker, KLM-SVM embedder; 실행 초기에 일치 assertion |
| 학습 seed | 1234, 같은 KLM pretrain·초기 G/D state |
| 연구 적용 | 별도 PYTHONPATH/import hook으로 해당 프로세스에만 B′ 연결 |
| 모델·출력 | 모두 새로운 v10 모델 이름과 연구 결과 폴더 |

운영 Python은 `~/DevSpace/applio-lab/.venv-rocm10/bin/python`으로 고정했다. 연구 실행기가 이 경로로 실제 운영 코드를 호출한다. 운영 소스·런처·venv·공용 MIOpen DB를 수정하지 않았으며, 기존 모델 이름으로 cleanup을 실행하지 않았다. source·selection snapshot과 전체 명령·환경은 각 실행 폴더의 `command.json`, `result.json`, `source/`에 보존된다.

[PREDECLARED_GATES.json](PREDECLARED_GATES.json)에 GPU 실행 전에 다음 기준을 저장했다.

| 관문 | 고정 기준 |
|---|---|
| 개별 연산 | seed 20261108, 독립 FP32 참조, y/dX/dW finite, NRMSE ≤ 0.002, max error/reference RMS ≤ 0.02 |
| 새 native 대체 방향 | 가장 빠른 정확한 native와 비교, event ≥ 1.10배·wall ≥ 1.05배 |
| 융합 Conv 쌍 | 기존 B의 forward+backward 전체와 비교, event·wall 모두 ≥ 1.03배 |
| 운영 짧은 교대 | B/B′/B/B′ 각 8 epoch, 사전 지정 epoch 3–8, 두 인접 쌍 모두 개선·통합 ≥ 1.02배 |
| 최종 검증 | 관문 통과 후 100 epoch씩, 같은 네 문장 익명 사용자 청취 |
| GPU 보호 | GpuAudit, 외부 GPU 작업과 겹친 시간 무효, 운영 학습 KFD 큐 ≤ 7 |

NORMAL·FP32 참조에는 `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0`과 `MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53=0`을 함께 사용했다. 새 커널 실험에는 `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1`을 둔다.

## 3. 실제 운영 프로파일에서 줄어든 비용

정본은 [B profile](profile_B_target/result.json), [B′ profile](profile_Bprime/result.json)다. 각각 실제 운영 경로의 step 40–44를 계측했고, 조건·export·GPU audit·보호 검사를 통과했다. profiler를 켠 epoch 시간은 정상 속도 판정에 포함하지 않았다.

| kernel 계열, 5 step 합 | B ms | B′ ms |
|---|---:|---:|
| Conv2d, native+신규 | 700.494 | 565.941 |
| Conv1d, native+v08 direct+신규+융합 | 653.687 | 656.517 |
| ConvTranspose, native+신규 | 29.603 | 12.376 |
| optimizer | 76.983 | 76.983 |
| reduction·normalization | 65.609 | 70.237 |
| elementwise | 44.652 | 42.710 |
| copy·cast kernel | 39.760 | 40.046 |
| activation | 22.855 | 19.374 |
| 기타 GEMM | 22.179 | 22.277 |
| 그 외 | 23.631 | 23.841 |
| **전체 kernel 합** | **1,679.453** | **1,530.302** |

전체 kernel 시간은 **8.881% 감소**했다. 큰 절감은 period Conv2d와 ConvTranspose에서 나왔다. 융합은 activation·잔차 연산을 Conv 안으로 옮기므로, standalone activation 시간 감소를 그대로 융합의 순이익으로 세지 않는다. 융합과 native 교체를 각각 분리한 epoch 실험은 하지 않았다.

연산자 표의 self-device 분모는 B 1,685.778 ms·B′ 1,536.640 ms다. B′에서 `convolution_backward` 39.85%, `miopen_convolution` 20.63%, `_CandidateConv` 4.18%, `_FusedPairBackward` 4.12%였다. **이 operator 귀속 시간과 위 kernel 시간을 더하지 않는다.** [B operator/host 요약](profile_B_target/profile_summary.json), [B′ operator/host 요약](profile_Bprime/profile_summary.json), [최종 kernel 표](gap_Bprime_final/kernels.csv)에 전체 자료가 있다.

프로파일 window는 2.18808 → 1.99909초, 기록된 GPU 활동이 없는 구간은 0.50230 → 0.46245초였다. `hipMemcpyWithStream`의 CPU 대기 합 0.92280 → 0.66489초에는 GPU 작업을 기다린 시간도 포함된다. 계측·annotation 비용이 있으므로 이 수치를 전부 제거 가능한 host 비용으로 해석하지 않는다.

B 77,026개·B′ 63,555개 kernel을 HIP runtime correlation으로 모두 연결했다. fallback·파싱 오류는 0이다. GPU timing은 그대로 두고 CPU 귀속만 수정했으며, 방향별 FLOP·조건부 roofline 정본은 [GAP_5070_KO.md](GAP_5070_KO.md)다.

## 4. ResBlock 융합: 구현과 수치·메모리 검증

기존 `leaky_relu → Conv1 → leaky_relu → Conv2 → + skip`에서 각 Conv의 입력 load에 LeakyReLU를, Conv2 store에 skip 덧셈을 합쳤다. Conv1·Conv2 자체는 각각 별도 kernel이다.

역전파에서는 Conv2 dX store에 두 번째 LeakyReLU mask를 적용하고, Conv1 dX store에 첫 mask와 skip gradient 합산을 적용한다. dW는 저장된 activation 대신 preactivation에서 LeakyReLU를 계산한다. FP32 split partial과 고정 순서 reduction을 사용하며 새 atomic 경로를 추가하지 않았다. native로 남는 backward 방향에만 필요한 activation을 다시 만든다.

AMP·weight_norm 경계는 기존 B와 맞췄다. dot2는 FP32로 누산하고 **Conv 결과를 FP16으로 반올림한 뒤 bias를 더하는 순서**를 보존한다. mask와 잔차 gradient의 반올림 순서도 유지한다. weight_norm graph는 custom 함수 밖에 남고, master parameter와 해당 gradient는 FP32다. 실제 선택은 검증된 shape·입력 dtype에 한정된다.

| 최종 source의 검증 | 결과 |
|---|---|
| 전체 대상·선택 | 18쌍 검증, C32/C64 12쌍 선택 |
| 개별 연산 FP32 참조 | 108/108 metric 통과 |
| 실제 ResBlock AMP·weight_norm 계약 | 144/144 metric 통과 |
| 연산 최악 수치 | NRMSE 0.000438903, max/reference RMS 0.004425903 |
| 선택된 쌍 event 배속 | 1.039106–1.134607배 |
| 선택된 쌍 wall 배속 | 1.039146–1.140568배 |
| 제외 | C64/K11 세 쌍은 1.03배 관문 미달, C128/K11 세 쌍은 더 느림 |

전체 sweep의 입력은 **FP16**이고 FP32는 연산 참조와 master/gradient dtype이다. FP32 입력 전체 sweep까지 수행했다는 뜻은 아니다. 개별 Conv의 FP32 참조는 AMP에서 반올림된 동일 activation을 FP32로 확장한 기준이며, 전체 ResBlock을 중간 반올림 없이 FP32로 계산하는 참조와 구분한다. 검증된 고정 입력의 B/B′ 호출 경계 y/dX/master·bias gradient는 모두 바이트 일치했다. 전체 학습이나 모든 입력의 바이트 동일성을 의미하지 않는다.

대표 B8/C32/T12800/K3/d1에서 현재 source의 pair event 중앙값은 0.785988 → 0.705912 ms, kernel 수는 **28 → 22**, allocator peak 증가량은 **33,180,672 → 26,633,728 bytes**였다. 실제 allocator 차이는 6,546,944 bytes다. standalone LeakyReLU·backward mask·add가 사라졌고, 정적 tensor read/write 감소 모델은 forward·backward 각각 39,321,600 bytes다.

이 대표 쌍에 한해 추가로 현재 gfx1030의 `FETCH_SIZE`·`WRITE_SIZE` 하드웨어 카운터를 수집했다. warmup 3회 뒤 forward+backward 한 쌍을 계측하고, B/B′와 읽기/쓰기를 별도 프로세스로 직렬 실행했다. 네 팔의 input·dy·FP32 master·출력/gradient SHA와 source가 같고, 실제 kernel 28/22개가 모두 named ROCTX 구간 안에 있다. counter/marker PID와 kernel의 dispatch·agent·queue·이름·시작/종료 시각을 엄격히 대조했다.

| 대표 쌍의 GL2C/EA 인터페이스 요청량 | B bytes | B′ bytes | 감소 |
|---|---:|---:|---:|
| 읽기 (`FETCH_SIZE`) | 132,801,536 | 93,509,760 | 29.587% |
| 쓰기 (`WRITE_SIZE`) | 52,968,064 | 22,658,816 | 57.222% |
| 읽기+쓰기 | **185,769,600** | **116,168,576** | **69,601,024 bytes, 37.466%** |

이 결과로 해당 계측 구간에서 실제 인터페이스 요청량이 줄었음을 확인했다. **GL2C/EA 요청량을 물리 GDDR6/DRAM bytes와 동일시하지 않는다.** arm·metric당 한 쌍이므로 반복 변동은 측정하지 않았으며, 12개 융합 쌍 전체에 이 감소율을 적용하지 않는다. 계측 중 시각은 기존 비계측 event/epoch 성능을 대체하지 않는다. 정본은 [fusion_traffic_comparison_02.json](fusion_traffic_comparison_02.json), [수집·복구·재현 메모](FUSION_TRAFFIC_PREPARATION_KO.md), [CPU 집계기](../../v10_summarize_fusion_traffic.py)다.

정본은 [fusion_full/result.json](fusion_full/result.json), [selection.json](fusion_full/selection.json), [상세 융합 메모](FUSION_IMPLEMENTATION_NOTES_KO.md)다. 선택 파일은 source SHA `c824ee32ccaf69d6b17b605f68687bc556b4958c2d951663e0563b481ee780b6`에 묶인다.

## 5. native 잔류 Conv: 전체 검증과 남은 방향

세 완료 실험을 source·benchmark SHA와 gate가 같은지 확인해 병합했다. 중복 shape와 선언한 169방향의 누락은 0이었다. 이미 v0.8 직접 경로로 선택된 방향을 다시 측정 대상에 넣지는 않았다. [native_selected/summary.json](native_selected/summary.json), [direction_summary.csv](native_selected/direction_summary.csv)가 정본이다.

| 범위 | 측정 방향 | 새로 선택한 방향 |
|---|---:|---:|
| 잔류 Conv1d | 13 | 2 |
| discriminator period Conv2d | 144 | 59 |
| ConvTranspose1d | 12 | 7 |
| **합계, 60 shape** | **169** | **68** |

후보 **531/531**이 독립 FP32 연산 참조를 통과했고 실행 오류·native 수치 실패는 0이다. 선택된 31 shape의 39개 입력 dtype 변형에서 AMP·weight_norm 계약 **195/195**가 통과했다. C192의 실제 논리 입력 FP32와 AMP 계산 입력 FP16을 구분해서 검사했다.

새 dW는 `CI×K`를 출력 축으로 묶고 전체 batch를 reduction에 포함한다. split=1이면 FP32 scratch와 후처리 reduction을 생략한다. ConvTranspose는 동치 normal-convolution의 adjoint와 stride phase로 표현하고, period Conv2d는 NCHW를 유지하며 높이·period 위치를 함께 타일링한다. 적용 범위를 벗어난 geometry는 기존 경로로 간다.

대표 두 dW에서 native는 작은 batch별 GEMM 8회·im2col 8회·누적 초기화 1회로 **17 kernel**, 신규 구현은 **2 kernel**이었다. 출력 matrix·MT·GSU를 교차 확인하면 native의 batch별 작은 GEMM은 출력 tile 하나인 반면, 신규 partial은 256 workgroup으로 reduction을 분산한다. 따라서 큰 개선은 im2col 제거와 부족한 병렬도를 함께 고친 결과다. trace의 grid 인수 단위만으로 workgroup 수를 단정하지 않았다.

| 대표 방향 | 가장 빠른 정확한 native ms | 새 후보 ms | event 배속 |
|---|---:|---:|---:|
| period3 C1→32/K5/s3 dW | 0.906734 | 0.033158 | 27.3458배 |
| ConvTranspose C64→32/T6400/K4/s2 dW | 2.399234 | 0.110454 | 21.7216배 |

micro 배속을 전체 학습 배속으로 확대하지 않았다. 방향별 native 기준은 초기 측정과 recheck 중 빠른 값을 사용했고, 할당·partial·reduction을 포함한 event·wall을 모두 판정했다. source와 trace 근거는 [NATIVE_CONV_NOTES_KO.md](NATIVE_CONV_NOTES_KO.md)에 있다.

이번에도 native로 남긴 방향은 다음과 같다.

- C128/K11 d1 dX, d1/d3/d5 dW: 새 변형이 이기지 못했다.
- C192→384/K5 dW 다섯 shape: native 유지. T74의 최선 1.0954배는 1.10배 관문 미달이었다. 새로 발견한 T220·T298은 forward만 추가했고 dX는 native다.
- ConvTranspose forward 네 shape와 C512→256 dX: native가 더 빨랐다.
- period Conv2d 144방향 중 85방향: native 유지. 큰 channel의 Winograd 경로 등이 포함된다.

이는 이번 후보가 이기지 못했다는 결과다. hardware 최적값을 증명한 것은 아니다. Winograd/FFT에 직접 Conv FLOP을 대입한 ISA 활용률을 만들지 않았고, 논리 tensor bytes를 실제 DRAM traffic으로 취급하지 않았다.

최종 [native dispatch](native_selected/dispatch.json) SHA는 `37f1fede6701d2cc79efb60a1eb81afb7d568678df74cd11ba2a1fbd1d81a705`다. 기존 v0.8의 선택 방향에 새 68방향을 더했으며 융합 대상과 native 교체 대상의 충돌을 CPU로 검토했다. B′ 학습에서 선택된 fusion 12쌍·native 31 shape의 실제 hit를 모두 확인했다.

## 6. 운영 교대 비교와 100 epoch 연결

[short_epoch_protocol/result.json](short_epoch_protocol/result.json)은 B1 → B′1 → B2 → B′2를 각 8 epoch 실행한 결과다. 사전 지정한 epoch 3–8을 arm당 12개씩 모은 중앙값은 **12.54207456 → 11.44752336초, 1.09561467배**였다. 인접 쌍은 1.09488874배·1.09436873배로 모두 개선돼 1.02배 관문을 통과했다.

| 순서 | epoch 3–8 중앙값 초 | 8 epoch 합 초 | 프로세스 전체 초 |
|---|---:|---:|---:|
| B1 | 12.52697527 | 104.35860133 | 121.56132659 |
| B′1 | 11.44132257 | 115.03412008 | 132.05544478 |
| B2 | 12.54604340 | 104.31990671 | 121.56849851 |
| B′2 | 11.46418297 | 96.03488183 | 114.06549995 |

B′1 첫 epoch 34.881초에는 최초 연구 커널 컴파일이 포함됐다. 이 비용은 전체 시간에 남기고 정상 성능은 미리 정한 epoch 3–8로 판정했다. 네 run의 초기 G/D SHA는 같고 loss 비유한은 없었다. AMP scale 감소는 B1 10회, 나머지 11회였다.

100 epoch 시작 직전 **B 약 21분·B′ 약 19분, 합계 40–42분**의 예상을 고지했다. [FINAL_ETA_KO.md](FINAL_ETA_KO.md)에 기록돼 있다. 이후 양쪽 학습과 export를 완료했고 §1의 실제 시간을 얻었다. 짧은 교대의 중앙값×100 환산 20.9035→19.0792분은 이 최종 실측과 구분한다.

| 완료 모델 | 파일 크기 | SHA256 |
|---|---:|---|
| `logs/v10_final_B1_20261108/v10_final_B1_20261108_100e_3100s.pth` | 55,180,900 B | `c65dcbafeb9f08a10a2373a3a738943ce64f617ce14350b82f4641b766cd3fad` |
| `logs/v10_final_Bprime1_20261108/v10_final_Bprime1_20261108_100e_3100s.pth` | 55,183,215 B | `51bf001cad39a7baae1f9a9c67e78589429213aeea5a56c328dd04edd0fa2d6b` |

두 경로의 기준은 `~/DevSpace/applio-lab/`이다. source·selection·raw result SHA는 최종 protocol에 고정돼 있다. 완료된 학습과 청취 생성을 재개 작업으로 다시 실행할 필요는 없다.

## 7. 익명 청취 자료: 기술 완료, 사용자 판정 대기

공개 LISTEN_BLIND.html (`audio_eval/LISTEN_BLIND.html`, 연구 폴더에만 있음) 한 파일에 기존 네 문장의 음성 8개가 들어 있다. 파일 크기는 4,318,552 bytes, SHA256은 `d11c480728be685565578c3c3c66ab9d5f40ca38be0017e54dbf691643ad9db8`이다. WAV header는 모두 **40,000 Hz·mono·PCM16**임을 CPU로 확인했다. 청취 기호와 학습 모델의 대응은 이 보고서에 포함하지 않는다.

최종 100 epoch/3,100 step 모델을 native FP32 inference로 읽었다. 생성 시 weight load 완전성, generator의 FP32 입력·가중치·출력, 문장별 같은 입력·난수 조건을 검사했다. 학습 후보 patch는 inference에 적용하지 않았다.

`audio_eval/result.json`의 공개 가능한 상태·보호 필드만 다시 읽어 다음을 확인했다. 익명 대응 필드와 private key 파일은 열지 않았다.

| 기술 확인 | 결과 |
|---|---|
| completed / valid_for_listening | true / true |
| native weight load / FP32 / same generator inputs | 모두 true |
| 모델·source 자산 변경 | 없음 |
| GPU audit / foreign client | 유효 / 0 |
| 생성 전 보호 파일 | 1,022개, 변경 0·누락 0·읽기 오류 0 |
| 생성 후 보호 파일 | 1,022개, 변경 0·누락 0·읽기 오류 0 |
| 새 다운로드 | 0 |
| quality_judgment | pending_user_blind_listening |

이 결과는 청취 자료가 기술적으로 준비됐다는 판정이다. **사용자가 B/B′를 구분하지 못했다거나 음질이 동등하다는 v1.0 판정은 아직 없다.**

## 8. 진단 실패와 수정 근거

| 진단 자료 | 문제와 처리 | 최종 근거 |
|---|---|---|
| `profile_B/` | 새 모델의 model_info 부재로 109 speaker/contentvec fallback; 추가 scale 조회가 step당 동기화 하나를 만듦; native exporter의 metadata 누락 분기에서 local 변수 오류 | 원본 metadata를 runtime 읽기 경로로 지정, 추가 scale sync 제거, 연구 프로세스 사본의 누락 export 분기만 보완. [진단 설명](profile_B/DIAGNOSTIC_NOTES_KO.md) 보존 |
| `profile_B_faithful/` | spawn/runpy에서 함수 globals와 모듈 dictionary가 달라 의도한 설정이 연결되지 않음 | 자체 프로세스 그룹을 중지하고 실제 `run.__globals__`에 연결; 첫 epoch 전 metadata assertion. 완료된 `profile_B_target/`만 기준선으로 사용 |
| `fusion_smoke/` | 구형 source의 autocast가 bias sum을 FP32로 승격 | 기존 B의 FP16 bias reduction 경계를 명시하고 새 source로 전체 18쌍 검증. 구형 28→20 kernel 결과는 현재 28→22 결과와 분리 |
| `gap_B_target/`, `gap_B_target_reviewed/` | 같은 autograd backward가 External id를 공유해 일부 dX를 dW에 잘못 귀속 | GPU kernel→HIP launch correlation→CPU scope로 CPU 재분석. B 77,026개·B′ 63,555개 전부 연결 |
| `gap_Bprime_steady/` | C32 융합 5개 행에서 cold-DRAM 모델이 실제 시간보다 길어짐 | 논리 bytes와 실제 DRAM traffic을 구분하고 해당 조건부 합계에서 제외. `gap_Bprime_final/`이 정본 |
| `fusion_traffic_FETCH_SIZE_B_01/` | AQL profiler가 요구한 unversioned 라이브러리 이름이 wheel에 없어 초기화 실패·SIGABRT handler 재진입 | 실패 소유 프로세스만 정리. 연구 폴더 symlink로 같은 wheel `.so.1`을 가리켜 복구; 운영 파일·라이브러리·driver 변경 없음. 새 `_02` 네 팔 완료 |
| `fusion_traffic_batch_02/`의 최초 CPU summary | 실제 kernel CSV에는 `Process_Id` 열이 없어 기존 요약기가 실패 | GPU 결과는 정상 완료였다. 별도 CPU 집계기로 PID-bearing counter/marker와 엄격 join하여 기존 CSV만 재분석. 성공한 GPU 팔을 재실행하지 않음 |

진단용 raw 파일은 보존했다. 조건이 복구되기 전 profile epoch, 수정 전 source의 선택 파일, 폐기된 방향 귀속 모델을 최종 성능 근거에 섞지 않았다.

## 9. 최종 보호 검사와 전체 연구 판정

이번 baseline은 [protected_before.json](protected_before.json)의 **1,022개 파일**이다. v0.8 SHA와 다른 Claude 롤아웃 파일 4개는 이번 시작 상태에 포함했으며 원복하지 않았다. 학습 실행별 추적 SHA와 음성 생성 전후 전체 1,022개 검사에서 변경이 없었다.

모든 GPU 실행이 끝난 뒤 **2026-11-08 11:51:03 UTC**에 `v10_prepare.py after`를 실제 실행했다. [protected_after.json](protected_after.json)은 **1,022개·3,241,275,343 bytes, 변경 0·누락 0·읽기 오류 0, unchanged=true**다. 운영 Applio source·런처·기존 보호 자산·공용 MIOpen 파일과 원 카탈로그·가이드의 시작 SHA를 유지했다. [gpu_idle_after.json](gpu_idle_after.json)의 최종 GpuAudit는 유효하고 foreign client·오류·KFD client가 모두 0이다. 완료 결과를 새 세션에서 다시 실행할 필요가 없다.

| 질문 | 최종 상태 | 확인한 결과 또는 남은 한계 |
|---|---|---|
| A — Applio 남은 비용 | 연산·실제 학습·추론 검증 완료, 사용자 청취 대기 | 12개 융합 쌍과 새 native 대체 68방향. 실제 100 epoch 프로세스 시간 113.019초 절감 |
| B — RTX 5070과의 차이 | 조건부 모델 완료, 유일한 인과 분할은 미식별 | 같은 알고리즘의 ceiling 모델을 제시했지만 외부 12분의 조건·시간 범위가 같다는 증거가 없음 |
| C — TunableOp 비결정 | Qwen·Anima의 원인 경로와 연구 필터 검증 완료 | FP32 atomic split-K 행만 Default로 변경. Anima 새 2프로세스 cand 6장 동일, off 대비 1.78945배 유지. Qwen 같은 프로세스 3장 동일, 필터 warm 시간 비용 약 1.32% |
| D — Strata warm | 재개 근거 없음, 서버/GPU 실행 0회, 미판정 | 최종 CPU 점검 RAM 35.9092 GiB·expert cache 0%, 이전 실패보다 불리. 서비스 inactive·KFD 0이어도 원래 재개 조건은 별도로 유지 |

C의 Qwen 필터 새 프로세스 반복과 MOSS는 이번에 실행하지 않았다. D의 작은 운영 파일은 별도 41개 SHA 전후가 모두 같았으며, 156 GB의 대용량 입력은 cache를 바꾸지 않도록 읽지 않았다. 운영 경로에 B′나 TunableOp 필터를 적용하지 않았다.

- 5070 차이의 조건부 분해·미식별 범위: [GAP_5070_KO.md](GAP_5070_KO.md).
- TunableOp 원인 재현·필터 속도·PNG 검증: [TUNABLEOP_DETERMINISM_KO.md](TUNABLEOP_DETERMINISM_KO.md).
- Strata 자원 사전점검·원래 보호 관문·재개 여부: [STRATA_WARM_KO.md](STRATA_WARM_KO.md).
- 카탈로그·가이드에 제안할 문안: [V10_CATALOG_GUIDE_PROPOSAL_KO.md](V10_CATALOG_GUIDE_PROPOSAL_KO.md).

A의 시간 개선은 확인된 사실이다. 융합과 native 교체의 개별 epoch 기여량은 분리 측정하지 않았고, 외부 RTX 5070의 12분과 남은 차이를 하드웨어/소프트웨어로 유일하게 나누지 못했다. 새 B′의 지각 품질은 사용자 청취 후 판정한다. 다음 세션은 완료한 GPU 실험을 반복하지 않고 이 청취 응답과 원자료를 이어받는다.

## 10. 구현·원자료·재현 입구

| 내용 | 정본·실행기 |
|---|---|
| 운영 경로 주입·학습 프로토콜 | [v10_run_operating.py](../../v10_run_operating.py), [v10_epoch_protocol.py](../../v10_epoch_protocol.py), [v10_core_hook.py](../../v10_core_hook.py) |
| 융합 구현·재현 조건 | [v10_fused_resblock.py](../../v10_fused_resblock.py), [bench_v10_fusion.py](../../bench_v10_fusion.py), [융합 상세 메모](FUSION_IMPLEMENTATION_NOTES_KO.md) |
| native 구현·재현 조건 | [v10_native_conv.py](../../v10_native_conv.py), [bench_v10_native_conv.py](../../bench_v10_native_conv.py), [native 상세 메모](NATIVE_CONV_NOTES_KO.md) |
| native 원 측정 3개 | [C192 smoke](native_c192_smoke/result.json), [family smoke](native_family_smoke/result.json), full sweep (`native_full/result.json`, 연구 폴더에만 있음) |
| 선택 manifest | [fusion selection](fusion_full/selection.json), [native dispatch](native_selected/dispatch.json) |
| 정상 속도·100 epoch | [short protocol](short_epoch_protocol/result.json), [final protocol](final_epoch_protocol/result.json) |
| 실제 운영 명령·환경 | [B command.json](final_B1/command.json), [B′ command.json](final_Bprime1/command.json) |
| CPU profile·roofline 분석 | [v08_analyze_profile.py](../../v08_analyze_profile.py), [v10_gap_analysis.py](../../v10_gap_analysis.py) |
| 사용자 청취 | 공개 LISTEN_BLIND.html (`audio_eval/LISTEN_BLIND.html`, 연구 폴더에만 있음) |

운영 GPU 실행 파일은 `~/DevSpace/applio-lab/.venv-rocm10/bin/python`이다. GPU 재현의 새 out·DB·seed·dtype·round·rep·선택 기준은 상세 메모와 저장된 command/result를 따른다. 이 문서의 아래 확인 명령은 `/usr/bin/python3`와 표준 라이브러리만 쓰며 torch를 import하지 않는다. 운영 파일·기존 결과를 쓰거나 GPU 작업을 시작하지 않는다.

## 11. Claude가 GPU 없이 확인할 핵심 숫자 5개

1. 실제 B 100 epoch 프로세스 시간: **1,285.245366560초**.
2. 실제 B′ 100 epoch 프로세스 시간: **1,172.226508294초**.
3. 짧은 운영 교대의 통합 steady 배속: **1.095614673배**.
4. 새로 선택된 native 대체 방향: **68개**.
5. 최종 source에서 선택된 융합 쌍: **12개**.

아래 명령을 실제 실행해 다섯 값을 확인했다. 마지막 assertion은 음성 결과에서 보호·기술 상태만 검사하며, 청취 라벨 대응은 읽거나 출력하지 않는다.

~~~bash
/usr/bin/python3 -B - <<'PY'
import json
from pathlib import Path
root=Path('~/DevSpace/rdna2-pro-research-20261102/rdna2-ops/results/v10')
def read(relative):
    return json.loads((root/relative).read_text())
b=read('final_B1/result.json')
bp=read('final_Bprime1/result.json')
short=read('short_epoch_protocol/result.json')
native=read('native_selected/summary.json')
fusion=read('fusion_full/selection.json')
assert b['completed'] and b['valid_for_timing']
assert bp['completed'] and bp['valid_for_timing']
assert short['completed'] and short['meaningful_speedup_pass']
assert native['completed'] and native['coverage_complete'] and fusion['strict_gate_pass']
print('1 B100 process seconds:',b['process_wall_s'])
print('2 Bprime100 process seconds:',bp['process_wall_s'])
print('3 short steady speedup:',short['steady_speedup'])
print('4 selected new native directions:',native['counts']['selected_new_directions'])
print('5 selected fused pairs:',len(fusion['pair_keys']))
audio=read('audio_eval/result.json')
assert audio['completed'] and audio['valid_for_listening']
for phase in ('operating_before','operating_after'):
    protection=audio[phase]
    assert protection['count']==1022 and protection['unchanged']
    assert not protection['changed'] and not protection['missing'] and not protection['errors']
PY
~~~
