# v1.0 카탈로그·가이드 반영안 — 검증 결과와 남은 판정

작성: 2026-11-08. **원 카탈로그·가이드는 수정하지 않았다.** Claude가 원자료를 확인하고 재현한 뒤 반영할 문안이다. 실제 운영 8 epoch B/B′/B/B′ 교대 비교는 **12.5421→11.4475 s/epoch, 1.09561배**였다. 이어서 운영 100 epoch B/B′를 모두 완료했고, 프로세스 전체는 **21.4208→19.5371분**, epoch 시간 합은 **21.0284→19.2228분**이었다. 같은 네 문장의 익명 WAV 8개와 청취 HTML도 생성·전달했다. **Applio 기술 검증은 완료했고 사용자 청취 판정은 대기 중**이다. C/D와 대표 pair의 GL2C/EA 실측까지 마친 최종 운영 보호 검사는 **1,022개 변경·누락·오류 0개**였고 GPU는 유휴 상태였다.

이 문서의 **연산 검증 완료**, **운영 학습 검증 완료**, **사용자 청취 대기**, **운영 반영**은 서로 다른 상태다. v0.8 B의 기존 사용자 청취 통과를 새 v1.0 B′에 승계하지 않는다.

## 1. 지금 확정된 근거와 보류 칸

| 항목 | 현재 상태 | 정본 |
|---|---|---|
| 운영 B의 실제 step 40–44 프로파일 | 완료. 대상 metadata·93 step·최종 export·KFD 최대 7큐 확인 | [APPLIO_V10_KO.md](APPLIO_V10_KO.md), [profile_B_target/result.json](profile_B_target/result.json) |
| ResBlock 융합 | 18쌍 검증, FP32 참조 108/108, AMP·weight_norm 계약 144/144, 12쌍 선택 | [fusion_full/result.json](fusion_full/result.json), [선택 파일](fusion_full/selection.json) |
| 대표 pair의 실제 memory-interface traffic | read/write × B/B′ 네 팔 완료. GL2C/EA 합 185,769,600→116,168,576 B/pair, 37.4663% 감소. 물리 DRAM bytes는 미식별 | [FUSION_TRAFFIC_PREPARATION_KO.md](FUSION_TRAFFIC_PREPARATION_KO.md), [비교 정본](fusion_traffic_comparison_02.json) |
| 남은 native Conv | 60 shape·169방향·531 후보 수치 통과, AMP 계약 195/195, 31 shape·68 새 방향 선택 | [native_selected/summary.json](native_selected/summary.json), [방향별 표](native_selected/direction_summary.csv) |
| 실제 운영 B/B′ 교대 8 epoch | 완료. 12.5421→11.4475 s/epoch, 1.09561배, 시간 8.727% 감소 | [APPLIO_V10_KO.md](APPLIO_V10_KO.md), [short protocol](short_epoch_protocol/result.json) |
| 운영 B′ 프로파일 | 완료. 5-step kernel 1,530.302 ms, 63,555 kernel. 최종 방향 귀속·gap 분석 완료 | [profile_Bprime/profile_summary.json](profile_Bprime/profile_summary.json), 최종 gap 원자료 (`gap_Bprime_final/gap_analysis.json`, 연구 폴더에만 있음) |
| 운영 100 epoch B/B′ | 완료. 각각 3,100 step, process wall 1,285.2454→1,172.2265초, steady 1.09574배 | [final protocol](final_epoch_protocol/result.json), [B 원자료](final_B1/result.json), [B′ 원자료](final_Bprime1/result.json) |
| 익명 네 문장 청취 | 기술 검증·생성·공개 전달 완료. 익명 WAV 8개와 HTML, `valid_for_listening=true`. 사용자 판정 대기 | 청취 HTML (`audio_eval/LISTEN_BLIND.html`, 연구 폴더에만 있음), [AUDIO_PREPARATION_NOTES_KO.md](AUDIO_PREPARATION_NOTES_KO.md) |
| RTX 5070 gap | B/B′ 최종 프로파일과 조건부 모델 완료. 외부 12분과의 인과적 분할은 미식별 | [GAP_5070_KO.md](GAP_5070_KO.md), 최종 gap 원자료 (`gap_Bprime_final/gap_analysis.json`, 연구 폴더에만 있음) |
| TunableOp 원인 | 고정 입력 GEMM·실제 모델의 비결정 경로와 선택 필터 검증 완료. Anima는 새 2프로세스 재현성·1.78945배 유지, Qwen은 동일 프로세스 3회 재현성 회복. Qwen 필터 새 프로세스·MOSS는 미측정 | [TUNABLEOP_DETERMINISM_KO.md](TUNABLEOP_DETERMINISM_KO.md) |
| Strata warm | 최종 CPU 점검 완료. MemAvailable 35.9092 GiB·expert cache 0%로 기존 실패보다 나쁜 조건, 재개 근거 없어 서버 미실행·병목 미판정 | [STRATA_WARM_KO.md](STRATA_WARM_KO.md), [최종 preflight](strata_preflight_02.json) |
| 운영 보호 | 2026-11-08 11:51:03 UTC 최종 감사 완료. 전후 각 1,022개, 변경·누락·오류 0개. 11:51:04 UTC KFD client 0·GpuAudit 유효 | [protected_before.json](protected_before.json), [protected_after.json](protected_after.json), [gpu_idle_after.json](gpu_idle_after.json) |

`profile_B/`와 `profile_B_faithful/`는 조건 복구 중 진단 자료다. B 기준선 수치에는 검증된 `profile_B_target/`만 사용하고, B′는 `profile_Bprime/`를 사용한다. 진행 중인 단일 run의 epoch 수치도 최종 채택 숫자로 넣지 않는다.

## 2. 카탈로그 반영 문안

### 2.1 §4.3 합성곱과 §4.4 활성화의 추가 행

기존 v0.8 행은 당시의 검증·청취 이력으로 유지하고 다음 행을 추가한다. 새 B′는 **연산·운영 short·100 epoch 학습·추론 검증 완료, 사용자 청취 대기** 상태다.

| 연산·설정 | 상태 | 수치·경계 검증 | 성능·선택 범위 | 근거 |
|---|---|---|---|---|
| Applio ResBlock의 LeakyReLU·Conv·잔차 융합 | ⚠️ 연산·통합 100 epoch 학습·추론 검증 완료, 사용자 청취 대기 | 18쌍, FP32 참조 결과 108/108, 실제 ResBlock AMP·FP32 weight_norm master/gradient 144/144 통과. 이번 입력의 B 호출 경계 y/dX·master gradient는 바이트 일치 | event/wall 모두 1.03배 이상인 C32/C64 12쌍 선택. event 1.03911–1.13461배. C64/K11 세 쌍과 C128/K11 세 쌍은 제외 | A, 제안 가이드 §9.22, `rdna2-ops/results/v10/APPLIO_V10_KO.md` |
| Applio 잔여 Conv 방향별 직접 커널 | ⚠️ 연산·통합 100 epoch 학습·추론 검증 완료, 사용자 청취 대기 | 60 shape·169방향, 후보 531/531 수치 통과. 실제 논리 입력 dtype 39변형의 AMP·weight_norm 계약 195/195 통과 | 31 shape의 새 68방향: Conv1d 2, period Conv2d 59, ConvTranspose1d 7. 가장 빠른 정확한 native 대비 event≥1.10배·wall≥1.05배만 선택 | A, 제안 가이드 §9.22, `rdna2-ops/results/v10/native_selected/summary.json` |
| C128/K11 dW·d1 dX, C192→384/K5 dW, ConvTranspose1d forward 재검토 | ✅ native 유지 | 정확한 native를 이기는 선택 관문을 통과하지 못함 | 직접 커널 미선택. Conv1d 신규 선택 2방향은 C192→384/K5의 T220·T298 forward | A, `rdna2-ops/results/v10/native_selected/direction_summary.csv` |

위 행의 “바이트 일치”는 검증한 fusion 호출 경계의 결과다. 전체 학습 가중치·최종 오디오의 바이트 일치나 모든 입력에 대한 동일성을 뜻하지 않는다. Native 후보의 수치 통과와 fusion의 해당 경계 바이트 일치도 서로 다른 결과로 기록한다.

### 2.2 §4.7 학습의 추가 행

| 항목 | 상태 | 비고 |
|---|---|---|
| v0.8 B 적용 후 실제 운영 남은 비용 | A, 프로파일 완료 | `core.py train`, `voice_model_A`, 40 kHz, batch 8, KLM49, FP16 AMP. step 40–44 kernel 합 1,679.453 ms. native Conv2d 41.71%, native Conv1d 29.40%, 기존 direct Conv 9.52%, optimizer 4.58%, ConvTranspose 1.76%. Operator 표와 kernel 표는 별도 분모 |
| Applio v1.0 B′(v0.8 B + 융합 12쌍 + 새 Conv 68방향) | ⚠️ 운영 short·100 epoch 학습·추론 완료, 사용자 청취 대기 | 운영 B/B′/B/B′ 8 epoch씩, steady epoch 3–8 통합 중앙값 12.5421→11.4475 s, 1.09561배·시간 8.727% 감소. 두 인접 쌍 1.09489/1.09437배. 별도 100 epoch의 process wall은 21.4208→19.5371분, 113.0189초 절감. 묶음 전체 이득이며 각 최적화의 단독 epoch 기여는 미분리. 같은 네 문장의 익명 청취 파일을 전달했으며 최종 채택은 사용자 판정 대기 |

기존 “Conv backward 55.7%, forward 24.6%” 행에는 **v0.8 당시 프로파일**이라는 조건을 붙인다. 이 비율을 이미 최적화한 운영 B의 현재 병목 표로 재사용하지 않는다. 9월 운영 프로파일에는 워커 GPU 큐 문제가 있었고, 하네스·운영·최적화 적용 전후의 수치를 구분한다.

### 2.3 §4.6 Strata의 서로 다른 조건을 표시

현재 카탈로그에는 Claude의 warm 병목 확인 행과 GPT의 자원 차단 미판정 행이 함께 있다. 두 행을 단순 합치지 않고 **swap 보호 조건**을 명시한다.

| 항목 | 상태 | 비고 |
|---|---|---|
| Claude의 Strata warm 프리필 선행 측정 | A, 별도 8,192 MiB swap 관문 | 2026-11-07 기록. J on/off 모델 0.991/0.996/0.991배와 token ID 일치, warm phase 분해는 그 조건의 결과. 앞선 세 팔 추가 swap-out 2,243.6/1,319.7/556.6 MiB로 이번 256 MiB 조건과 다름. `REVIEW_V07_CLAUDE_KO.md` §2.3 |
| v1.0의 256 MiB 관문 아래 Strata warm 병목 | ❔ 자원 조건으로 미판정, 서버 미실행 | 2026-11-08 11:47:58 UTC 최종 CPU 점검: MemAvailable 35.909168 GiB, swap 사용 15.504971 GiB, expert cache 0%, KFD client 0, 서비스 inactive. 이전 92.78% cache·약 44.23 GiB에서도 중단됐으므로 같은/나쁜 조건에서 반복하지 않음. `rdna2-ops/results/v10/STRATA_WARM_KO.md` |

정본은 [strata_preflight_02.json](strata_preflight_02.json)이며, 앞선 30.44 GiB 관측도 [strata_preflight_01.json](strata_preflight_01.json)에 보존했다. 최종 점검의 서비스·포트·빌드·보호 관문이 통과한 것과 충분한 RAM/cache를 확보한 것은 별개다. 이번에는 새 서버나 GPU 요청을 실행하지 않았으므로 warm GPU 계산·H2D·host 비율과 J on/off 처리량을 새로 채우지 않는다.

### 2.4 §6 남은 칸의 교체 문안

> v1.0은 Applio 운영 B 위에 올릴 ResBlock 융합 12쌍과 새 Conv 68방향의 연산·AMP 계약을 검증했다. 실제 운영 short 교대 비교는 12.5421→11.4475 s/epoch, 1.09561배였고, 각각 100 epoch·3,100 step을 완료한 프로세스 전체 시간은 21.4208→19.5371분이었다. 네 문장의 익명 WAV 8개와 청취 HTML을 생성·전달했고 사용자 판정을 기다린다. RTX 5070의 공개 사양으로 조건부 모델은 계산했으나, 외부 12분 기록과 이 PC 사이의 차이를 하드웨어 필연분·회수 가능한 소프트웨어분으로 유일하게 식별하지는 못했다. TunableOp는 선택된 FP32 GSU4 atomic 경로에서 같은 입력·솔루션의 비결정을 재현했고 실제 모델의 호출과 연결했다. 해당 행만 Default로 바꾼 Anima는 새 두 프로세스의 cand 6장이 동일하고 기존 경로의 off 대비 1.78945배 이득을 유지했다. Qwen은 동일 프로세스의 25-step 3장이 동일해졌고 필터의 관측 warm 비용은 원본 대비 1.32% 증가했다. Qwen 필터의 새 프로세스 재현성과 MOSS는 미측정이다. Strata는 최종 CPU 점검에서도 원래 256 MiB 관문을 만족할 자원 재개 근거가 없어 새 서버를 실행하지 않았다. 정본은 `rdna2-ops/results/v10/`의 항목별 보고서다.

사용자 청취 결과가 도착하면 Applio의 지각 판정과 채택 상태를 추가한다. 연구자가 기술 검증을 완료한 시점과 사용자가 음질을 판단한 시점을 같은 “완료”로 합치지 않는다. C의 확인 범위와 D의 자원 미판정도 현재 상태 그대로 유지한다.

## 3. 가이드 새 §9.22 초안

### 9.22 Applio v1.0: 남은 Conv·ResBlock 융합과 실제 운영 평가

v0.8 B는 이미 사용자 익명 청취를 통과하고 운영에 적용된 기준선이다. v1.0은 그 위에 새 후보 B′를 얹는다. 대상은 `voice_model_A`, KLM49 HFG, 40 kHz, batch 8, FP16 AMP, FP32 master·gradient·optimizer 상태, 31 step/epoch다. 연구는 운영 `core.py train` 진입점을 그대로 쓰며 새 모델 이름과 연구 import hook에서만 연결한다. 운영 코드·런처·venv·공용 MIOpen DB는 수정하지 않는다. 상세는 `results/v10/APPLIO_V10_KO.md`에 있다.

#### 실제 운영 기준선과 계측

정본 `profile_B_target/`는 원래 1 speaker/KLM-SVM metadata를 확인한 실제 운영 run이다. step 40–44의 kernel 합은 1,679.453 ms, GPU activity union은 1,685.778 ms다. native Conv2d·native Conv1d가 kernel 시간의 약 41.71%·29.40%로 남아 있었다. KFD 큐 최대 7개, foreign client 0, 완료 시점 보호 파일 동일을 확인했다.

프로파일 window 2.18808 s 중 기록된 GPU 활동이 없는 구간은 0.50230 s였다. 이 시간을 전부 제거 가능한 host overhead라고 하지 않는다. CPU의 GPU 대기는 GPU 실행과 겹치고, profiler·annotation 비용도 있다. Operator self-device와 kernel 합, wall을 서로 더하지 않는다. backward의 같은 External ID가 여러 launch에 공유되므로 실제 runtime correlation ID로 연결했다. 77,026 kernel의 fallback·파싱 오류는 0이다.

새 모델의 config/filelist만 복사할 때도 원래 metadata 읽기 경로를 유지해야 한다. 이 대상은 metadata가 없으면 109 speakers/contentvec로 바뀌었다. spawn/runpy에서는 함수 `__globals__`와 모듈 dictionary가 다를 수 있으므로 실제 `run.__globals__`에 연구 연결을 적용하고 초기 state·metadata를 검증한다. 조건이 맞지 않았던 초기 진단 trace는 최종 성능 기준선에서 제외한다.

#### LeakyReLU prologue·잔차 epilogue와 backward

HiFi-GAN ResBlock의 `leaky_relu → conv → leaky_relu → conv → + x`에서 LeakyReLU를 Conv 입력 load에, residual add를 두 번째 Conv output에 합쳤다. backward는 입력 부호 mask와 residual gradient 분기를 함께 처리한다. FP16 Conv 반올림 뒤 bias, LeakyReLU dtype 경계, weight_norm과 FP32 master gradient를 보존한다. atomic dW나 FP16 partial reduction으로 바꾸지 않았다.

18쌍의 FP32 참조 결과 108/108과 실제 AMP 계약 144/144가 통과했다. 두-Conv pair의 fwd+backward를 기존 B와 번갈아 측정해 event/wall 모두 1.03배 이상인 12쌍만 선택했다. 선택 범위는 C32/C64이며 event 배속은 1.03911–1.13461이다. C64/K11은 약 1.023배로 기준 미달, C128/K11은 0.992–0.997배로 제외했다.

현재 source의 대표 C32/K3에서 kernel 28→22개, allocator peak 증가량 33,180,672→26,633,728 B를 확인했다. 중간 tensor 저장·kernel 제거의 근거이며 실제 DRAM bytes counter 측정은 아니다. bias reduction dtype 수정 전 smoke의 28→20개를 현재 source 결과에 섞지 않는다. 선택 파일은 kernel source SHA 및 v0.8 manifest SHA와 일치할 때만 적용된다.

별도 rocprofv3 availability에서는 gfx1030의 GL2C 계열 및 `FETCH_SIZE`·`WRITE_SIZE`가 실제 노출됐다. 첫 수집의 aqlprofile loader 초기화 실패는 같은 wheel 라이브러리에 대한 연구 폴더 alias로 한 번 복구했다. 성공한 GPU 표본은 재실행하지 않았고, PID 열이 없는 kernel CSV는 별도 CPU 도구에서 PID가 있는 counter·marker와 정확한 dispatch identity/time으로 검증했다. **metric 이름 노출과 실제 값 수집 성공은 별개이며, 이번 값은 후자의 관문까지 통과한 결과**다.

| 대표 pair의 GL2C/EA 요청량 | B bytes/pair | B′ bytes/pair | 감소율 |
|---|---:|---:|---:|
| read | 132,801,536 | 93,509,760 | 29.5868% |
| write | 52,968,064 | 22,658,816 | 57.2217% |
| 합계 | **185,769,600** | **116,168,576** | **37.4663%** |

대상은 B8/C32/T12800/K3/d1의 FP16 두-Conv pair forward+backward다. 각 arm/metric에서 warmup 3회 후 pair 1회만 수집했다. 네 팔의 입력/master/출력/source SHA는 모두 같았고 kernel 수는 28/22개였다. 반복 분산은 측정하지 않았다. 이 지표는 GL2C/EA 요청 traffic이며 물리 DRAM 전송량으로 부르지 않고, counter를 켠 실행 시간도 기존 성능 표본과 섞지 않는다. 단위·복구 이력·정확한 재현 명령은 [FUSION_TRAFFIC_PREPARATION_KO.md](FUSION_TRAFFIC_PREPARATION_KO.md), 합계와 원자료 SHA는 [fusion_traffic_comparison_02.json](fusion_traffic_comparison_02.json)에 있다.

#### 남은 native Conv의 방향별 선택

60 shape·169방향을 조사했고 후보 531/531이 FP32 참조 관문을 통과했다. NRMSE≤0.002, max error/reference RMS≤0.02, finite 조건은 실행 전에 고정했다. 39개 논리 입력 dtype 변형에 대한 AMP·weight_norm 계약 결과 195/195도 통과했다. C192의 실제 FP32 입력 경계를 FP16 micro 입력과 구분한다.

31 shape의 68 새 방향을 선택했다. Conv1d 2, discriminator period Conv2d 59, ConvTranspose1d 7이다. 새 Conv1d 선택은 C192→384/K5 T220·T298 forward이며, C128/K11 dW·d1 dX와 C192→384/K5 dW, ConvTranspose forward는 native로 남긴다. 비교 기준은 현재 세션에서 재측정한 가장 빠른 정확한 native 정책이다. 각 후보가 event≥1.10배·wall≥1.05배를 모두 만족해야 선택된다.

선택한 fusion C32/C64와 새 native 대체 범위는 겹치지 않는다. 기존 v0.8의 다른 winning 방향과 조건 밖 fallback을 보존한다. 이득이 없었던 방향을 “하드웨어상 개선 불가능”이라고 단정하지 않는다. 주소 계산·타일 낭비·register/LDS·launch·cache 조건이 함께 시간을 정한다.

#### 운영 epoch·100 epoch·최종 음질

실제 운영 B/B′/B/B′를 각각 8 epoch 실행하고, 프로파일을 끈 정상 epoch 3–8의 중앙값을 비교했다. B1/B′1/B2/B′2는 각각 12.52698 / 11.44132 / 12.54604 / 11.46418 s였다. 통합 정상 중앙값은 **B 12.54207456 s, B′ 11.44752336 s, 1.09561467배·시간 8.727% 감소**다. 두 인접 쌍도 **1.09488874배 / 1.09436873배**로 개선했고 사전 1.02배 관문을 통과했다. 초기 G/D SHA가 같음을 확인했으며 기술 세부 관문과 raw 결과는 `results/v10/short_epoch_protocol/result.json`에 있다. 이 수치는 두 최적화 묶음의 통합 이득으로, 각각의 단독 epoch 기여를 측정한 값은 아니다.

100 epoch 실행 직전에 예상 시간을 사용자에게 알린 뒤 B→B′ 순서로 각각 한 번씩 실행했다. 두 run 모두 **100 epoch·3,100 step 학습과 최종 export를 완료**했고 `completed`, `valid_for_epoch_timing`, `valid_for_timing`은 모두 true다. 정본은 [final_epoch_protocol/result.json](final_epoch_protocol/result.json)이다.

| 같은 측정 기준 | B | B′ | B→B′ 차이 |
|---|---:|---:|---:|
| 실제 100 epoch 시간 합 | 1,261.701447725초 / 21.0284분 | 1,153.365163803초 / 19.2228분 | 108.336283922초 절감, 시간 8.5865% 감소 |
| 초기화·종료를 포함한 프로세스 벽시계 | 1,285.245366560초 / 21.4208분 | 1,172.226508294초 / 19.5371분 | 113.018858266초 절감, 시간 8.7936% 감소 |
| epoch 3–100 steady 중앙값 | 12.565548658초 | 11.467649102초 | 1.09573885배, 시간 8.7374% 감소 |

초기 G/D state SHA는 두 run에서 같고 모든 기록 loss가 finite였다. AMP scale 감소는 B 12회·B′ 11회, 마지막 scale은 32·64였다. 이 차이를 숨기거나 전체 학습 경로의 바이트 동일성으로 표현하지 않는다. 두 run 모두 KFD 큐 최대 7개·foreign client 0이며, 각 run의 보호 검사는 변경 0개였다. C/D와 대표 counter 수집을 포함한 전체 종료 감사도 1,022개 변경·누락·오류 0개로 완료했다.

프로토콜 SHA256은 `33cdc0f310e1813a7f89e45568b29af3a8083971f56386cf74a8153cfd4e735a`다. 기록된 B/B′ result SHA256 `35ef68d05dbb3f83afa64aefad9bbc1f2ead348ae75b23f5855553b36b33df69` / `ea388480fb504714fd301383ba00f44c5420c3d2f7bd9f2aae9fa028f27b3def`가 실제 [B 파일](final_B1/result.json)·[B′ 파일](final_Bprime1/result.json)과 일치함을 CPU로 확인했다.

같은 네 로컬 문장·KLM-SVM·RMVPE·index와 동일 seed/input, native FP32 추론으로 **익명 WAV 8개와 청취 HTML (`audio_eval/LISTEN_BLIND.html`, 연구 폴더에만 있음)을 생성·전달했다.** `audio_eval/result.json`의 `completed`와 `valid_for_listening`은 true다. 가중치 엄격 로딩, native FP32 경로, generator 입력과 난수 상태 동일성, finite 출력 검사를 통과했다. 파형 수치는 참고이며 사용자 청취가 최종 지각 판정이다. 아직 사용자의 구분 가능 여부·선호·채택 응답은 없다.

공개 HTML은 **4,318,552 bytes**, SHA256 `d11c480728be685565578c3c3c66ab9d5f40ca38be0017e54dbf691643ad9db8`다. 문서 갱신 시 공개 HTML의 크기·SHA와 익명 WAV 8개만 확인했고 비공개 라벨 대응은 열지 않았다. 오디오 단계의 `operating_before.count`·`operating_after.count`는 각각 **1,022**이며 변경·누락·오류는 각각 **0**이다.

| 최종 기입 항목 | 현재 상태 |
|---|---|
| B / B′ 교대 steady epoch 중앙값·두 쌍 배속 | 12.5421 / 11.4475 s, 1.09489 / 1.09437배, 관문 통과 |
| 100 epoch B / B′ process wall·epoch 합 | 완료. process wall 1,285.2454 / 1,172.2265초, epoch 합 1,261.7014 / 1,153.3652초 |
| 3,100 step·초기 SHA·finite loss·AMP 상태 | 각 3,100 step, 초기 SHA 동일, nonfinite loss 0, scale 감소 12 / 11회 |
| 동일 네 문장 익명 청취 HTML·WAV 생성 | 완료·전달. WAV 8개와 HTML, native FP32·엄격 로딩·동일 입력/난수·finite 통과 |
| 사용자 청취 결과·최종 채택 | 아직 판정 없음 |
| 운영 보호 | 전체 종료 검사 완료. 1,022개·3,241,275,343 bytes, 변경·누락·오류 0, unchanged=true. 종료 시 KFD client 0 |

Applio의 상태는 **기술 연구 완료 / 사용자 청취 대기**다. 청취 대기를 GPU 실험 미완료로 쓰거나, 파일 생성만으로 지각 통과·운영 채택을 선언하지 않는다.

#### RTX 5070과의 시간 차이

공식 사양·계측 분해·조건부 분 단위 표는 `results/v10/GAP_5070_KO.md`에 둔다. RTX 5070의 같은 누산 계약에 사용할 값은 dense FP16 곱·FP32 누산 61.7 TFLOP/s와 672 GB/s다. 광고의 sparse/FP4 AI TOPS 또는 FP16 누산 값을 이 학습의 처리량으로 대신 쓰지 않는다. [NVIDIA Blackwell 백서](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)의 구분을 따른다.

이 PC 교정값 48.725 TFLOP/s·436.745 GB/s는 실측 ceiling이고 5070 값은 공표 peak다. 직접 Conv FLOP과 최소 논리 bytes로 만든 floor는 같은 알고리즘·traffic·상한 도달이라는 가정을 둔다. Winograd에는 직접 Conv FLOP을 실제 활용률처럼 붙이지 않는다. Cache 때문에 논리 bytes/DRAM 대역폭을 나눈 값이 실제 시간보다 큰 행은 물리 하한으로 채택하지 않는다.

최종 B′ 프로파일의 kernel 합은 **1.530302 s/5 step**, 63,555 launch다. B의 1.679453 s/77,026 launch보다 kernel 시간은 8.881% 줄었으며, 이는 실제 운영 steady epoch의 8.727% 감소와 분모가 다르다. 최종 모델은 1,211 workload 행 중 330행, **전체 kernel 시간의 53.367%**를 덮는다. 나머지 행에 측정하지 않은 FLOP·DRAM 활용률을 채워 넣지 않는다. 원자료는 gap_Bprime_final/gap_analysis.json (`gap_Bprime_final/gap_analysis.json`, 연구 폴더에만 있음)이다.

| 같은 알고리즘·최소 traffic을 가정한 B′ 모델 범위 | 분 / 100 epoch 환산 | 해석 |
|---|---:|---|
| 모델 범위의 현재 GPU kernel 시간 | 8.4390 | 측정한 5 step을 3,100 step으로 환산 |
| RDNA2 실측 ceiling에 도달한 조건부 하한 | 1.5008 | 실제 구현이 달성했다는 뜻이 아님 |
| RTX 5070 공표 peak에 도달한 조건부 하한 | 1.1530 | 같은 알고리즘·traffic과 peak 도달 가정 |
| 두 조건부 하한의 차이 | 0.3478 | 실제로 줄일 수 없는 하드웨어 몫으로 식별되지 않음 |
| 현재 RDNA2 구현과 해당 ceiling 사이 | 6.9382 | 전부 회수 가능한 소프트웨어 비용으로 해석하지 않음 |

이 표와 별개로 B′가 5070에서도 같은 ceiling 활용 비율을 유지한다고 가정하면 capacity 차이에 해당하는 절감은 2.0739분이다. 이는 추가 가정을 둔 대안 시나리오이며 0.3478분과 더하지 않는다. 실제 운영 short 비교에서 확인한 B→B′ 절감은 steady 중앙값의 100 epoch 환산 **1.8243분**이다. 별도 전체 100 epoch의 절감 실측은 **epoch 합 1.8056분, process wall 1.8836분**으로 위 학습 표에 구분했다.

외부의 5070 “12분”은 데이터·step·batch·AMP·라이브러리·CPU·시간 범위가 일치한다는 증거가 없다. **이 PC에서 실제로 줄인 몫은 B→B′ 운영 실측으로 확인하고, 줄일 수 없는 몫의 인과적 분할은 현재 자료로 미식별**이라고 적는다. 조건부 capacity 차이와 현재 구현의 ceiling까지 거리를 전부 회수 가능한 시간으로 바꾸어 말하지 않는다.

## 4. 기존 가이드에 보충할 문안

### §9.20 TunableOp

CPU 소스·CSV·설치 커널 조사에서는 Qwen 29 GEMM 중 25개를 정확히 매핑했다(3개 Default, 1개 reserved −9). Anima는 57개 모두 매핑했다. 선택된 GSU4는 Qwen 3개·Anima 12개이며, 그중 FP32는 각각 2개·5개다. Qwen entry16과 Anima entry2의 실제 설치 FP32 커널에는 atomic compare-and-swap 명령이 있다. 반면 확인한 Half GSU4 예시는 atomic이 없으므로 **GSU4와 비결정을 동의어로 쓰지 않는다**. 근거와 정확 버전 소스는 [TUNABLEOP_DETERMINISM_KO.md](TUNABLEOP_DETERMINISM_KO.md)에 있다.

음수 rocBLAS solution ID는 그 자체가 오류가 아니며, 이 설치 버전에서는 `Tensile_index = -rocBLAS_solution_index - 11`로 매핑한다. Qwen의 모델 INT8 표기와 이 CSV의 실제 float/Half GEMM dtype도 구분한다. CSV에는 m/n/k·전치·leading dimension·batch count가 있지만 입력 값·batch stride·alpha/beta·pointer alignment는 없다.

고정 입력 GPU probe 13개를 완료했다. Qwen entry0·16과 Anima FP32 5행은 같은 storage·signature·솔루션으로 각각 32회 반복해 모두 다른 on 결과를 냈고 off는 한 SHA였다. Qwen16·Anima2는 새 프로세스 p0/p1을 합친 64회도 on 64종·off 1종이었다. 원본 trace의 커널 이름이 설치된 FP32 GSU4 atomic 커널과 일치했고, 개별 Default 대조는 반복 일치 및 on/off 오차 0이었다. 같은 CSV 안의 같은 솔루션에서도 비결정이 생긴다는 사실을 확인한 것이다. atomic 합산 순서의 변화를 매 실행 직접 추적한 것은 아니므로 그 내부 순서까지 관측했다고 쓰지 않는다. Half GSU4의 Qwen19는 별도 PostGSU4 합산으로 32회 일치했으므로 GSU>1 전체를 제외하는 필터는 근거가 없다.

| 실제 모델의 연구 필터 | 재현성 결과 | warm denoise 및 범위 |
|---|---|---|
| Anima FP32 GSU4 5행 → Default | 같은 설정의 새 2프로세스에서 token·conditioning·같은 mode의 latent/RGB 일치. cand 6장 모두 같은 SHA | filtered 45.106621 s(표본 4개), original Tunable+manual_vt 45.231108 s(2개), off native 80.715881 s(2개). 같은 세션 off 대비 **1.789446배**로 기존 결합 경로의 속도 이득 유지 |
| Qwen FP32 GSU4 2행 → Default | 동일 프로세스의 25-step tbase 3장 latent/RGB 일치. 원본 3장은 모두 달랐으며 conditioning·입력은 같았음 | original 55.715586→filtered 56.448651 s(각 warm 2개), 관측 **1.32% 증가**. 같은 조건의 off 57.286772 s 대비 filtered **1.01485배**. 적은 표본의 차이를 확정 손실로 일반화하지 않음 |

Anima에서 해당 5행은 실제 CLIP에 모두 나타났고 sampler hit는 0이었다. 각 프로세스가 처음 만든 conditioning을 반복 재사용하므로, 프로세스 안의 출력이 같았다는 종전 관측과 새 프로세스의 출력 차이가 함께 설명된다. 원본 모델의 프로세스 비교는 한쪽이 진단 설정이라는 제한을 남기며, 필터 p0/p1은 같은 비진단 설정으로 검증했다. Qwen은 같은 conditioning 아래 실제 sampler의 해당 호출을 확인하고 두 행 치환 후 반복 일치를 얻었다. **Qwen 필터의 새 프로세스 재현성과 MOSS는 미측정**이다. 두 필터 모두 연구 CSV이며 운영에 적용하지 않았고 지각 품질 우열도 판정하지 않았다.

재현성과 속도의 정본 원자료는 [Anima 재현성](tunable_analysis/anima_filter_replay_comparison.json), [Anima 속도](tunable_analysis/anima_speed_observation.json), [Qwen 최종 비교](tunable_analysis/qwen_final_observation.json)다. 각 실행 명령·소스 SHA·CSV 경로·개별 GEMM 및 실제 모델의 범위는 [TUNABLEOP_DETERMINISM_KO.md](TUNABLEOP_DETERMINISM_KO.md) §10–13에 있다.

### §9.18 Strata

v1.0은 원래 프로세스 기준 추가 swap-out 256 MiB 관문, 1초 간격 연속 증가 두 표본 조건을 유지한다. Claude의 8,192 MiB 조건에서 얻은 선행 warm 결과는 조건을 붙여 인용한다. 최종 점검은 2026-11-08 11:47:58 UTC이며 MemAvailable **35.909168 GiB**, swap 사용 **15.504971 GiB**, expert cache **0%**, KFD client **0**, 서비스 **inactive**였다. 이전 92.78% cache·약 44.23 GiB에서도 관문 중단이 있었으므로 §8의 재개 근거를 충족하지 않는다. 새 서버를 실행하지 않았고 GPU 계산·PCIe·host 비율도 새로 확정하지 않았다. 서비스 조작, swap 관문 완화, 대형 SHA 읽기로 cache를 미리 채우는 절차는 하지 않았다. 상세와 CPU 원자료는 [STRATA_WARM_KO.md](STRATA_WARM_KO.md), [strata_preflight_02.json](strata_preflight_02.json)에 있다.

### §10 체크리스트 추가

- [ ] 학습 모델의 원본 metadata·speaker 수·embedder·초기 G/D SHA를 맞췄고, spawn 함수가 실제 참조하는 globals에 연구 연결이 적용됐는지 확인했다.
- [ ] Conv+activation+residual 융합은 forward만이 아니라 부호 mask·skip gradient·bias reduction dtype·FP32 master gradient까지 검증했다.
- [ ] 새 직접 Conv·fusion을 동시에 넣을 때 선택 범위가 겹치는지 확인했고, 다른 context의 경로를 조용히 우회하지 않는다.
- [ ] backward kernel 귀속은 공유 External ID만으로 결정하지 않고 runtime launch correlation을 대조했다.
- [ ] roofline의 FLOP 정의·누산 dtype·실측 ceiling/공표 peak·논리 bytes/실제 DRAM traffic·Winograd 제외 범위를 적었다.
- [ ] operator 검증, 운영 epoch, 100 epoch 기술 완료, 사용자 최종 청취, 운영 적용을 각각의 상태로 기록했다.

## 5. 반영 상태와 남은 판정

| 근거 | 반영 위치 | 현재 상태 |
|---|---|---|
| short protocol 최종 result | §1, §2.2, §3 epoch 문단 | 반영 완료: 12.5421→11.4475 s, 1.09561배 |
| 100 epoch protocol 최종 result | §1·§2·§3 최종 표 | 반영 완료: 각 3,100 step, process wall 1,285.2454→1,172.2265초, epoch 합 1,261.7014→1,153.3652초, 초기 SHA 동일·loss finite·AMP 감소 12/11회 |
| audio result/public HTML | §1·§2·§3 최종 표 | 반영 완료: WAV 8개+HTML 생성·전달, native FP32·엄격 로딩·동일 입력/난수·finite 통과. 사용자 판정 대기 |
| 사용자 청취 응답 | §2.2 상태, §3 최종 판정 | 구분 불가/차이 관측/선호 등 사용자가 실제 말한 결과 |
| 최종 B′ profile·gap 보고 | §1, §3 RTX 문단 | 반영 완료: kernel 1.530302 s, 모델 coverage 53.367%, 조건부 하한 차이 0.3478분·ceiling까지 거리 6.9382분. 실제 short 회수분 1.8243분/100 epoch 환산 |
| C 담당 보고서 | §1·§2.4·§4 | 최종 GPU 결과 반영 완료. Anima 필터 새 2프로세스·1.789446배, Qwen 같은 프로세스 3회 일치·관측 비용 +1.32%. Qwen 필터 새 프로세스·MOSS 미측정 |
| D 마지막 CPU preflight | §1·§2.3·§4 | 반영 완료. 11:47:58 UTC MemAvailable 35.909168 GiB·expert cache 0%, 재개 근거 없어 서버 미실행 |
| 대표 pair GL2C/EA counter | §1·§3 fusion 문단 | 완료. 네 팔 엄격 검증 후 read+write 185,769,600→116,168,576 B/pair, 37.4663% 감소. 물리 DRAM bytes·반복 분산은 미측정 |
| 전체 보호 after | §1·§3 | 완료. 11:51:03 UTC 전후 1,022개·변경/누락/오류 0, 11:51:04 UTC KFD client 0 |

이번 연구 실행과 전체 보호 감사는 종료했다. 사용자 청취가 도착하면 Applio 지각 판정·채택 상태만 실제 응답에 맞춰 갱신한다. Qwen 필터 새 프로세스·MOSS·자원 부족으로 미판정인 Strata의 warm 분해는 이번에 확인한 결과로 확대하지 않는다.

## 6. 현재 확정된 핵심 숫자 5개 — GPU 없이 읽기

```bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
/usr/bin/python3 -B - <<'PY'
import json
from pathlib import Path
p = Path('results/v10')
read = lambda name: json.loads((p / name).read_text())
profile = read('profile_B_target/profile_summary.json')
fusion = read('fusion_full/result.json')
selection = read('fusion_full/selection.json')
native = read('native_selected/summary.json')
short = read('short_epoch_protocol/result.json')
final = read('final_epoch_protocol/result.json')
print('1 B 5-step kernel ms:', profile['accounting']['kernel_sum_us'] / 1000)
print('2 Fusion pairs measured / selected:', len(fusion['cases']), len(selection['pair_keys']))
print('3 Fusion FP32 / AMP metrics:', sum(len(c['operator']['metrics']) for c in fusion['cases']), sum(len(c['contract']['metrics']) for c in fusion['cases']))
print('4 Native shapes / candidates / new selected directions:', native['counts']['cases'], native['counts']['candidates'], native['counts']['selected_new_directions'])
print('5 Operating short / 100-epoch steady speedups:', short['steady_speedup'], final['steady_speedup'])
print('  Actual 100-epoch seconds:', {r['arm']: {'epoch_sum': r['sum_epoch_seconds'], 'process_wall': r['process_wall_s']} for r in final['runs']})
PY
```

현재 기대값은 **1,679.453 ms / 18·12쌍 / 108·144 / 60·531·68 / short 1.09561배·100 epoch steady 1.09574배**다. 실제 100 epoch 시간 합은 **B 1,261.7014초·B′ 1,153.3652초**, process wall은 **B 1,285.2454초·B′ 1,172.2265초**다. 최종 반영 시에는 항목별 보고서의 추론·TunableOp·보호 핵심 숫자와 함께 읽는다.
