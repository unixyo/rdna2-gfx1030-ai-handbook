# FSR4 교대 측정 분석: clamp_all

생성 UTC: 2026-11-09T18:48:28.798446+00:00
상태: **complete**. 유효 timing 실행 14/14.

reference와 candidate는 같은 community exact DLL/provider를 사용한다. candidate의 SPIR-V override 목록과 SHA256은 trial declaration에 고정되어 있다. 실제 override hit 증거는 별도 런타임/ISA 검증이 필요하다.

각 실행의 프레임 중앙값을 먼저 계산하고 같은 round끼리 비교한다. 양의 이득은 candidate가 빠르다는 뜻이다. MAD는 비보정 중앙절대편차이고 bootstrap은 프레임이 아닌 paired round를 10,000회 재표집한다. 미완료 round를 0으로 대체하지 않는다.

| reference→candidate | case | 짝 수 | 이득 중앙값 % | MAD %p | 95% CI % | timing gate |
|---|---|---:|---:|---:|---|---|
| reference→candidate | 1920x1080/still | 7 | 1.945 | 0.038 | [-0.458, 1.982] | False |
| reference→candidate | 1920x1080/moving | 7 | 1.913 | 0.064 | [1.849, 2.001] | True |
| reference→candidate | 3840x2160/still | 7 | 1.374 | 0.114 | [1.259, 1.507] | True |
| reference→candidate | 3840x2160/moving | 7 | 1.433 | 0.128 | [1.305, 1.645] | True |

timing gate는 계획한 7개 pair 확보, 이득>0, 이득>2×MAD, bootstrap 하한>0의 수치 조건이다. 출력 exact 판정과 사용자 품질 채택은 별개다. 배경 엔진 증가가 있는 자료도 삭제하지 않았으며 아래에 표시한다.

## 전체 시간과 실행 내부의 변화

| variant | case | round 중앙값의 중앙값 ms | 첫→끝 사분구간 변화 범위 % |
|---|---|---:|---:|
| reference | 1920x1080/still | 0.738480 | -0.317–4.427 |
| reference | 1920x1080/moving | 0.742120 | -0.586–0.508 |
| reference | 3840x2160/still | 3.205320 | -0.235–0.160 |
| reference | 3840x2160/moving | 3.201120 | -0.259–0.109 |
| candidate | 1920x1080/still | 0.723840 | -0.349–32.838 |
| candidate | 1920x1080/moving | 0.727920 | -0.049–0.351 |
| candidate | 3840x2160/still | 3.162260 | -0.230–0.148 |
| candidate | 3840x2160/moving | 3.154360 | -0.160–0.174 |

사분구간 변화는 시간순 처음 50프레임과 마지막 50프레임 중앙값의 상대 변화다. 제거 기준으로 사용하지 않았다. 절대 변화가 가장 큰 두 구간은 다음과 같다.

- clamp_all_r04_candidate, 1920x1080/still: 0.725080 → 0.963180 ms (+32.838%). 전체 200프레임 중앙값 0.983440 ms.
- clamp_all_r00_reference, 1920x1080/still: 0.701100 → 0.732140 ms (+4.427%). 전체 200프레임 중앙값 0.716560 ms.

## GPU 겹침과 관측 범위

| run | 상태 | 배경 엔진 증가 | game/KFD 최대 | fault | clock 표본 sclk/mclk |
|---|---|---|---|---:|---|
| clamp_all_r00_reference | valid | False | 0/0 | 0 | 10/14, 10/14 |
| clamp_all_r00_candidate | valid | False | 0/0 | 0 | 13/13, 13/13 |
| clamp_all_r01_candidate | valid | False | 0/0 | 0 | 13/13, 13/13 |
| clamp_all_r01_reference | valid | False | 0/0 | 0 | 13/13, 13/13 |
| clamp_all_r02_reference | valid | False | 0/0 | 0 | 13/13, 13/13 |
| clamp_all_r02_candidate | valid | False | 0/0 | 0 | 13/13, 13/13 |
| clamp_all_r03_candidate | valid | False | 0/0 | 0 | 13/13, 13/13 |
| clamp_all_r03_reference | valid | False | 0/0 | 0 | 13/13, 13/13 |
| clamp_all_r04_reference | valid | False | 0/0 | 0 | 13/13, 13/13 |
| clamp_all_r04_candidate | valid | False | 0/0 | 0 | 13/13, 13/13 |
| clamp_all_r05_candidate | valid | False | 0/0 | 0 | 13/13, 13/13 |
| clamp_all_r05_reference | valid | True | 0/0 | 0 | 14/14, 14/14 |
| clamp_all_r06_reference | valid | False | 0/0 | 0 | 13/13, 13/13 |
| clamp_all_r06_candidate | valid | False | 0/0 | 0 | 13/13, 13/13 |

엔진 증가량은 before/audit/after 전체 창의 max−min ns이며 JSON에 PID·엔진별로 보존한다. setup/teardown까지 포함하므로 특정 timed case와의 동시성을 이 값만으로 특정할 수 없다. Xwayland처럼 엔진 카운터가 없는 클라이언트, null clock 표본은 관측 한계다.


| run | background PID/name | engine | max−min ms |
|---|---|---|---:|
| clamp_all_r05_reference | 13433 / steamwebhelper | drm-engine-compute | 5.129051 |
| clamp_all_r05_reference | 13433 / steamwebhelper | drm-engine-dma | 48.756348 |
| clamp_all_r05_reference | 13433 / steamwebhelper | drm-engine-gfx | 78.914063 |

이 배경 활성 표시는 수치 timing gate와 별도로 해석해야 한다. sclk/mclk의 전체 실행 분포에는 업로드·캡처·idle 시점도 들어가므로 이를 timed shader의 평균 클록으로 사용하지 않는다.

## 출력 바이트와 lossy 오차

warmup 400, timed 200를 거친 최종 frame599만 비교한다. RGBA16F의 행 padding을 제외한 전체 파일을 끝까지 읽는다. lossy 수치 오차는 선형 RGB 값이며 지각 품질 판정이 아니다.

| reference→candidate | case | 비교 round | 전체 바이트 일치 | RGB MAE 범위 |
|---|---|---:|---|---|
| reference→candidate | 1920x1080/still | 7 | True | — |
| reference→candidate | 1920x1080/moving | 7 | True | — |
| reference→candidate | 3840x2160/still | 7 | True | — |
| reference→candidate | 3840x2160/moving | 7 | True | — |

| variant | case | 실행 수 | round 반복 바이트 재현성 |
|---|---|---:|---|
| reference | 1920x1080/still | 7 | True |
| reference | 1920x1080/moving | 7 | True |
| reference | 3840x2160/still | 7 | True |
| reference | 3840x2160/moving | 7 | True |
| candidate | 1920x1080/still | 7 | True |
| candidate | 1920x1080/moving | 7 | True |
| candidate | 3840x2160/still | 7 | True |
| candidate | 3840x2160/moving | 7 | True |

전체 프레임 분포(p01/p05/p25/중앙값/p75/p95/p99, MAD, IQR, 표본 표준편차), 시간순 첫·마지막 사분구간 중앙값, jitter 부호별 분포, round별 값과 raw SHA256은 clamp_all_analysis.json 및 clamp_all_rounds.csv에 있다.

일반 SDK 직접 경로의 결과이며 Fallout 4에서 provider가 선택되는지, 게임 FPS가 얼마나 바뀌는지는 이 자료가 입증하지 않는다.
