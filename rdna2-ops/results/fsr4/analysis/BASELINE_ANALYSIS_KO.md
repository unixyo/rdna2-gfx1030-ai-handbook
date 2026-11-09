# FSR4 교대 baseline 분석

생성 UTC: 2026-11-09T18:38:29.628644+00:00
상태: **complete**. 유효 timing 실행 21/21.

SDK baseline은 공식 4.1.2 DLL의 연구 사본에 CPU IsSupported 6바이트만 바꾼 경로다. 셰이더 원본 동일성은 MEASUREMENT_MAPPING.json의 qualification을 따른다.

각 실행의 프레임 중앙값을 먼저 계산하고 같은 round끼리 비교한다. 양의 이득은 candidate가 빠르다는 뜻이다. MAD는 비보정 중앙절대편차이고 bootstrap은 프레임이 아닌 paired round를 10,000회 재표집한다. 미완료 round를 0으로 대체하지 않는다.

| reference→candidate | case | 짝 수 | 이득 중앙값 % | MAD %p | 95% CI % | timing gate |
|---|---|---:|---:|---:|---|---|
| sdk412→exact | 1920x1080/still | 7 | 6.584 | 0.051 | [6.533, 6.735] | True |
| sdk412→exact | 1920x1080/moving | 7 | 7.424 | 0.213 | [7.288, 8.043] | True |
| sdk412→exact | 3840x2160/still | 7 | 6.074 | 0.103 | [6.006, 6.291] | True |
| sdk412→exact | 3840x2160/moving | 7 | 6.618 | 0.053 | [6.400, 6.665] | True |
| sdk412→lossy | 1920x1080/still | 7 | 21.112 | 0.326 | [20.772, 21.332] | True |
| sdk412→lossy | 1920x1080/moving | 7 | 21.186 | 1.019 | [16.957, 22.205] | True |
| sdk412→lossy | 3840x2160/still | 7 | 27.767 | 0.324 | [27.340, 27.971] | True |
| sdk412→lossy | 3840x2160/moving | 7 | 27.647 | 0.080 | [27.546, 27.727] | True |
| exact→lossy | 1920x1080/still | 7 | 15.552 | 0.279 | [15.215, 15.651] | True |
| exact→lossy | 1920x1080/moving | 7 | 14.899 | 0.700 | [9.694, 15.598] | True |
| exact→lossy | 3840x2160/still | 7 | 23.096 | 0.101 | [22.636, 23.181] | True |
| exact→lossy | 3840x2160/moving | 7 | 22.561 | 0.055 | [22.475, 22.575] | True |

timing gate는 계획한 7개 pair 확보, 이득>0, 이득>2×MAD, bootstrap 하한>0의 수치 조건이다. 출력 exact 판정과 사용자 품질 채택은 별개다. 배경 엔진 증가가 있는 자료도 삭제하지 않았으며 아래에 표시한다.

| run | 상태 | 배경 엔진 증가 | game/KFD 최대 | fault | clock 표본 sclk/mclk |
|---|---|---|---|---:|---|
| baseline_r00_sdk412 | valid | False | 0/0 | 0 | 13/17, 13/17 |
| baseline_r00_exact | valid | False | 0/0 | 0 | 17/17, 17/17 |
| baseline_r00_lossy | valid | False | 0/0 | 0 | 14/14, 14/14 |
| baseline_r01_lossy | valid | False | 0/0 | 0 | 10/10, 10/10 |
| baseline_r01_exact | valid | False | 0/0 | 0 | 10/10, 10/10 |
| baseline_r01_sdk412 | valid | False | 0/0 | 0 | 10/10, 10/10 |
| baseline_r02_exact | valid | False | 0/0 | 0 | 10/10, 10/10 |
| baseline_r02_sdk412 | valid | False | 0/0 | 0 | 10/10, 10/10 |
| baseline_r02_lossy | valid | False | 0/0 | 0 | 10/10, 10/10 |
| baseline_r03_lossy | valid | False | 0/0 | 0 | 11/11, 11/11 |
| baseline_r03_sdk412 | valid | False | 0/0 | 0 | 10/10, 10/10 |
| baseline_r03_exact | valid | False | 0/0 | 0 | 10/10, 10/10 |
| baseline_r04_sdk412 | valid | False | 0/0 | 0 | 11/11, 11/11 |
| baseline_r04_lossy | valid | False | 0/0 | 0 | 10/10, 10/10 |
| baseline_r04_exact | valid | False | 0/0 | 0 | 10/10, 10/10 |
| baseline_r05_exact | valid | False | 0/0 | 0 | 10/10, 10/10 |
| baseline_r05_lossy | valid | False | 0/0 | 0 | 10/10, 10/10 |
| baseline_r05_sdk412 | valid | False | 0/0 | 0 | 10/10, 10/10 |
| baseline_r06_sdk412 | valid | False | 0/0 | 0 | 10/10, 10/10 |
| baseline_r06_exact | valid | False | 0/0 | 0 | 10/10, 10/10 |
| baseline_r06_lossy | valid | False | 0/0 | 0 | 10/10, 10/10 |

엔진 증가량은 before/audit/after 전체 창의 max−min ns이며 JSON에 PID·엔진별로 보존한다. setup/teardown까지 포함하므로 특정 timed case와의 동시성을 이 값만으로 특정할 수 없다. Xwayland처럼 엔진 카운터가 없는 클라이언트, null clock 표본은 관측 한계다.

## 출력 바이트와 lossy 오차

최종 frame239만 비교한다. RGBA16F의 행 padding을 제외한 전체 파일을 끝까지 읽는다. lossy 수치 오차는 선형 RGB 값이며 지각 품질 판정이 아니다.

| reference→candidate | case | 비교 round | 전체 바이트 일치 | RGB MAE 범위 |
|---|---|---:|---|---|
| sdk412→exact | 1920x1080/still | 7 | True | — |
| sdk412→exact | 1920x1080/moving | 7 | True | — |
| sdk412→exact | 3840x2160/still | 7 | True | — |
| sdk412→exact | 3840x2160/moving | 7 | True | — |
| sdk412→lossy | 1920x1080/still | 7 | False | 0.0032881442–0.0032881442 |
| sdk412→lossy | 1920x1080/moving | 7 | False | 0.0059188298–0.0059188298 |
| sdk412→lossy | 3840x2160/still | 7 | False | 0.0030270019–0.0030270019 |
| sdk412→lossy | 3840x2160/moving | 7 | False | 0.0056145624–0.0056145624 |
| exact→lossy | 1920x1080/still | 7 | False | 0.0032881442–0.0032881442 |
| exact→lossy | 1920x1080/moving | 7 | False | 0.0059188298–0.0059188298 |
| exact→lossy | 3840x2160/still | 7 | False | 0.0030270019–0.0030270019 |
| exact→lossy | 3840x2160/moving | 7 | False | 0.0056145624–0.0056145624 |

| variant | case | 실행 수 | round 반복 바이트 재현성 |
|---|---|---:|---|
| sdk412 | 1920x1080/still | 7 | True |
| sdk412 | 1920x1080/moving | 7 | True |
| sdk412 | 3840x2160/still | 7 | True |
| sdk412 | 3840x2160/moving | 7 | True |
| exact | 1920x1080/still | 7 | True |
| exact | 1920x1080/moving | 7 | True |
| exact | 3840x2160/still | 7 | True |
| exact | 3840x2160/moving | 7 | True |
| lossy | 1920x1080/still | 7 | True |
| lossy | 1920x1080/moving | 7 | True |
| lossy | 3840x2160/still | 7 | True |
| lossy | 3840x2160/moving | 7 | True |

전체 프레임 분포(p01/p05/p25/중앙값/p75/p95/p99, MAD, IQR, 표본 표준편차), 시간순 첫·마지막 사분구간 중앙값, jitter 부호별 분포, round별 값과 raw SHA256은 baseline_analysis.json 및 baseline_rounds.csv에 있다.

일반 SDK 직접 경로의 결과이며 Fallout 4에서 provider가 선택되는지, 게임 FPS가 얼마나 바뀌는지는 이 자료가 입증하지 않는다.
