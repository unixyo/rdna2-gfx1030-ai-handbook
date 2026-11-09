# Sparse v0.7 모델 품질 진단 계획 — RGB 실행 전 고정

후보의 채택 상태는 실패다. late step 24/layer 31의 same-mask 계산 gate를 두 tile/모든 전략이 통과하지 못했으므로 이 모델 실험은 자동 적용 또는 배포를 승인하는 검증이 아니다. 출력 품질이 어느 정도 바뀌는지 실제 25-step workflow와 사전 지각 절차로 확인하는 명시적 진단이다. strict/compute 기준을 완화하지 않는다.

## 고정 후보와 선택 이유

- 2D 공간 거리에서 착안한 image-only radial r4, q32/k64/d16/pv16/w8. 원 temporal Radial의 공식 포팅으로 부르지 않는다.
- target Q2400/K4964 FP32, 실제 40×60 geometry, H32/D128의 800 target 호출에만 적용. reference 32회는 dense alpha, masked 64회는 원래 native callable 그대로다. step/layer 제한을 관측 몇 점만 보고 일반 안전 규칙으로 만들지 않는다.
- 겹침 없는 step 12/layer 15 operator: alpha 22.703846 ms, r4 18.523929 ms(1.225650배). 같은 지점 Sparge .95는 selector 포함 18.250787 ms(1.243993배)이나 native 대비 relative RMSE .241145로 r4 .175240보다 크다. r4는 정적 mask로 지점별 selector 오버헤드/밀도 변화가 없고, map 유지 밀도 32.726%다. r8/r16 및 더 높은 Sparge 보존 threshold는 이 지점에서 alpha보다 느렸다.
- 9개 실제 map에서 r4의 retained probability 평균은 72.029–97.982%이고 최악 query는 0.669%다. 이 질량은 후보 선택 진단일 뿐 RGB 기준을 대체하지 않는다.
- late step 24/layer 31: r4 same-mask max 1.864433e-4 / RMSE 5.088961e-6로 고정 atol=3e-5, rtol=3e-4 elementwise gate 실패. all-true도 실패했고, native/streaming을 각각 FP64와 비교한 worst-absolute-difference 3개 row는 raw results에 있다. 선택 row 통과를 전체 통과로 승격하지 않는다.
- 기존 all-true model control은 이미 RGB max .428385437 / RMSE .006812409로 strict 실패했다. 따라서 이번 RGB 차이를 전부 pruning의 효과라고 설명하거나 두 차이를 선형으로 더해 분해하지 않는다.

## 실행 순서와 시간 해석

| seed | 같은 서버에서의 요청 순서 | 지각 평가에 쓰는 후보 | 시간 해석 |
|---|---|---|---|
| 20261107 | baseline → dense_alpha → r4(first) → r4(warm) → dense_alpha | 두 번째 r4 | baseline 초회 제외, first/warm 각각 보존; warm 후보와 앞뒤 alpha 대조 |
| 20261108 | baseline → dense_alpha → r4(first) | 첫 r4 | baseline 초회 제외; r4는 first이므로 warm 속도 표에 합산하지 않음 |
| 20261109 | baseline → dense_alpha → r4(first) | 첫 r4 | 동일 |

총 11요청이다. native/alpha의 반복 바이트 일치는 이미 여러 차례 확인했으며 후보가 채택 불가인 상황에서 모든 seed의 warm 반복을 늘리지 않는다. 07의 first/warm 차이·복원 control, 나머지 2 seed의 별도 출력으로 남은 품질 위험을 확인한다. 새 prewarm 코드는 넣지 않는다.

- 원 workflow 25 Euler/simple steps, cfg 1, 같은 prompt·공통 입력·conditioning cache 유지. 모델 다운로드/운영 설정 변경 없음.
- 각 seed의 baseline과 alpha는 latent/RGB raw float bytes가 같아야 비교가 유효하다. conditioning SHA도 함께 확인한다. seed07의 r4 두 결과도 비교하고 차이가 있으면 그대로 남긴다.
- strict RGB 최대 2/255와 RMSE .25/255를 동시에 만족해야 한다. 사전 gate 실패는 image가 좋아 보여도 바꾸지 않는다.

## 익명 지각 평가

- 2 독립 판정자 × 3 seeds × 좌우 배치 양방향 = 12결정. 방향별 fresh context, 원응답은 root가 보존한다.
- 기준/후보 정체와 seed/성능/strict 숫자를 감춘 파일명. 공통 실제 prompt/negative prompt 및 공통 입력 이미지는 public 자료로 제공하되 생성 결과와 구분한다.
- full/crop 모두 무손실 PNG. crop은 같은 좌표와 원본 1:1 픽셀을 유지한다. CodexPro2 view_image schema가 제한하는 파일당 2,000,000 bytes를 넘으면 같은 위치에서 분할한다. (도구 입력 검증에서 확인한 decimal MB 한도이며 crop/판정 기준은 그대로다.) 정확 원본 PNG도 보존한다.
- 고정 crop: upper (.05,.02,.95,.42), middle (.05,.30,.95,.75), lower (.05,.58,.95,.98), 실제 크기에 같은 반올림을 적용한다. 결과를 보고 crop을 고르지 않는다.
- severity≥2 동일 결함이 같은 seed의 양방향에서 재현되면 실패. 순서 의존/판정자 충돌은 미확정. 10/12 이상 동등·우세이며 전 seed에서 반복 열화가 없을 때만 제한적 지각 통과. strict/compute 실패를 지각 통과로 지우지 않는다.

## 소스와 원자료

- 연구 config: research_configs/qwen_sparse_radial4_v07.json
- 실제 map: results/sparse_v07/capture25_seed20261107_retry/sparse_capture/
- 중후반 유효 operator: results/sparse_v07/bench_remaining3/results.json 및 gpu_audit.json
- bench_actual5는 JSON NaN 직렬화 실패로 종료 audit가 누락됐다. 첫 두 case 시간은 탐색용으로만 보존하고 채택 숫자로 쓰지 않는다. bench_complete3의 완전 audit 재측정으로 대체한다.
- masked-text의 최초 causal 제외 가정은 actual attention_split의 bool 0/1 additive 동작과 달랐다. 원 failure를 보존하고 명시적 prefix_bias_offset 옵션으로 계약을 보존한다. 모델 adapter는 이 옵션을 masked 호출에 자동 적용하지 않는다.
- kernel SHA: 76007ff7087dd251353dbf8018bf960c48e27792f46fdefaffe233305cece606
- adapter SHA: 27cd0d14b73aa68d6d60d75b87a003286ed6eaa5f1df538bda7f7da0479512eb
- benchmark SHA: 12d7a8805243d6a00c7bfdc9f46989e9e80280383a912b62517e52eef8ddd7e9

RGB 실행 전 원자료 저장 확정: candidate latent/RGB float Tensor도 instrument_sparse_outputs_v07.py로 저장해 PNG 양자화 전 strict 수치를 독립 재계산할 수 있게 한다. 기존 decoder 호출의 반환값은 바꾸지 않는다. 저장은 하네스의 VAE stage clock 뒤이므로 denoise/VAE 시간에 포함되지 않으며, request wall에는 포함된다. config SHA: 310b052d797c5dcf3fd1b9d7d57f9910040171fcdb7c45bd4cdda148e5d82a85; archive instrument SHA: 6ad10d0db8cb64f07dbb43978aae095fd4558bb3d67547140ee9c4eb93d73773
