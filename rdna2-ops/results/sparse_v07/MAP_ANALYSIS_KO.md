# Sparse v0.7 실제 Qwen 확률 지도 진단

상태: 25-step 실제 workflow의 9개 step/layer 수집 완료. 이 문서는 후보 선택 전 진단이며 속도·최종 채택 판정은 후속 operator/model A/B에 둔다. capture25_seed20261107_retry는 Strata CPU 빌드와 겹친 quality_only=true 실행이므로 모든 wall/event 시간은 성능표에서 제외한다. GPU audit는 foreign client 0, 1,110 samples였다.

메모리 계약: streaming kernel은 token score 행렬을 저장하지 않는다. 출력은 O(BHQD), 작업 tile은 고정 크기지만, Boolean block map은 O(BH·QK/(128·64))다. 따라서 전체 저장공간을 O(S)라고 부르지 않는다. 후속 allocator peak는 준비된 input/map을 제외한 incremental allocation이며 driver scratch는 포함하지 않는다.

## 1. 기준 경로와 실제 geometry

- Qwen FP32 attention_split; unmasked baddbmm(alpha=D^-0.5) 접기 경로 832회(참조 image 32회, target 800회), 원래 masked 호출 64회 유지.
- Seed 20261107, 원 workflow 25 Euler/simple steps. Conditioning embedding SHA가 warm/capture/all-true/native 반복에서 동일했다. alpha capture와 마지막 native 반복은 latent/RGB 바이트가 warm과 동일했다.
- 실제 target Q=[1,32,2400,128], K/V=[1,32,4964,128]. Q image grid는 40×60이며 실제 build_sequence에서 읽었다.
- K segment: text [0,8), reference image [8,2408) 40×60, text [2408,2564), target image [2564,4964) 40×60. image-only radius 후보는 두 image segment의 공간 위치를 쓰며 text는 전부 보존한다.
- 별도 실제 shape: reference Q2400/K2408, masked text Q156/K2564. text는 prefix offset 2408의 실제 mask를 보존한다.

## 2. 실제 확률 질량의 oracle 희소성

아래 oracle은 각 head·128-query block의 평균 post-softmax block 질량을 큰 순서로 고른 값이다. 64-key block 단위다. 최악 query의 보존 보장 또는 model RGB 통과를 뜻하지 않는다. 값은 유지해야 하는 block 비율(%)이며, 9개 선택 호출은 전체 800회 workload의 빈도 가중 평균이 아니다.

| step | layer | oracle 90% 질량 | oracle 99% 질량 | oracle 99.9% 질량 | 평균 entropy(nat) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 16.55 | 63.44 | 90.66 | 1.4562 |
| 0 | 15 | 34.63 | 65.82 | 83.07 | 4.4570 |
| 0 | 31 | 18.81 | 36.39 | 53.21 | 4.4882 |
| 12 | 0 | 6.53 | 26.94 | 65.87 | 1.0518 |
| 12 | 15 | 26.13 | 58.04 | 78.69 | 4.1340 |
| 12 | 31 | 16.78 | 32.43 | 48.98 | 3.9760 |
| 24 | 0 | 4.22 | 10.38 | 27.29 | 0.8304 |
| 24 | 15 | 21.72 | 53.34 | 77.58 | 3.8754 |
| 24 | 31 | 13.69 | 29.01 | 42.28 | 3.1797 |

## 3. Compressed CDF 95% ≠ 실제 probability mass 95%

Sparge-inspired 후보는 block mean QK의 CDF를 .95까지 고르고, Q/K block 평균 pairwise cosine이 .1 미만이면 해당 row/column을 dense로 보존한다. 아래 질량은 모든 head·실제 query row에 대해 full native softmax로 다시 잰 값이다. p01은 각 호출의 전체 head/query row 하위 1% 분위다. 낮은 cosine 보호가 step 0/layer 0을 100% dense로 돌리는 것은 의도한 보호 동작이며, 이 호출에서 희소성 이득이 있다는 뜻이 아니다.

| step | layer | 유지 block(%) | 실제 질량 평균(%) | p01(%) | 최악 query 질량 | Q cosine<.1(%) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 100.00 | 100.00 | 100.00 | 0.999999 | 100.00 |
| 0 | 15 | 37.90 | 83.53 | 20.39 | 0.004024 | 0.00 |
| 0 | 31 | 24.28 | 85.02 | 13.78 | 1.726e-6 | 0.00 |
| 12 | 0 | 77.99 | 94.50 | 0.00 | 5.189e-18 | 0.00 |
| 12 | 15 | 31.50 | 82.47 | 21.10 | 0.002803 | 0.00 |
| 12 | 31 | 21.40 | 85.47 | 15.24 | 2.234e-4 | 0.00 |
| 24 | 0 | 67.02 | 88.77 | 0.00 | 1.289e-14 | 0.00 |
| 24 | 15 | 30.50 | 82.52 | 10.73 | 0.004175 | 0.00 |
| 24 | 31 | 19.45 | 85.42 | 7.46 | 9.733e-4 | 0.00 |

## 4. 후보별 9개 관측 범위

각 행은 같은 9개 실제 호출에서의 범위이며, 모델 전체의 평균이 아니다. 확률 질량이 높아도 작은 수의 중요한 query를 잃으면 RGB가 변할 수 있다. 진단 수치는 기준을 완화하지 않는다.

| 후보 | block 유지 범위(%) | 실제 질량 평균 범위(%) | p01 범위(%) | 전체 최악 질량 |
|---|---:|---:|---:|---:|
| sparge_p0.9_c0.1 | 15.48–100.00 | 75.62–100.00 | 0.00–100.00 | 3.231e-18 |
| sparge_p0.95_c0.1 | 19.45–100.00 | 82.47–100.00 | 0.00–100.00 | 5.189e-18 |
| sparge_p0.99_c0.1 | 27.88–100.00 | 89.64–100.00 | 16.50–100.00 | 2.737e-17 |
| sparge_p0.95_c0.5 | 26.86–100.00 | 85.19–100.00 | 0.00–100.00 | 1.290e-14 |
| fixed_50 | 53.37–53.37 | 57.89–89.07 | 0.00–31.27 | 1.357e-19 |
| radial_r4 | 32.73–32.73 | 72.03–97.98 | 19.86–62.08 | 0.006694 |
| radial_r8 | 47.77–47.77 | 83.27–98.59 | 36.53–69.75 | 0.006736 |
| radial_r16 | 72.13–72.13 | 93.39–99.28 | 57.42–80.35 | 0.007272 |

`radial_r4/r8/r16`은 원 Radial 구현이 아니라 2D 공간 거리에서 착안한 별도 image-only 후보다. 원 논문 식(4)은 같은/인접 frame의 중앙 band가 dense여서 f=1/2에서 이득이 없다. `radial_formula_cpu.json`은 이 사실을 CPU 식 검증으로 남겼다. 기존 이미지 편집의 reference와 target을 시간 영상 프레임이라고 간주하지 않는다. 원 temporal Radial의 gfx1030 video 포팅/모델 검증을 했다고 주장하지 않는다.

## 5. Sparse 생략 손실과 dense 계산 경로 변화의 대조

같은 FP32 streaming kernel의 all-true mask를 target 800회에 적용했다. prefix/cache와 masked 호출은 유지했다. 이때 생략 block은 0개다.

- Native 반복/alpha capture: latent와 RGB byte equal.
- All-true streaming: latent max 1.8612377643585205, RMSE 0.06196142174956846.
- All-true streaming: RGB max **0.428385437** (109.2383/255), RMSE **0.006812409** (1.7372/255).
- 사전 strict gate max 2/255, RMSE .25/255: 실패. Sparse 후보 오차를 모두 key 생략 탓으로 해석할 수 없다. native와 streaming 사이에는 FP32 dot 누적 순서, online softmax/exp 및 P·V 누적 방식 차이가 있다. 실제 operator 정확도 진단으로 계산 오차 범위를 따로 확인한다.
- 두 종류의 model RGB 차이는 비선형 INT8 경계를 지나는 결과이므로 수치적으로 더하거나 빼서 오차 기여도를 분해하지 않는다. native↔all-true와 all-true↔sparse를 각각 대조한다.

## 6. 원자료와 상태

- `capture25_seed20261107_retry/runs.json`, `gpu_audit.json`, `sparse_capture/map_summaries.json`, `block_mass_step{00,12,24}_layer{00,15,31}.pt`에 전체 head/query 통계·mask·block mass·selector cosine과 SHA를 보존했다.
- `capture25_seed20261107` 첫 시도는 sort의 PyTorch API 인자 오류로 중단됐다. stable=True를 사용할 때 dim을 keyword로 전달하도록 수정하고 CPU 회귀 확인 후 재실행했다. 환경/하드웨어 불지원으로 분류하지 않는다. 이전 소스도 `sources_initial_smoke/`에 보존했다.
- Layer 0 QKV 3개만으로 동적 selector 성능을 판정하면 밀도 100% 데이터에 치우친다. 추가로 native step 12/layer 15 및 step 24/layer 31의 full H32 QKV를 저장해 operator benchmark에 포함한다.

## 1차 문헌

- SpargeAttention 논문 §3.2–3.4: https://arxiv.org/html/2502.18137v6
- 공식 구현/API: https://github.com/thu-ml/SpargeAttn
- Radial Attention 논문 §4.2, 식(4): https://arxiv.org/html/2506.19852v1

이 코드의 Sparge-inspired selective compression은 독립 FP32 구현이다. 전통적인 off-diagonal pairwise cosine 평균, partial block token count 보정, crossing block 포함을 쓴다. 원 논문의 정규화 유사도 식과 모든 CUDA 세부 구현까지 동일하지 않다. Sage 양자화/PV warp skip/Hilbert 정렬의 공식 포팅 성공으로 쓰지 않는다.
