# Sparse Attention v0.7 — gfx1030 FP32 연구 결과

## 1. 판정과 적용 범위

**자동 적용하지 않는다.** FP32 block-sparse 커널과 선택기는 RX 6900 XT / Triton 3.8에서 실제 컴파일·실행됐고, score 행렬을 저장하지 않아 단위 연산 메모리를 크게 줄였다. 그러나 후반 Qwen QKV의 **같은 mask에 대한 native 재현 tolerance**를 통과하지 못했다. 최종 25-step 모델의 엄격 RGB 기준도 통과하지 못했다. 속도가 빠른 부분과 출력 계약 실패를 같은 표에 남긴다.

이 실험은 설치된 Qwen 이미지 편집 경로의 실제 K=4964를 택했다. Anima 고해상도나 새 영상 모델을 받지 않았다. Sparge 선택기와 이미지 공간 radius는 아래에 정의한 독립 FP32 참고 구현이다. 원 Sparge/Sage CUDA 전체 또는 원 temporal Radial의 gfx1030 포팅 성공으로 읽으면 안 된다.

| 칸 / 후보 | 성공·실패 판정 | 핵심 원인 / 관측 | 증거 |
|---|---|---|---|
| FP32 streaming block-sparse core | 실행·작은 독립 FP64 검증 성공, 후반 native 재현 gate 실패 | q32/k64/d16/pv16/w8, 실제 D128에서 spill 0; 후반 누적·softmax 순서 차이 | A, 실제 3 shape + 중후반 2 QKV |
| Sparge-inspired mean/cosine + CDF | 선택기 자체 검증 성공, 일반 최적화 채택 실패 | compressed CDF .95가 실제 질량 .95를 보장하지 않음; layer 0은 보호 조건으로 dense | A; 알고리즘 출처는 B/C |
| 이미지 2D radius r4 | 단위 일부 속도 성공, 엄격 모델 실패·채택 불가 | 32.726% block 유지, 일부 중요한 query 질량 소실; 계산 경로 자체 차이도 존재 | A, 3-seed 실제 25-step 모델 |
| fixed block 약 50% | 목표 long shape에서 속도 실패 | 실제 유지 53.374%, 28.454 ms로 alpha 22.666 ms보다 느림 | A |
| 원 temporal Radial | 이 image/prefix 조건에서 적용 범위 부적합 | 원식의 f=1/2 중앙 band는 dense; reference image를 시간 frame으로 바꿔 부르지 않음 | C + CPU 식 재현, 영상 GPU 미검증 |

A는 이 PC 실측, B는 실제 소스 확인, C는 1차 문헌이다. 아래의 계산·모델 실패를 지각 평가로 성공으로 바꾸지 않는다.

## 2. 실제 기준 커널과 호출 계약

기준 callable은 `comfy.ldm.modules.attention.attention_split`이다. source SHA는 `9cafaafaf93ff53e8cbefb5e4a204014019985df2da8c1996bf40f1235fc2960`. 실제 연구 config와 runs metadata에서 module, qualname, source 경로와 SHA를 보존했다.

모델의 unmasked 기준은 기존 FP32 `attention_split`에 후행 scale을 GEMM alpha로 접은 경로다. Q/K/V를 head-contiguous로 만들고, rocBLAS FP32 QK, native softmax, FP32 PV를 쓴다. masked 64회는 native callable에 원 mask 그대로 전달했다.

| 실제 호출 종류 | BHQD / BHKD | 모델당 호출 수 | sparse 모델 실험의 처리 |
|---|---|---:|---|
| target 이미지 | 1·32·2400·128 / 1·32·4964·128 | 800 | r4 또는 all-true streaming |
| reference prefix 이미지 | 1·32·2400·128 / 1·32·2408·128 | 32 | dense alpha |
| masked text prefix | 1·32·156·128 / 1·32·2564·128 | 64 | native 그대로 |
| **합계** | — | **896** | 800 + 32 + 64 |

새 adapter의 hits=832에는 prefix 32가 이미 들어 있다. fallbacks=64와 합치면 896이다. 최초 capture adapter의 hits=832, fallbacks=96에는 prefix 32가 겹쳐 있었으므로 그 둘을 더하지 않는다. 원자료의 옛 카운터는 수정하지 않고 의미를 이 표로 고정했다.

실제 `build_sequence`에서 읽은 target grid는 40×60이다. K의 segment는 text [0,8), reference image [8,2408), text [2408,2564), target image [2564,4964)다. 이미지 공간 후보는 두 image segment의 좌표를 쓰고 text key는 전부 보존한다. 파일명이나 이미지 해상도만 보고 token geometry를 추정하지 않는다.

### 실제 Boolean mask의 함정

`actual_qkv_156_2564.pt`에 보존한 실제 mask는 torch.bool [156,2564], false 12,090개 / true 387,894개다. raw bytes SHA는 `6963aa8ff515b246078ad28edb9468a2aa7db6ab0d858b55db4a870381d62d01`. 실제 callable은 `scores += mask`를 실행한다. 따라서 이 경계에서 bool mask는 **0/1 additive bias**이며 논리적 causal 제외가 아니다.

초기 micro가 이를 causal 제외로 읽어 native 대비 max 5.48432를 냈다. Sparge .90/.95가 선택한 block과 논리적 visible 영역이 겹치지 않는 row에서는 dense 참조 softmax가 all-minus-inf의 NaN이 됐다. streaming 출력은 finite였고 empty row 반환값 0이었다. 이 실패를 FP32 overflow라고 쓰지 않는다.

교정은 명시적 `prefix_bias_offset` 옵션으로 제한했다. corrected reference는 native와 bit 일치했고 corrected 3-case는 nonfinite metric 0이었다. 모델의 masked 64회에 새 옵션을 자동 연결하지 않았다. 실제 입력 mask 증거, callable 소스 증거, 만들어 낸 논리 mask의 실패를 서로 구분한다.

## 3. 구현과 사전 기준

재사용 모듈은 `rdna2_ops/sparse_attention_v07.py`다. FP32 BHLD 입력, D64/128, gfx1030 forward eager에 대한 명시적 연구 API다. dropout/GQA/backward/자동 dispatch를 제공하지 않는다. finite 입력·score 및 신뢰할 수 있는 Boolean block map이 호출자 계약이다. 논리 causal를 쓸 때는 유효 query마다 유지된 visible key가 하나 이상 있어야 한다. `causal_offset`과 `prefix_bias_offset`은 서로 다른 의미며 동시에 쓸 수 없다.

QK는 FP32 IEEE FMA를 d16 단위로 누적하고, online softmax의 분모·PV도 FP32다. PV는 d16 조각으로 계산한다. Triton q32/w8과 q16/w4 두 tile만 비교했다. q32는 실제 long shape에서 187 VGPR, spill 0, LDS 16 KiB였고, q16은 207 VGPR, spill 0, LDS 8 KiB였다. q16은 dense 108.670 ms로 q32 54.377 ms보다 느렸다. register spill 제거만으로 dense ROCm GEMM을 이긴다고 볼 수 없다. 여기서 occupancy의 정확한 병목은 별도 하드웨어 카운터로 확정하지 않았다.

선택기는 128-query × 64-key block이다. Sparge-inspired 선택기는 fused block mean·전통적 off-diagonal pairwise cosine, 낮은 cosine의 Q/K block dense 보호, partial key block token 수 보정, CDF crossing block 포함을 쓴다. 공식 구현의 모든 유사도 수식과 같다고 주장하지 않으며 Sage 양자화, PV warp skip, Hilbert 정렬은 이식하지 않았다.

첫 GPU 전에 [PREREGISTERED_KO.md](PREREGISTERED_KO.md)를 저장했다. 독립 CPU FP64 small checks는 D64/128, tail, strided, causal-prefix를 포함한 3 shape에 corrected bool-additive 동일 shape 검사를 더했다. 두 tile 모두 통과했고, 최악 CPU FP64 대비 max는 1.595e-8 미만이었다. fused selector mean/cosine은 CPU 참조 최대 절대오차 각각 1e-5 이하가 기준이다.

같은 mask의 elementwise 계산 gate는 **atol=3e-5, rtol=3e-4, 전값 finite**다. dense 대비 생략 오차는 별도 max/RMSE/relative RMSE로 기록한다. 모델 float RGB 기준은 **max 2/255, RMSE 0.25/255 동시 충족**이다. probability mass 수치가 RGB 기준을 대체하지 않는다.

## 4. 희소성부터 측정한 결과

25 step 중 0/12/24, layer 0/15/31의 9개 실제 호출에서 모든 head·query의 post-softmax 질량을 측정했다. QKV와 block mass, head/query 하위 분위·최악값을 보존했다. 이 9개 호출은 전체 800회의 빈도 가중 평균이 아니다.

[MAP_ANALYSIS_KO.md](MAP_ANALYSIS_KO.md)에 9개 oracle·CDF·분위 전체 표가 있다. oracle은 각 128-query block에서 평균 질량이 큰 key block부터 고른 값이다. query별 최악 보존 보장은 아니다. 99% oracle에 필요한 평균 block 비율도 관측 위치에 따라 10.38–65.82%, 99.9%는 27.29–90.66%였다.

**Compressed CDF 95% ≠ 실제 probability mass 95%.** Sparge .95/.1은 step 0/layer 0의 낮은 cosine을 감지해 100% dense로 돌아갔다. 이것은 보호 동작 성공이며 희소성 성능 이득은 아니다. step 12/layer 15에서는 31.50% block을 유지했지만 실제 평균 질량은 82.47%, 하위 1% query 질량은 21.10%였다. step 24/layer 31은 19.45% block 유지 / 평균 질량 85.42% / p01 7.46%다. 일부 지점 최악 query는 거의 모든 질량을 잃었다.

| 후보 | 9개 실제 호출의 유지 block 범위 | 실제 평균 질량 범위 | 전체 최악 query 질량 |
|---|---:|---:|---:|
| Sparge .95/.1 | 19.45–100.00% | 82.47–100.00% | 5.189e-18 |
| fixed 약 50% | 53.37% | 57.89–89.07% | 1.357e-19 |
| image radius r4 | 32.73% | 72.03–97.98% | 0.006694 |
| image radius r8 | 47.77% | 83.27–98.59% | 0.006736 |
| image radius r16 | 72.13% | 93.39–99.28% | 0.007272 |

이 radius는 token 공간의 Euclidean 거리와 block bounding box를 쓰는 image-only 참고 구현이다. 원 Radial 논문 식(4)은 시간 frame 간 거리에 따른 band다. `radial_formula_cpu.json`에서 f=1/2는 dense로 퇴화함을 확인했다. 기존 reference와 target을 영상 frame이라 부르거나, 이 CPU 식 확인을 영상 모델 GPU 검증으로 승격하지 않았다.

## 5. 실제 연산 3 shape + 중후반 QKV 2개

다음은 gpu_guard가 유효하고 다른 KFD client가 관측되지 않은 측정이다. correctness 이후 2 warmup, 순서를 회전·반전한 3 round × 2 call의 일반 호출 wall 중앙값이다. HIP event와 첫 native/candidate 호출도 raw에 따로 있다. 첫 correctness 뒤의 timing을 cold JIT라고 부르지 않는다. 동적 Sparge 시간은 selector를 매번 포함하고, 정적 r4의 mask 준비는 별도다.

| 실제 Q/K, 위치 | 실제 기준 (ms) | all-true q32 (ms) | Sparge .95 포함 (ms) | r4 q32 (ms) | r4 배속 | r4 same-mask |
|---|---:|---:|---:|---:|---:|---|
| 2400/4964, step 0/layer 0 | 22.666 | 54.377 | 54.434 | 18.600 | 1.219× | 통과 |
| 2400/2408, reference layer 0 | 12.867 | 25.584 | 24.316 | 8.201 | 1.569× | 통과 |
| 156/2564, masked text layer 0 | **1.892 native** | 1.862 | 1.746 | — | — | geometry 부적합 |
| 2400/4964, step 12/layer 15 | 22.704 | 54.310 | 18.251 | 18.524 | 1.226× | 통과 |
| 2400/4964, step 24/layer 31 | 22.757 | gate 실패 | gate 실패 | gate 실패 | — | 실패, 시간 제외 |

masked text 기준은 실제 `attention_split`이다. generic alpha wrapper의 2.219 ms를 분모로 써서 이득을 부풀리지 않는다. [OPERATOR5_SUMMARY.json](OPERATOR5_SUMMARY.json)은 이 기준으로 배속을 다시 계산했다. 원 micro JSON의 `speedup_vs_actual_split_alpha`는 일관된 alpha 대비 필드라 masked 실제 baseline 배속과 구분해야 한다.

`results.status=completed`와 exit 0은 수집이 끝났다는 뜻이다. 개별 후보의 gate 실패가 성공으로 바뀌지는 않는다.

원자료는 corrected [bench_complete3/results.json](bench_complete3/results.json) 및 중후반 [bench_remaining3/results.json](bench_remaining3/results.json)이다. remaining3의 첫 masked case는 옛 잘못된 계약이므로 실패 증거로만 쓴다. 유효 5-case 집계에는 corrected masked case로 교체했다. 각 소스 버전·SHA·trace·audit는 그대로 보존했다.

### 후반 실패의 뜻: native 재현과 수학적 정확도를 구분

step 24/layer 31은 finite FP32 출력이지만 두 tile의 모든 streaming 전략이 native 재현 tolerance에 미달했다. r4의 같은-mask max 1.864433e-4, RMSE 5.088961e-6다. all-true도 max 1.854897e-4였다. 이후 시간 경쟁 표에서 이 후보들은 제외했다.

worst absolute difference 3개 row를 골라 CPU FP64와 native/streaming을 각각 대조했다. r4 head 23/query 1164에서 streaming→FP64 max 약 2.102e-5, rocBLAS 참조→FP64 약 1.478e-4로 streaming이 더 가까웠다. query 440은 반대였다. 따라서 모든 실패를 overflow나 수학식 오답으로 단정할 수 없다. native 참조에도 반올림 오차가 있다. **선택한 3개 row의 FP64 결과로 전체 tensor gate를 통과로 바꾸지 않는다.**

### 메모리와 커널 프로파일

long shape에서 alpha incremental torch allocator peak는 **3102.594 MiB**, r4는 **38.000 MiB**였다. input과 미리 준비한 map은 제외한 추가 할당이며 driver scratch나 전체 모델 VRAM 수치가 아니다. 출력은 O(BHQD), tile workspace는 고정 크기지만 block metadata는 O(BH·QK/(128·64))다. 따라서 이 구현의 총 저장공간을 O(S)라고 부르지 않는다.

별도 1회 진단 trace에서 alpha GPU 작업 합은 21.673 ms였다. softmax 6.973 ms(32.17%), rocBLAS `Cijk_Alik_Bljk_SB_MT128x128x8...WGM8.kd` 5.667 ms, `Cijk_Ailk_Bljk_SB_MT128x64x8...WGM4.kd` 4.673 ms, head/output copy 4회 합 4.361 ms다. r4 trace는 단일 `_sparse_fp32_stream.kd` 18.449 ms였다. 전체 실제 커널 이름과 Chrome trace SHA는 raw JSON에 있다. 이 trace 1회 값을 일반 호출 중앙값과 섞지 않고 CPU 귀속 시간을 GPU kernel 시간에 더하지 않았다.

## 6. 실제 모델 A/B와 all-true 대조

모델 실험 전에 [MODEL_DIAGNOSTIC_PLAN_KO.md](MODEL_DIAGNOSTIC_PLAN_KO.md)로 r4를 고정했다. 중간 layer의 속도와 Sparge보다 작은 근사 relative RMSE, 정적 밀도를 근거로 **명시적 품질 진단**을 택했다. 후반 native 재현 gate 실패는 이미 알려져 있었고 자동 채택 자격은 없었다. 관측 몇 개 step/layer를 근거로 일반 fallback 안전 범위를 만들지 않았다.

처음의 all-true 모델 대조는 target 800회에 같은 FP32 streaming을 적용하되 block을 하나도 생략하지 않았다. alpha/native 반복은 latent/RGB byte 일치였고 conditioning SHA도 같았다. all-true만으로 RGB max **0.428385437 (109.2383/255)**, RMSE **0.006812409 (1.7372/255)**로 엄격 기준에 실패했다. latent max 1.861237764, RMSE 0.061961422였다. 이 capture는 Strata CPU 빌드와 겹친 quality_only 실행이므로 모든 시간은 성능 증거에서 제외했다.

따라서 r4의 최종 차이를 전부 key 생략 탓으로 돌리지 않는다. native↔all-true 계산 경로 변화와 native↔sparse를 각각 대조한다. INT8 activation 경계를 거친 RGB는 비선형 결과라 두 오차를 더하거나 빼서 기여도를 분해하지 않는다.

동일 workflow의 실제 25 Euler/simple step, cfg 1과 conditioning cache를 유지했다. seed07은 baseline → alpha → r4(first) → r4(warm) → alpha 5요청, seed08/09는 각각 baseline → alpha → r4(first) 3요청이다. 총 11요청이며 새 prewarm 코드를 넣지 않았다. 여기서 first는 해당 소유 프로세스의 첫 후보 요청이다. 디스크 Triton cache까지 비어 있는 cold compile을 뜻하지 않는다.

| seed | native 초회 denoise (s, 분모 제외) | alpha denoise (s) | r4 denoise (s) | 비교 상태 / 비율 | RGB max (/255) | RGB RMSE (/255) | strict |
|---:|---:|---:|---:|---|---:|---:|---|
| 20261107 | 66.8956 | 50.8797 / 50.8693 | 46.3941 | warm 1회 1.09657× | 243.4411 | 49.7999 | 실패 |
| 20261108 | 65.6997 | 50.7787 | 46.4120 | first 관측 1.09409× | 240.1403 | 48.2224 | 실패 |
| 20261109 | 65.6149 | 51.0179 | 46.5498 | first 관측 1.09599× | 241.3618 | 54.0203 | 실패 |

seed07의 첫 r4는 46.4318 s, 반복 r4는 46.3941 s였다. 두 후보의 latent/RGB raw float bytes는 일치했고, 마지막 alpha를 복원하니 원 baseline/첫 alpha와도 bit 일치했다. alpha 앞뒤 두 값의 중앙값 50.8745 s를 warm r4 46.3941 s로 나누면 **1.09657185배**다. seed08/09 후보는 first라 이 warm 평균에 합치지 않는다. 각 seed baseline과 alpha는 latent/RGB byte 일치, 모든 요청의 conditioning embedding SHA도 동일했다.

후보 VAE 단계는 각각 2.6143 / 2.6232 / 2.6234 s였다. 후보 float latent/RGB는 `candidate_*_tensors.pt`로 별도 저장했다. 저장은 stage clock 뒤에서 실행해 denoise/VAE에는 들어가지 않으나 request wall에는 포함된다. 따라서 request wall을 저장 코드가 없는 운영 서버 응답 시간처럼 사용하지 않는다.

세 run의 GPU audit는 foreign client 0, sample count 1168 / 742 / 743였고 timing valid다. 모델 프로세스 종료 뒤 `model_cleanup.json`에서 guard 재획득·해제, KFD client 없음, 소유 PID 잔류 없음을 확인했다. 원자료 경로와 SHA, 모든 요청의 stage·출력 SHA·strict 수치는 [MODEL3_SUMMARY.json](MODEL3_SUMMARY.json)에 모았다.

**결과는 3/3 seed strict 실패**다. 관측된 약 1.095배 디노이징 시간 감소를 사용 가능한 무손실 최적화로 올리지 않는다.

## 7. 익명 지각 평가

사전 절차는 **2 독립 판정자 × 3 seed × 좌우 배치 2방향 = 12결정**이다. 각 방향을 fresh context로 보고, 실제 공통 prompt/negative prompt와 공통 입력을 제공하되 원본/후보 정체·seed·속도·오차는 숨긴다. full과 같은 좌표의 C1/C2/C3 crop을 모두 평가한다. crop은 결과 전에 정한 upper (.05,.02,.95,.42), middle (.05,.30,.95,.75), lower (.05,.58,.95,.98)이고 1:1 원본 픽셀이다.

패널은 실제 3-seed 출력에서 만들었다. seed07은 사전 계획의 warm index3, seed08/09는 index2 후보를 사용했다. 익명 자료에는 이 매핑이 없으며 coordinator만 PRIVATE_MAPPING을 읽는다. [public packet index](blind_panels/JUDGE_PACKET_INDEX_KO.md), [PUBLIC_CASES.json](blind_panels/PUBLIC_CASES.json), [PUBLIC_TASK.json](blind_panels/PUBLIC_TASK.json), [JUDGE_PROMPT_KO.txt](blind_panels/JUDGE_PROMPT_KO.txt)가 판정 입구다.

full/crop 24개와 공통 입력 1개, 총 25 PNG를 보존했다. 모든 출력은 PNG metadata를 제거했고 원 이미지 영역의 저장·재로드 픽셀 byte 일치를 검증했다. resize는 하지 않았다. 가장 큰 파일도 1,830,270 bytes로 CodexPro2의 실제 2,000,000 bytes 한도 안이다. full은 모두 A/B를 나란히 배치할 수 있었다. 정확한 원 PNG와 float tensor는 모델 결과 폴더에도 남아 있다.

판정 항목은 얼굴/손·해부, 텍스트/작은 물체, 윤곽, 질감/세부, 색/명암, 전체 구성과 공통 지시 충실도다. severity 0–3, 두 결과의 공유 결함과 한쪽만의 열화를 구분한다. 같은 후보 열화가 severity≥2로 양방향에 반복되는 seed는 실패, 순서 의존이나 판정자 충돌은 미확정이다. 12결정 중 최소 10개 동등·우세이며 전 seed에서 반복 열화가 없을 때만 제한적 지각 통과다. 엄격 RGB/native 재현 실패는 이와 별도로 유지한다.

**실행 완료: 12/12결정 native 선호, 후보의 상대 severity 모두 3, 지각 gate 실패.** 두 논리 판정자 J1/J2를 좌우 순서별 새 문맥으로 총 네 번 실행했다. 세 seed 각각에서 두 판정자 모두 양방향에 같은 인물·얼굴의 다층 반복과 신체/소품 연결 붕괴를 지적했다. 6/6개의 판정자×seed 쌍에서 동일한 큰 결함이 반복됐다. 후보 동등·우세 및 severity≤1은 0/12이며, 선호 반전·후보 채택 결론 충돌은 각각 0이다.

모델 표시는 네 세션 모두 GPT-6 Astra Pro, 내부 revision은 unknown이다. 같은 계열 모델의 독립 문맥이며 서로 다른 아키텍처 검증이라고 주장하지 않는다. 각 세션의 viewed_files는 공통 입력·세 full·아홉 crop, 13개 고유 파일을 포함하고 배정 목록과 일치한다. native 쪽 푸딩·손/소매의 작은 지시 차이를 지적한 응답도 원문에 남겼다. 이 세부 관찰의 차이가 후보의 severity 3 및 native 선호라는 결론을 바꾸지는 않았다.

[12개 전체 표·양방향 확인·한계](blind_judgements/BLIND_EVALUATION_KO.md), [verdict JSON](blind_judgements/UNBLINDED_VERDICT.json), [동결 프롬프트](BLIND_JUDGE_PROMPT_V07.json), [실제 네 세션의 지시문](blind_judgements/SESSION_PROMPTS.json)을 보존했다. 원응답은 [J1/A](blind_judgements/J1_session_A_raw.json), [J1/B](blind_judgements/J1_session_B_raw.json), [J2/A](blind_judgements/J2_session_A_raw.json), [J2/B](blind_judgements/J2_session_B_raw.json)다.

이 결과는 단순 픽셀 차이 외에도 이번 r4가 전역 장면 일관성을 크게 손상시킨다는 출력 증거다. 국소 mask의 전역 관계 손실과 일관되지만 원인별 ablation을 완료한 것은 아니다. all-true 대조의 strict 실패도 있어 전체 오차를 pruning 하나에 귀속하지 않는다. 기존 strict 3/3 실패·late native 재현 tolerance 실패·채택 불가는 그대로 유지한다.

## 8. 실패·수정 이력과 원자료 보존

| 실패 | 원인 / 복구 | 처리 |
|---|---|---|
| 첫 25-step capture 중단 | installed PyTorch stable sort의 dim을 positional로 전달; dim= 키워드로 수정 | 첫 결과와 이전 소스 보존, CPU API 회귀 확인 후 retry |
| 최초 5-case bench 종료 | 잘못된 causal 참조의 NaN을 allow_nan=False JSON으로 저장하다 실패 | NaN 경로·finite 상태를 표준 JSON 문자열+failure 필드로 보존하도록 수정 |
| 위 run의 종료 audit 누락 | finally에서 같은 serializer가 다시 실패 | 첫 두 case 속도는 탐색용으로만 보존; corrected 완전 audit 3-case로 대체 |
| masked text 계약 오독 | 실제 bool 0/1 additive 입력을 논리적 제외로 읽음 | raw mask/source 대조, 명시적 bias 옵션 추가; native masked 모델 경로 유지 |
| 후반 same-mask gate 실패 | FP32 계산 경로 간 재현 오차; 모든 출력 finite, FP64는 row에 따라 승패 다름 | gate 유지, 후보 시간 제외, 모델은 채택 불가 품질 진단으로만 진행 |
| r4 strict RGB 실패 | 큰 근사 오차와 계산 경로 변화가 최종 모델에 전달 | 3 seed·all-true·blind 결과와 함께 실패로 보존, 운영 자동 연결 없음 |

`sources_initial_smoke/`에 수정 전 소스를 보존했다. 초기 `bench_actual5/`의 첫 두 timing은 최종 audit가 없어 유효 성능표에서 제외한다. `capture25_seed20261107_retry/`는 GPU 겹침은 없었어도 CPU 빌드 중 품질 전용이므로 시간을 버렸다. 유효 timing run은 각각 gpu_audit를 별도 파일로 남겼다. audit는 KFD client 주기 관측 및 research flock으로 직렬성을 확인하며 그래픽 부하나 polling 사이의 짧은 외부 client까지 증명하는 장치는 아니다.

## 9. 재사용 파일·SHA와 재현 명령

| 파일 | SHA256 |
|---|---|
| rdna2_ops/sparse_attention_v07.py | `76007ff7087dd251353dbf8018bf960c48e27792f46fdefaffe233305cece606` |
| sparse_switches_v07.py | `27cd0d14b73aa68d6d60d75b87a003286ed6eaa5f1df538bda7f7da0479512eb` |
| bench_sparse_attention_v07.py | `12d7a8805243d6a00c7bfdc9f46989e9e80280383a912b62517e52eef8ddd7e9` |
| instrument_sparse_capture_v07.py | `f176735171f714d634f7a0eeca7a63c5ff8763196e4bd919afec4d0811e9d1a7` |
| instrument_sparse_outputs_v07.py | `6ad10d0db8cb64f07dbb43978aae095fd4558bb3d67547140ee9c4eb93d73773` |
| prepare_sparse_blind_v07.py | `99010b69ce7f784032fb9c2cb38f2ddf00de6bedb05b74dc830bf28d93a87c1e` |
| research_configs/qwen_sparse_radial4_v07.json | `310b052d797c5dcf3fd1b9d7d57f9910040171fcdb7c45bd4cdda148e5d82a85` |

현재 corrected source는 실제 bool additive 계약만 명시적으로 추가했다. 중후반 2-case 원자료의 kernel SHA는 이전 `f5d556a26151abb0e6501999c1125269e6e7df05836d50cebd0186291d7d3039`이고 source snapshot을 보존했다. unmasked 수식은 그대로지만 생성 ASM의 SHA는 달라졌으므로 바이너리 동일이라고 주장하지 않는다. 실제 각 실험에서 사용한 전체 SHA는 해당 results의 `source_sha256`이 기준이다.

실행 환경 원자료: RX 6900 XT gfx1030, Torch 2.12.0+rocm10.2.0a20261011, HIP 7.17.26405, Triton 3.8.0. 운영 Comfy/custom node/config를 변경하지 않고 연구 adapter를 소유한 별도 프로세스에서만 설치하고 복원했다. root의 v0.7 보호 파일 manifest가 운영 전후 SHA를 확인한다.

연구 디렉터리에서 기존 Qwen venv를 쓴다. 아래 출력 디렉터리는 재현 때 새 이름으로 바꾸고, GPU lease와 guard가 비어 있는 상태에서 **한 명령씩** 실행한다. 모델·패키지를 다운로드하거나 설치하는 단계는 없다.

```bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
export HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0
export MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53=0

~/DevSpace/qwen-image-21-rdna2-lab01/.venv/bin/python \
  bench_sparse_attention_v07.py --out results/sparse_v07/repro_operator5 \
  --captures \
  results/sparse_v07/capture25_seed20261107_retry/sparse_capture/actual_qkv_2400_4964.pt \
  results/sparse_v07/capture25_seed20261107_retry/sparse_capture/actual_qkv_2400_2408.pt \
  results/sparse_v07/capture25_seed20261107_retry/sparse_capture/actual_qkv_156_2564.pt \
  results/sparse_v07/selected_qkv_seed20261107/sparse_capture/representative_qkv_step12_layer15.pt \
  results/sparse_v07/selected_qkv_seed20261107/sparse_capture/representative_qkv_step24_layer31.pt

~/DevSpace/qwen-image-21-rdna2-lab01/.venv/bin/python \
  research_comfy_v05.py --spec research_configs/qwen_sparse_radial4_v07.json \
  --out results/sparse_v07/repro_model20261107 --steps 25 --seed 20261107 \
  --modes baseline,dense_alpha,sparse_radial,sparse_radial,dense_alpha --port 8487
```

seed 20261108/20261109는 각각 새 out에 `--modes baseline,dense_alpha,sparse_radial`로 실행한다. 다른 seed의 first candidate를 warm 평균에 합치지 않는다. source 변경 전 기록과 현재 corrected source의 SHA를 함께 비교한다.

블라인드 panel은 CPU 준비기 `prepare_sparse_blind_v07.py --pairs <private manifest> --out <fresh directory>`로 재생성한다. manifest와 PRIVATE_MAPPING은 판정자에게 전달하지 않는다.

## 10. 새 모델에서 이 칸을 읽는 절차

1. 실제 호출 경계·dtype·heads·Q/K/D·mask 의미·prefix/cache를 확인한다. bool이라고 logical mask라고 단정하지 않는다.
2. 원래 최적화된 dense 기준을 먼저 확보한다. scale alpha 접기 같은 이미 bit 일치하는 개선을 누락한 느린 기준과 경쟁하지 않는다.
3. 최소 3개 실제 shape에 더해 early/mid/late·layer·head/query 하위 질량을 잰다. 압축 CDF와 실제 full-softmax retained mass를 구분한다.
4. 같은 mask 계산 오차와 key 생략 근사 오차를 분리하고, 최소 한 번은 같은 커널 all-true로 모델을 대조한다.
5. dynamic selector·layout copy·mask 준비 비용을 포함할 위치를 명확히 한다. metadata 복잡도와 allocator peak 범위를 함께 쓴다.
6. 실제 최종 출력의 사전 gate와 여러 seed A/B를 끝낸 뒤 적용을 판단한다. native 반복·conditioning SHA가 흔들리면 후보 효과로 해석하지 않는다.
7. 엄격 실패와 지각 평가는 별도 판정한다. 관측 몇 지점의 통과를 자동 fallback 범위로 일반화하지 않는다.
8. 실패 후보와 source/raw/results를 남기고 운영 파일을 바꾸지 않는다.

## 1차 문헌

- SpargeAttention 논문 §3.2–3.4: https://arxiv.org/html/2502.18137v6
- 공식 SpargeAttn 소스/API: https://github.com/thu-ml/SpargeAttn
- Radial Attention 논문 §4.2, 식(4): https://arxiv.org/html/2506.19852v1

실측 재현의 입구는 [OPERATOR5_SUMMARY_KO.md](OPERATOR5_SUMMARY_KO.md), [MAP_ANALYSIS_KO.md](MAP_ANALYSIS_KO.md), 모델 각 seed의 runs/audit 및 본 보고서의 blind 원응답 경로다.
