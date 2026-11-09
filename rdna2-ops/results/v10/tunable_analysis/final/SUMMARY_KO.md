# TunableOp CPU 결과 요약

GPU를 실행하지 않고 저장된 JSON/trace를 읽은 결과다. 누락은 pending이다.

warm 제외 규칙: first-per-mode. 각 속도 비율에는 arm별 warm 2회 이상이 필요하다.

| 프로브 | 상태 | signature | solution | on 출력 종류 | off 출력 종류 | 선택 kernel 일치 |
|---|---|---|---|---:|---:|---|
| qwen16_original_p0 | verified | tn_64_2400_4096_ld_4096_4096_64 | Gemm_Rocblas_-1086325192 | 32 | 1 | matched |
| qwen16_original_p1 | verified | tn_64_2400_4096_ld_4096_4096_64 | Gemm_Rocblas_-1086325192 | 32 | 1 | matched |
| qwen16_default_p0 | verified | tn_64_2400_4096_ld_4096_4096_64 | Default | 1 | 1 | pending |
| qwen0_original_p0 | verified | tn_4096_164_4096_ld_4096_4096_4096 | Gemm_Rocblas_-1086325182 | 32 | 1 | matched |
| qwen0_default_p0 | verified | tn_4096_164_4096_ld_4096_4096_4096 | Default | 1 | 1 | pending |
| qwen19_original_p0 | verified | nt_640_640_1152_ld_640_640_640 | Gemm_Rocblas_-1086324823 | 1 | 1 | matched |
| anima2_original_p0 | verified | tn_1024_32_1024_ld_1024_1024_1024 | Gemm_Rocblas_-1086325251 | 32 | 1 | matched |
| anima2_original_p1 | verified | tn_1024_32_1024_ld_1024_1024_1024 | Gemm_Rocblas_-1086325251 | 32 | 1 | matched |
| anima2_default_p0 | verified | tn_1024_32_1024_ld_1024_1024_1024 | Default | 1 | 1 | pending |
| anima5_original_p0 | verified | tn_1024_32_2048_ld_2048_2048_1024 | Gemm_Rocblas_-1086325257 | 32 | 1 | matched |
| anima7_original_p0 | verified | tn_1024_32_3072_ld_3072_3072_1024 | Gemm_Rocblas_-1086325257 | 32 | 1 | matched |
| anima13_original_p0 | verified | tn_1024_156_2048_ld_2048_2048_1024 | Gemm_Rocblas_-1086325182 | 32 | 1 | matched |
| anima15_original_p0 | verified | tn_1024_156_3072_ld_3072_3072_1024 | Gemm_Rocblas_-1086325182 | 32 | 1 | matched |

| 모델/역할 | 상태 | mode | warm n | denoise 중앙값 s | RGB 반복 | 조건 반복 | 속도 판정 가능 |
|---|---|---|---:|---:|---|---|---|
| anima_diagnostic/original | verified | base | 0 | pending | pending | pending/pending | False |
| anima_diagnostic/original | verified | cand | 1 | 45.507516364974435 | bitwise_equal | bitwise_equal/bitwise_equal | False |
| anima_original_p0/original | verified | base | 0 | pending | pending | pending/pending | True |
| anima_original_p0/original | verified | cand | 2 | 45.231108354986645 | bitwise_equal | bitwise_equal/bitwise_equal | True |
| anima_filtered_p0/filter | verified | base | 0 | pending | pending | pending/pending | True |
| anima_filtered_p0/filter | verified | cand | 2 | 45.08772157599742 | bitwise_equal | bitwise_equal/bitwise_equal | True |
| anima_filtered_p1/filter | verified | base | 0 | pending | pending | pending/pending | True |
| anima_filtered_p1/filter | verified | cand | 2 | 45.11531504600134 | bitwise_equal | bitwise_equal/bitwise_equal | True |
| anima_off_p0/off | verified | base | 2 | 80.71588143499685 | bitwise_equal | bitwise_equal/bitwise_equal | True |
| qwen_diagnostic/original | verified | tbase | 1 | 5.155496433988446 | different | bitwise_equal/bitwise_equal | False |
| qwen_original_p0/original | verified | base | 1 | 57.00392360400292 | bitwise_equal | bitwise_equal/bitwise_equal | True |
| qwen_original_p0/original | verified | tbase | 2 | 55.71558646099584 | different | bitwise_equal/bitwise_equal | True |
| qwen_filtered_p0/filter | verified | base | 1 | 57.56962112599285 | bitwise_equal | bitwise_equal/bitwise_equal | True |
| qwen_filtered_p0/filter | verified | tbase | 2 | 56.448651260492625 | bitwise_equal | bitwise_equal/bitwise_equal | True |

| 속도 기준 | 후보 | 상태 | 기준 warm n | 후보 warm n | 배율 |
|---|---|---|---:|---:|---:|
| anima_off_p0:base | anima_diagnostic:cand | pending | 2 | 1 | pending |
| anima_off_p0:base | anima_original_p0:cand | verified | 2 | 2 | 1.784521413924142 |
| anima_off_p0:base | anima_filtered_p0:cand | verified | 2 | 2 | 1.7901965017005024 |
| anima_off_p0:base | anima_filtered_p1:cand | verified | 2 | 2 | 1.789101580088619 |

프로브 상태는 측정 관문이며 선택 kernel trace의 확인 상태는 별도다.
비교·관문·SHA·stage별 실제 CSV hit·trace 이름과 누락 경로는 analysis.json에 있다.
