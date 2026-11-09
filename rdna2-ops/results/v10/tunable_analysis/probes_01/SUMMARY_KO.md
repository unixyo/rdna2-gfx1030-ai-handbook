# TunableOp CPU 결과 요약

GPU를 실행하지 않고 저장된 JSON/trace를 읽은 결과다. 누락은 pending이다.

warm 제외 규칙: first-per-mode. 각 속도 비율에는 arm별 warm 2회 이상이 필요하다.

| 프로브 | 상태 | signature | solution | on 출력 종류 | off 출력 종류 | 선택 kernel 일치 |
|---|---|---|---|---:|---:|---|
| qwen16_original_p0 | verified | tn_64_2400_4096_ld_4096_4096_64 | Gemm_Rocblas_-1086325192 | 32 | 1 | mismatch |
| qwen16_default_p0 | verified | tn_64_2400_4096_ld_4096_4096_64 | Default | 1 | 1 | pending |

| 모델/역할 | 상태 | mode | warm n | denoise 중앙값 s | RGB 반복 | 조건 반복 | 속도 판정 가능 |
|---|---|---|---:|---:|---|---|---|

| 속도 기준 | 후보 | 상태 | 기준 warm n | 후보 warm n | 배율 |
|---|---|---|---:|---:|---:|

프로브 상태는 측정 관문이며 선택 kernel trace의 확인 상태는 별도다.
비교·관문·SHA·stage별 실제 CSV hit·trace 이름과 누락 경로는 analysis.json에 있다.
