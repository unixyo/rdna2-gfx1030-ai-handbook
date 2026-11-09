# TunableOp CPU 결과 요약

GPU를 실행하지 않고 저장된 JSON/trace를 읽은 결과다. 누락은 pending이다.

warm 제외 규칙: first-per-mode. 각 속도 비율에는 arm별 warm 2회 이상이 필요하다.

| 프로브 | 상태 | signature | solution | on 출력 종류 | off 출력 종류 | 선택 kernel 일치 |
|---|---|---|---|---:|---:|---|
| missing_probe | pending | pending | pending | pending | pending | pending |

| 모델/역할 | 상태 | mode | warm n | denoise 중앙값 s | RGB 반복 | 조건 반복 | 속도 판정 가능 |
|---|---|---|---:|---:|---|---|---|
| anima_legacy/original | pending | base | 1 | 60.86959405399102 | pending | pending/pending | False |
| anima_legacy/original | pending | cand | 1 | 45.19573462801054 | pending | pending/pending | False |
| qwen_legacy/original | pending | base | 1 | 57.05722161900485 | pending | pending/pending | False |
| qwen_legacy/original | pending | tbase | 2 | 56.354807138995966 | pending | pending/pending | False |
| qwen_legacy/original | pending | cand | 0 | pending | pending | pending/pending | False |
| qwen_legacy/original | pending | tcand | 2 | 50.643935724991024 | pending | pending/pending | False |
| missing_model/off | pending | pending | 0 | pending | pending | pending | False |

| 속도 기준 | 후보 | 상태 | 기준 warm n | 후보 warm n | 배율 |
|---|---|---|---:|---:|---:|
| missing_model:base | anima_legacy:pending-mode | pending | 0 | 0 | pending |
| missing_model:base | qwen_legacy:pending-mode | pending | 0 | 0 | pending |

프로브 상태는 측정 관문이며 선택 kernel trace의 확인 상태는 별도다.
비교·관문·SHA·stage별 실제 CSV hit·trace 이름과 누락 경로는 analysis.json에 있다.
