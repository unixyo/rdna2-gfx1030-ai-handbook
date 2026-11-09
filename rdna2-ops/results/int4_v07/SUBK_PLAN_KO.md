# G128 sub-K 분리 — 추가 실험 실행 전 고정

기존 full sweep의 dot4 G128에서 256 VGPR / spill 136–137을 관측했다. source는 그룹 G 전체를 한 dot/load K 폭으로 사용하는 구조다. 이 직접 관측에 대응하는 변형 한 개만 추가한다.

- `dot4_split`: quant group G는 유지하고 compute BK=min(G,64)로 분리한다. 각 그룹 안에서 INT32 dot partial을 먼저 정확하게 합친 뒤 기존과 같은 순서로 FP32 group scale을 한 번 적용한다.
- G≤128 정수합 범위는 기존 계약 안이다. 원래 dot4 커널은 삭제하거나 변경하지 않는다. 이전 source는 각 완료 run의 source/에 이미 보존했다.
- 제한된 검증: G128의 대칭/비대칭 × NK/KN × nibble low/high 작은 CPU INT64 oracle, 그리고 기존 대표 projection 3개(MOSS/Qwen/Anima), KN·low-first·G128의 sym/aff만 측정한다. tile=64×64, 3 round×3회 교대.
- 기존 dot4 전체 output byte 일치도 함께 기록한다. 같은 저장값 numeric gate, native 대비 1.05 배 gate는 그대로다.
- 개선된 ISA의 VGPR/spill과 기존 원시시간을 함께 비교한다. spill을 줄여도 native 관문을 넘지 않으면 추가 타일 탐색이나 9×6 확장은 하지 않는다.
- 현재는 CPU 구현·AST 검사만 완료했다. GPU 실행은 다음 독점 lease에서만 한다.
