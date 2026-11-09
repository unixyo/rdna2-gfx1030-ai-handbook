# Sparse v0.7 사전 실험 기준

이 파일은 첫 GPU 실행 전에 저장한다. 상위 기준은 `results/V07_PREREGISTERED_CRITERIA_KO.md`다.

- 실제 기준선: 기존 Qwen ComfyUI `attention_split` 및 후행 scale alpha 접기. FP32 Q/K/V head-contiguous 복사, native softmax, FP32 P·V를 유지한다. masked prefix는 원함수로 처리한다.
- 후보 구현: FP32 스트리밍 block-sparse, mask 128×64, QK/PV 모두 FP32 IEEE FMA. 후보의 같은 mask 계산 오차와 dense 대비 생략 오차를 분리한다.
- 계산 정확도: 독립 CPU FP64의 tail/strided/causal-prefix 3문제와 같은 mask의 rocBLAS 참조에 `atol=3e-5, rtol=3e-4`, 전값 finite. 실패 커널은 성능/모델 채택 후보에서 제외한다.
- selector 블록 mean+pairwise cosine 융합은 별도 CPU PyTorch 참조와 mean/cosine 최대 절대오차 각각 ≤1e-5로 검증한다. 이 gate는 fused selector의 첫 GPU 실행 전에 추가했다.
- 원모델 근사 오차는 max, RMSE, relative RMSE, relative L1로 보고한다. 확률 질량 또는 연산 오차는 최종 RGB 기준을 대체하지 않는다.
- 최종 RGB: float RGB 최대 2/255 및 RMSE 0.25/255, 같은 conditioning·seed native 반복 SHA로 비교 기준을 확인. seed 20261107/20261108/20261109, 원 workflow의 25 Euler/simple steps.
- 실제 희소성: step 0/12/24 × layer 0/15/31, 모든 head/query에 대한 post-softmax block 질량과 per-query retained mass min/p01/median/p99/mean. 단일 head 평균만으로 전체 희소성을 주장하지 않는다.
- 후보 격자: Sparge-inspired mass .90/.95/.99와 cosine .1, mass .95/cosine .5. 보존 조건은 low-cosine Q/K block dense. fixed block fraction .5. 이미지 공간 거리 radius 4/8/16 token, 텍스트는 전체 유지. block geometry는 실제 build_sequence에서 읽는다.
- 원 Radial 논문 식(4): f=1/2의 중앙 band는 dense다. CPU 식 검증으로 기록하고 image-only 2D 거리 실험과 구분한다. 영상 temporal Radial의 효과를 검증했다고 부르지 않는다.
- 선택 규칙: 같은 mask 계산을 통과한 후보 중 실제 selector 포함 시간을 기준으로 비교한다. 근사 후보의 모델 검증은 성능 통과 후 자동 채택이 아니라 명시적 품질 진단이다. 매 단계 실패도 보존한다. 모델에 연결할 격자는 최초 질량 진단 뒤 확정 파일로 남기고 그 뒤 RGB 결과를 보고 완화하지 않는다.
- 모델의 독립 지각 판정은 root가 2명, 3 seeds, 좌우 양방향(12회)과 같은 crop/full로 실행하고 원문을 보존한다. severity≥2 동일 결함 양방향 재현 seed는 실패. 판정자 충돌·순서 의존은 미확정. 10/12 이상 동등/우세+반복 열화 없음일 때만 제한적 통과. strict 실패를 지각 통과로 고쳐 쓰지 않는다.
- warm: 첫 호출(JIT 포함)을 별도 기록한 뒤 2 warmup, 순서 회전/반전 3round×2call. static map 첫 준비 시간은 별도, 동적 Sparge selector는 전체 호출 시간에 포함한다. GPU 독점 GpuAudit가 불가/겹침이면 시간 증거 폐기.
- 세 shape는 실제 Qwen target 2400×4964, reference 2400×2408, masked text 156×2564를 수집한다. 세 번째가 현재 prompt의 tokenizer 결과로 달라지면 실제 shape를 그대로 기록한다. 공간 radial은 image query에만 적용하며 text query는 원래 mask를 유지한다.

## 1차 문헌

- SpargeAttention (ICML 2025), https://arxiv.org/html/2502.18137v6 , §3.2–3.4. 이 구현은 selective compression만 참고한 독립 FP32 구현이다. upstream Sage quantization/PV warp skip을 이식했다고 주장하지 않는다.
- 공식 API/지원, https://github.com/thu-ml/SpargeAttn , mask 128×64, upstream CUDA 경로.
- Radial Attention (NeurIPS 2025), https://arxiv.org/html/2506.19852v1 , §4.2 식(4). f=1/2 dense 중앙 band를 확인했다.
