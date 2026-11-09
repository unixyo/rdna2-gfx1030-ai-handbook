# MOSS INT4 최종 출력 진단 — 실행 전 고정

상태: 모델 실행 전. 기존 512²·4-step SFT workflow와 seed 20261105를 쓴다. g32, affine(0을 포함하는 min/max), KN, low-first를 품질 우선으로 고정했다. 출력 엄격 기준은 RGB 최대 2/255, RMSE 0.25/255, finite다.

## 세 계산 경로

1. `int8_fused`: 기존 W8A8 가중치와 활성화, v0.5 FP32 dequant 융합 기준선. 처음과 끝의 기준 요청이 latent/RGB byte 일치해야 한다.
2. `int4_reference`: 같은 원래 W8 복원값에서 고정 규칙으로 만든 W4를 사용한다. 각 K 그룹의 중심화 정수 q-z와 A8을 torch FP32 GEMM으로 곱한다. G≤128에서 모든 정수 부분합의 절댓값은 128·15·128=245760 < 2^24라 FP32로 정확히 표현된다. 이후 `total = total + integer_dot * group_scale`을 그룹 증가 순서로 별도 torch mul/add로 계산하고, 마지막에 activation row scale과 bias를 적용한다. 후행 scale과 그룹 합산의 FP32 반올림은 이 참조 계약에 포함된다. 이는 속도 기준선이 아닌 독립 정확성 oracle이다.
3. `int4_dot4` 또는 `int4_dot4_words`: 동일한 W4 저장값과 동일한 A8 활성화에서 직접 커널을 쓴다. 연산 선별 결과로 선택한 경로명을 run 기록에 명시한다.

원본 대비 W4 참조의 최종 RGB 변화와, W4 참조 대비 직접 커널의 계산 차이를 별도로 기록한다. 후자가 byte 일치하면 모델의 차이는 커널의 계산 오류 없이 W4 표현 변경을 포함하는 경로에서 생겼다고 판정한다. 후자까지 다르면 두 차이를 합쳐 표현 손실이라고 부르지 않는다.

기존 input activation, ConvRot, activation quantizer, bias, residual는 MOSS v0.5 adapter로 보존한다. pack은 매 호출의 범위 안에서 새로 만들며 빌린 weight pointer cache를 유지하지 않는다. 이 진단의 시간에는 그 준비 비용이 포함되고, reference의 다수 GEMM 비용을 속도 최적화 성과로 쓰지 않는다. 모든 운영 파일과 checkpoint는 수정하지 않는다.

실행 순서는 `warm,int8_fused,int4_reference,int4_dot4_words,int8_fused`를 기본으로 한다. byte/수치 비교와 source SHA, 실제 치환 hit·shape, 보호 복원, 원본/후보 이미지, gpu_guard를 남긴다. 실제 커널 선택이 dot4_words가 아니라 dot4이면 해당 토큰만 바꾸고 기록한다. 첫 요청/컴파일 준비와 warm 수치를 섞지 않는다.
