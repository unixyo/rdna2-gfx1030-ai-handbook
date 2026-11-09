# RDNA2 v0.7 사전 판정 기준

기록 시점: 첫 v0.7 GPU 실험 이전. 대상 PC의 기존 모델과 Triton 3.8을 사용한다.

## 공통

- 기존 모델 경로를 기준선으로 삼고, 커널 / 준비·변환 포함 연산 / 모델 전체 시간을 나눈다.
- 초회와 warm을 분리한다. warmup 횟수, prewarm shape, 교대 순서, 원시 sample을 보존한다.
- GPU는 gpu_guard_v06.GpuAudit의 공통 lock과 KFD 관측으로 직렬 실행한다. 외부 compute PID 또는 관측 실패가 있으면 timing을 채택하지 않는다. 관측 간격보다 짧은 외부 작업과 graphics 부하는 이 감사만으로 배제할 수 없음을 명시한다.
- MIOPEN_FIND_MODE=NORMAL 및 FP32 Conv 참조에는 MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0, MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53=0을 함께 지정한다.
- 모델 RGB strict: normalized [0,1] 최대 절대오차 ≤ 2/255, RMSE ≤ 0.25/255, 모두 finite. dtype·shape와 원본 bytes/SHA도 별도 기록한다.
- operator 성공을 모델 채택으로 승격하지 않는다. 동일 dtype·값을 쓴 계산 경로 오차와 값 자체를 바꾼 표현 손실을 분리한다.

## INT4

- 실제 MOSS INT8 융합, Qwen patientx, Anima FP16 rocBLAS를 각각 기준선으로 삼는다.
- W4A8 dot8은 A8의 signed nibble 분해와 보정 비용을 포함한다. W4A4는 별도 활성화 근사다.
- 그룹 g32/64/128, 대칭/비대칭, packing 순서와 레이아웃, 준비 및 호출당 dequant 비용을 명시한다.
- 정수 부분은 CPU 정수 참조와 정확 일치해야 한다. 부동소수점 scale·합산은 독립 참조 대비 오차를 기록하고, 후보 구현의 사전 gate를 첫 실행 전에 별도 INT4 계획 파일에 확정한다.
- 원본→W4 및 원본 activation→A8/A4 손실은 계산 경로 오차와 별도 표로 남긴다. 모델 채택은 위 RGB strict를 통과해야 한다.

## Sparse Attention

- 실제 Qwen long-sequence를 먼저 조사한다. 후보별 보존한 확률 질량과 block 수, step/layer 변화를 기록한다. 확률 질량 지표는 품질 통과를 대신하지 않는다.
- Sparge식·Radial식·고정 block 후보의 설계 충실도와 실제 구현 차이를 명시한다.
- sparse kernel은 동일 mask의 dense 참조와 먼저 비교한다. 모델 근사 오차는 mask 없는 실제 baseline과 비교한다.
- 모델 seed는 20261107, 20261108, 20261109를 기본으로 사용한다. 실제 harness가 허용하는 기존 workflow의 동일 입력·conditioning을 재사용한다.
- native 반복 재현이 실패하면 후보의 causal 품질 판정은 미확정으로 남긴다.

### 독립 지각 판정: strict와 별도

- 두 독립 판정 세션에 익명 full image와 동일 좌표 crop(얼굴·손/경계·질감 등)을 제공한다.
- 각 seed를 좌/우 순서를 바꾼 두 번의 fresh-context 평가로 제시한다. 후보 이름·시간·수치 오차·어느 쪽이 원본인지 알리지 않는다.
- 판정 항목: 구도/의미·얼굴/손·텍스트(있을 때)·경계·질감·색. 차이 심각도는 0(없음), 1(작음), 2(명확한 열화), 3(심한 파손).
- 원본/후보 공개 뒤, 같은 판정자가 양방향에서 후보의 동일 결함을 severity≥2로 지적한 seed가 하나라도 있으면 지각 gate 실패. 양방향 의견이 뒤집히거나 독립 판정자끼리 갈리면 미확정. 3 seed 모두 일관된 명확한 열화가 없고 12개 판정 중 적어도 10개에서 후보가 동등/우세(severity≤1)일 때에만 '제한된 표본에서 지각 gate 통과'라고 기록한다.
- 개별 원응답·프롬프트·이미지 SHA·익명화 대응표·판정자의 확인 가능한 모델명/날짜를 보존한다. 내부 모델 revision이 노출되지 않으면 미확인이라고 쓴다.
- 연구자의 육안, Claude 기존 육안, 단일 비교, 작은 표본의 통과를 보편적 품질 동등으로 부르지 않는다. strict 실패는 지각 결과로 고쳐 쓰지 않는다.

## Strata prefill

- 운영 unit/drop-in/launcher/engine/pack 전후 SHA와 서비스 상태를 기록한다. 서비스 시작·중지·재시작하지 않는다.
- 현재 inactive인 서비스 상태가 model run 직전에도 유지될 때만 별도 worktree/build/port로 연구 인스턴스를 실행한다.
- stage timer의 GPU event 구간에 host scheduling/page fault가 포함되는지 확인하고, 실제 copy bytes·host stall·compute와 구분해 해석한다.
- 실제 layer pack operator는 비트 일치 또는 기존 Q8_1의 사전 tolerance를 적용한다.
- 모델은 동일 9.7K·19.8K·119K fresh prompt의 tok/s, temperature 0 생성 token ID 일치로 판정한다. timing profile과 무계측 A/B 시간을 구분한다.
- decode 최적화·운영 설정 변경은 이번 실험에 포함하지 않는다.
