# Sparse v0.7 유효 연산 5-case 요약

3개 실제 shape와 같은 long shape의 중후반 실제 QKV 2개다. 첫 3개는 bool-additive 계약 수정 후 완전 audit로 재측정했고, 중후반 2개는 bench_remaining3의 겹침 없는 결과다. 모든 source SHA·세부 통계·ISA·FP64 진단은 OPERATOR5_SUMMARY.json 및 연결 raw results에 있다. 운영 파일/기존 masked 경로는 수정하지 않았다.

| case | Q/K | 실제 baseline (ms) | Sparge .95 selector 포함 (ms) | radial r4 (ms) | r4 배속 | r4 same-mask gate | r4 max / RMSE |
|---|---|---:|---:|---:|---:|---|---|
| target_step0_layer0 | 2400/4964 | 22.666 | 54.434 | 18.600 | 1.219× | 통과 | 2.027e-5 / 2.579e-7 |
| reference_prefix_layer0 | 2400/2408 | 12.867 | 24.316 | 8.201 | 1.569× | 통과 | 2.432e-5 / 3.540e-7 |
| masked_text_prefix_layer0 | 156/2564 | 1.892 | 1.746 | — | — | image geometry 아님 | — |
| target_step12_layer15 | 2400/4964 | 22.704 | 18.251 | 18.524 | 1.226× | 통과 | 3.862e-5 / 1.615e-6 |
| target_step24_layer31 | 2400/4964 | 22.757 | — | — | — | 실패; 시간 제외 | 1.864e-4 / 5.089e-6 |

masked-text의 실제 baseline은 attention_split이다. generic alpha forwarding wrapper의 더 큰 wall time을 기준으로 배속을 부풀리지 않는다. 원자료에는 두 함수 모두 남겨 두었다.

## 메모리·tile 및 계산 상태

| case | baseline incremental MiB | r4 incremental MiB | dense q32 regs/spills | dense q16 regs/spills | q16 dense 시간(ms) |
|---|---:|---:|---|---|---:|
| target_step0_layer0 | 3102.594 | 38.000 | 187/0 | 207/0 | 108.670 |
| reference_prefix_layer0 | 1523.688 | 37.500 | 189/0 | 207/0 | 42.179 |
| masked_text_prefix_layer0 | 182.652 | — | 208/0 | 212/0 | 1.815 |
| target_step12_layer15 | 3102.094 | 37.500 | 187/0 | 207/0 | 102.362 |
| target_step24_layer31 | 3102.094 | — | 187/0 | 207/0 | — |

Peak는 input/미리 만든 mask를 제외한 torch allocator incremental allocation이다. driver scratch·전체 model VRAM이 아니다. token score는 저장하지 않으나 block metadata는 고정 block 크기에서 O(QK)다.

late step24/layer31에서는 두 tile의 모든 streaming 전략이 같은-mask gate 실패했다. finite FP32 계산이지만 재결합/online softmax 경로의 차이를 고정 tolerance 안에 넣지 못했다. FP64 worst-absolute-difference 3개 row는 native/streaming 각각 비교했다. 선택 row의 pass를 full tensor pass로 승격하지 않는다.

## 실제 Boolean mask 증거

실제 .pt의 torch.bool [156,2564] mask에는 false 12,090개, true 387,894개가 있다. Raw bytes SHA256: 6963aa8ff515b246078ad28edb9468a2aa7db6ab0d858b55db4a870381d62d01. 실제 캡처 .pt SHA와 source metadata는 JSON에 있다.

현재 attention_split callable source SHA256: 9cafaafaf93ff53e8cbefb5e4a204014019985df2da8c1996bf40f1235fc2960. 해당 함수가 scores += mask로 처리하므로 이 bool 입력은 0/1 additive bias다. corrected reference는 native와 bit 일치했다. 논리적 causal 제외로 읽었던 이전 third-case는 native max 차이 5.48432를 냈고, Sparge .90/.95 선택 후 visible key가 없는 row가 생겨 dense reference softmax가 NaN이었다. Streaming 출력 자체는 finite이며 empty-row 반환값 0이었다. 원인은 mask 계약 오독과 empty-row 참조 계약 실패이지 FP32 overflow가 아니다.

source/계약 교정은 prefix_bias_offset 옵션에 한정했다. 모델에서는 masked 64회를 원래 callable에 전달한다. 후보의 모델 3-seed 단계는 strict/계산 gate가 미달인 상태의 명시적 품질 진단이며, 자동 적용하지 않는다.
