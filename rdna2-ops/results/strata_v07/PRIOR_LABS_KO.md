# v0.7 Strata 프리필: 기존 랩에서 이어받은 근거

검토: 2026-11-07 KST. 새 GPU 실험 전에 읽은 정본이다.

| 기존 실험 | 확인한 결론 | 이번 처리 |
|---|---|---|
| strata-mmq-typical-lab-20261025/MMQ_TYPICAL_J_GFX1030_2026-10-25.md | 실제 layer 35 IQ3_XXS GU 1.105배, IQ4_NL down 1.066배. 4개 전문가의 counts={1,2,3,52}, resident 연산 1개 조건. 모델 A/B 없음 | 기존 수치를 재실험 성과로 세지 않는다. 실제 프리필 단계와 분포를 먼저 확인하고 범위를 확장 |
| strata-rdna2-gfx1030-recipe/README.ko.md, results/README.md, rejected.md | 현재 IQ3_XXS·K8V4·256K의 fresh 9.7K–119K 약 1,200–1,300 tok/s. RAM pin·resident KV 확대는 page cache를 빼앗아 실패. decode 묶음은 무이득/프리필 손해 | 동일 운영 설정 사본을 baseline으로 사용. 운영 서비스·메모리 정책 변경 없음 |
| strata-lookup16-rdna2-lab-20261025/LOOKUP16_GFX1030_PR4040_20261025.md | 다른 기반 엔진 0.1.48. greedy 한 쌍 +13.2%는 물리 I/O/major fault 차이가 큼. sampled 결과 불완전. 주로 decode lookup | 프리필 범위 밖이므로 반복 안 함 |
| strata_v01403_port/lab-map-20261031/04-layout-lookup16.md | native LDS mode 7/8은 decode T1에서 거의 무이득, T3에서 15–20% 손해. mode2 T3 약 2배 느림. 폴더의 모델 프로파일은 native-LDS 바이너리가 아닌 pinned engine | 이 decode용 LDS 실험을 그대로 되풀이하지 않음. 프리필 MMQ의 코드북/타일은 별도 |
| strata-qsa-k8v4-lab-20261025/QSA_K8V4_RDNA2_2026-10-25.md | QSA attention 연산 및 단계 24–28% 단축, 전체 prefill은 페이지 캐시/I/O에 오염. 계측 자체도 스케줄링을 바꿈 | attention 재포팅 대신 MoE/전송의 실제 비용부터 분해 |
| strata-qsa-k8v4-shipcandidate-20261026/QSA_K8V4_CLEAN_CANDIDATE_20261026.md + recipe/rejected.md | 119K 연산 116.974→88.451 ms와 전체 무이득 구분. 이후 운영 시도에서 GPU memory fault 1회, 철회 | 기각 이력 유지. 이번 후보에 섞지 않음 |

## 소스 확인으로 바로잡을 점 (B)

1. pinned ggml 3cf03257의 gfx1030 MMQ 구성은 `mmq-config-rdna2.cuh`에서 IQ3_XXS/IQ4_NL J={8,16,24,32,40,48,64}이다. 오래된 typical-J 문서의 "J128까지" 추론을 gfx1030의 실제 선택으로 그대로 옮길 수 없다. 실제 선택은 shared-memory 제한까지 포함해 보고한다.
2. `prefill.cpp`의 `STRATA_PREFILL_TIMING`은 같은 compute stream의 인접 HIP 이벤트 간격이다. GPU가 host grouping/copy issue를 기다린 시간도 한 단계에 귀속된다. 독립 kernel duration 합계와 다르다.
3. MMQ 경로의 `dequant` 구간은 `gather_native_group`의 양자화 바이트 복사도 포함한다. IQ3_XXS 코드북 lookup과 dot4는 실제 GU/down MMQ kernel 안에서 수행한다. "dequant가 크다"만으로 IQ 계산 병목이라고 판정할 수 없다.
4. IQ3_XXS는 코드북에서 부호 있는 byte 수준 값을 꺼낸다. 예: grid 0x3e=62. IQ4_NL의 16개 lookup 값은 -127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113이다. 코드 인덱스 4비트를 그대로 dot8에 넣는 것은 같은 계산이 아니다. INT4 diffusion GEMM의 결과를 이 IQ 형식으로 바로 옮기지 않는다.

소스 경로:
- strata-mmq-typical-lab-20261025/src/prefill/prefill.cpp (PfTimer, grouped gather, GU/down 호출)
- strata_v01403_port/build-hip-gfx1030-rocm10-juud/_deps/strata_llamacpp-src/ggml/src/ggml-cuda/mmq.cuh
- 같은 폴더 mmq-config-rdna2.cuh, mmq-load-tiles.cuh
- strata-prefill-v07-lab-20261107/third_party/ggml/ggml-common.h

## 원본 보존과 분리

- 기존 lab 파일은 읽기만 했다.
- 새 worktree: strata-prefill-v07-lab-20261107, HEAD a6fe1183805bc75fa6dfd4aae741a75b187a9cdf + 기존 typical-J 실험 diff의 보존 사본.
- 기존 dirty diff는 inherited_typical_j.patch와 worktree_provenance.json으로 고정했다.
- baseline은 운영 pinned ELF의 byte-identical copy, SHA dd6753203cec325a2a89bc1c9a96d2a847c0bbcf3e08622d9cebb62484977466.
- 실행 중인 운영 서비스에 어떤 signal도 보내지 않는다. 연구 인스턴스만 별도 포트와 별도 로그로 실행한다.
