# Strata 프리필 v0.7 사전 판정 기준

작성: 2026-11-07 KST. 새 GPU 실행 전에 확정. 디코드 최적화는 범위 밖이다.

## 보호와 실행 조건

- 운영 서비스 strata-verified-256k.service는 상태 조회만 한다. 시작·중지·재시작하지 않는다.
- 모델 실행 직전과 각 팔 사이에 inactive/dead, MainPID 0을 다시 확인한다.
- GPU는 root의 독점 허가와 gpu_guard_v06.GpuAudit(공통 flock) 아래에서만 쓴다.
- 다른 ComfyUI·Strata GPU 프로세스가 있으면 시작하지 않는다. 발견한 겹침은 그 측정 전체를 무효로 한다.
- 새 worktree strata-prefill-v07-lab-20261107, build-v07-gfx1030에서만 실험한다.
- 현재 운영 pinned ELF의 바이트 동일 사본을 연구 기준선으로 쓴다. 모델 파일은 읽기 전용이다.
- 운영 SHA는 unit/drop-ins/launcher/config/engine/feasibility packs/실제 IQ3_XXS pack 48개를 전후 비교한다.
- 최초 보호 해시를 위해 약 50.67 GB를 읽었다. 이후 측정을 cold disk라고 부르지 않는다. 첫 요청과 warm, process start와 파일 캐시 상태를 각각 기록한다.

## 1. 병목 가설 검증

- 같은 운영 설정 사본에 STRATA_PREFILL_TIMING=1을 연구 환경에서만 켠다.
- 9.7K·19.8K·119K 범위의 합성 프롬프트를 쓰며 실제 read/reused 토큰 수를 로그로 기록한다.
- fresh는 reused=0이다. 재사용이 생긴 표본을 fresh 처리량에 포함하지 않는다.
- 단계 GPU timeline의 합과 host staging, host setup, sync, 전체 prefill wall은 서로 겹친다. 모두 더하지 않는다.
- 특히 기존 dequant 라벨은 MMQ 경로에서 실제 dequant만이 아니라 GGUF 블록 복사와 host issue 대기를 포함한다. gemm gate/up/down 내부에 IQ 코드북 lookup 및 dot4가 있다.
- GPU 계산이 주병목이라는 판정에는 (a) warm 구간의 GU/down 비중, (b) copy 대기·호스트 staging·물리 I/O, (c) 실제 bytes/시간에 따른 전송 가능량을 함께 쓴다. PCIe 이론 대역폭만으로 단정하지 않는다.
- 파일 I/O, major faults, swap I/O, memory PSI를 팔별로 기록한다. 스토리지/메모리 차이가 결과를 설명하면 순수 커널 가속 근거로 채택하지 않는다.
- 계측한 실행의 tok/s는 계측 없는 실행과 별도로 둔다.

## 2. 연산 단위

- 실제 설치 pack의 실제 IQ3_XXS gate/up 및 IQ4_NL down 바이트를 쓴다. layer 35가 출발점이다. 다른 타입은 이름과 층을 구분한다.
- 기존 typical-J 한 개 치우친 그룹 결과는 다시 성과로 재지 않는다. 모델에서 관찰한 분포 또는 세 가지 이상 새로운 크기/치우침으로 범위를 확인한다.
- J 타일 선택만 바꾸는 후보는 기준과 출력의 uint8 바이트 전부 일치해야 한다. finite와 모든 출력 쓰기도 확인한다.
- 독립 FP64 곱은 같은 저장 가중치와 원래 FP32 활성화를 쓴다. Q8_1 활성화 반올림 오차 허용치는 기존 관문 rel_l2 ≤ 0.04, max_abs/ref_rms ≤ 0.35, zero row ≤ max(1e-5, 1e-4·ref_rms)이다.
- 속도는 같은 resident input/output 상태에서 warmup 후 A/B 순서를 교대하고 중앙값 및 개별값을 보존한다. 최솟값 한 개를 대표값으로 쓰지 않는다.
- >= 1.03배와 반복간 일관성이 있어야 속도 후보로 잡는다. 3% 안쪽은 실익 미확정으로 둔다.

## 3. 모델 A/B

- 같은 프롬프트, 모델 pack, KV/컨텍스트/라우팅/샘플러, temperature 0.
- native 로그의 fresh prefill tok/s를 사용한다. API wall·부팅·짧은 생성 시간은 별도 항목이다.
- 9.7K·19.8K·119K 각 길이에서 기존/후보 출력 token ID 시퀀스가 같아야 정확도 통과다. 실제 token ID를 얻을 수 없으면 텍스트 SHA 일치만 별도 표시하고 token 일치로 대체하지 않는다.
- 채택 가능은 실제 변경 경로 hit와 모델 출력 일치, 그리고 paired median >= 1.03배. 계측/페이지 캐시/순서에 오염되면 모델 이득 미확정이다.
- 작은 연산 이득이 모델에 안 옮겨가도 실패 원인과 적용 범위를 기록한다.
- chunk cap 후보는 chunk 경계의 연산 반올림이 달라질 수 있다. 출력 불일치면 속도와 별개로 strict 실패다.
- 운영 반영은 하지 않는다. 연구 후보와 실패 결과 모두 원자료와 재현 명령을 남긴다.

## 4. 첫 요청 swap 관측 후 추가한 운영 보호 중단 조건

도입 시점: `pinned_stage_9k_first_retry` 완료 후, 2026-11-07 KST. 최초 요청에서 global `pswpout`이 584,870 → 2,082,464 페이지로 늘어난 것을 보고 후속 실행부터 보강했다. 위의 사전 정확도·속도 기준을 소급 변경하지 않는다.

- 이후 연구 프로세스 시작 전 `pswpout`을 기준으로 추가 swap-out이 256 MiB를 넘고, 1초 간격 두 표본에서 계속 증가하면 연구 소유 process group만 종료한다.
- 각 요청 시작 대비 증가량도 함께 기록한다. 프로세스 전체 기준이 더 엄격한 중단 조건이다.
- 이 경우 `resource-contaminated`로 분류하고 모델 처리량·개선율 표본으로 채택하지 않는다. 중단 원본과 관측값은 보존한다.
- 운영 서비스, swap 설정, OS cache, GPU clock을 바꾸지 않는다. 이미 swap된 다른 앱을 강제로 unswap하지 않는다.
- 운영 서비스는 실행 중에도 5초마다 읽기 전용으로 확인한다. 사용자가 켜면 연구 인스턴스만 종료하고 해당 측정을 무효로 한다.
- warm은 같은 엔진에서 별도의 priming 요청 후 nonce를 앞에서 바꾼 fresh 요청으로 측정한다. 측정 팔끼리의 프롬프트는 동일하며 native `reused=0`을 확인한다.
