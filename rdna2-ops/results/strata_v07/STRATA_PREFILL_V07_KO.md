# Strata 프리필 v0.7 — J 타일 범위, 계측 해석, 모델 자원 차단

작성: 2026-11-07 KST. 대상: Radeon RX 6900 XT / gfx1030. 디코드 최적화는 범위 밖이다.

## 1. 판정

**실제 layer 35 가중치를 쓰는 연산 실험은 성공했다. 새로 만든 3개 라우팅 분포, gate/up·down, 9개 타일 선택을 합친 54행이 모두 기준과 출력 바이트가 같았다.** 치우친 분포에서 typical-J 평균 힌트는 gate/up 1.090배, down 1.080배였다. 탐색한 최저 중앙값은 각각 J40의 1.148배, J32의 1.224배였다. 이 최적 J는 모델에 자동 적용한 정책이 아니다.

**모델 전체 성공으로 승격하지 않는다.** 운영 pinned ELF의 첫 fresh 요청은 실행했지만 스토리지 읽기·swap이 컸다. 이후 도입한 운영 보호 조건 아래에서 연구 바이너리의 priming 요청이 두 번 중단됐다. 9.7K·19.8K·119K의 같은 연구 바이너리 J off/on 출력·처리량 A/B, warm 단계 분해, 운영 pinned ELF와 새 빌드 사이의 출력 bridge는 미완료다. 이는 자원 보호에 따른 차단이며, J 후보의 속도 실패나 GPU 커널 오류가 아니다.

Claude의 “warm fresh 1,200–1,300 tok/s에서 GPU 계산이 주병목” 가설은 **이번 자료로 확인도 반증도 하지 못했다.** 첫 요청의 199.2 tok/s를 기존 warm 수치의 반증에 사용하지 않는다.

| 항목 | 결과 | 근거 등급 / 범위 |
|---|---|---|
| layer 35 IQ3_XXS GU + IQ4_NL down, 3개 새 분포 | 54행 byte exact, Q8_1 오차 관문 통과 | A / 실제 저장 가중치 + 합성 FP32 입력·분포 |
| 치우친 분포의 평균 J 힌트 | GU 1.090배, down 1.080배 | A / 연산만 |
| medium·large 분포의 평균 J 힌트 | 모두 실제 J64로 동일, 가속 없음 | A / 같은 커널 음성 대조 |
| 직접 dot8로 IQ 인덱스 곱하기 | 표현이 다르므로 이식 불가 | B / 코드북과 ISA, dot8 GPU 실험 결과는 아님 |
| warm 모델 병목, 3길이 J off/on | 미측정 / 자원 차단 | A / 보호 중단 원자료 |
| 운영 반영 | 하지 않음 | 전후 SHA 별도 감사 |

사전 정확도·속도 기준은 [PROTOCOL_KO.md](PROTOCOL_KO.md), 이전 랩 검토는 [PRIOR_LABS_KO.md](PRIOR_LABS_KO.md)에 있다. 실패 파일과 중단 원본을 지우지 않았다.

## 2. 기준 경로와 소스 계보

운영 서비스는 `strata-verified-256k.service`다. 서비스 조작 없이 inactive/dead, MainPID 0을 확인한 상태에서 포트 18477의 연구 인스턴스만 시작했다. 실행 중에도 5초마다 운영 상태를 읽고, 운영이 켜지면 연구 소유 process group만 종료하도록 했다.

| 구분 | 고정된 정체성 |
|---|---|
| 운영 pinned ELF와 바이트 동일 연구 사본 | SHA-256 `dd6753203cec325a2a89bc1c9a96d2a847c0bbcf3e08622d9cebb62484977466` |
| 별도 worktree | `~/DevSpace/strata-prefill-v07-lab-20261107` |
| 연구 base HEAD | `a6fe1183805bc75fa6dfd4aae741a75b187a9cdf` |
| 연구 엔진, J off/on 공통 실행 파일 | SHA-256 `31ecea0716fd7cba638d004b2fc7987efa1b179f9a9cf67ca96f8d42b4c263b5` |
| 새 연산 하네스 ELF | SHA-256 `6879df8c5daf458e826d8a895cf81b2ffe935067797d1d11bdf5d5236ef0d1f8` |
| 읽기 전용으로 재사용한 ggml 소스 | revision `3cf03257f219afbe7334045ff7c6a06ac68c627d` |
| 연구 소스 전체 차이 | [v07_research_source.patch](v07_research_source.patch), SHA-256 `f7df27a2cfb8443f64ff578b554b759076c84dd4d756448d12b2499ffadccd3d` |

새 worktree는 typical-J 랩과 같은 base에서 시작해 그 랩의 미커밋 변경을 보존했다. 현재 운영 checkout의 HEAD도 같지만, 이것만으로 역사적 pinned ELF를 재현한다고 주장할 수 없다. pinned ELF에는 별도 미커밋 진단 패치가 들어간 이력이 있다. 따라서 **성능 본비교는 같은 연구 ELF의 J off/on**으로 설계했고, pinned→연구 ELF 비교는 별도 bridge로 나눴다. 이번에는 자원 차단 때문에 bridge와 본비교 모두 끝내지 못했다.

빌드는 ROCm 10의 기존 컴파일러와 기존 ggml 소스·의존성 cache만 썼다. `CARGO_NET_OFFLINE=true`, `FETCHCONTENT_FULLY_DISCONNECTED=ON`, `FETCHCONTENT_UPDATES_DISCONNECTED=ON`, `nice 12`, `-j2`를 사용했다. 모델·패키지 다운로드 및 환경 재구축은 하지 않았다. 빌드 중 GPU 문맥을 만들지 않았고, 성능 측정 때 컴파일러는 종료된 상태였다.

증거: [worktree_provenance.json](worktree_provenance.json), [source_and_pack_provenance.json](source_and_pack_provenance.json), [build_record.json](build_record.json), [build_incremental_record.json](build_incremental_record.json), [source_patch_manifest.json](source_patch_manifest.json).

## 3. pack 이름과 실제 연산 타입을 구분한다

설치 pack 이름은 SC117-IQ3_XXS지만 모든 층이 IQ3_XXS는 아니다. 실제 pack index의 48개 층을 읽은 결과다.

| 역할 | 실제 타입 | 타입 번호 | 층 수 |
|---|---|---:|---:|
| gate/up | IQ2_XXS | 16 | 9 |
| gate/up | IQ2_XS | 17 | 10 |
| gate/up | IQ2_S | 22 | 10 |
| gate/up | IQ3_S | 21 | 13 |
| gate/up | IQ3_XXS | 18 | **6** |
| down | IQ4_NL | 20 | 18 |
| down | Q2_0 | **42** | 30 |

타입 42의 이름은 빌드에 사용한 `strata_v01403_port/build-hip-gfx1030-rocm10-juud/_deps/strata_llamacpp-src/ggml/include/ggml.h`의 `GGML_TYPE_Q2_0 = 42`로 확인했다. 타입 18·20도 같은 enum과 대조했다. index의 층별 offset·blob 크기·타입 원자료는 [source_and_pack_provenance.json](source_and_pack_provenance.json)에 있다.

연산 실험은 layer 35, experts 0–15의 서로 다른 실제 저장 바이트를 읽는다. 이 층의 offset은 29,792,665,600 B, expert당 2,176,000 B, GU 타입 18, down 타입 20이다. 따라서 이 연산의 가속을 전체 48개 층이나 모델 처리량에 그대로 곱하면 안 된다.

IQ3_XXS의 작은 코드값은 signed INT4 수치가 아니라 코드북 인덱스다. lookup 후 int8 값·부호·스케일을 적용한다. IQ4_NL도 비선형 값의 표를 쓴다. 코드북 인덱스를 바로 `v_dot8_i32_i4`에 넣는 것은 같은 곱셈이 아니다. 동일값을 여러 nibble 곱과 교차항으로 재구성하는 후보는 별도 설계와 측정이 필요하며 이번에 실행하지 않았다.

## 4. 실제 커널·J 선택과 새 연산 실험

### 4.1 ISA와 리소스

현재 gfx1030 IQ3/IQ4 설정의 유효한 일반 J는 8/16/24/32/40/48/64다. 예전 랩 문서의 “큰 프롬프트에서 J128” 표현을 이 빌드에 적용하면 틀린다. J56 및 J72 이상 심벌에는 `no_device_code` stub가 있어서 유효한 타일로 세지 않았다.

컴파일한 IQ3_XXS 코드에 `v_dot4c_i32_i8`가 실제로 존재한다. 유효 커널에서 dot8은 없다. 이는 컴파일 산출물 근거 B이며, 연산 실행 결과 A와 따로 표시한다. 정적 opcode 개수는 동적 실행 횟수나 활용률이 아니다.

| 일반 커널 J | VGPR | SGPR | private segment | wave 크기 |
|---:|---:|---:|---:|---:|
| 8 | 191 | 52 | 0 B | 32 |
| 16 | 183 | 38 | 0 B | 32 |
| 24 | 187 | 38 | 0 B | 32 |
| 32 | 183 | 38 | 0 B | 32 |
| 40 | 187 | 38 | 0 B | 32 |
| 48 | 256 | 42 | 16 B | 32 |
| 64 | 233 | 38 | 0 B | 32 |

J48의 private segment와 높은 VGPR은 타일마다 비용이 같지 않다는 단서다. 이것만으로 정확한 occupancy나 동적 spill 병목을 계산하지 않았다. `group_segment_fixed_size=0`도 동적 LDS가 없다는 뜻이 아니다.

원자료: iq3_gfx1030.asm (`iq3_gfx1030.asm`, 연구 폴더에만 있음), [iq3_gfx1030_notes.txt](iq3_gfx1030_notes.txt), [iq3_isa_manifest.json](iq3_isa_manifest.json), [iq3_isa_summary.json](iq3_isa_summary.json), [iq3_registers.json](iq3_registers.json).

### 4.2 최소 3개 분포와 정확도

기준은 각 expert 그룹의 최대 행 수로 J를 고르는 기존 경로다. typical-J는 `ceil(total_rows/experts)`를 타일 선택 힌트로 주며, grid coverage는 기존 최대 행 수를 유지한다. 다른 수학식이나 Q8_1 반올림을 도입하지 않는다.

각 분포는 16 expert를 쓴다. 이 분포들은 새 screening용 합성 라우팅 분포이며, 실제 모델에서 캡처한 라우팅 히스토그램이라고 표시하지 않는다.

| 분포 | expert별 행 수 | 총 행 / 평균 올림 / 최대 |
|---|---|---:|
| skew_tail | 1, 1, 2, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 64, 512 | 702 / 44 / 512 |
| prefill_medium | 61, 80, 96, 112, 128, 140, 160, 180, 200, 220, 240, 256, 300, 320, 384, 512 | 3,389 / 212 / 512 |
| prefill_large | 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 640, 768, 1,024 | 6,592 / 412 / 1,024 |

GU의 M=1,280, K=2,560, down의 M=2,560, K=640이다. 출력 FP32 바이트 전체를 비교하고 finite·sentinel 잔류·zero row도 확인했다. 같은 저장 가중치와 원래 FP32 활성화를 CPU FP64 matmul로 계산한 sampled reference는 각 후보에서 816개 출력 값을 확인했다. 전체 출력의 후보 간 byte 비교와 sampled FP64 참조를 혼동하지 않는다.

54행 전부 byte exact, zero row 오차 0이었다. sampled relative L2 최댓값은 0.002893044, max_abs/ref_rms 최댓값은 0.012302054였다. 사전 Q8_1 관문인 0.04 / 0.35를 통과했다. 이 오차는 원래 FP32 활성화를 Q8_1로 바꾸는 기존 경로의 계산 오차이며, 새로운 모델 양자화 손실을 측정한 것이 아니다.

### 4.3 속도

후보마다 prewarm 3회, 순서를 교대한 7라운드, 라운드당 20회 실행의 중앙값이다. GPU는 독점 `GpuAudit` 아래에서 실행했다.

| 분포 / 역할 | 기준 J64 | 평균 힌트 실제 J / 시간 | 평균 힌트 비율 | 탐색 최저 J / 시간 / 비율 |
|---|---:|---:|---:|---:|
| skew_tail GU | 440.676 μs | J48 / 404.290 μs | **1.090배** | J40 / 383.902 μs / 1.148배 |
| skew_tail down | 197.342 μs | J48 / 182.684 μs | **1.080배** | J32 / 161.284 μs / 1.224배 |
| medium GU | 858.253 μs | J64 / 852.647 μs | 같은 커널 | J64 |
| medium down | 424.496 μs | J64 / 414.185 μs | 같은 커널 | J64 |
| large GU | 1,450.748 μs | J64 / 1,431.574 μs | 같은 커널 | J64 |
| large down | 740.747 μs | J64 / 718.611 μs | 같은 커널 | J64 |

medium·large에서 “max”와 “mean”의 실제 J가 같으므로 프로그램이 같은 커널을 두 번 잰 것이다. large down의 1.0308이라는 비율은 **같은 커널의 최대 3.1% 흔들림**이지 개선이 아니다. 미리 정한 1.03배 기준만 기계적으로 적용하면 이 음성 대조를 성과로 오판할 수 있다. 경로 hit와 출력·소스 확인이 함께 필요하다.

실용 규칙은 “평균으로 무조건 선택”이 아니라 **expert 행 수 분포와 실제 선택 J를 기록하고, 선택이 바뀌는 범위만 실측한다**다. skew_tail의 최적 J40/J32도 하나의 고정 모델 정책으로 일반화하지 않았다.

원자료: [operator_screen/operator.jsonl](operator_screen/operator.jsonl), [operator_screen/result.json](operator_screen/result.json), [operator_summary.json](operator_summary.json). raw JSONL SHA-256: `bfd7f6e855c7c48c17a861ec553b7c9b67d0da8e0a21a996a095c131e6286f5b`.

## 5. 모델 단계 분해 — 첫 요청에서 확인한 것과 확인 못 한 것

### 5.1 pinned ELF 첫 fresh 요청

운영 ELF와 바이트 동일인 연구 사본으로 실행했다. server ready 10.012 s, actual prompt 9,716 tokens, read 9,716 / reused 0, 생성 8 tokens였다. native prompt 48.774 s, 199.2 tok/s, API 전체 49.141 s다. 이는 **동일 엔진 내 priming 없는 첫 요청**이며, 파일 cache를 통제한 cold disk 실험은 아니다.

가장 큰 9,711-token chunk는 다음과 같다.

| GPU event 구간 | 시간 | 이 chunk timeline 비중 |
|---|---:|---:|
| dequant라는 이름의 구간 | 29,209 ms | 74.92% |
| gemm down | 2,143 ms | 5.50% |
| gemm gate/up | 1,338 ms | 3.43% |
| gdn | 1,115 ms | 2.86% |
| qsa attention | 1,019 ms | 2.61% |
| hc read | 921 ms | 2.36% |
| router + shared | 754 ms | 1.93% |
| qsa projection | 666 ms | 1.71% |
| wait copy | 476 ms | 1.22% |
| 그 밖의 구간 | 원자료 참조 | 원자료 참조 |

timeline 38,989 ms, wall 39,026 ms, host staging 누계 38,758 ms다. 마지막 4-token chunk의 timeline은 1,059 ms, wall 1,076 ms다. GPU event 구간 합에는 host가 다음 kernel을 enqueue하지 못해 생기는 idle gap도 들어간다.

**dequant 74.92%를 IQ lookup GPU 계산이라고 해석하면 틀린다.** 현재 MMQ 경로에서 그 구간에는 GGUF 블록 group gather, copy 발행 대기 `wait_issued` 등이 포함되고, 실제 IQ 코드북 lookup·dot4는 GU/down 커널 내부에 있다. 이 첫 요청에서는 메모리·스토리지 지연이 그 구간에 잡힐 수 있다.

timeline, host staging, wall을 더하지 않는다. host staging은 overlap되며 누계다. `src/prefill/prefill.cpp`의 `PfTimer::fold()`는 연속 event i→i+1 시간을 오직 `ms[ph[i]]`에 더한다. `kPfGdn`은 qkv/gate/alpha/beta **입력 projection**이고, 그 뒤 `kPfGdnConv` → `kPfGdnRec` → `kPfGdnOut`으로 mark가 바뀐다. 따라서 gdn 1,115 ms와 conv/gates 80 ms, recurrence 330 ms, out projection 363 ms는 배타 구간이며, GDN 전체 묶음은 1,888 ms다. 중복 부모·자식 이벤트처럼 빼지 않는다. main chunk의 printed phase 합은 38,989 ms로 timeline과 같고, tail의 1,061 ms 대 1,059 ms 차이는 개별 정수 출력의 반올림 오차다. 소스 위치는 PfTimer 1916–1954행, GDN mark 2584–2599행이며 파일 SHA는 `82542fffe384a2d9e1363955fb2c1501a2787c5fe823022cf12640ad6c5dbec8`이다.

### 5.2 물리 I/O의 정확한 범위

시작 시 PCIe probe는 28.5 GB/s였다. 이것만으로 workload가 PCIe 천장에 근접했는지 알 수 없다. 이 pinned 실행에는 실제 expert H2D 발행 bytes counter가 없어서 요약의 `issued_h2d_bytes`는 **null**이다. bytes/PCIe bandwidth로 프리필 전송 한계를 정량 계산할 수 없는 표본이다.

기존 native 로그의 “OS read 47,202.9 MB”는 첫 요청에서 **engine 프로세스 시작 → 첫 요청 완료** 누계다. `generate.cpp`의 static `ProcIo io_prev`가 0에서 시작하고 요청 끝에만 갱신되기 때문이다. 여기에는 기동·프리필·8-token 생성이 들어간다. prefill만의 I/O로 쓰지 않는다.

요청 도중 관찰한 native PID는 OS read 약 47.0 GB, D 상태, `folio_wait_bit_common`을 보였다. 요청 전후 global swap-out은 1,497,594 pages 증가했다. 4,096 B/page 기준 **6,134,145,024 B = 6.134 GB = 5.713 GiB**다. 첫 요청에 파일 읽기와 메모리 회수 비용이 컸다는 근거다. 이 자료는 warm GPU 계산 병목 가설을 반증하지 않는다.

후속 하네스는 `STRATA_PHASE_IO=1`의 실제 prefill/decode 경계, HTTP 요청 경계의 native `/proc/io`, 추가 H2D expert-copy bytes를 구분해 저장하도록 구현했다. 그러나 warm 요청이 보호 관문으로 끝나지 않아 새 경계의 완료값을 얻지는 못했다. 누락값을 0 B로 채우지 않았다.

원자료: [pinned_stage_9k_first_retry/result.json](pinned_stage_9k_first_retry/result.json), pinned_stage_9k_first_retry/engine.log (`pinned_stage_9k_first_retry/engine.log`, 연구 폴더에만 있음), [pinned_stage_9k_first_retry/proc_io_observation.jsonl](pinned_stage_9k_first_retry/proc_io_observation.jsonl), [pinned_stage_first_summary.json](pinned_stage_first_summary.json).

## 6. 모델 A/B가 멈춘 이유

첫 요청의 swap 관측 뒤, 정확도·속도 기준을 바꾸지 않고 후속 실행의 운영 보호 조건을 추가했다. 연구 프로세스 시작 대비 추가 swap-out이 256 MiB를 넘고 1초 간격 두 표본에서 계속 늘면 연구 소유 process group만 종료한다. 운영 서비스·다른 앱·swap 설정·OS cache·GPU clock은 건드리지 않는다.

| 실행 | 상태 | 추가 swap-out / 소유자 정리 |
|---|---|---|
| `research_stage_9k_primed` | priming 중 36.2 s에 보호 종료 | 관문 표본 608,624,640 B = 580.43 MiB, 남은 연구 PID 없음 |
| `research_stage_9k_primed_retry` | priming 중 18.7 s에 보호 종료 | 288,772,096 B = 275.39 MiB, 남은 연구 PID 없음 |

두 번째 시도는 첫 종료 뒤 CPU read-only `mincore`로 expert 파일의 92.78%가 이미 page cache에 있음을 확인해 상태 변화 근거가 생겼을 때 한 번만 허용했다. 기준은 그대로 유지했다. 재차 관문에 걸려 더 반복하지 않았다.

중단 뒤 첫 snapshot에서는 actual experts.bin 42.913 GB 중 39.814 GB가 resident였다. MemAvailable 47.49 GB 중 대부분은 회수 가능한 file cache였고, 물리 free는 1.91 GB, 익명 페이지는 12.39 GB였다. **MemAvailable이 커도 큰 모델 cache를 추가로 채울 때 다른 앱의 anon이 swap될 수 있다.** 이 snapshot은 원인과 일관되는 관측이며, 전역 swap 증가 전부의 소유 프로세스를 확정한 추적은 아니다.

두 번째 관문 순간 native engine RSS 약 20.00 GB 중 file 18.71 GB, anon 0.607 GB, shared 0.679 GB였다. native process 자체 swap은 0 B였다. 거대한 연구 anon 버퍼 누수라는 가설보다 file cache를 채우는 동안의 시스템 회수 압력과 일관된다. 단순히 swap 한도를 늘려 실험을 통과시키지 않았다.

| 원래 모델 검증 칸 | 실제 상태 |
|---|---|
| 9.7K warm fresh J off/on tok/s·32 token 일치 | 자원 차단, 미측정 |
| 19.8K warm fresh J off/on tok/s·32 token 일치 | 선행 관문 차단으로 미실행 |
| 119K warm fresh J off/on tok/s·32 token 일치 | 선행 관문 차단으로 미실행 |
| pinned→연구 ELF 출력 bridge | 미완료 |
| 실제 모델 라우팅 분포에 대한 J hit 비율 | 미측정 |
| warm 계산 vs PCIe 병목 판정 | 미판정 |

모든 실행에서 관측된 외부 KFD client 겹침은 없었다. 첫 성공 요청 guard 245개 표본, 두 보호 종료는 각각 148개·78개 표본이며, 연구 엔진·vision child가 남지 않았음을 확인했다. `GpuAudit`은 graphics load나 polling 사이의 아주 짧은 외부 실행까지 증명하는 도구는 아니다.

원자료: [research_stage_9k_primed/result.json](research_stage_9k_primed/result.json), [research_stage_9k_primed_retry/result.json](research_stage_9k_primed_retry/result.json), [resource_after_abort.json](resource_after_abort.json). 첫 시도의 top_k=0 config 검증 실패도 `pinned_stage_9k_first`에 남아 있으며 GPU 엔진 기동 전 오류다. top_k=20으로 고친 뒤 원본을 덮어쓰지 않고 새 결과 폴더에 재실행했다.

## 7. 반복하지 않은 후보

| 후보 / 이전 랩 | 이번 처리 | 이유 |
|---|---|---|
| 기존 typical-J 4-expert 한 분포 | 새 16-expert 3개 분포로 범위 확대 | 기존 1.105배 / 1.066배를 새 성과로 재계상하지 않음 |
| native LDS layout mode 8/7/2 | 재실행하지 않음 | 기존 연산 역효과·불일치한 모델 profile 계보가 정리돼 있음 |
| lookup16 / PR4040 | 프리필 성과로 옮기지 않음 | 다른 v0.1.48 계보, decode 중심, 품질·copy 조건 불충분 |
| QSA k8v4 attention 개선 | 재실행하지 않음 | 이미 연산 24–28% 개선과 모델 무이득·첫 요청 fault 철회 기록 존재 |
| 일반 INT4 dot8을 IQ3 코드에 그대로 적용 | 소스 분석에서 제외 | 코드북 인덱스와 signed INT4 값이 다름 |
| 모든 expert가 IQ3_XXS라는 전제 | 기각 | 실제 GU IQ3_XXS는 6/48층 |
| medium·large의 1.03배를 J 성과로 채택 | 기각 | 실제로 같은 J64 kernel |

각 기존 랩의 결론·경로는 [PRIOR_LABS_KO.md](PRIOR_LABS_KO.md)에 정리했다. 후보를 실행하지 않은 경우 B 또는 이전 근거로 표시했으며 새 A 실패로 꾸미지 않았다.

## 8. 재현 명령과 재실행 조건

아래 명령은 기존 설치와 이번 연구 worktree/build를 사용한다. 서버는 직접 포트 18477에서 연구용으로만 뜬다. 운영 서비스를 꺼 주는 명령은 포함하지 않았다.

### 8.1 Claude가 재현할 핵심 연산 숫자 — 54행, 약 10초

다른 GPU 작업이 없는 독점 slot에서 실행한다. 출력 폴더는 새 이름이어야 한다.

~~~bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
/usr/bin/python3 -u strata_operator_v07.py \
  --out results/strata_v07/reproduce_operator_01
~~~

기대 확인: `operator.jsonl`의 54개 operator 행 모두 `bit_exact=true`, 최종 `pass=true`; skew_tail에서 평균 힌트 GU 약 1.09배 / down 약 1.08배. J40/J32 탐색값은 약 1.15배 / 1.22배지만 모델 채택값이 아니다. 동일 J64인 medium·large는 가속 판정에서 제외한다. 실제 각 라운드 원값을 비교하고 가장 좋은 한 번만 고르지 않는다.

### 8.2 단계 하네스 — 자원 조건이 충족될 때만

조건: 운영 service가 이미 inactive/dead, 다른 ComfyUI·모델 프로세스와 GPU·대용량 RAM 작업이 없음, 별도 port 18477이 비어 있음, 같은 256 MiB swap 보호 조건 유지. 단순 MemAvailable 통과만으로 warm 측정 가능하다고 보지 않는다. 이번에 두 번 실패한 조건이 그대로라면 반복하지 않는다.

~~~bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
/usr/bin/python3 -u strata_prefill_v07.py \
  --exe ~/DevSpace/strata-prefill-v07-lab-20261107/build-v07-gfx1030/strata \
  --out results/strata_v07/reproduce_stage_01 \
  --prime-rows 390 --rows 390 --repeats 1 --max-new 8 \
  --timing --port 18477 --env STRATA_MMQ_TYPICAL_J=0 \
  --arm research-joff-stage
~~~

첫 요청은 priming이며, 두 번째 요청의 실제 native `reused=0`을 확인한 뒤 warm fresh로 분류한다. `resource_valid_for_timing=false` 또는 guard 불통과·소유자 잔류이면 성능 표에 넣지 않는다. raw log, 요청 nonce·SHA, 실제 prompt token 수, generated token IDs, 물리 I/O·swap 구간을 보존한다.

CPU 요약 명령:

~~~bash
/usr/bin/python3 strata_summarize_v07.py \
  results/strata_v07/reproduce_stage_01 \
  --out results/strata_v07/reproduce_stage_summary_01.json
~~~

향후 모델 J off/on은 같은 연구 ELF에서 `STRATA_MMQ_TYPICAL_J=0/1`만 바꾸고, 390/800/4800행 프롬프트의 실제 토큰 수를 확인한다. 주 처리량 비교는 `--timing`을 끈 상태에서, 별도 priming 뒤 같은 측정 nonce와 temperature 0, 32 생성 token, 서로 교대한 반복으로 한다. 현재까지 이 3길이 검증을 완료한 것처럼 쓰지 않는다.

## 9. 운영 보호 감사

사전 SHA는 [operating_before.json](operating_before.json)에 있다. 대상은 unit/drop-ins/launcher/config/실제 pinned ELF/feasibility packs/actual pack 등 48개 파일, 합계 50,673,888,720 B다. 운영 파일을 수정하는 코드는 실행하지 않았다. 새 하네스는 Python bytecode 쓰기도 꺼서 운영 경로의 __pycache__ 갱신을 방지한다.

모든 GPU 실험과 연구 서버 정리가 끝난 뒤 CPU I/O slot에서 [operating_after.json](operating_after.json)을 작성했다. **48/48개 SHA-256이 사전과 동일하며, added/missing/changed는 모두 0, comparison.pass=true**다. 사후 파일 SHA-256은 `443be60a4946b745a7d15bbfe5bc0af6b3f3680362166d5e7c6a124850cff64f`다. 감사 전후 `strata-verified-256k`는 inactive/dead, MainPID 0이었다. 연구가 서비스를 시작·중지·재시작한 적은 없다. 이는 명시한 48개 파일의 보존 확인이며 OS page cache·전체 파일 시스템 불변 주장은 아니다.

같은 범위를 다시 확인할 명령:

~~~bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
/usr/bin/python3 -u strata_audit_v07.py \
  --out results/strata_v07/operating_after.json \
  --compare results/strata_v07/operating_before.json
~~~

## 10. 카탈로그에 넣을 적용 판단

새 MoE 프리필에서 IQ라는 모델 이름만 보고 kernel을 골라서는 안 된다. 실제 pack layer type histogram, expert별 행 수 분포, 선택된 J를 먼저 기록한다. J 변경은 기존 Q8_1 산술을 유지하면서 좁은 분포에서 byte exact 이득을 얻을 수 있는 후보지만, 큰 fresh 프롬프트에 자동 이득이 생기는 것은 아니다.

단계 로그의 dequant 이름보다 실제 event 경계와 copy·host wait 위치가 우선이다. OS I/O 첫 누계와 요청 구간, 첫 요청과 같은 process의 priming 뒤 fresh 요청, 연산 성공과 모델 성공을 각각 분리한다. 자원 보호에 막히면 관문을 완화해 숫자를 채우지 않고 미검증 칸을 남긴다.
