# rdna2-ops v0.7 검수·재현 기록 (2026-11-07)

GPT v0.7 보고(카탈로그 v0.3, 가이드 §9.16–9.19)가 제시한 "Claude 재현용 핵심 숫자 5개"를 이 PC에서 다시 실행했다. 수행자는 Claude다. 산출물은 `rdna2-ops/results/claude_repro_20261107/`에 있다. 모든 실행에서 `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0`, `MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53=0`, `OMP_NUM_THREADS=2`를 지정했다.

## 1. 재현 결과

| # | 항목 | GPT 값 | Claude 재현 | 판정 |
|---|---|---|---|---|
| ① | INT4 G128 sub-K (Qwen proj, 대칭) | 7.165 → 3.525 ms (2.032배), native 대비 0.663배 | 7.245 → 3.584 ms (**2.02배**), native 대비 **0.653배**. 이전 W4 출력과 바이트 일치, `n_spills` 18 | 재현 |
| ② | MOSS 같은 W4: reference ↔ dot4 | latent·RGB 바이트 일치 | **바이트 일치** | 재현 |
| ② | MOSS INT8 → W4 경로 | RGB max 240.370/255, RMSE 45.288/255 | max 241.4/255, RMSE **54.78/255** | 엄격 실패는 재현. 크기는 다름(§2.1) |
| ③ | Sparse r4 연산 (Q2400/K4964) | 22.666 → 18.600 ms (1.219배), 추가 peak 3,102.594 → 38.000 MiB | 22.563 → 18.549 ms (**1.216배**), 3,102.6 → **38.0 MiB** | 재현 |
| ④ | Qwen 25단계 warm denoise | alpha 50.8745 s → r4 46.3941 s (1.09657배), RGB 243.441/255, RMSE 49.800/255 | alpha 50.766/50.790 s → r4 46.331/46.343 s (**1.096배**), RGB **243.441/255, RMSE 49.800/255** | 재현. conditioning·native·r4 이미지 SHA가 GPT 실행과 모두 같음 |
| ⑤ | Strata MMQ typical-J 연산 | 54행 바이트 일치, skew GU 1.090배, down 1.080배 | 54행 **바이트 일치**, skew GU **1.051배**, down **1.098배**. J가 바뀌지 않는 큰·중간 분포는 1.003–1.027배 | 재현(같은 대역, 반복 변동 약 ±4%) |

r4 이미지는 GPT 판정과 같이 인물이 반복된다(같은 인물이 3명씩 두 쌍). Claude 육안으로도 실패가 분명하다. 다만 이 육안은 판정자 수에 넣지 않는다.

## 2. 새로 확인한 사실

### 2.1 MOSS는 프로세스마다 conditioning이 달라지고, 그 차이만으로 최종 RGB가 크게 갈라진다

②를 재현하다가 INT8 native 이미지 SHA가 GPT 실행과 다르다는 것을 발견했다. spec은 출력 폴더와 포트 말고는 같았다(환경변수 포함).

| 실행 | positive conditioning SHA | latent SHA | 이미지 SHA |
|---|---|---|---|
| GPT `int4_v07/moss_model01` | `55a7df454f` | `a36c9b2b5e` | `62dcd98312` |
| Claude `moss_int4_model` | `d862db1c39` | `64c8e2f175` | `ce597c0467` |
| Claude `moss_int8_only_b` | `2207cd7eb1` | `ce9f7612ad` | `d7f609689b` |
| 참고: v0.5 `v05_moss_int8_tuned_ab` | `251d19f7de` | `57ccb0ce64` | — |

- 확인된 사실: 텍스트 인코더 출력부터 프로세스마다 다르다. 같은 프로세스 안에서는 ComfyUI가 conditioning을 재사용하므로 native 반복이 바이트 일치한다. GPT 하네스의 프로세스 내 A/B는 유효하다.
- 확인된 사실: native끼리의 8-bit PNG 차이가 RMSE **29.7–37.5/255**, max 228–238이다. INT8 → W4 차이(45.3, 54.8/255)와 같은 자릿수다.
- 확인된 사실: Qwen은 같은 비교에서 conditioning과 이미지 SHA가 GPT 실행과 같았다. 이 현상은 MOSS에만 있다.
- 추론: MOSS 4단계 INT8 ConvRot 경로는 입력의 아주 작은 차이를 크게 키운다(v0.5에서 본 경계 증폭과 같은 성격). 그래서 MOSS에서 엄격 기준은 사실상 "비트 일치냐 아니냐" 시험이 된다. W4의 엄격 실패는 맞지만, 그 RMSE 크기를 품질 열화의 정도로 읽으면 안 된다. 품질 판단에는 지각 평가가 필요하다. W4는 native보다 느려서 채택 결론은 바뀌지 않는다.
- 운영 영향: MOSS 운영 결과를 이전 실행 이미지와 바이트로 비교하는 회귀 검사는 성립하지 않는다. 비교는 같은 프로세스에서 하거나, conditioning을 저장해 고정한 뒤 해야 한다.
- 원인(텍스트 인코더 내부의 어느 연산인지)은 찾지 않았다.

### 2.2 Sparse r4가 인물을 복제하는 이유 (추론)

r4는 각 위치가 가까운 블록만 보게 한다. 먼 위치에 이미 인물이 그려지고 있다는 정보가 끊기므로, 프롬프트를 받은 각 영역이 각자 인물을 그린다. spec(`qwen_sparse_radial4_v07.json`)에는 초반 단계나 앞쪽 층을 dense로 남기는 조건이 없어, 구도가 정해지는 초반부터 희소화가 걸렸다. 초반 단계와 앞쪽 층을 dense로 두는 변형은 복제를 줄일 가능성이 있지만 이득도 1.096배보다 줄어든다. 미검증이다.

### 2.3 Strata warm 프리필: 모델 A/B 완료, 병목은 GPU 계산이지만 한 커널에 몰려 있지 않다

GPT이 swap 보호 관문(프로세스 시작 이후 swap-out 256 MiB 초과)에 걸려 남긴 미완료 항목을 이어서 측정했다. GPT 실행기를 `rdna2-ops/strata_prefill_v07_claude.py`로 복사해 swap 한도만 인자(`--swap-limit-mib`)로 뺐고 8,192 MiB로 실행했다. 램 가용 2 GiB 미만 중단과 운영 서비스 감시는 그대로다. 연구 빌드는 `strata-prefill-v07-lab-20261107/build-v07-gfx1030/strata`(같은 ELF로 `STRATA_MMQ_TYPICAL_J` off/on), 팔마다 서버를 새로 띄우고 untimed 9.7K priming 1회 뒤 fresh 요청(native `reused=0`)을 쟀다. 운영 서비스는 내내 inactive였다.

| 팔 | 9.7K | 19.8K | 119K | 프로세스 내 최대 swap-out | 최소 MemAvailable |
|---|---:|---:|---:|---:|---:|
| a1 J off | 1,085.3 | 1,148.7 | 1,311.6 | 2,243.6 MiB | 40.1 GiB |
| b1 **J on** | **1,220.9** | **1,255.4** | **1,294.9** | 1,319.7 MiB | 43.1 GiB |
| a2 J off | 1,231.5 | 1,260.1 | 1,307.3 | 556.6 MiB | 44.0 GiB |
| t1 J off + `STRATA_PREFILL_TIMING` | 1,218.3 | 1,244.5 | 1,287.8 | 6.3 MiB | 43.8 GiB |

단위는 native 로그의 fresh 프리필 tok/s다.

- 확인된 사실(정확도): 네 팔 모두 같은 프롬프트 ID SHA에서 생성 32토큰 ID가 세 길이 전부 같았다(9.7K·19.8K `340eb49d04…`, 119K `e8ac542acc…`). J on/off는 출력이 같다.
- 확인된 사실(속도): J on은 인접한 J off(a2) 대비 0.991 / 0.996 / 0.991배다. 사전 기준 1.03배에 못 미쳐 **모델 이득 없음**이다. a1의 짧은 길이가 낮은 것은 첫 팔의 파일 캐시 상태 영향으로 본다(a1의 priming은 260.7 tok/s, 이후 팔은 670–992 tok/s).
- 확인된 사실(자원): swap-out은 팔을 거듭할수록 2,244 → 6 MiB로 줄었고, MemAvailable은 내내 40 GiB 이상이었다. 256 MiB 관문이었다면 앞의 세 팔은 모두 중단됐다. 이 기계에서 warm 측정은 가능하다.

#### 단계 분해 (t1, warm fresh 9,711 토큰 / 118,357 토큰)

| 항목 | 9.7K | 119K |
|---|---:|---:|
| GPU timeline / wall | 7,310 / 7,341 ms | 91,189 / 91,277 ms |
| wait copy (PCIe 대기) | 43 ms (0.6%) | 518 ms (0.6%) |
| OS가 스토리지에서 읽은 양 | 0.6 MB | 11.1 MB |
| 전문가 경로 합(gather·wait copy·dequant·GEMM gate/up·down·combine) | 2,604 ms (35.6%) | 30,522 ms (33.5%) |
| 그중 GEMM gate/up + down | 1,892 ms (25.9%) | 22,106 ms (24.3%) |
| QSA attention | 1,062 ms (14.5%) | 14,345 ms (15.7%) |
| GDN | 1,083 ms (14.8%) | 12,594 ms (13.8%) |
| HC read | 904 ms (12.4%) | 10,995 ms (12.1%) |

- 판정: Claude의 "warm 프리필은 PCIe가 아니라 GPU 계산이 병목" 가설은 **확인**됐다. GPU timeline이 wall과 거의 같고, PCIe 대기는 1% 미만, 스토리지 읽기는 사실상 0이다.
- 다만 GPU 시간은 한 커널에 몰려 있지 않다. 전문가 GEMM은 약 25%이고, QSA attention·GDN·HC가 합쳐 약 40%를 차지한다. 그래서 전문가 MMQ 커널을 2배로 만들어도 전체는 약 12% 상한이다(추론). typical-J가 모델에서 무이득인 것도 같은 맥락이다. 큰 덩어리에서는 J가 바뀌지 않는다.
- `dequant` 라벨: warm에서는 5% 안팎인데, 파일 캐시가 덜 찬 첫 측정 요청(OS 6.4 GB 읽음)에서는 13.3%, GPT의 cold 첫 요청(47 GB 읽음)에서는 74.9%였다. GPT이 소스로 지적한 대로 이 라벨은 스토리지·host 대기를 흡수한다.
- 계측 비용: timing 팔의 처리량은 a2 대비 1.1–1.5% 낮다.

## 3. 무결성과 운영 상태

- `RDNA2_PACKAGE_V07_MANIFEST.json` 1,153개, `rdna2-ops/SHA256SUMS` 1,152개가 모두 현재 파일과 일치한다. v0.6에서 지적한 3개 불일치(`kernels.hip`, FP8 config 2개)는 고쳐졌다.
- 운영 파일: `results/v07_operating_files/after.json` 31개가 지금도 그대로다. Strata `operating_after.json` 48개 중 47개가 그대로이고, 43 GB `experts.bin` 하나는 해시를 건너뛰었다.
- `strata-verified-256k.service`는 inactive다.
- GPT Strata 실험의 흔적으로 swap 10 GB가 사용 중이다(램 가용 41 GB). 기능 문제는 아니다.
- 카탈로그 v0.3에 Claude의 §0.1–0.2, 재현 표시, MIOpen 변수 행이 보존돼 있다. 링크 깨짐 0, 숫자·단위 붙여 쓰기 0.

## 4. 재현하지 않은 범위

- INT4의 나머지 sweep(9 shape × g32/64/128 × 6 포맷, 324 관문), dot8 경로
- Sparse seed 08·09, 익명 판정 12건 재실행
- Strata 첫 요청(cold) 단계 분해(48.774 s, `dequant` 라벨 74.92%)의 재실행. warm A/B와 단계 분해는 §2.3에서 완료했다
