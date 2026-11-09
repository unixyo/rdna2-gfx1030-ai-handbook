# Strata warm 프리필 v1.0 — CPU 사전점검, 서버 미실행

최종 갱신: 2026-11-08 11:49 UTC. **v1.0 D는 자원 재개 근거가 없어 미판정이며, 이번 연구의 Strata GPU 실행·서버 시작은 0회다.** CPU 사전점검을 갱신하고 지정한 작은 운영 파일 41개의 SHA 일치를 확인했다. 운영 서비스는 읽기 전용으로 조회했다.

## 1. 재개 판단

원본 [STRATA_PREFILL_V07_KO.md §8](../strata_v07/STRATA_PREFILL_V07_KO.md)과 [PROTOCOL_KO.md](../strata_v07/PROTOCOL_KO.md)는 같은 자원 조건에서 관문에 걸린 실험을 반복하지 않도록 한다. Applio A/B 완료 후 갱신한 **2026-11-08 11:47:58 UTC** 사전점검에서는 expert 파일 cache가 **0%**, MemAvailable이 **35.91 GiB**였다. 이전 재시도 때는 expert cache가 **92.78%**, MemAvailable이 **47.49 GB = 44.23 GiB**였는데도 256 MiB swap-out 관문에 다시 걸렸다. 현재 값은 재개를 뒷받침하는 개선 근거가 아니다.

이번에는 KFD client가 0이고 서비스가 꺼져 있어 기본 사전점검의 `prerequisite_checks_pass=true`다. 이 값은 원본 §8의 자원 재개 근거를 대신하지 않는다. GPU idle 또는 MemAvailable 14 GiB 조건만으로 서버를 시작하지 않았으며, 대용량 보호 해시로 모델 cache를 미리 채우지도 않았다. **새 실행 중 관문 중단이 발생한 것이 아니라, 서버를 시작하기 전에 재개 근거가 없다고 판단했다.**

| 관측 항목 | 이번 CPU 점검 | 상태 |
|---|---:|---|
| 운영 service | `inactive / dead`, MainPID `0` | 전후 동일, 조작 없음 |
| MemAvailable | 38,557,175,808 B = 35.9092 GiB | 실행 가능성의 충분조건 아님 |
| swap 사용량 | 16,648,335,360 B = 15.5050 GiB | 기존 사용량, 이번 관측 증가와 구분 |
| 관측 중 추가 swap-out | 0 B | 짧은 CPU 점검 구간 |
| OS Cached | 27,293,552,640 B = 25.4191 GiB | 다른 파일 cache 포함 |
| 실제 expert 파일 cache | 0 / 42,912,972,800 B | `mincore` 0% |
| 실제 dense 파일 cache | 0 / 1,538,035,200 B | `mincore` 0% |
| native GGUF 2개·PLE cache | 모두 0% | 파일 내용을 읽지 않고 관측 |
| 포트 8080 / 연구 포트 18487 | 둘 다 비어 있음 | `/proc/net/tcp*` 조회 |
| KFD | 0 clients / 0 queues | 조회 당시 GPU 작업 없음 |
| 새 Strata 서버 / GPU 실행 | 0 / 0 | CPU 준비만 수행 |

최신 원자료: [strata_preflight_02.json](strata_preflight_02.json). 각 값은 한 시점의 관측이며 이후 실행 조건을 대신하지 않는다. 초기 [strata_preflight_01.json](strata_preflight_01.json)은 10:14 UTC Applio 실행 중의 KFD 6 clients / 7 queues, MemAvailable 30.44 GiB를 기록한 자료로 보존한다. 최신 02는 그 실행 종료 후 별도로 수행한 CPU 점검이다.

## 2. 이번에 확인한 사실과 남은 질문

확인된 사실:

- [v10_strata_preflight.py](../../v10_strata_preflight.py)는 Python 표준 라이브러리만 사용하며 GPU runtime을 import하지 않는다. 실행 결과 `torch_imported=false`, `gpu_execution=false`, `server_execution=false`다.
- 파일 cache는 `PROT_NONE / MAP_PRIVATE / mincore`로 관측했다. 매핑한 페이지를 역참조하거나 `read`하지 않으며 `madvise`, cache flush, swap 설정 변경을 하지 않는다. 대상 22개 조회가 모두 성공했다. 파일 소유권·현재 UID의 쓰기 가능 여부도 기록했다.
- 초기 사전점검 때 0%가 도구의 고정 출력인지 확인하기 위해 SHA를 읽은 직후의 21,010,736 B 연구 ELF를 같은 함수로 조회했다. 5,130 / 5,130 pages, 100% resident였다. 모델 파일 자체를 읽어서 이 검사를 하지 않았다. 이 진단을 최신 02에서 반복한 것으로 계산하지 않는다.
- 원본 `strata_prefill_v07.py`를 AST로 확인했다. `swap_limit_bytes = 256 << 20`이 그대로이며 CLI로 한도를 바꾸는 경로가 없다.
- 연구 ELF와 pinned 기준 사본의 SHA가 기존 v0.7 기록과 일치한다. 재빌드하지 않았다.

미검증:

- 현재 조건의 warm fresh 9.7K 처리량, J off/on 차이, token ID 동일성.
- 현재 조건의 GPU 계산·H2D·host stall 비율과 실제 routing/J hit 분포.
- 모든 대용량 운영 입력의 이번 before/after SHA. 현재 단계에서는 큰 파일을 읽지 않았다.

추론:

- 이전보다 적은 가용 RAM과 비어 있는 expert cache는, 동일한 256 MiB 관문 아래에서 새 warm 측정을 성공시킬 재개 근거가 되지 않는다. 이 판단은 GPU 계산 병목 또는 PCIe 병목을 판정한 것이 아니다.

## 3. Claude의 선행 결과는 조건이 다르다

[REVIEW_V07_CLAUDE_KO.md §2.3](../../../REVIEW_V07_CLAUDE_KO.md)는 복사본 `strata_prefill_v07_claude.py`에 `--swap-limit-mib 8192`를 주어 실행했다. J off/on 비교와 warm 단계 분해가 기록되어 있으므로 그 선행 자료를 없던 것으로 취급하지 않는다. **이번 지시의 256 MiB 보호 조건을 충족한 새 실행 완료 근거로 사용하지도 않는다.**

그 기록의 비계측 a1/b1/a2는 프로세스 기준 추가 swap-out이 각각 2,243.6 / 1,319.7 / 556.6 MiB였다. 모두 이번 256 MiB보다 크다. 뒤의 timing 팔 t1은 6.3 MiB였지만, 앞의 넓어진 관문 아래에서 형성된 자원 상태를 이어받았다. 따라서 기존 보고의 “GPU 계산이 병목”은 별도 조건의 선행 결론으로 남기고, v1.0 D의 결과 칸은 미판정으로 둔다.

## 4. 보호 범위와 SHA

현재 v1.0 Applio 보호 baseline 1,022개에는 Strata 경로가 **0개**다. D를 실제 실행하게 되면 Strata의 독립 보호 기록이 필요하다. 이번에는 목록을 모으고 작은 필수 파일만 SHA로 검사했다.

| 범위 | 이번 점검 |
|---|---|
| 기존 v0.7 보호 48개 | 목록/stat 확인. 전체 약 50.67 GB SHA 재읽기 안 함 |
| 현재 config가 참조하는 전체 권장 범위 | 64개, 156,451,275,508 B. SHA 완료가 아닌 경로·크기 inventory |
| 작은 파일 + 운영 pinned ELF | **41개, 27,275,498 B** SHA 읽기 |
| 위 41개 중 v0.7 SHA와 비교 가능 | **32개 일치, 변경 0개** |
| 기존 목록에 없던 작은 파일 | **9개** 새 current SHA 기록 |
| 이번 before/after 비교 | **41 / 41 일치, 변경 0개·누락 0개·오류 0개** |
| 연구 ELF | `31ecea0716fd7cba638d004b2fc7987efa1b179f9a9cf67ca96f8d42b4c263b5` |
| pinned 연구 사본 | `dd6753203cec325a2a89bc1c9a96d2a847c0bbcf3e08622d9cebb62484977466` |

작은 파일 SHA 원자료: before [strata_small_files_sha_01.json](strata_small_files_sha_01.json), after [strata_small_files_sha_after_02.json](strata_small_files_sha_after_02.json). 11:49:04 UTC after 검사에서는 before에 명시된 정확한 41개 경로만 총 **27,275,498 B** 읽었고, 크기와 읽는 동안의 변경 여부도 검사했다. 9개 신규 baseline을 포함해 41개 전부 같았다. 11:47:58 UTC cache 관측은 이 작은 파일 after SHA 읽기보다 먼저 수행했다. 대용량 expert·dense·GGUF 입력의 SHA를 읽거나 cache를 채우지 않았다.

필수 운영 파일에는 systemd unit/drop-in, launcher, private config의 **내용이 아닌 SHA**, 운영 ELF, server source 및 작은 인덱스·설정이 포함된다. 비밀번호·키·토큰·전체 환경변수를 출력하거나 보고서에 저장하지 않았다. after JSON의 SHA256은 `d899f1d5316e52d30ad76fdc8eaa2a746337bd6139334b0fa52924ad26143a3d`다.

64개 권장 목록은 `strata_preflight_02.json → protection.recommended_files`에 있다. 기존 48개 외에 현재 실행이 읽는 native GGUF 2개, PLE GGUF, MTP 파일, expert-profile, draft vocabulary, tokenizer, shadow server Python source를 더했다. 이 **156.45 GB 범위의 전체 SHA 검사는 수행하지 않았다.** 향후 D를 재개할 때 실제 범위와 I/O 순서를 정하고, 앞뒤 SHA 감사가 cache 상태에 주는 영향도 구분해야 한다. Stat 일치는 SHA 일치를 대신하지 않는다.

## 5. 재개할 수 있을 때 사용할 명령

아래 내용은 준비한 명령이며 **실행 기록이 아니다**. 서버 실행은 root가 GPU를 독점하고 원본 자원 재개 조건을 충족했을 때만 한다. 운영 service는 시작·중지·재시작하지 않는다.

자원 상태가 실제로 달라져 재개를 검토할 때 사용할 다음 CPU 사전점검 명령이다. 완료한 02를 덮어쓰지 않도록 **03**을 사용한다. 이 03 명령은 아직 실행하지 않았다.

```bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
/usr/bin/python3 -B v10_strata_preflight.py \
  --out results/v10/strata_preflight_03.json \
  --port 18487 --run-prefix strata_warm_01
```

준비한 9개 명령은 사전점검 JSON의 `commands`에 argv와 shell 문자열로 저장된다. 위의 03 사전점검을 실행한 뒤 이를 **출력만** 하는 CPU 명령:

```bash
/usr/bin/python3 -B - <<'PY'
import json
from pathlib import Path
r = json.loads(Path('results/v10/strata_preflight_03.json').read_text())
for command in r['commands']:
    print(command['label'], command['shell'])
PY
```

| 순서 | 용도 | J | 측정 방법 |
|---|---|---:|---|
| a1 → b1 → b2 → a2 → a3 → b3 | 주 처리량 | 0/1 교대 | 각 서버에 priming 390행 1회, fresh 390행 1회, 32 생성 token, 비계측 |
| timing_a → timing_b | 단계 분해 | 0/1 | 같은 요청에 `--timing`, 주 처리량 중앙값에서 제외 |
| routing_b | 선택 분포, 선택적 | 1 | `STRATA_MMQ_V07_TRACE=1`, 주 처리량 중앙값에서 제외 |

각 팔은 **같은 기존 연구 ELF**를 쓰고 별도 `results/v10/strata_warm_01_*` 출력 디렉터리, 포트 **18487**를 사용한다. 원본 하네스는 이미 존재하는 출력 디렉터리를 거부한다. priming nonce는 `v07-prime-390-0`, 측정 nonce는 `v07-390-0`이며 팔 간 동일하다. 실제 native `reused=0`과 prompt ID SHA를 확인해야 warm fresh로 인정한다. 390행을 “정확히 9,700 token”이라고 가정하지 않고 실제 prompt token 수를 사용한다.

준비 명령의 공통 환경은 `PYTHONDONTWRITEBYTECODE=1`, `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1`, 두 MIOpen 안전 변수 0, `OMP_NUM_THREADS=2`다. 비계측 팔에 routing trace가 상속되지 않도록 shell 환경에서 제거한다. 운영 config의 routing trace도 사전점검에서 없는지 확인한다. 운영 파일을 바꾸는 코드, 서비스 조작 명령, 8,192 MiB 관문을 쓰는 Claude 복사본은 이 실행 계획에 없다.

원본 하네스가 유지하는 조건:

- `gpu_guard_v06.GpuAudit`의 공통 GPU lock과 소유 PID 확인.
- 프로세스 시작 이후 system `pswpout` 증가가 **256 MiB 초과**이고, **1초 간격 두 표본에서 계속 증가**하면 연구 소유 process group만 종료.
- 실행 중 MemAvailable **2 GiB 미만**이면 종료.
- 5초마다 운영 service를 읽고 켜졌으면 연구 소유 process group만 종료.
- 요청 구간별 swap·physical I/O·major faults와 전체 프로세스 기준을 함께 보존.

첫 팔이 관문에 걸리면 뒤의 명령을 순서대로 계속 실행하지 않는다. raw 결과를 남기고 미판정으로 종료한다. 관문을 완화하거나 cache를 강제로 채워 통과시키지 않는다.

## 6. 결과를 읽을 때 지킬 구분

`strata_summarize_v07.py`는 saved result/log만 읽는 CPU 요약기다. native `prompt_ms`, `tok_s`, `reused`, generated token ID, prompt ID SHA, `issued_h2d_bytes`, phase I/O와 swap delta를 만든다. **요약기에 값이 있다는 사실만으로 timing-valid가 되는 것은 아니다.** 원본 `resource_valid_for_timing`, `guard.valid_for_timing`, 오류 유무, 소유 PID 잔류, fresh 여부를 먼저 확인한다.

주 처리량은 비계측 a1/a2/a3와 b1/b2/b3의 중앙값 및 짝별 비율을 함께 쓴다. priming과 timing/routing 팔은 제외한다. 동일 요청 SHA와 prompt token ID SHA, 생성 token ID 배열을 모두 비교한다. 텍스트 SHA는 token ID 일치의 대체물이 아니다.

`issued_h2d_bytes`는 연구 source가 실제 issue한 expert H2D blob bytes이며 전체 프로세스의 모든 PCIe 전송을 포괄한다고 주장하지 않는다. phase GPU timeline은 event 사이 idle도 포함한다. `host staging`은 겹치는 누계이므로 wall·timeline·host staging을 더하지 않는다. `wait copy`와 host grouping, 물리 I/O, phase 비중을 함께 읽고, event 구간 전체를 순수 GPU 계산 시간으로 바꾸어 적지 않는다. 별도 trace 없이 정확한 전체 host stall 비율을 확정할 수 없으면 그 항목은 미식별로 남긴다.

## 7. Claude가 GPU 없이 확인할 숫자 5개

| # | 숫자 | 의미 |
|---|---:|---|
| 1 | 35.9091682434 GiB | 최신 02의 11:47:58 UTC MemAvailable |
| 2 | 15.5049705505 GiB | 같은 시점의 swap 사용량 |
| 3 | 0 / 42,912,972,800 B = 0% | 실제 expert cache residency |
| 4 | 268,435,456 B = 256 MiB | 유지한 추가 swap-out 관문 |
| 5 | 41 / 41 일치, 변경 0개 | 작은 운영 파일의 이번 before/after, 누락·오류도 0개 |

아래 명령은 실제 실행해 위 5개 숫자와 서버·GPU 실행 0회를 확인했다. 저장된 작은 JSON 두 개만 읽으며 GPU를 사용하지 않는다.

```bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
/usr/bin/python3 -B - <<'PY'
import json
from pathlib import Path
p = Path('results/v10')
r = json.loads((p / 'strata_preflight_02.json').read_text())
s = json.loads((p / 'strata_small_files_sha_after_02.json').read_text())
print('1 MemAvailable GiB:', r['before']['mem_available_bytes'] / (1 << 30))
print('2 Swap used GiB:', r['before']['swap_used_bytes'] / (1 << 30))
for item in r['file_cache']:
    if item['path'].endswith('/pack-strata-v0142/experts.bin'):
        print('3 Expert resident / file bytes:', item['resident_bytes'], item['bytes'], item['resident_fraction'])
print('4 Swap guard bytes:', r['guard']['swap_out_limit_bytes'])
print('5 Matched / expected / changed / missing / errors:', s['matching_count'], s['expected_count'], len(s['changed']), len(s['missing']), len(s['errors']))
print('Execution:', r['gpu_execution'], r['server_execution'], 'service actions:', r['operating_service_actions'])
PY
```

최종 상태: Applio A/B 완료 후 새 CPU 사전점검 02까지 수행했으나 **자원 재개 근거 없음 → Strata 서버·GPU 실행 0회 → warm 병목 미판정**이다. 이번에 새 swap-out 관문 중단이 발생한 것은 아니다. 원본 §8을 뒷받침할 자원 상태 변화가 생길 때 다음 사전점검 03부터 검토한다.
