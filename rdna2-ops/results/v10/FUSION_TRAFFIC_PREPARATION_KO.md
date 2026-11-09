# 대표 ResBlock의 GL2C/EA traffic 실측

작성: 2026-11-08. **대표 C32/K3/d1 pair의 GL2C/EA read/write × B/B′ 네 pass 수집과 엄격 CPU 후처리를 완료했다.** pair당 합계는 **185,769,600→116,168,576 bytes, 37.4663% 감소**했다. 첫 loader 실패는 같은 wheel 파일을 가리키는 연구 폴더 alias로 한 번 복구했고 GPU 표본은 재실행하지 않았다. 운영 파일·venv·기존 연구 source는 수정하지 않았다. 물리 DRAM 전송량은 이 지표로 식별하지 않는다.

## 설치 자료에서 확인한 것

- 실제 도구는 `~/DevSpace/applio-lab/.venv-rocm10/lib/python3.12/site-packages/_rocm_sdk_core/bin/rocprofv3`다. 설치 source의 버전은 1.3.5, git revision `6b0e43f341195e203754e08f850e437ff2fc09f9`, ROCm 10.0.0이다. console trampoline 대신 이 파일을 기존 Python으로 직접 실행하면 devel package의 자동 확장 경로를 거치지 않는다.
- 같은 SDK의 `share/rocprofiler-sdk/config.yaml`에는 gfx1030에 적용되는 정적 metric 정의가 89개 있으며 GL2C 이름 20개와 `FETCH_SIZE`, `WRITE_SIZE`가 있다. 이 **설치 정의 수**와 아래 실제 availability의 metric 수를 별도로 확인했다. 이름 노출과 유효한 counter 값 수집은 서로 다른 단계다.
- standalone `~/DevSpace/rocm10-gfx1030/install/`의 CLI·config는 Applio venv 사본과 SHA가 같다. config SHA256은 `63542b5a2a285098f6ccb1d355574c8559ca75d3d5f80c82c5f02be7e3c8f960`다.
- `snapshots/guide-stack.md` §5.8과 `snapshots/vllm-profile.md` §10의 2026-08-19 기록은 78개와 cache counter 부재를 말한다. 현재 정의에는 TCC/TCP 대신 GL2C 계열이 있으므로 옛 문장만으로 현재 지원 불가를 단정할 수 없다. 이름 누락인지 당시 설치/장치 제약인지는 원 availability 원자료가 없어 미확정이다.
- 설치된 rocprof-compute의 SoC 파일에는 gfx1030 전용 구성이 없다. 이 조사에서는 rocprofv3의 기존 counter 경로를 사용하며 설치·재빌드를 제안하지 않는다.

## 실제 availability — 2026-11-08 완료

메인 작업이 availability.log (`fusion_traffic_availability_01/availability.log`, 연구 폴더에만 있음)를 생성했다. [result.json](fusion_traffic_availability_01/result.json)은 `completed=true`, exit 0, GpuAudit 유효, foreign client·audit error 0을 기록한다. `--list-avail -f csv`의 이 실행은 stdout log를 남겼으며 별도 CSV를 생성하지 않았다. 로그의 `GPU:0 / Name:gfx1030` 구간만 집계했고 뒤의 gfx1036 구간과 섞지 않았다.

| 집계 대상 | 수 | 의미 |
|---|---:|---|
| 설치 config의 gfx1030 metric 정의 | 89 | CPU 파일에서 읽은 정의 |
| 실제 availability의 이름 전체 | 148 | agent 상수 포함 |
| 실제 agent 상수 | 59 | 장치 속성값, 하드웨어 event가 아님 |
| 실제 metric 이름 | 89 | block event 45 + expression 44 |
| `GL2C_`로 시작하는 이름 | 20 | 기본 event와 derived sum을 합한 수 |
| 별도 관련 GRBM 신호 | 1 | `GRBM_GL2CC_BUSY`; 위 GL2C prefix 수에는 포함하지 않음 |
| 요청한 derived metric | 2 | `FETCH_SIZE`, `WRITE_SIZE` 모두 실제 노출 |

기본 GL2C event는 `DIMENSION_INSTANCE[0:31]`, 두 요청 metric은 reduce 후 `DIMENSION_INSTANCE[0:0]`으로 나온다. 각 항목의 `SPM: Not Supported`는 streaming monitor 표시이며 dispatch PMC metric의 이름 부재를 뜻하지 않는다. [메인 관측 JSON](fusion_traffic_batch_01/availability_observation.json)의 GL2C 관련 21개는 관련 GRBM 신호까지 포함한 분모이며, prefix 기준 20개와 구분한다.

로그 SHA256은 `b58bfd22f615236e022987efe74877f93c3e87c44642fadffbdad68e95db69b8`, 실행 result SHA256은 `21d870c3a4fb15d0b1edd56455b710fc82a47e35391908253a2be2a2050d0157`다. 현재 노출 결과는 옛 “cache counter 없음” 문장을 이 설치에 그대로 적용할 수 없음을 확인한다. 옛 78개 결과가 달랐던 이유 자체는 미확정으로 유지한다.

## 지표의 정확한 분모

| 이름 | gfx1030 설치 정의 | 단위·보고 범위 |
|---|---|---|
| `FETCH_SIZE` | `(32*GL2C_EA_RDREQ_32B_sum + 64*GL2C_EA_RDREQ_64B_sum + 96*GL2C_EA_RDREQ_96B_sum + 128*GL2C_EA_RDREQ_128B_sum)/1024` | KiB, GL2C/EA read-request traffic |
| `WRITE_SIZE` | `((GL2C_MC_WRREQ_sum-GL2C_EA_WRREQ_64B_sum)*32 + GL2C_EA_WRREQ_64B_sum*64)/1024` | KiB, GL2C/EA write-request traffic; 정의상 atomic/CMPSWAP가 같은 인터페이스에 포함될 수 있음 |

`basic_counters.xml`의 gfx10 정의와 gfx1030 상속, `config.yaml`의 해당 항목을 읽었다. `*_sum`은 해당 GL2C instance의 합이다. SDK 설명은 video memory라는 표현을 쓰지만 기반 event가 직접 세는 것은 GL2C/EA 요청이다. gfx1030의 MALL/Infinity Cache miss 이후 GDDR6 전송과 같다고 판정할 별도 MALL/UMC/DRAM 전용 metric을 이번 설치 정의에서 찾지 못했다. **결과는 GL2C/EA interface bytes라고 보고하고 물리 DRAM bytes라고 부르지 않는다.**

## 최소 driver와 구간

[v10_fusion_traffic.py](../../v10_fusion_traffic.py)는 기존 `bench_v10_fusion.pairs`·안전 변수·SHA helper와 실제 v0.8 dispatch/v1.0 fusion context를 재사용한다. 대상은 **B8/C32/T12800/K3/d1, FP16 입력, FP32 weight_norm master, seed 20261108** 하나뿐이다. 기존 `one()`과 같은 CPU RNG 순서로 parameter·x·dy를 만들며 네 pass의 x/dy/master SHA가 같아야 한다.

한 프로세스는 B 또는 B′ 하나만 실행한다. warmup 3회 이후 synchronize하고 `roctxProfilerResume(0)`에서 pair forward+backward를 실행한다. 완료 synchronize 뒤 `roctxProfilerPause(0)`로 닫는다. 설치 `rocprofiler-sdk-roctx/roctx.h`는 **0이 프로세스의 모든 스레드**라고 명시하므로 autograd worker도 포함된다. label은 기존 trace 파일 이름과 같은 `B8_C32_T12800_K3_D1_float16_B` / `B8_C32_T12800_K3_D1_float16_B_prime`의 process range다. thread-local `--kernel-rename`에 backward 귀속을 의존하지 않는다.

ROCTX library는 실제 wheel의 `lib/librocprofiler-sdk-roctx.so.1`을 사용한다. standalone 배포의 `.so.1.3.5` 파일명과 다르므로 존재 여부도 CPU preflight에서 확인한다.

`GpuAudit`에 진입하고 exclusivity를 확인한 뒤에만 torch를 import한다. MIOpen 안전 변수 두 개와 `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1`, 연구 DB·cache를 사용한다. 전체 sweep·FP32 참조·100 epoch는 재실행하지 않는다. 현재 source의 기존 108/144 검증과 selection/evidence SHA가 맞아야 시작한다. 수집 결과는 성능 표본이 아니며 기존 pair/epoch 속도를 대체하지 않는다.

초기 pause는 idempotent 전역 제어를 사용한다. `--selected-regions-ref-count`와 자동 multi-pass는 이 driver 계약에 포함하지 않으며 실행 전에 거부한다. `--kernel-trace`가 빠진 경우도 GPU 작업 전에 중단한다.

기존 profiler에서 이 pair는 B **28 launch**, B′ **22 launch**였다. 별도 CPU `v10_summarize_fusion_traffic.py`는 실제 수집 kernel 수가 `28/22 × reps`인지 확인하고 각 dispatch에 요청한 derived metric이 정확히 하나씩 있는지 검사한다. 누락·중복·음수·nonfinite·전체 0인 traffic은 유효 판정하지 않는다. 이 검사를 통과하기 전 run의 `valid_for_counters`는 false이며, 유효 판정은 후처리 `traffic_summary.json`에만 기록한다.

## 메인 작업의 명령과 후속 재현

메인 작업은 availability `_01`, 최초 수집 실패 `_01`, 연구 alias를 통한 수집 `_02`를 순서대로 남겼다. 수집용 DB는 기존 `fusion_bench_db`를 별도 `fusion_traffic_db_01`로 복사했다. driver SHA256 `b236e122cd424dd57ff1d95bf6ce1df0707d657af950b148d85a22f415f47acb`는 바꾸지 않았다. 아래는 완료된 `_02`를 덮어쓰지 않는 **새 `_03` 출력의 후속 재현 예시**이며 이번 연구에서 추가 실행한 명령이 아니다. GPU 작업은 직렬로 실행하고, driver 자체 GpuAudit와 같은 lock의 외부 guard를 중첩하지 않는다.

```bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
TRAFFIC_PY=~/DevSpace/applio-lab/.venv-rocm10/bin/python
TRAFFIC_SDK=~/DevSpace/applio-lab/.venv-rocm10/lib/python3.12/site-packages/_rocm_sdk_core
TRAFFIC_ALIAS=~/DevSpace/rdna2-pro-research-20261102/rdna2-ops/results/v10/fusion_traffic_loader_03

# CPU only: evidence/shape/source 확인, torch import·KFD 조회 없음.
/usr/bin/python3 -B v10_fusion_traffic.py --arm B --counters FETCH_SIZE

# CPU only: wheel 자체를 수정하지 않고 dlopen용 이름만 연구 폴더에 둔다.
mkdir "$TRAFFIC_ALIAS"
ln -s "$TRAFFIC_SDK/lib/libhsa-amd-aqlprofile64.so.1" "$TRAFFIC_ALIAS/libhsa-amd-aqlprofile64.so"

# 두 derived metric은 기반 event가 총 6개라 동시 수집 가능 여부를 가정하지 않는다.
# 각 arm/read/write는 별도 프로세스·별도 새 출력으로 순차 실행한다.
for traffic_metric in FETCH_SIZE WRITE_SIZE; do
  for traffic_arm in B B_prime; do
    traffic_run="fusion_traffic_${traffic_metric}_${traffic_arm}_03"
    env LD_LIBRARY_PATH="$TRAFFIC_SDK/lib:$TRAFFIC_ALIAS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      ROCPROFILER_REGISTER_ENABLED=1 HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 \
      MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0 MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53=0 \
      "$TRAFFIC_PY" -B "$TRAFFIC_SDK/bin/rocprofv3" \
      --selected-regions --kernel-trace --marker-trace --pmc "$traffic_metric" \
      -f csv -d "results/v10/${traffic_run}_rocprof" -o "$traffic_run" -- \
      "$TRAFFIC_PY" -B v10_fusion_traffic.py --execute --arm "$traffic_arm" \
      --counters "$traffic_metric" --seed 20261108 --warmup 3 --reps 1 \
      --manifest results/applio_v08/direct_selected_final/dispatch.json \
      --selection results/v10/fusion_full/selection.json \
      --evidence results/v10/fusion_full/result.json --db results/v10/fusion_traffic_db_01 \
      --out "results/v10/${traffic_run}" || break 2
    /usr/bin/python3 -B v10_summarize_fusion_traffic.py \
      --run "results/v10/${traffic_run}" \
      --counter-dir "results/v10/${traffic_run}_rocprof" || break 2
  done
done
```

## 초기 실패, 동일 wheel alias 복구, CSV 형식 보완

최초 [batch `_01`](fusion_traffic_batch_01/)의 첫 FETCH_SIZE/B pass는 GPU 초기화에서 `aqlprofile API table load failed: HSA_STATUS_ERROR`로 abort했고 counter CSV는 없었다. 뒤 세 팔은 실행하지 않았다. `failed_before_collection` 이력과 정리 후 유효한 GpuAudit를 보존했다. **availability의 이름 노출만으로 실제 수집 성공이라고 할 수 없다는 사례**다.

CPU 확인에서 SDK/lib에는 `libhsa-amd-aqlprofile64.so.1`만 있고 loader가 찾는 unversioned `.so`가 없었다. 설치 `hsa_ven_amd_aqlprofile.h:416–420`과 HSA/rocprofiler binary의 이름이 그 `.so`를 가리켰다. `.so.1`의 필요한 API export와 SONAME은 존재했다. 연구 폴더 `fusion_traffic_loader_02/libhsa-amd-aqlprofile64.so`를 같은 wheel `.so.1`로 연결하고 `LD_LIBRARY_PATH=wheel/lib:연구alias:기존값`만 추가했다. rocprofv3는 기존 LD 경로에 자기 lib를 append하므로 이 우선순위를 유지한다. 다른 배포의 라이브러리를 섞거나 venv에 symlink를 만들지 않았다. 대상 파일 SHA256은 `0001175c61c8ad3491e75a12a056697261b6ffe86b3bdd65c153834602f1b686`다.

그 뒤 첫 B/FETCH와 나머지 세 팔은 모두 exit 0, 완료·input contract·전후/실행 중 GpuAudit를 통과했다. 첫 성공 B/FETCH의 기존 CPU 후처리는 kernel CSV에 `Process_Id` 열이 없어 `KeyError`로 중단됐다. 이 [protocol](fusion_traffic_batch_02/protocol.json)의 `summary_failed` 이력은 그대로 두고, **성공한 GPU 표본을 재실행하지 않았다**. [나머지 세 팔 protocol](fusion_traffic_remaining_02/protocol.json)도 보존한다.

새 [v10_summarize_fusion_traffic.py](../../v10_summarize_fusion_traffic.py)는 stdlib만 사용하는 CPU 도구다. kernel CSV의 Thread_Id를 PID로 오인하지 않고 PID가 있는 counter CSV와 `Dispatch_Id·Agent_Id·Queue_Id·Kernel_Id·Kernel_Name·Start/End_Timestamp·Correlation_Id·Thread_Id`를 모두 정확히 join한다. counter/marker의 PID가 driver PID와 같아야 하며, 명명된 ROCTX range와 전역 pause/resume 사이에 모든 kernel이 들어 있어야 한다. 28/22 launch, metric별 dispatch 누락·중복 없음, finite·비음수·전체 양수, 동일 입력/master/출력/source SHA 관문도 통과했다. 기존 driver와 run result는 수정하지 않았고 각 run의 `traffic_summary.json`을 후처리 정본으로 추가했다.

## 유효한 실측 결과

정본은 [fusion_traffic_comparison_02.json](fusion_traffic_comparison_02.json)이다. 아래 값은 FP16 입력의 **B8/C32/T12800/K3/d1 두-Conv pair forward+backward 1회당**이며 optimizer·전체 epoch가 아니다.

| GL2C/EA 지표 | B bytes/pair | B′ bytes/pair | 감소 bytes/pair | 감소율 |
|---|---:|---:|---:|---:|
| read 요청 `FETCH_SIZE` | 132,801,536 | 93,509,760 | 39,291,776 | 29.5868% |
| write 요청 `WRITE_SIZE` | 52,968,064 | 22,658,816 | 30,309,248 | 57.2217% |
| read+write 합 | **185,769,600** | **116,168,576** | **69,601,024** | **37.4663%** |

네 팔은 각각 warmup 3회 뒤 pair 1회만 수집했으며 read/write가 별도 프로세스다. 입력·dy·FP32 master뿐 아니라 y/dX/master gradient SHA도 네 팔에서 같았다. launch는 read/write 모두 B 28개, B′ 22개였다. 반복 분산은 측정하지 않았고 counter 계측 시간은 속도 표본으로 쓰지 않는다. 기존 kernel 시간·allocator peak·논리 traffic 모델과 이 표를 서로 다른 분모로 유지한다. 논리상 제거한 tensor read/write 78,643,200 B와 실제 GL2C/EA 감소 69,601,024 B가 같아야 한다고 가정하지 않는다.

| 원자료 | 유효 후처리 |
|---|---|
| [FETCH B result](fusion_traffic_FETCH_SIZE_B_02/result.json) | [summary](fusion_traffic_FETCH_SIZE_B_02/traffic_summary.json) |
| [FETCH B′ result](fusion_traffic_FETCH_SIZE_B_prime_02/result.json) | [summary](fusion_traffic_FETCH_SIZE_B_prime_02/traffic_summary.json) |
| [WRITE B result](fusion_traffic_WRITE_SIZE_B_02/result.json) | [summary](fusion_traffic_WRITE_SIZE_B_02/traffic_summary.json) |
| [WRITE B′ result](fusion_traffic_WRITE_SIZE_B_prime_02/result.json) | [summary](fusion_traffic_WRITE_SIZE_B_prime_02/traffic_summary.json) |

CPU summarizer SHA256은 `271b00e60bc401eb2f1eaf2927e1afd900ca96bbc250d1894a86c3e0a6671bd2`, 비교 정본 SHA256은 `7b9d04799ad44e061ec0bce702504650eaab64ac027a82e831259a8a5495bc7a`다. 각 summary는 result와 세 CSV의 SHA 및 정확한 join 항목을 포함한다. 다음은 저장된 결과를 읽기만 하는 CPU 명령이다.

```bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
/usr/bin/python3 -B - <<'PY'
import json
from pathlib import Path
r = json.loads(Path('results/v10/fusion_traffic_comparison_02.json').read_text())
print('valid:', r['valid_for_counters'])
print('scope:', r['memory_scope'])
print('launches:', r['kernel_count_per_pair'])
for metric, values in r['metrics'].items():
    print(metric, values)
PY
```

이 문서를 갱신한 워커는 CPU 파일 조사·집계만 수행했다. availability와 GPU counter 실행은 메인 작업이 소유했다. 설치·다운로드·운영 파일 수정은 없으며, 수집 후 최종 [protected_after.json](protected_after.json)은 1,022개 변경·누락·오류 0, [gpu_idle_after.json](gpu_idle_after.json)은 KFD client 0을 기록한다.
