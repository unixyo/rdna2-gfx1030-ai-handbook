# v0.8 MIOpen 방향별 solver 선택 — 구현과 검증 메모

최초 작성 범위는 CPU 소스/DB 조사와 재사용 하네스였다. **2026-11-08 재개 시 rank_00~04 FAST replay는 완료**했으며, workspace 수정과 최종 합성 replay 상태는 아래 §7을 따른다. GPU 실행은 루트의 직렬 lease로만 수행한다.

## 1. 재사용할 파일과 이미 확인한 차이

- 기준 DB는 루트가 운영 DB를 읽기 전용으로 복사한 `results/applio_v08/db_fast_native/gfx1030_40.HIP.3_6_0_8d1ae90e.ufdb.txt`다.
- 복사본 SHA256: `a5a7fa17513f541a46184072658ebed7abc808ae0370d30914fd485d8e4286b8`, 3,132 key. 운영 원본은 루트 확인 결과 `~/.config/miopen/`에 있다. `.cache/miopen`과 User FindDb를 혼동하지 않는다.
- 기존 v0.6 필터 DB: `results/conv_catalog_v06/hybrid_cache/filtered_db/gfx1030_40.HIP.3_6_0_8d1ae90e.ufdb.txt`, 11,734 bytes. 일반 Conv1D FP16 F의 V4R1 entry만 제거했고 기존 B/W entry는 이미 포함했다.
- `filter_miopen_finddb_v06.py`는 특정 원본 SHA와 forward 제거 규칙이다. 방향별 최적 선택·검증 도구 자체는 아니므로 이번 새 도구와 구분한다.
- `bench_conv_catalog_v06.py`는 fwd+dX+dW 합산 측정이었다. v0.8에서는 세 방향을 각각 시간/오류로 분리한다.
- `applio_epoch_v06.py`의 기본 bench40k·일반 HiFi-GAN pretrain은 이번 voice_model_A·KLM49와 다르므로 루트가 실제 설정을 맞춘다.

## 2. 소스 근거 B

설치 헤더
`applio-lab/.venv-rocm10/lib/python3.12/site-packages/torch/include/ATen/miopen/Descriptors.h`
의 `ConvolutionDescriptor::set`은 `benchmark=true`일 때
`miopenSetConvolutionFindMode(..., miopenConvolutionFindModeNormal)`을 호출한다.
따라서 환경변수 FAST만 기록하고 effective FindMode가 FAST라고 단정하지 않는다.
기존 v0.6 하네스는 benchmark=False를 고정했고, 현재 런타임 CPU import에서 PyTorch 기본값도 False였다.
실제 Linux 학습의 effective 값은 루트 프로파일 결과로 확인한다.

보존된 MIOpen 소스
`results/conv_catalog_v06/source/find_controls.cpp`의
`GetFindModeValue`는 function-template의 `static const`로 환경변수값을 최초에 읽는다.
NORMAL seed와 FAST replay는 별도 프로세스다. Conv마다 환경변수를 바꾸지 않는다.

User FindDb는 선택 후보/기록 시간/workspace/알고리즘이고, PerfDb는 solver tuning 값이다.
레코드 예:

```text
...-NCHW-FP16-F=ConvBinWinogradRxSf2x3g1:0.0804998,0,miopenConvolutionFwdAlgoWinograd;...
```

키와 entry 순서가 시간순이라는 보장은 없다. 새 도구는 entry의 기록 시간을 파싱해 후보 순서만 정하며, 그 시간을 최종 성능이라고 쓰지 않는다.

## 3. 방향 key를 잘못 만들면 엉뚱한 DB를 검증하게 된다

일반 Conv의 F key는 input → output 순서, B/W key는 output → input 순서다.
ConvTranspose의 forward는 MIOpen B, dX는 F, dW는 W에 대응한다.
grouped key에는 `_g4` 같은 suffix가 있다.

새 `v08_solver_select.py:db_key`는 실제 v0.6 normal/grouped/transpose DB와 대조했다.
24개 shape × 3방향 = **72개 key가 모두 일치**, 누락 0.
현재 native 복사본 3,132 key도 파싱에 성공했고 SHA가 위 원본 복사 기록과 같다.
차단할 solver entry는 기존 DB에 24개 들어 있었지만, 런타임의 두 차단 변수는 그대로 유지한다.
이는 CPU 구조 검증이며 GPU 수치/속도 성공이 아니다.

CPU 검증 durable job:
`726e8dd1-c1ab-4bd8-9207-493b80c02f2d`, 종료 코드 0.
PyTorch schema/default benchmark CPU 조회:
`d5cee996-e011-4693-b891-5e657b0caa60`, 종료 코드 0.

## 4. 새 코드의 사용 계약

`v08_solver_select.py`의 CPU subcommand는 torch를 import하지 않는다.

1. `prepare`: NORMAL seed 또는 허용된 연구 DB 사본에서 안전 후보를 정렬한다. rank별로 각 key에 solver 한 개만 남긴 별도 DB를 만든다. `--ranks 0`은 기록된 모든 rank를 포함한다. 두 알려진 오답 solver 이름은 제거한다.
2. `bench`: 부모 GPU lease와 `gpu_guard_v06.GpuAudit` 안에서 새 프로세스의 F/dX/dW를 독립 실행한다. seed 63117, FP16 반올림 입력/가중치/grad-output, FP32 동일 입력 참조, NRMSE ≤ 0.002 및 max error/reference RMS ≤ 0.02를 고정한다. reference 자체 finite도 필수다.
3. `merge`: 유효 audit·동일 runtime/seed/gate·FP32 gate 통과·단일 DB entry 유지가 확인된 후보만 고른다. 기본 최소 speedup 1.05를 event와 wall 모두에 적용한다. 실제 kernel 이름·launch count가 동일한 몇 % 차이는 노이즈로 보고 baseline을 보존한다. 수치 gate는 그대로다. **미측정 key는 baseline DB 선택을 보존**한다.
4. 합성 DB는 새 FAST replay와 실제 epoch A/B를 다시 통과해야 한다. 합성 직후 상태는 `operator_screened_requires_fresh_FAST_replay_and_model_AB`다. 자동 운영 채택은 없다.

`bench --profile`은 방향별 실제 kernel trace를 저장한다.
`bench --log --probe-only`는 방향마다 FP16 Conv를 정확히 한 번 호출해 MIOpen native stderr를 `miopen_stderr.log`에 저장한다. warm·timing·FP32 reference Conv는 실행하지 않으며,
로그 오버헤드를 성능 근거에 섞지 않도록 **valid_for_timing=False**로 표시한다.
`requested_single_solver`는 DB가 요구한 solver 이름이지 native 로그로 확정한 실행 이름이라고 과장하지 않는다.
최종 solver 표는 로그/trace를 함께 대조한다.

환경변수 두 개는 `bench`가 torch import 전에 명시적으로 지정한다.

```text
MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0
MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53=0
```

`NORMAL` seed도 FP32 참조까지 같은 두 변수로 보호한다. 기존 이름 `...OCL_WRW53`은 사용하지 않는다.

## 5. 재현 명령 예시

아래는 실행 예시이며 이 메모 작성 시 GPU 명령은 아직 실행하지 않았다.
출력 경로는 새 폴더를 사용한다. 실제 측정 case JSON은 `bench_conv_catalog_v06.py`와 같은 shape/weight/kind/groups/stride/padding/dilation 필드의 list 또는 `{"cases":[...]}`다.

```bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
PY=~/DevSpace/applio-lab/.venv-rocm10/bin/python
$PY v08_solver_select.py inventory --source results/applio_v08/db_fast_native/gfx1030_40.HIP.3_6_0_8d1ae90e.ufdb.txt
$PY v08_solver_select.py bench --cases results/applio_v08/conv_cases.json --mode NORMAL --db results/applio_v08/solver_seed_db --out results/applio_v08/solver_seed --label normal_seed
$PY v08_solver_select.py prepare --source results/applio_v08/solver_seed_db/gfx1030_40.HIP.3_6_0_8d1ae90e.ufdb.txt --out results/applio_v08/solver_ranks --ranks 0
$PY v08_solver_select.py bench --cases results/applio_v08/conv_cases.json --mode FAST --db results/applio_v08/solver_ranks/rank_00 --out results/applio_v08/solver_rank00 --label rank00 --profile
```

baseline에는 current DB의 **별도 연구 사본**을 사용한다. 위 rank별 bench 결과를 받은 다음:

```bash
$PY v08_solver_select.py merge --baseline results/applio_v08/solver_baseline/result.json --candidate results/applio_v08/solver_rank00/result.json --base-db results/applio_v08/db_fast_native/gfx1030_40.HIP.3_6_0_8d1ae90e.ufdb.txt --out results/applio_v08/solver_selected_db
```

롤백은 연구 프로세스를 끝내고 연구 DB override를 넘기지 않는 것이다. 운영 파일/전역 DB를 교체하거나 캐시를 삭제하지 않는다.

## 6. 한계와 다음 검증

- 단일 entry를 요구한 DB가 실제 실행에서 사용됐는지는 native 로그와 방향별 kernel 이름으로 확인한다. fallback이 있으면 그 후보 이름으로 성과를 기록하지 않는다.
- 3 rank만 실행하면 모든 MIOpen solver를 탐색한 것이 아니다. `--ranks 0`도 해당 NORMAL seed가 발견한 후보 범위다.
- isolated dX/dW의 성능이 full autograd backward에서 그대로 유지되는지는 합성 replay·epoch에서 확인한다.
- kernel time, event time, CPU operator attribution을 서로 더하지 않는다.
- native 비교도 finite/gate가 실패하면 후보 선택을 멈춘다. 참조 gate를 완화하지 않는다.

## 7. 2026-11-08 재개 — rank 결과와 workspace 0 원인

### 완료된 rank FAST replay

`results/applio_v08/solver_rank00`부터 `solver_rank04`까지 5개 result.json은 모두
`status=completed`, `valid_for_timing=true`다. 재실행 없이 이 결과를 사용한다.

| rank | case 수 | 검사한 방향 | singleton DB 후보 방향 | 정확도 통과 |
|---:|---:|---:|---:|---:|
| 00 | 23 | 69 | 69 | 69/69 |
| 01 | 18 | 54 | 42 | 54/54 |
| 02 | 12 | 36 | 36 | 36/36 |
| 03 | 12 | 36 | 12 | 36/36 |
| 04 | 9 | 27 | 9 | 27/27 |

총 222방향에는 missing-rank native fallback control 54방향이 포함된다.
유효한 singleton 후보는 168방향이다. `solver_selected_rank47_db/selection_manifest.json`의
첫 merge는 6방향을 고른다. 이 6방향의 연산 speedup 범위는 1.0716–1.5173배다.
최종 DB는 workspace seed 후보까지 합친 뒤 새 FAST replay로 검증한다.
이 값들은 아직 전체 epoch 배속이나 운영 채택 결과가 아니다.

### workspace 0의 소스 원인

설치 PyTorch의 `torch/version.py` commit은 `c9ab269368d46a826d64705192fef55594c8b698`다.
해당 공식 `Conv_miopen.cpp`는 세 방향 모두 GetWorkSpaceSize를 조회한 뒤 그 크기를 할당하고
Find에 전달한다. 호출부가 언제나 workspace를 생략한다는 설명은 이 코드와 맞지 않는다.

MIOpen DB key의 commit `8d1ae90e`와 같은 공식 소스를 보존했다.
`convolution.cpp:GetWorkSpaceSizeGEMM`은 **dilation 중 하나라도 1보다 크면 0을 반환**한다.
반면 같은 구성의 `GemmFwdRest`, `GemmBwdRest`, `GemmWrwUniversal`은 실제 Find 시
workspace를 요구한다. 기존 NORMAL seed log의 required 2,457,600–9,011,200 bytes /
provided ptr 0, size 0 경고와 일치한다. FP32 reference의 요구량은 두 배다.

같은 commit의 `GetWorkSpaceSizeRange`는 활성화되고 applicable한 convolution solver 전체를
순회하여 workspace의 최소·최대를 직접 구한다. 설치 `libMIOpen.so.1`에
Forward/BackwardData/BackwardWeights의 Range C API 세 심볼이 export된 것도 `nm`으로 확인했다.

공식 원자료와 파일별 URL·크기·SHA는
`results/applio_v08/source/miopen_workspace/source_manifest.json`에 있다.
다운로드는 소스 7개, 합계 **621,004 bytes**다. 운영 패키지를 설치하거나 교체하지 않았다.

### 연구 프로세스에만 적용하는 수정

`v08_miopen_workspace.c`와 연구 폴더의 `libv08_miopen_workspace.so`는 workspace query 3개만
interpose한다. 다음 두 조건을 모두 만족할 때만 반환값을 `max(original, Range.max)`로 바꾼다.

- `APPLIO_RDNA2_FIX_FIND_WORKSPACE=1`
- `MIOPEN_FIND_MODE=NORMAL` 또는 `1`

그 외에는 원래 함수의 결과를 그대로 반환한다. 계산 커널·tensor·학습 코드는 바꾸지 않는다.
`V08_WORKSPACE` log는 방향, 원래 bytes, Range 최소·최대, 최종 반환 bytes를 남긴다.
Range 조회 실패나 심볼 누락은 성공으로 무시하지 않고 MIOpen 오류를 반환한다.
FAST replay와 학습에서는 LD_PRELOAD와 opt-in 변수를 주지 않는다.

CPU `cc -std=c11 -O2 -fPIC -shared -Wall -Wextra -Werror` 컴파일과 export 확인은 통과했다.
**GPU smoke도 완료**했다. `solver_workspace_smoke1/result.json`은 completed,
GPU audit clean(관측 foreign 0), fwd/dX/dW 정확도 3/3 통과다.
FP16 세 방향 모두 original 0 → Range.max/returned **5,734,400 bytes**였고,
같은 FP32 참조는 **11,468,800 bytes**였다. MIOpen 실제 Find 입력과도 일치한다.
native log의 중간 이후 `578000`과 `af0000`은 hex 표기이며 각각 이 두 byte 수다.
`IsEnoughWorkspace` 경고는 0건이다.

| smoke 방향 | NRMSE | max/reference RMS | 생성된 FP16 FindDb entry |
|---|---:|---:|---|
| fwd | 0.000207731 | 0.00193927 | GemmFwdRest |
| dX | 0.000293896 | 0.00242476 | GemmBwdRest |
| dW | 0.000440136 | 0.00372423 | GemmWrwUniversal, ConvHipImplicitGemmV4R1WrW |

나머지 23개 NORMAL seed도 종료 코드 0, 정확도 69/69, audit clean으로 완료했다.
smoke와 합쳐 **24 shape × 3방향 = 72/72** 통과, workspace 경고는 양쪽 모두 0이다.
`solver_workspace_seed_summary.json`에 원자료 SHA와 합계를 보존했다.
새 DB는 FP16 72개 key·90개 entry와 FP32 reference 72개 key를 포함하며,
이전에 누락된 FP16 72방향이 모두 생성됐다.
NORMAL seed에서 보고된 시간은 최종 성능 선택에 쓰지 않는다.

`solver_workspace_ranks24`와 `solver_workspace_rank_cases24/plan.json` 준비까지 완료했다.
shim 없는 별도 FAST replay의 대상은 다음과 같다.

| workspace rank | case 수 | singleton 후보 방향 | 측정 전체 방향 |
|---:|---:|---:|---:|
| 00 | 24 | 72 | 72 |
| 01 | 18 | 18 | 54 |

후보는 총 90방향이며 rank01의 나머지 36방향은 fallback control이다.
추가 FAST replay도 완료했다. rank00은 72/72, rank01은 54/54 정확도 통과이며
둘 다 `valid_for_timing=true`, GPU audit clean이다. 기존 5개와 추가 2개, 총 7개 rank를
merge한 결과 추가 dilated solver 채택은 0개이고 기존 6방향을 유지했다.
`solver_selected_final_db`의 새 FAST replay 결과는 아래 §8과 같다.

작업 폴더 rdna2-ops에서 root GPU lease로 1개 smoke를 실행한다.

```bash
env APPLIO_RDNA2_FIX_FIND_WORKSPACE=1 \
 LD_PRELOAD=~/DevSpace/rdna2-pro-research-20261102/rdna2-ops/results/applio_v08/source/miopen_workspace/libv08_miopen_workspace.so \
 ~/DevSpace/applio-lab/.venv-rocm10/bin/python \
 v08_solver_select.py bench \
 --cases results/applio_v08/solver_workspace_smoke1.json \
 --mode NORMAL --db results/applio_v08/solver_workspace_seed_db24 \
 --out results/applio_v08/solver_workspace_smoke1 \
 --label normal_explicit_range_smoke --warmup 1 --rounds 1 --reps 1 --log
```

smoke는 C64/T6400/K7/d3의 FP16 결과와 FP32 참조 세 방향을 모두 검사했다.
native 로그의 실제 Find workspace와 V08_WORKSPACE 반환값을 대조했다.
`--log`이므로 `valid_for_timing=false`인 것이 의도된 동작이며 성능 후보로 merge하지 않는다.

같은 opt-in·DB 설정에 `--cases results/applio_v08/solver_workspace_remaining23.json`,
새 출력 `--out results/applio_v08/solver_workspace_seed23`으로 나머지 23개를 이어서 seed한다.
전체 24개 목록은 `solver_workspace_cases24.json`이고, smoke와 나머지 목록은 겹치지 않는다.
seed 결과는 singleton rank DB로 다시 나눈 뒤 **shim 없는 별도 FAST 프로세스**에서
정확도·실제 kernel·event/wall 시간을 검증한 것만 기존 merge에 추가한다.

## 8. 최종 통합 DB FAST replay — 연산 검증 완료

원자료는 `results/applio_v08/solver_selected_final_replay/result.json`이며,
기준은 `solver_baseline144/result.json`, 선택 내역은
`solver_selected_final_db/selection_manifest.json`이다.
새 FAST 프로세스에서 FP16 카탈로그 144개 구성 × 3방향 = **432/432 정확도 통과**,
`status=completed`, `valid_for_timing=true`, GPU audit clean을 확인했다.
workspace shim과 opt-in 없이 실행한 replay다.

| 선택 구성 / 방향 | 현재 운영 DB ms | 최종 DB ms | event 배속 | wall 배속 |
|---|---:|---:|---:|---:|
| Conv1D C128/T3200/K7/d1, dX | 0.489507 | 0.455531 | 1.0746× | 1.0751× |
| Conv1D C256/T320/K11/d1, dW | 0.387294 | 0.349765 | 1.1073× | 1.1013× |
| Conv1D C256/T320/K7/d1, dW | 0.313029 | 0.262761 | 1.1913× | 1.1908× |
| Conv2D C128/H712/W2→O512/K5×1/s3×1, dW | 0.598203 | 0.452818 | 1.3211× | 1.3211× |
| ConvTranspose1D C256/T320→O128/K16/s10, dW | 0.397647 | 0.310295 | 1.2815× | 1.2810× |
| ConvTranspose1D C512/T32→O256/K16/s10, dW | 0.191433 | 0.125158 | 1.5295× | 1.5210× |

DB에 선택한 solver ID는 앞의 한 방향 `GemmBwdRest`, 다음 두 방향 `GemmWrwUniversal`,
마지막 세 방향 `ConvHipImplicitGemmV4R1WrW`다. 이는 DB 선택과 실제 kernel 검증의 근거이며,
native 로그에서 ID를 직접 관측한 것과 구분한다(§9).
6방향 모두 singleton DB entry가 실행 전후 유지됐고, 실제 kernel 이름·launch count가
선택된 개별 rank 때와 같으며 기존 기준선과 달랐다.
최종 replay의 선택 6방향은 모두 event와 wall에서 1.05배 이상 이득을 유지했다.
선택 6방향 중 최악 NRMSE는 **0.000441372**, max/reference RMS는 **0.00539934**로 gate 안이다.

선택되지 않은 control 중 가장 큰 시간 증가는 같은 kernel의 Conv2D
C1/H1164/W11→O32/K5×1/s3×1 fwd에서 0.108476→0.125260 ms(+0.016784 ms)였다.
그 밖의 상위 시간 증가도 실제 kernel 구도가 유지된 항목이며, 선택 6방향의 회귀는 없었다.
한 번씩 호출한 방향 시간의 단순 합이나 이러한 control 변동을 전체 학습 배속으로 환산하지 않는다.
이 절의 판정 범위는 **선택 solver DB의 연산 통과**다. 후속 실제 epoch·100 epoch 진행 상태는
`results/applio_v08/APPLIO_V08_KO.md`를 정본으로 삼는다.

## 9. 전체 432방향의 실제 solver ID 보완

이 절의 짧은 결과 경로는 `results/applio_v08/` 기준이다. 실행 명령은 `rdna2-ops`에서 사용한다.

### 9.1 기존 증거의 범위

원 지시 §2.1·§5의 실제 solver·시간 표를 확인하다가
`solver_roofline/shape_direction.csv`의 432행 모두가
`pending_native_log_crosscheck`였음을 확인했다. 방향별 시간·오류·실제 kernel은 보존돼 있으나,
baseline·final replay·rank 실행은 모두 `logging_enabled=false`여서 runtime solver ID 로그는 없었다.
`requested_single_solver`가 있는 129행도 DB 요청일 뿐 관측 ID로 승격하지 않는다.

CPU 원자료 집계:

| 기존 근거 | 방향 수 | 의미 |
|---|---:|---|
| 실제 kernel trace | 432 | Winograd 126, GEMM 306 |
| 실제 kernel 계열과 호환하는 User FindDb 후보 1개 | 183 | 후보가 하나인 것은 실제 runtime 선택 증명이 아님 |
| 호환하는 User FindDb 후보 복수 | 126 | Winograd ID 후보를 모두 보존 |
| 해당 계열 User FindDb 기록 없음 | 123 | FAST fallback 가능. 실행 solver 부재라는 뜻이 아님 |
| singleton rank와 실제 kernel 이름·launch count가 같은 방향 | 141 | 요청 ID와 kernel의 대응 실험 근거 |
| 위 141 중 서로 다른 요청 ID가 같은 kernel을 낸 방향 | 7 | `ConvBinWinogradRxSf2x3` / `ConvBinWinogradRxSf2x3g1` 구별 불가 |
| 기존 FAST native 로그로 직접 확인한 ID | 0 | 별도 metadata probe 필요 |

공식 보존 소스 `source/miopen_workspace/mlo_dir_conv.cpp`의 34–46행은 여러 GEMM wrapper,
100–109행과 157–165행은 여러 Winograd solver를 별도로 등록한다.
이는 후보가 복수임을 보여 주며 Tensile kernel명이나 f2x3 kernel명 하나를 전역 유일한 solver ID로
해석할 근거가 아니다. 특히 위 7방향은 실제 rank kernel이 같아서 이름으로 둘을 합칠 수 없다.
소스 commit·URL·SHA는 `source/miopen_workspace/source_manifest.json`에 있다.

`shape_direction.csv`와 `roofline.json`에 `observed_solver_status`,
`compatible_db_solver_ids`, `solver_candidate_status`, `same_kernel_rank_requested_solvers`,
`rank_request_evidence`, `solver_evidence_scope`, `solver_ambiguity_reason`를 분리했다.
원래 시간·정확도 값과 `requested_single_solver`는 그대로 보존하고 원본 사본도
`*.before_solver_attribution.*`로 남겼다. 집계·행별 근거는 `solver_attribution.json`에 있다.

### 9.2 B100 종료 뒤 완료한 최소 native metadata probe

GPU 실행은 root만 한다. CPU에서 다음 입력을 준비했다.

- `solver_roofline/baseline_metadata_cases144.json`: baseline의 144 case와 원래 `conv_cases.json`이 전부 같음을 확인. 물리 연산은 FP16이며 논리 FP32 경계 metadata를 보존.
- `solver_roofline/baseline_metadata_db/`: baseline DB와 .time의 새 사본. DB 714,381 bytes, SHA `a5a7fa17513f541a46184072658ebed7abc808ae0370d30914fd485d8e4286b8`.
- `solver_roofline/baseline_metadata_probe_plan.json`: 정확한 argv·입력 SHA·새 출력 경로와 실행 전 읽을 코드 구간.

root가 `rdna2-ops`에서 아래 명령을 B100 종료 뒤 한 번 실행해 종료 코드 0으로 완료했다.
완료 DB·timing 결과는 덮어쓰지 않았다. 아래는 실행 기록이며 같은 출력으로 재실행하지 않는다.
native stderr는 harness가 새 output의 `miopen_stderr.log`로 분리한다.

~~~bash
env -u LD_PRELOAD -u APPLIO_RDNA2_FIX_FIND_WORKSPACE \
  ~/DevSpace/applio-lab/.venv-rocm10/bin/python \
  v08_solver_select.py bench \
  --cases results/applio_v08/solver_roofline/baseline_metadata_cases144.json \
  --db results/applio_v08/solver_roofline/baseline_metadata_db \
  --out results/applio_v08/solver_roofline/baseline_metadata_probe144 \
  --cache results/applio_v08/miopen_cache \
  --mode FAST --seed 63117 --label baseline144_native_solver_metadata \
  --log --probe-only
~~~

확인할 코드는 `v08_solver_select.py` 286–335(env·dtype), 337–415(FP16 생성·3방향 호출·marker·reference 생략),
434–452(audit·timing 무효 표시), 558–590(CLI)다.
`--probe-only`는 방향마다 한 번, **144 × 3 = 432회**만 호출한다.
warmup·timed rounds·profiler·FP32 reference 호출은 0회다.
두 안전 변수를 torch import 전에 설정하고 `GpuAudit` 안에서 직렬 실행한다.
관측 probe는 의도적으로 `valid_for_timing=false`이며 기존의 유효한 event·wall 시간을 유지한다.

### 9.3 로그 파싱과 FP32·복수 후보 구별

CPU 도구 `solver_roofline/update_solver_attribution.py`는 각
`V08_DIRECTION_BEGIN`–`V08_DIRECTION_READY` 사이만 읽는다.
`FillFindReturnParameters`의 `FW/BWD/BWrW Chosen Algorithm`과 뒤따른 실제 convolution API 호출을
대조한다. ConvTranspose의 fwd는 BWD, dX는 FW다.
`EvaluateInvokers Selected`는 알고리즘별 후보이므로 최종 실행 ID로 사용하지 않는다.

실제 NORMAL smoke를 parser에 입력해 FP16 구간의 최종 Chosen 3개를 읽었고,
구간 밖 FP32 reference의 Chosen 3개와 알고리즘별 Selected 4개를 제외했다.
dW에서 2279행의 implicit GEMM Selected와 2665행의 최종 GemmWrwUniversal Chosen이 다른 사례다.
이 smoke 관측을 FAST baseline의 관측으로 옮기지는 않는다.

probe 완료 뒤 다음 CPU 명령을 실행해 보완했다.

~~~bash
python3 results/applio_v08/solver_roofline/update_solver_attribution.py \
  --probe results/applio_v08/solver_roofline/baseline_metadata_probe144
~~~

도구는 완료·audit clean·FP16·FAST·동일 seed·안전 변수·runtime·DB SHA·144개 descriptor와
432개 고유 log block을 먼저 확인한다. event/wall은 null, reference gate는 미측정이어야 한다.
각 block에 올바른 물리 방향의 Chosen ID가 하나이고 실제 API 호출이 있을 때만
`confirmed_native_metadata_replay`로 채운다.
불일치·복수 ID·누락은 `ambiguous_or_missing_native_log`와 이유를 남긴다.
이 ID는 같은 조건의 **별도 metadata replay 관측**이며 과거 timed 호출 자체의 로그라는 뜻은 아니다.
baseline 표와 최종 선택 DB 표의 범위도 구분한다.

### 9.4 최종 native 관측 결과 — 432/432 확정

`baseline_metadata_probe144/result.json`은 `status=completed`,
FP16·FAST·seed 63117·동일 runtime·안전 변수·144개 descriptor를 확인했다.
432개 고유 방향 구간 각각에서 **올바른 물리 방향의 최종 Chosen Algorithm 1개와
실제 convolution API 호출 1개**가 대응했다. FP32 problem 혼입·복수 ID·누락은 0개다.
모든 출력 finite, GPU audit clean이며 timing은 의도대로 false다.
reference gate와 event/wall은 미측정으로 유지했다.

| 실제 관측 solver ID | 방향 수 |
|---|---:|
| `ConvBinWinogradRxSf2x3g1` | 69 |
| `ConvBinWinogradRxSf3x2` | 37 |
| `ConvBinWinogradRxSf2x3` | 20 |
| `GemmFwdRest` | 73 |
| `GemmBwdRest` | 75 |
| `GemmWrwUniversal` | 83 |
| `GemmFwd1x1_0_1` | 28 |
| `GemmBwd1x1_stride1` | 29 |
| `GemmWrw1x1_stride1` | 18 |
| **합** | **432** |

CSV·roofline JSON의 상태는 **432/432 `confirmed_native_metadata_replay`, 미확정 0**이다.
기존 후보가 복수였던 항목도 runtime 로그로 구별했다. 같은 rank kernel 때문에 모호했던
7방향은 실제 관측 결과 f2x3 5개 / g1 2개였다.
호환 User FindDb 기록이 없던 123방향도 각각 실제 ID를 확인했다.
후보가 있던 행에서 관측 ID가 해당 후보 목록과 충돌한 경우는 0개다.

원 baseline DB, 새 probe DB의 실행 전·후 SHA는 모두
`a5a7fa17513f541a46184072658ebed7abc808ae0370d30914fd485d8e4286b8`로 같다.
native log SHA는 `708d75f3f1af455bc3df6350ee25f9c32dfebef186298f843a799351c46b6456`,
probe result SHA는 `c8072a2e0e4de728b5d8f0111ea63d8ff2f0b8a33595c56b9b64448ccc41e248`다.
원래 432행의 event·wall·오류·roofline·requested 필드가 보존 사본과 같음을 별도 확인했다.
이번 약 9.5초의 실행 시간은 관측 작업의 경과이며 성능 비교에 사용하지 않는다.

정확한 증거 범위는 **동일 조건의 별도 FAST metadata replay에서 관측한 baseline solver ID**다.
과거 timed 호출 자체의 로그나 최종 선택 DB의 새 native ID 로그로 확대하지 않는다.
전체 행의 로그 경로·line·방향·API는 `solver_roofline/solver_attribution.json`에 보존했다.
