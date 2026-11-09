# exact warm 프로파일의 packing 관련 명령 계열

작성: 2026-11-09 UTC. GPU 추가 실행 없이 기존 `exact_warm_profile_isa.json`을 집계했다. 원본 JSON SHA256은 `fdd365643df11c8925ca6f27402926da7c4fd22e61d6d2ba2ec5bee90d142c9e`이다.

아래 값은 최종 ISA 코드에 들어 있는 **명령 계열별 정적 개수**다. **packing만 분리한 실행 비용이나 실행 시간은 아니다.** 주소 계산, unpack, clamp 및 기타 연산이 섞여 있다. 정적 코드 개수에는 루프 반복 수, 활성 lane 수, 경계 분기와 실행 빈도를 곱하지 않았다. 특히 network pass 7/8/9의 동적 실행량을 이 표의 정적 개수와 혼동하지 않는다.

## 주요 15개 pass

1080p Native와 4K Native의 아래 열은 15개 pass 모두 일치한다. 표 하나를 두 해상도에 공통으로 적용한다. padding 전용 dispatch 2, 4, …, 26은 이 주요 pass 표에서 제외했다.

| Pass | dispatch | med3_i32 | ashrrev_i32 | 논리 shift 계열 | AND/OR/XOR 계열 | SDWA | perm | cvt_pk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exposure/SPD | 0 | 0 | 0 | 56 | 28 | 44 | 0 | 0 |
| pre | 1 | 16 | 0 | 21 | 6 | 16 | 0 | 0 |
| network 1 | 3 | 64 | 64 | 18 | 0 | 80 | 0 | 0 |
| network 2 | 5 | 64 | 64 | 18 | 0 | 80 | 0 | 0 |
| network 3 | 7 | 32 | 0 | 3 | 0 | 24 | 0 | 0 |
| network 4 | 9 | 112 | 96 | 18 | 0 | 132 | 0 | 0 |
| network 5 | 11 | 112 | 96 | 18 | 0 | 132 | 0 | 0 |
| network 6 | 13 | 64 | 0 | 3 | 0 | 48 | 0 | 0 |
| network 7 | 15 | 12 | 8 | 3 | 0 | 13 | 0 | 0 |
| network 8 | 17 | 12 | 8 | 3 | 0 | 13 | 0 | 0 |
| network 9 | 19 | 12 | 16 | 13 | 0 | 28 | 0 | 0 |
| network 10 | 21 | 112 | 112 | 34 | 0 | 148 | 0 | 0 |
| network 11 | 23 | 176 | 80 | 49 | 98 | 228 | 0 | 0 |
| network 12 | 25 | 64 | 64 | 18 | 0 | 80 | 0 | 0 |
| post | 27 | 64 | 64 | 79 | 10 | 64 | 0 | 0 |

## 원본 JSON key와 집계 정의

원본 경로는 `analysis/exact_warm_profile_isa.json`이다. 각 행은 `shaders[]`의 한 원소이며, `case.size`로 해상도, `pipeline`으로 위 dispatch 번호를 구분한다. 아래에서 `o`는 해당 원소의 `opcodes` 사전이다. 없는 key의 값은 0으로 센다.

| 표의 열 | 정확한 JSON key 또는 집계식 |
|---|---|
| med3_i32 | `o.get("v_med3_i32", 0)` |
| ashrrev_i32 | `o.get("v_ashrrev_i32", 0)` |
| 논리 shift 계열 | `v_add_lshl_u32`, `v_lshl_add_u32`, `v_lshl_or_b32`, `v_lshlrev_b32`, `v_lshrrev_b32`, `v_lshrrev_b64` 여섯 key의 합 |
| AND/OR/XOR 계열 | key가 `v_and`, `v_or`, `v_xor` 중 하나로 시작하는 모든 항목의 합. 관측된 key는 `v_and_b32`, `v_and_or_b32`, `v_or_b32`; XOR는 0 |
| SDWA | 원소의 `sdwa_instructions` |
| perm | `o.get("v_perm_b32", 0)` |
| cvt_pk | key가 `v_cvt_pk`로 시작하는 모든 항목의 합. 관측된 항목 없음 |

논리 shift 계열은 단순 shift 외에 add/OR와 융합된 명령도 포함한다. `v_add_lshl_u32`는 exposure와 post에 각각 12개다. 엄격히 `v_lshl*`/`v_lshr*` 접두사만 집계하는 경우 두 행은 각각 **44/67**이 된다. 표에서는 이 융합 shift를 누락하지 않도록 **56/79**로 표기했다.

`v_and_or_b32`는 AND와 OR를 함께 수행하지만 명령 한 개로 센다. `v_lshl_or_b32`는 논리 shift 열에만 포함하며 AND/OR/XOR 열에 중복 포함하지 않는다. 이 분류는 opcode 이름을 기준으로 하므로, OR를 수행하는 모든 명령의 의미론적 개수와 동일하지는 않다.

`parse_isa.py`의 opcode 정규화는 마지막 `_e32`, `_e64`, `_sdwa` 접미사를 제거한다. SDWA 열은 별도 opcode 이름이 아니라, 원본 명령 word에서 VOP1/VOP2/VOPC 형식과 Src0 값 249를 확인한 개수다. 따라서 **SDWA 열은 다른 opcode 열과 겹칠 수 있으며, 모든 열을 더해 총 명령 개수로 사용하면 안 된다.**

`v_med3_f32`는 med3_i32 열에 포함하지 않는다. post에 12개가 있다. `v_max_i32`, scalar shift/bitwise, bitfield 등도 위 열에 자동 포함되는 것이 아니다. 따라서 이 표는 모든 비-dot 연산을 빠짐없이 합친 표가 아니다.

## packing 비용과의 구분

- 이 표로 확인할 수 있는 것: 최종 코드의 shift·clamp·bitwise 계열 및 SDWA 사용량, perm/cvt_pk 사용 여부.
- 이 표만으로 확인할 수 없는 것: packing에만 귀속되는 명령 개수, 각 계열의 동적 실행 횟수·지연·GPU 시간, 전체 시간에서 packing이 차지하는 비율.
- 실제 최적화의 근거: `CANDIDATE_ISA_COMPARISON.json`과 대응 문서의 같은 PSO 전후 machine-code 차이, 출력 바이트 비교, 별도의 전체 업스케일러 시간 비교를 함께 사용한다. 위 계열 수가 감소한 것을 같은 비율의 시간 감소로 바꾸지 않는다.

원본 RADV 역어셈블의 일부 SDWA/SMEM 경계 오류는 `parse_isa.py`에서 raw word로 복원했다. 코드 크기와 ACO instruction 통계의 일치를 확인했으며 원본·정규화 assembly와 raw byte를 모두 보존했다. PSO와 vkd3d hash는 DXBC SHA256으로 연결된다. ISA block과 PSO의 순서 기반 대응은 post/all override 시 예상한 2/20개 block만 바뀌고 나머지 54/36개가 byte 동일했던 실험으로 추가 뒷받침된다. 자세한 근거는 `ISA_FINDINGS_KO.md`, `CANDIDATE_ISA_COMPARISON_KO.md`에 있다.

## 재현 명령

아래 명령은 GPU를 사용하지 않는다. 각 해상도별 15개 행을 출력하고 표의 열이 두 해상도에서 같은지도 확인한다.

```bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops/results/fsr4
python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path('analysis/exact_warm_profile_isa.json').read_text())
shift_keys = ('v_add_lshl_u32', 'v_lshl_add_u32', 'v_lshl_or_b32',
              'v_lshlrev_b32', 'v_lshrrev_b32', 'v_lshrrev_b64')
by_case = {}
for row in data['shaders']:
    dispatch = row['pipeline']
    if dispatch != 0 and dispatch % 2 == 0:
        continue
    o = row['opcodes']
    counts = (
        o.get('v_med3_i32', 0),
        o.get('v_ashrrev_i32', 0),
        sum(o.get(k, 0) for k in shift_keys),
        sum(v for k, v in o.items() if k.startswith(('v_and', 'v_or', 'v_xor'))),
        row['sdwa_instructions'],
        o.get('v_perm_b32', 0),
        sum(v for k, v in o.items() if k.startswith('v_cvt_pk')),
    )
    by_case.setdefault(row['case']['size'], {})[dispatch] = counts
    print(row['case']['size'], dispatch, *counts, sep=',')
assert len(by_case) == 2
first, second = by_case.values()
assert len(first) == len(second) == 15 and first == second
print('Verified: both resolutions have identical 15-pass counts.')
PY
```

## resource 표에 함께 붙일 subgroup 각주

권장 표기는 **RADV 자원 계산값(32-lane 환산)** 이다. `aco_stats["Subgroups per SIMD"]`를 그대로 보존한 값이며 실제 resident wave 수나 달성 occupancy의 관측값이 아니다.

Mesa 26.2.4 `radv_get_max_waves`는 초기값 16에 VGPR/LDS 등의 `MIN2` 자원 제한을 적용하고, GFX10 이상에서 마지막에 `wave_size / 32`를 곱해 반환한다. wave64 해석에서 로그 18/20/24는 내부 계산값 9/10/12를 32-lane 단위로 환산한 값이다. **최종 환산값에 16을 다시 상한으로 적용하지 않는다.** 그러므로 “uncapped”라고만 쓰면 하드웨어/자원 제한을 전혀 적용하지 않았다고 오해할 수 있다. “최종 환산값은 16으로 다시 제한하지 않으며 실제 resident occupancy를 나타내지 않는다”라고 구체화한다.

근거: [Mesa 26.2.4 radv_shader.c](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/vulkan/radv_shader.c)의 `radv_get_max_waves`, [ac_gpu_info.c](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/common/ac_gpu_info.c)의 GFX10.3 이상 초기값 16. 파일 SHA는 기존 `ISA_METHOD_KO.md`와 `ISA_ADDITIONAL_SOURCE_FETCHES.json`에 보존한 값과 동일하게 재확인했다.

