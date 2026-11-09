# FSR4 RDNA2 ISA/ACO 분석 방법
작성: 2026-11-10 (Asia/Seoul). 범위: CPU와 원문 확인. 이 파일 작성 시 GPU 실행은 하지 않았다.

## 확인된 사실
- AMD RDNA2 ISA에는 signed packed-byte dot4와 destination-accumulate dot4c, v_med3_i32, v_perm_b32, v_cvt_pk_i16_i32, SDWA가 있다. signed/unsigned opcode와 saturating clamp를 구분해야 한다. 정수 dot4 1회는 multiply 4회+add 4회, 즉 이 보고서에서는 8 integer ops로 센다. RDNA2 ISA §6.5, §6.8, §12.8, §12.10, §12.12, §12.19가 근거다. [AMD 공식 ISA](https://docs.amd.com/api/khub/documents/Et~wpu9g~Ffl7d9q0QZ~Og/content)
- Mesa 26.2.4의 ACO는 sdot_4x8_iadd를 gfx1030에서 v_dot4_i32_i8로 선택한다. saturating NIR opcode는 같은 명령의 clamp를 켠다. 일반 opcode에는 clamp가 꺼져 있다. 이는 실제 FSR 셰이더가 이 opcode로 내려갔다는 증거와 구분한다. [해당 버전 ACO 소스](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/compiler/instruction_selection/aco_select_nir_alu.cpp)
- 같은 버전 optimizer는 pre-GFX11에서 상수 경계의 min/max clamp를 v_med3_i32로 결합하는 규칙을 이미 갖고 있다. 단순한 clamp 표기 변경만으로 새 이득을 예상하지 않는다. [ACO optimizer](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/compiler/aco_optimizer.cpp)
- NIR pack_sint_2x16은 두 signed32를 signed16 범위로 포화한 뒤 패킹한다. ACO는 이를 v_cvt_pk_i16_i32로 매핑한다. pack_32_4x8은 byte 성분을 조립하는 형태로 처리한다. [NIR opcode 정의](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/compiler/nir/nir_opcodes.py)
- upstream model_clamp.py는 rounding 뒤 clamp를 shift 앞으로 옮기고, shift=8이면 byte1을 직접 선택한다. model_tail.py는 정확한 power-of-two scaling을 정수화한다. 이미 exact 기준선에 포함되는 최적화다. [커뮤니티 설명](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/blob/d1c3b30db656bf6a38634642542ee7736127d211/docs/how-it-works.md)

## 덤프 명령과 해석
Mesa 26.2.4 radv_instance.c에서 nocache, shaders, shaderstats, cs, asm 플래그를 직접 확인했다. 덤프 전용 실행에서는 환경 변수 RADV_DEBUG=nocache,cs,asm,shaderstats를 사용하면 compute disassembly와 통계를 얻을 수 있다. 전체 NIR/ACO IR까지 필요하면 RADV_DEBUG=nocache,shaders,shaderstats를 사용한다. 덤프 로그를 timing 측정에 섞지 않는다.
[26.2.4 플래그 테이블](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/vulkan/radv_instance.c)

vkd3d-proton의 VKD3D_SHADER_DUMP_PATH는 hash별 SPIR-V/DXIL/DXBC를 내보낸다. VKD3D_SHADER_OVERRIDE는 같은 hash의 .spv 파일을 읽는다. DXIL 파일만 override 폴더에 두면 적용되지 않는다. 기존 dump 파일은 덮어쓰지 않으므로 run별 새 디렉터리를 쓴다.
[vkd3d-proton README](https://github.com/HansKristian-Work/vkd3d-proton/blob/master/README.md)

커뮤니티 prebuilt는 4.1.1.2740 해시·번역 레이아웃에서 만들어졌다. 체크 스크립트와 이 PC의 실제 dump를 대조해야 한다. RX6900XT에서 어두워지는 사례는 root-constant layout 차이가 관련되어 있고, 같은 시스템의 dump로 다시 만든 override에서 정상화됐다는 보고다.
[Linux 적용 문서](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/blob/d1c3b30db656bf6a38634642542ee7736127d211/docs/linux.md)

RGP가 필요하면 MESA_VK_TRACE=rgp, window 없는 compute 작업이면 MESA_VK_TRACE_PER_SUBMIT=1을 사용할 수 있다. trace 실행은 별도 표본으로 취급한다. trace 시 profiling pstate에 영향을 줄 수 있어 일반 timing과 직접 혼합하지 않는다.
[Mesa 환경 변수 문서](https://docs.mesa3d.org/envvars.html)

## 통계를 읽을 때의 기준
| 항목 | 해석 |
|---|---|
| VGPR | 사용 수와 allocation 수를 구분한다. wave32는 먼저 8개 단위 정렬을 하고 gfx1030의 실제 granularity를 추가 적용한다. |
| SGPR | gfx10 이상에서는 radv_get_max_waves가 SGPR 수로 occupancy를 제한하지 않는다. |
| Max Waves | 정적 자원 제한 계산값이다. gfx10 이상 반환값은 wave_size/32로 정규화되므로 실제 달성 occupancy와 구분한다. |
| LDS | gfx1030 allocation granularity는 1024 bytes. hardware register encoding 단위 512 bytes와 혼동하지 않는다. |
| Scratch | wave당 private scratch allocation. LDS로 spill하거나 SGPR을 VGPR로 spill하는 경우까지 'scratch=0이면 spill=0'으로 읽지 않는다. |
| ACO latency/throughput | compiler 내부 추정치다. GPU 실측 latency나 실측 dot4 상한으로 기재하지 않는다. |

근거: [radv_shader.c](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/vulkan/radv_shader.c), [ac_shader_util.h](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/common/ac_shader_util.h), [aco_statistics.cpp](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/compiler/aco_statistics.cpp).

## 정적 명령 수와 이론 하한
아래는 실제 pass ISA를 판정할 때 사용할 분석 규칙이다. 성능 결과는 이후 GPU 측정으로 채운다.

1. ISA disassembly 구간만 파싱한다. ACO IR과 최종 ISA에 같은 opcode가 두 번 나오는 것을 합산하지 않는다.
2. dot4와 dot4c를 함께 세되 signed/unsigned, clamp 유무를 따로 기록한다. static count는 코드에 들어 있는 개수이며 dispatch 전체 실행 횟수가 아니다.
3. 실행량은 loop 반복 수, lane/workitem 수, tile당 출력 수, boundary mask를 확인해서 계산한다. 루프가 남아 있으면 static count에 반복 수를 곱한다. branch 양쪽 코드를 무조건 모두 실행한 것으로 세지 않는다.
4. dot4 하한은 같은 정의의 integer-ops를 empirical dot4 TOPS로 나눈다. 최소 메모리 하한에는 입력/출력의 필수 bytes를 쓴다. ISA load 개수×폭은 캐시 재사용 때문에 DRAM bytes와 같지 않다.
5. 직렬 누산기 개수와 독립 누산기 수를 함께 확인한다. saturation이 있으면 재결합 순서가 결과를 바꿀 수 있다. plain modulo32 정수 덧셈과 각 단계 saturation을 구별한다.

## 추가 후보: 우선순위와 미검증 부분
| 순위 | 후보 | 현재 판정 |
|---|---|---|
| 1 | 실제 ISA에 남은 별도 dot4+integer add, shift/mask/merge 제거 | 실제 pass ISA에 패턴이 남아 있을 때만 수정한다. |
| 2 | n=8 출력에서 signed16 포화 두 쌍+high-byte permutation | 수학적 후보. 비-ReLU에서 clamp32(x,-32768,32767)의 byte1은 saturated16(x)의 high byte와 같다. 네 값을 2×v_cvt_pk_i16_i32+1×v_perm_b32로 모을 여지가 있다. |
| 3 | 독립 누산기 증가/일찍 store해 live range 감소 | VGPR allocation 경계와 occupancy 손실을 함께 비교한다. |
| 낮음 | min/max를 단순히 med3 표기로 변경 | ACO가 이미 결합한다. actual dump가 결합 실패를 보여줄 때만 검토한다. |

2번은 해당 연산의 ISA 존재와 NIR 의미만 확인했다. 일반 HLSL/SPIR-V 표현으로 현재 ACO가 그 명령을 선택하게 할 수 있는지, 기존 exact가 이미 같은 결과를 내는지, 전체 시간 이득은 미검증이다. driver 수정은 이번 구현 범위에 넣지 않는다. ReLU가 있는 층은 하한 clamp도 유지/융합해야 한다.

## Vulkan dot4 상한 프로브
ceiling/dot4_ceiling.c와 dot4.comp는 upstream timing-kit/src/bench.c의 buffer/command helper를 include해 사용한다. FSR용 harness를 새로 만들지 않는다.

- gfx1030에 해당하는 RX6900XT/NAVI21 discrete GPU와 RADV만 선택하고 signed packed 4x8 acceleration feature를 확인한다.
- wave32를 명시하고 4/8/16/32 독립 누산기를 교대로 실행한다.
- 각 누산기가 다음 dot4의 packed operand와 누산 입력에 모두 쓰인다. 입력은 runtime buffer에서 읽고 모든 누산기 출력은 저장해 compiler hoisting/상수접기/죽은 코드 제거를 막는다.
- GPU timestamp만으로 dispatch 시간을 측정한다. 초기 전송, CPU reference 계산, readback은 timestamp 밖이다.
- 기본 6 warmup rounds 후 15 measured rounds. 매 round에서 variant 순서를 뒤집는다. raw ticks/ms와 variant 중앙값을 JSONL로 출력한다.
- CPU signed-byte dot4 reference와 64개 분산 thread×32 accumulator를 비교한다.
- 기재하는 상한은 variant 중앙값 중 최대의 관측 throughput이다. 실제 gfx1030 ISA에서 dot4 형태와 동적 반복 수를 확인한 뒤 하한 계산에 사용한다. synthetic workload에는 FSR의 packing·tensor traffic·다중 pass 비용이 없다.

CPU 검증 상태: gcc build 성공, glslang Vulkan1.3 compile 4개 성공, spirv-val 4개 성공, --help 실행 성공. GPU 실행과 성능값은 주 작업에서 별도로 기록한다.

## 원문 열람 식별 정보
모든 항목은 읽기 전용으로 가져왔으며 아래 별도 명시가 없는 경우 메모리에서만 읽었다. archive hash는 공식 Mesa 26.2.4 release notes와 일치한다.

| 파일 | bytes | SHA256 |
|---|---:|---|
| mesa-26.2.4.tar.xz | 68613948 | bce5f7fbebb934373b86c999a064d52fb5065878dc57f287f95346648ec832e9 |
| aco_select_nir_alu.cpp | 177011 | 1c3e092dd6d82c27e08fca55b11f15d89724a0509c7044b2c441351afd4b4a5b |
| aco_optimizer.cpp | 223082 | 88da817fea2011af66f44c439ca5ced65c33524f2d8c313a08b81ec1eab97242 |
| radv_instance.c | 15848 | 9d6db5588cdffd670fbbb58b75ccd448249c01052100a895b4d9a9fbd63c6ea8 |
| radv_shader.c | 156593 | 48da66a47a6cc66da6ecb998cd203bc940988adf1db465bde060addffb99d415 |
| ac_shader_util.h | 13198 | c911fc892f9a995570ba14287f98eb3c9ef3d5348130221054cd5b71382191ba |
| ac_shader_util.c | 64333 | 91b28d0dc39b95b8fa737ab2a3bd3732518c0641481cb8168af908289e8dcc64 |
| aco_statistics.cpp | 25709 | e67cfaea74c07d595ebe8892a364e707030737b7ed07e263e08b4a9c9bb18868 |
| nir_opcodes.py | 87677 | c8ed942f9c130627f79109228980f232ffc0259ae6d1f6a26a17443fe1f7bfd8 |
| nir_opt_algebraic.py | 254471 | f63c37ce8072425e8f490546b9800699363be01fda0431ec950c1686b605adbf |
| tools/model_clamp.py @ d1c3b30 | 5988 | 5baffb9b88088a4c03f63919ba6703a371e1945c72edde71aaed65f09a547c69 |
| tools/model_tail.py @ d1c3b30 | 4429 | 817ef9a929a81d47cdf2984bc904a1f37cd712046ecc2a420d3da5c52afaca61 |

Source archive: https://archive.mesa3d.org/mesa-26.2.4.tar.xz
Official hash: https://docs.mesa3d.org/relnotes/26.2.4.html
Source tag: https://gitlab.freedesktop.org/mesa/mesa/-/tree/mesa-26.2.4

