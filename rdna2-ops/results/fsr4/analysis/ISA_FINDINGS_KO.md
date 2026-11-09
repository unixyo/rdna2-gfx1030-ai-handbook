# 실제 4.1.2 exact의 gfx1030 ISA 분석

기준은 community `4.1.2-exact-r2-cyboman` DLL을 별도 timing kit로 실행한 exact_profile이다. 원자료는 `runs/exact_profile/stderr.log`, `measurements/exact_profile_passes.csv`, `pipelines/exact_profile/`, `dumps/exact_profile/`다. 이 분석 작업은 CPU만 사용했다. GPU 실행은 주 작업이 수행했다.

## 매핑의 증거와 한계

56개 바인딩된 PSO의 DXBC 전체 SHA256이 vkd3d `.dxil` dump와 일치한다. CASE/PIPE 로그, DXIL entrypoint를 함께 연결한 결과는 `exact_profile_pso_map.json`에 있다. dispatch0은 spdCounter와 SPD intermediate, luminance log2/exp2를 가진 exposure/SPD helper다. dispatch1은 prepass, 홀수3..25는 network1..12, dispatch27은 postpass다. 짝수2..26은 border padding pass다.

RADV의 ISA dump에는 vkd3d shader hash가 직접 실리지 않는다. 마지막60개 compile block이 해상도별30개 PSO의 생성 순서를 따른다는 대응을 사용했다. entrypoint별 코드 종류와 반복된 자원 통계가 이를 뒷받침하지만, **ISA block→PSO 관계는 생성 순서 추론**으로 남긴다. PSO→DXBC→vkd3d hash 관계의 exact SHA 증거와 구분한다.

## 자원·명령 표

표는 1080p 기준이며 4K의 대응 pass에서도 allocation, 총 instruction 수, dot4 수가 같다. SGPR allocation은 모두108이다. `Subgroups per SIMD`는 RADV의 wave32 슬롯 정규화 수치이고 실제 달성 occupancy가 아니다. network/post의64-bit EXEC/VCC 연산은 wave64 실행과 일치한다. DXIL에는 명시 WaveSize metadata가 없다.

| pass | dispatch | 명령 수 | static dot4 | VGPR | 정규화 subgroup/SIMD | LDS bytes | scratch bytes | SGPR spill | med3_i32 | SDWA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exposure/SPD | 0 | 1036 | 0 | 40 | 24 | 5120 | 0 | 0 | 0 | 44 |
| pre | 1 | 983 | 0 | 48 | 20 | 4096 | 0 | 0 | 16 | 16 |
| 1 | 3 | 1317 | 832 | 56 | 18 | 0 | 0 | 0 | 64 | 80 |
| 2 | 5 | 1318 | 832 | 56 | 18 | 0 | 0 | 0 | 64 | 80 |
| 3 | 7 | 1076 | 512 | 48 | 20 | 0 | 0 | 0 | 32 | 24 |
| 4 | 9 | 2470 | 1600 | 80 | 12 | 0 | 0 | 0 | 112 | 132 |
| 5 | 11 | 2469 | 1600 | 80 | 12 | 0 | 0 | 0 | 112 | 132 |
| 6 | 13 | 3475 | 2048 | 80 | 12 | 0 | 0 | 4 | 64 | 48 |
| 7 | 15 | 963 | 336 | 56 | 18 | 0 | 0 | 0 | 12 | 13 |
| 8 | 17 | 964 | 336 | 64 | 16 | 0 | 0 | 0 | 12 | 13 |
| 9 | 19 | 1227 | 400 | 72 | 14 | 0 | 0 | 0 | 12 | 28 |
| 10 | 21 | 2430 | 1600 | 80 | 12 | 0 | 0 | 0 | 112 | 148 |
| 11 | 23 | 4300 | 2112 | 48 | 20 | 0 | 0 | 0 | 176 | 228 |
| 12 | 25 | 1318 | 832 | 56 | 18 | 0 | 0 | 0 | 64 | 80 |
| post | 27 | 2922 | 960 | 128 | 8 | 16384 | 0 | 0 | 64 | 64 |

모든 pass의 VGPR spill은0이다. pass6은 SGPR4개를 spill하며 실제 `v_writelane_b32`와 `v_readlane_b32`가 각각4개다. 따라서 scratch0만으로 spill도0이라고 결론 내리면 이 비용을 놓친다. post의16KiB LDS는 공동 store를 위한 자원이다. 현재 workgroup에서 LDS만의 상한8 wave보다128VGPR에 따른4 wave 상한이 작아 VGPR 쪽이 우선한다. wave64 조건에서 log의 정규화 값8은 내부4 wave 계산에 대응한다. source 정의는 `OCCUPANCY_DEFINITION_KO.md`에 분리했다. SDWA열에는 byte pack뿐 아니라 unpack/shift/조건선택도 포함되므로 전부 packing 비용으로 합산하지 않는다.

`exact_profile_isa.csv`에는 padding13개를 포함한 모든28 dispatch×2해상도 표가 있고, JSON에는 전체 opcode histogram, 원래 stat, SHA, entrypoint, local size, 원문 line 위치와 raw ISA 경로가 있다. pass1의4K에서는 scalar address 계산이 `s_addk_i32` 한 개에서 `s_add_u32` 한 개로 바뀌지만 총 instruction 수와 자원은 같다.

## 역어셈블 문제를 어떻게 처리했는가

RADV dump와 별도 LLVM21/22 disassembler가 일부 gfx1030 SDWA와 SMEM encoding을 invalid로 표시했다. 다음32-bit word를 별도 fmac/interp/export 명령으로 오인하는 경우가 있어 텍스트 줄 수를 그대로 합산하면 잘못된다. 이것은 GPU hang을 관찰한 기록이 아니다. 해당 profile은 주 작업의 guard에서 정상 종료했다.

`parse_isa.py`는 최종 assembly에 함께 찍힌 machine word를 보존하고, 경계가 깨진 부분만 RDNA2 format 길이와 opcode field로 복원한다. SDWA는 Src0=249인 추가 word, SMEM/MUBUF는64-bit, VOP3/P의 literal은 추가 word다. 복원 후56개 shader 모두 raw byte 수가 ACO `Code size`와 같고 instruction 수가 ACO `Instructions`와 같다. 따라서 눈에 보이는 가짜 fmac를 실제 연산에 포함하지 않는다.

- network10: text상1599 dot4 → raw 경계복원1600.
- network11: text상2109 dot4 → raw 경계복원2112.
- network1: text1333 instruction 줄 → 실제1317 instruction.16개 SDWA의 두 번째 word가 별도 줄로 나온 차이다.

원본 `.original.s`, 복원 `.normalized.s`, 전체 `.bin`을 `exact_profile_isa/`에 모두 남겼다. 복원된 명령은 operand를 추측해 쓰지 않고 별도 표시한다. opcode와 경계를 복원한 것이며 완전한 새 ISA disassembler를 만든 것은 아니다. opcode field 근거는 [AMD RDNA2 공식 ISA](https://docs.amd.com/api/khub/documents/Et~wpu9g~Ffl7d9q0QZ~Og/content)이고, compiler stat 의미는 상위 `ISA_METHOD_KO.md`의 Mesa26.2.4 원문에 연결돼 있다.

## dot4·누산 의존성·실행량

실제 연산은 `v_dot4_i32_i8`와 destination-accumulate `v_dot4c_i32_i8`다. unsigned dot4 및 dot4 clamp는 관찰하지 않았다. 예를 들어 network1은64 explicit-acc dot4+768 dot4c이고, network11은176+1936이다. dot와 누산 add는 이미 ISA 한 명령으로 결합돼 있다. 이 층을 다시 붙이는 단순 변환에서 추가 이득을 예상할 근거는 없다.

register scheduling proxy를 보면 network1/4/5/10은32개 dot 명령 window에서 목적 누산 register가 중앙값8개, 최소4개다. network7/8/9는 중앙값4개다. 같은 accumulator를 다시 쓰는 간격 중앙값은 각각 약6 instruction과2~3 instruction이다. 긴 dot chain은36까지 보인다. 이것은 scheduling 관찰이며 수학적으로 독립된 accumulator의 정확한 동시 생존 수나 실제 stall 측정은 아니다. 실제 stall/IPC 계측이 없어 지연 원인을 이 숫자만으로 확정하지 않는다.

static dot4 수를 전체 dispatch 실행량으로 바로 곱할 수 없는 핵심 pass가7/8/9다. 실제 DXIL loop와 현재 groupId.z=0에서:

- pass7/8: 144×8+64×32+128×16=5248 dot4/유효 thread.
- pass9: 위 값+64×(2×2×8)=7296 dot4/유효 thread.
- 나머지 network와 post는 해당 유효 thread의 static dot4 수를 사용한다.

network의 유효 좌표는 실제 DXIL immediate bound로 확인했고 post는 cbuffer의 output W/H÷2 guard로 확인했다. post의 cbuffer raw 값은 별도 dump되지 않았으므로 Native case의 W/H 전달과의 대응 전제를 명시했다. 경계의 비활성 lane은 수학적 연산량에서 제외하며 부분 wave의 남는 issue 비용은 별도다. 자세한 근거는 `MEMORY_MODEL.json`에 있다.

12개 network+post의 합은1080p **3,396,556,800 dot4 =27.1724544 GOP**,4K **13,586,227,200 dot4 =108.6898176 GOP**다. wave32 synthetic 프로브의 지속 기준 **91.679754437 TOPS**로 나눈 dot-only 비교시간은 **0.296384 ms /1.185538 ms**다. 새 exact_warm_profile(warm40/timed200)의 해당 pass 진단 중앙값 합은 **0.534260 ms /2.183520 ms**다. 초기warm4/timed8에서 나온0.856220/2.324500ms와 구분하며 두 원자료를 보존했다. probe와 FSR은 wave size·packing·memory·clock 조건이 달라 이를 하드웨어 절대하한으로 단정하지 않는다. `DOT4_WORKLOAD_KO.md`와 JSON에 모든 pass의 계산과 조건을 남겼다.

## 실제 후보와 아직 미검증인 후보

현재 exact ISA에는 `v_perm_b32`와 `v_cvt_pk_i16_i32`가 없다. clamp는 이미 `v_med3_i32`로 합쳐져 있어 단순 min/max 표기 변경은 우선순위가 낮다. shader는 여전히 shift→signed-byte clamp→SDWA byte 조립 패턴을 가진다.

첫 실행 후보 `candidate_clamp_shift_post`는 현재 시스템에서 나온 post SPIR-V에 upstream `model_clamp.py`를 적용한다. 각해상도12 vec4 그룹, 총2개 hash override이며 원래 SPIR-V1.3 version을 유지하고 Vulkan1.3 `spirv-val`을 통과했다. 이 변환은4.1.1 community의 알려진 exact 최적화를 현재4.1.2 exact DLL에 옮기는 것이다. 새로운 최적화 원리를 발견했다는 주장이 아니다.

추가 CPU 후보 `candidate_clamp_shift_all`은 network1/2/4/5/7/8/9/10/12와 post를 포함한다. 각해상도114 vec4 그룹,20개 hash override다. network3/6/11은 패턴이 없어 그대로 두었다. 모두 SPIR-V 검증을 통과했다. 각 후보의 MANIFEST에 source/override SHA와 변경 수, provenance를 남겼다. GPU override hit, 실제 ISA, 전체 FSR 출력 byte일치, 교대 total timing은 주 작업이 별도로 판정한다.

signed16 포화 두 쌍+high-byte permutation 후보는 수학적 가능성으로 남긴다. shift8의 비-ReLU 출력을 saturated16 high byte로 얻을 수 있고 ISA에는 해당 명령이 있지만, ordinary HLSL/SPIR-V에서 현재 ACO가 `pack_sint_2x16`으로 선택하게 만드는 표현은 아직 검증하지 못했다. 이 명령이 존재한다는 사실만으로2×cvt_pk+1×perm 코드가 실행된다고 쓰지 않는다.

현재 병목 분류는 단일 원인으로 확정하지 않았다. 점유·live register·packing·scalar weight traffic·LDS store 설계가 함께 존재하고, 관측 dot throughput과의 차이만으로 compiler나 hardware 한계라고 확정할 수 없다. 최종 후보의 출력·전체시간 결과와 함께 주 보고서에서 판정한다.

후속 actual compile 비교는 `CANDIDATE_ISA_COMPARISON_KO.md`와 JSON에 있다. post는2개, all은20개 의도한 ISA만 바뀌고 다른 shader machine code는 byte동일함을 확인했다. source PSO DXBC56개는 모두 기준과 byte동일하다. 이는 override 적용과 ISA 순서 대응을 실제 변경으로 뒷받침한다. SDK/exact/lossy 자원·명령 비교는 `VARIANT_ISA_COMPARISON_KO.md`에 있다.

재현: `python3 analysis/parse_isa.py`, `python3 analysis/parse_isa.py exact_warm_profile`, `python3 analysis/build_dot_workload.py`, `python3 analysis/build_clamp_candidate.py post`, `python3 analysis/build_clamp_candidate.py all`. 이 명령들은 GPU를 사용하지 않는다.
