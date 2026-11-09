# RADV shader 통계의 subgroup 상한 정의

메인 표의 권장 이름은 **RADV 자원 계산값(32-lane 환산)**이다. 실제 GPU에서 달성한 occupancy, 실제 순간 resident wave 수, GPU 활용률을 측정한 값과 구분한다.

확인한 구현은 upstream Mesa **26.2.4 tag**다.

1. `radv_pipeline.c`의 shader 통계 조립은 `stats.maxwaves`에 `shader->max_waves`를 넣는다. 같은 함수가 VGPR/SGPR, LDS와 scratch 통계도 조립한다.
2. `radv_shader.c`의 `radv_get_max_waves`는 하드웨어/컴파일러 상한, VGPR allocation, LDS와 workgroup 배치를 적용한다. GFX10 이상에서는 마지막에 내부 wave 수에 `wave_size / 32`를 곱한다.
3. `ac_gpu_info.c`는 GFX10.3 이상에서 `max_waves_per_simd=16`, GFX10에서는20으로 초기화한다. 후자는 RDNA1에 해당하므로 RDNA2의 보편 상한을20이라고 쓰면 안 된다.

따라서 wave64 shader의 log 값8은 내부 계산값4 wave/SIMD에 대응한다. 값20은 내부 계산값10에 대응한다. wave32이면 multiplier가1이므로 나누지 않는다. 이 FSR의 network/post ISA는64-bit EXEC/VCC 연산을 사용하며 wave64 해석과 일치한다. DXIL 자체의 명시 WaveSize metadata는 없으므로 wave size 판정의 근거도 함께 보존한다.

로그18/20/24가16보다 크다는 사실을 실제 resident wave 수로 해석하지 않는다. 함수는 시작값16에 자원별 MIN2를 적용한 뒤 wave64에서2배로 환산하고, 최종 환산값에16을 다시 상한으로 적용하지 않는다. 따라서 “uncapped”라고만 부르기보다 “최종32-lane 환산값은16으로 다시 제한하지 않는 RADV 자원 계산값”이라고 명시한다. 자원 제한을 전혀 적용하지 않은 값이라는 뜻은 아니다.

post의 자원을 예로 들면 일반 Navi21의 wave64 VGPR 예산512, shader VGPR128에서 VGPR 한계는4 wave/SIMD다. 256-thread workgroup과16KiB LDS만 고려한 상한은8 wave/SIMD여서 이 경우 더 작은 VGPR 한계가 우선한다. 최종 log의 정규화 값8과 맞는다. 이것은 자원으로 계산한 상한이며 실제4 wave가 항상 resident하다는 관측은 아니다.

SDK post의72 VGPR·LDS0과 exact post의128 VGPR·LDS16KiB는 정규화 상한14→8로 바뀐다. 그럼에도 exact post의 warm 진단 시간이 줄었다. 메모리 쓰기 배치가 함께 바뀌므로 occupancy 수치 하나를 성능의 절대 기준으로 삼으면 안 되는 실제 사례다.

근거:

- [Mesa26.2.4 radv_pipeline.c](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/vulkan/radv_pipeline.c), shader 통계 조립의 maxwaves 대입.
- [Mesa26.2.4 radv_shader.c](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/vulkan/radv_shader.c), `radv_get_max_waves`와 wave 크기 정규화.
- [Mesa26.2.4 ac_gpu_info.c](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/common/ac_gpu_info.c), `ac_fill_compiler_info`의 max wave, physical VGPR, allocation granularity와 SIMD 수.
- 실제 관측: `exact_warm_profile_isa.json`, `VARIANT_ISA_COMPARISON.json` 및 원본 RADV log.

추가 열람 source의 URL·크기·SHA256은 `ISA_ADDITIONAL_SOURCE_FETCHES.json`에 있다. 파일은 네트워크에서 읽어 메모리에서 분석했으며 드라이버 설치나 수정은 하지 않았다.
