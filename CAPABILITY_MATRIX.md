# gfx1030 기능 공백·우회 구현 지도

범위: gfx1030의 하드웨어 ISA·메모리·동기화·라이브러리 제약에 대한 **보조 기술 지도**. 이번 Pro 연구의 목표는 LLM 및 이미지·영상 디퓨전에서 **실제 전체 처리 속도 개선**이고, 이 파일의 W4A8/INT4·MoE 설명은 최초 수집 단계에서 남은 사례이지 반드시 연구할 중심 주제가 아니다. Dense/Sage/sparse/Veda/Flash 등 Attention 포팅은 ATTENTION_PORTING_MAP.md, 성능 판정은 PERFORMANCE_CONTRACT.md가 우선이다. **GPU ISA의 모든 명령어 전수조사가 아니다.** 라이브러리 커널이 없다고 하드웨어 명령도 없다고 단정하지 않는다.

증거 인용 규칙: [snapshot-id]는 source_manifest.json의 id이며, 같은 이름의 snapshots/ 파일이 실제 원문이다. 실측값은 원저자의 보고이고 이 PC의 RX 6900 XT에서 재현한 수치가 아니다.

## 1. ISA·수치 연산

| 기능·문제 | gfx1030 판정 | 대체·응용 | 근거 |
|---|---|---|---|
| WMMA / MFMA 16×16 행렬 코어 | **하드웨어 없음** | wave32 SIMT 타일링 + FP16 dot2 / INT8 dot4; GEMV·GEMM 별도 설계 | [guide-silicon], [vllm-changes], [flash-forward], AMD rocWMMA 공식 지원표 |
| rocWMMA 라이브러리 | **gfx1030 근본적 미지원** | SIMT 커널 또는 rocBLAS/직접 특화 커널. 무작정 gfx11 WMMA 코드를 이식하지 말 것 | ROCm/TheRock issue #1944, AMD rocWMMA |
| BF16 네이티브 dot / FMA | **하드웨어 dot 없음**; BF16 저장/형변환 자체는 가능 | FP32 변환해 계산, 수치 범위 허용되는 shape만 FP16 재표현 | [guide-silicon], [strata-fp16-study], 로컬 bf16_fdot2 컴파일 실패 |
| FP8 네이티브 dot / matrix | **없음** | 소프트웨어 디코딩 후 FP16/INT8 등 지원 dtype 사용 | [guide-silicon], [comfy-capabilities], 공식 fp dot builtin 표 |
| MXFP4 / NVFP4 하드웨어 | **FP4 WMMA 및 microscaling 가속 없음** | 압축 저장·직접 언패킹·그룹 스케일 적용; 비용 분석 필요 | [comfy-capabilities] |
| INT8×INT8 signed dot | **있음** | __builtin_amdgcn_sdot4; 4개 INT8 × INT8 / INT32 누산 | [guide-silicon], gfx1030_compile_probes.json |
| UINT8×UINT8 dot | **있음** | __builtin_amdgcn_udot4 | gfx1030_compile_probes.json |
| INT4×INT4 signed / unsigned dot | **있음** | __builtin_amdgcn_sdot8 / udot8; 8쌍 nibble 계산 | [guide-silicon], gfx1030_compile_probes.json |
| INT4 weight × INT8 activation 직접 GEMM | **그 조합의 전용 matrix 유닛 없음** | packed INT4를 레지스터에서 signed INT8로 복원한 뒤 sdot4. 그룹별 보정·스케일, layout 재배치 | [vllm-w4a8-moe], [vllm-changes], MetaInfer 글 |
| mixed signed/unsigned dot(sudot4) | **현재 LLVM gfx1030 codegen 미지원**; dot8-insts target feature 필요 | 일관된 signed operand로 변환 + 수학적 보정 또는 별도 udot/sdot 경로 | gfx1030_compile_probes.json, Strata hip_compat/intrinsics.hpp |
| FP16×FP16 dot2 + FP32 누산 | **있음** | __builtin_amdgcn_fdot2 / v_dot2c_f32_f16; 직접 FlashAttention 구현 | [flash-forward], [guide-silicon], gfx1030_compile_probes.json |
| 바이트 재배열 v_perm_b32 | **있음** | __builtin_amdgcn_perm 으로 nibble→byte 복원·코드북 조회 | [vllm-w4a8-moe], Strata moe_fused_iq.cu, gfx1030_compile_probes.json |
| wave 협업 DPP·permlanex16 | **있음** | 가벼운 warp reductions / reorder. LDS 통신과 비용 비교 | Strata s26_tsum.cuh, gfx1030_compile_probes.json |
| VOPD와 최신 WMMA 결합 | **RDNA2에 없는 후속 세대 기능** | VALU 명령 독립성·register pressure·타일 비용 최적화 | AMD RDNA2 ISA, [guide-silicon]; 이 패킷에서 별도 실컴파일 미검증 |

**문서 충돌 주의:** AMD ROCm Optimization Guide의 정수 dot 내장함수 지원표 중 일부 버전은 RDNA2의 sdot4/8를 'No', sudot4/8를 'Yes'로 표시한다. 반면 실제 로컬 LLVM 22의 gfx1030 코드생성은 sdot4, udot4, sdot8, udot8이 **성공**, sudot4는 **dot8-insts 필요** 오류를 반환했다. [guide-silicon]도 이 실험과 일치한다. **공식 표를 진리로 취급하지 말고 표 자체를 검증 대상**으로 삼는다. 컴파일 성공은 실행 안정성과 속도를 보장하지 않는다.

## 2. 로드·캐시·동기화

| 문제 | 구분 | 실제 대안 | 근거 |
|---|---|---|---|
| LDS 작업그룹당 64 KiB | **물리적 용량 제한** | 작은 K/M/N 타일, streaming softmax, 2단계 split, Q/K와 partial 분리 | [guide-silicon], [llama-flash-fix], [flash-plan] |
| CUDA cp.async / TMA와 같은 최신 경로 | **그 API·ISA 그대로 없음** | 보통 load + barrier, 또는 **지원되는 buffer→LDS 직접 load**를 조건부 사용 | AMD Global-to-LDS 공식 문서 |
| global_load_lds generic builtin | **RDNA2 미지원** | raw_ptr_buffer_load_lds 경로의 리소스 디스크립터 사용 가능 여부 검증 | AMD Global-to-LDS 공식 문서 |
| raw_ptr_buffer_load_lds | **RDNA2 지원이라고 공식 문서에 기재** | 중간 VGPR를 피한 buffer→LDS; 주소/정렬/동기화 제약 검토 | AMD Global-to-LDS 공식 문서. 이 패킷에서는 로컬 codegen 미검증 |
| VGPR 소진·레지스터 스필 | **핵심 병목** | TILE_K=16/32 등 축소, 일부 직렬화, stage 축소, LDS bank-conflict 재배치 | [vllm-changes], [flash-plan], [guide-silicon] |
| wave32 / wave64 가정 차이 | **설계·포팅 문제** | shuffle/DPP/permlanex16 재구성, launch bounds와 실제 occupancy 검사 | [llama-flash-fix], [llama-progress], Strata s26_tsum.cuh |
| FP16/BF16 atomic accumulate | **native 지원 범위·성능은 연산형/주소공간/툴체인별 검증 필요** | 원자 누산 대신 FP32 별도 partial buffer와 2차 reduce, 필요 시 CAS | [guide-silicon], [flash-plan]; 이 항목은 광범위한 '미지원'으로 확정하지 않음 |
| Infinity Cache 128 MB 활용 | **임의 KV 캐시 데이터를 고정하는 일반적 API 없음** | working set 분석, sequential streaming, KV 압축, routing-aware locality | [guide-silicon], [guide-llamacpp] |
| Multi-GPU coherent shared memory | **V620 사례에서 장치 간 일관성 부재** | staged/push 기반 통신, 안정적인 이벤트, P2P 방향·PCIe root 확인 | [guide-multigpu], [vllm-changes]. RX 6900 XT 단일 카드에는 직접 우선순위 낮음 |

## 3. 라이브러리·컴파일러의 지원 공백 (하드웨어 부재와 혼동 금지)

| 현상 | 분류 | 근거 및 대응 |
|---|---|---|
| gfx1030 hipBLASLt 전용 GEMM 커널 부재 | **라이브러리 지원/배포 문제** | [strata-hip-guide]; 기존 hipBLAS/rocBLAS, 별도 HIP SIMT 커널 |
| FP16-in/FP32-out 및 BF16 GEMM이 느림 | **rocBLAS solution 선택/튜닝 문제** | [strata-fp16-study]: 한 장비에서 FP16 HH 37.7 TFLOPS vs HSS 5.6 TFLOPS; 결과 반올림·범위·KL 위험을 함께 측정 |
| Triton BF16 dot가 LLVM에서 죽음 | **아키텍처 기능 판별 오류** | unsloth issue #7922, PR #7682 및 zoo #1316 등; BF16 storage 허용과 BF16 compute 지원을 구분 |
| FlashAttention D=256/512에서 occupancy 0 | **커널 shape·launch config 버그** | [llama-flash-fix], [llama-progress]; 단순 VEC fallback은 prefill을 크게 악화시키기도 함 |
| W4A8의 큰 Triton tile에서 스필 | **컴파일러 register budget 문제** | [vllm-changes] W4A8, K≥64에서 190~390 VGPR spill 보고. 작은 tile과 정확도 비교 |
| NVIDIA 전용 FP4/FP8 API가 AMD에서 fallback | **백엔드 기능 협상 문제** | [comfy-capabilities]: WMMA가 없으면 직접 구현 지원만 광고하고 안전하게 fallback |
| custom quant에서 빠른 decode vs prefill | **형상별 지배 병목 차이** | [vllm-w4a8-moe] prefill only, [vllm-skinny-int4] decode, [guide-silicon] |

## 믿을 수 있는 직접 참고 링크

- AMD RDNA2 ISA 기준 문서: https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
- rocWMMA GPU 지원: https://rocmdocs.amd.com/projects/rocWMMA/en/latest/install/source-build.html
- rocWMMA gfx103x 비지원 이슈: https://github.com/ROCm/TheRock/issues/1944
- FP dot builtin별 가용성: https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/docs-1.0.0/compiler-builtins/cross-arch/arithmetic-ref/dot-float-builtins.html
- INT dot builtin별 가용성 (표 충돌 확인 필수): https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/docs-1.0.0/compiler-builtins/cross-arch/arithmetic-ref/dot-integer-builtins.html
- 직접 LDS 로드: https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/develop/compiler-builtins/cross-arch/direct-to-lds-builtins.html
- AMD GPUOpen occupancy: https://gpuopen.com/learn/occupancy-explained/
- Unsloth BF16 실제 실패와 수정 추적: https://github.com/unslothai/unsloth/issues/7922
- ROCm Linux gfx1030 SKU 공식 지원 구분: https://rocm.docs.amd.com/projects/install-on-linux/en/develop/reference/system-requirements.html

본 자료를 이용한 실험에서 '이 연산은 지원되지 않는다'는 진술은 최소한 **ISA/LLVM 코드생성/실제 런타임/라이브러리 dispatch 중 어느 층을 뜻하는지** 적시해야 한다.
