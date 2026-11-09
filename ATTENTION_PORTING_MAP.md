# 실제 사용자 성능 향상을 위한 RDNA2 Attention/Kernel 포팅 지도

작성 기준 2026-11-02. **이 문서가 연구 패킷의 핵심 출발점이다.** W4A8·MoE·저비트 가중치 연구는 중심 과제가 아니다. 학습 없이 모델을 더 빠르게 실행할 수 있는 dense/sparse Attention 및 전후 연산과, 필요한 gfx1030 대체 구현이 중심이다.

## 1. 3개 수준을 혼동하지 말 것

1. **알고리즘:** FlashAttention(정확한 softmax), SageAttention(Q/K 저정밀 근사), SpargeAttn/VSA/Radial/Veda(sparse mask/학습된 선택기) 등. 같은 목표지만 정확성/학습 필요성이 다르다.
2. **구현과 하드웨어:** CUDA/SM80/SM90/SM100, Hopper WGMMA/TMA, RDNA3 WMMA, CDNA MFMA, RDNA2 gfx1030의 wave32/VALU/dot2/dot4/LDS. 알고리즘이 공통이어도 실행 커널은 그대로 호환되지 않는다.
3. **실제 실행 경로:** PyTorch SDPA의 math/AOTriton, 설치된 FlashAttention의 CK/Triton/HIP, ComfyUI의 xformers 우선순위, 원래 엔진의 custom HIP 등이 선택될 수 있다. 옵션 문자열만 확인하지 말고 런타임에서 실제 선택된 커널을 밝혀야 한다.

이 세 층을 구분하지 않으면 'Flash 켰다'는 말이 무엇을 뜻하는지조차 알 수 없다. AMD의 2026-07-20 공식 ComfyUI 분석이 특히 유용하다:
https://rocm.blogs.amd.com/software-tools-optimization/comfyui-fa-backends/README.html

**주의:** 위 AMD 글에서 측정한 Navi31 / Navi48 / Strix Halo는 **gfx1030이 아니다.** 이 결과를 RX 6900 XT의 최적 설정으로 그대로 옮기지 말 것. 해당 글은 Sage가 때로 SDPA보다 느리고, xformers가 실행 경로를 가로채며, Flash CK 경로가 다른 RDNA에서는 빠를 수 있다는 점을 입증하는 **판단 방법**의 참고 자료다.

## 2. 실제 코드·포팅 난이도·성능 기회

| 기법 | gfx1030 현황 및 주요 장애 | 검증용 실제 코드/자료 | 예상 기회 / 실험 난이도 |
|---|---|---|---|
| **PyTorch SDPA** | 라이브러리 내 gfx1030용 AOTriton kernel image·dispatch 여부가 ROCm/PyTorch 빌드에 좌우됨. PyTorch math fallback과 혼동 쉬움 | snapshots/comfy-backend-selection.py, comfy-attention.py, AMD ROCm blog | **최우선 진단:** 실제 backend 확인과 올바른 빠른 경로 선택은 구현 부담 적음 |
| **RDNA2 네이티브 FlashAttention** | 상용 FlashAttention의 CK/SM 전용 구현과 달리 Tensor Core 없이 dot2 및 LDS 배치 직접 구현 필요 | snapshots/flash-forward.hip, flash-decode.hip, flash-plan.md, llama-flash-fix.md | **높은 실전 가치:** gfx1030 코드 이미 존재. 모델 shape·dtype·integration 문제 확인 |
| **Dense FlashAttention-2 Triton** | 트리톤/ROCm 버전에 따라 compile 미지원, LDS·VGPR 스필. 공식 FA2 ROCm Triton 설명과 gfx1030 가능 여부는 별개 | 2kiss HIP 대안, RDNA2 vLLM, 원조 DAO-AILab/flash-attention Triton backend 문서 | **중~상:** 동작하는 pinned Triton 또는 직접 HIP 유지 비용 분석 |
| **FlashAttention-3 / 4** | Hopper/Blackwell WGMMA, TMA, FP8/FP4 커널을 gfx1030에서 직접 실행 불가 | FA 프로젝트 docs/구현; 자료는 알고리즘과 데이터 이동 아이디어를 위한 것 | **높은 위험:** 전체 CUDA kernel 이식보다 dot2/HIP 기반으로 알고리즘만 참고 |
| **SageAttention** | QK INT8 등 주로 CUDA/Triton, GPU별 backend 의존. RDNA2 WMMA 부재·Triton 버전 걸림 | snapshots/sage-core.py, sage-triton-int8.py, sage-triton-quant.py, sage-readme.md | **중~상:** QK 근사+FP16 PV / 특화 SIMT 비교. INT8 재양자화 비용·화질 평가 필수 |
| **SpargeAttn** | 학습 없이 Q/K 블록 유사도와 softmax-aware filtering으로 희소 블록 선택, 현재 공식 CUDA 우선 | snapshots/sparge-readme.md, sparge-core.py, sparge-filter-example.py | **중~상:** 타일 mask 품질·찾는 비용을 포함한 HIP sparse path 필요. 긴 시퀀스에서 유망 |
| **Veda** | tile score **학습 predictor**와 top-k tile skipping. 공식 ComfyUI는 RDNA3/4 지원, gfx103x 후보를 제외함. CUDA TMA/WGMMA 커널을 통째로 재사용할 수 없음 | snapshots/veda-*.md, veda-hardware-gate.py, veda-sparse-int8-code.py, veda-triton-backend.py | **고위험·높은 잠재력:** 학습된 대상 모델·영상 길이·LDS·dot4 지원을 따로 따져야 함 |
| **VSA** | 학습된 sparse selection, CUDA 고성능 경로는 Hopper/Blackwell 전용 부분 존재. Triton fallback도 RDNA2 실동작 미검증 | snapshots/fastvideo-vsa-guide.md, fastvideo-sparse-triton.py, fastvideo-sparse-topk.py | **중~상:** 이미 학습된 model/checkpoint 없으면 다른 모델에 바로 적용 불가 |
| **Radial Attention** | 공간·시간 거리 기반 static sparse pattern, CUDA 계열 기존 구현 | snapshots/radial-readme.md, radial-wan-attn.py, radial-wan22-attn.py | **중:** predictor 학습 없이 mask 설계 참고 가능. 영상 DiT에서 정확도·모션 검사 |
| **Sliding/Window/Block-sparse** | sparse pattern이 엔진과 모델 조건에 맞으면 계산량 감소, 작은 sequence에서는 overhead가 승리 | snapshots/comfy-sparse-attention.py, fastvideo-sparse-triton.py, flash-plan.md | **중:** 우선 동일 shape에서 dense 대비 통합 비용 확인 |
| **Split / sub-quadratic fallback** | VRAM 부족 회피에 유리하지만 일반적으로 빠른 dense fused kernel보다 느릴 수 있음 | snapshots/comfy-attention.py, ComfyUI backend docs | **낮은 구현 비용:** 실제 용량 제약이 원인인지 확인 후 선택 |
| **Attention 외 GEMM·norm·VAE·conv** | Attention만 최적화해도 전체 실행을 지배하는 데이터 이동·VAE·FFN·weight streaming은 그대로 | snapshots/comfy-capabilities.md, comfy-backend-selection.py; 기존 로컬 소스 | **반드시 프로파일:** 전체 시간의 더 큰 구간이 있으면 그쪽부터 구현 |

## 3. 특히 중요한 '지원되지 않는 것'의 정체

**Veda 실제 차단 사유:** snapshots/veda-int8-kernel-design.md는 RDNA2의 INT8 WMMA 부재로 gfx103x를 후보에서 빼고 있음을 명시한다. snapshots/veda-hardware-gate.py도 함께 읽어라. **후보 목록을 속여서 RDNA2를 추가한다고 커널이 작동하는 것은 아니다.** WMMA 없는 dot4/SIMT sparse tile kernel이나 다른 표현으로 다시 설계해야 한다. 또한 model-specific predictor가 학습되어 있지 않으면 mask와 품질은 보장되지 않는다.

**Triton upstream 회귀:** 2026-09-30 병합된 https://github.com/triton-lang/triton/pull/12042 로 RDNA1/2 타깃을 upstream에서 제거했다. Triton v3.9 릴리스는 gfx10 미지원, 3.8 계열 또는 유지되는 포크/LLVM HIP 경로가 필요할 수 있다. Windows용 triton-windows 복구 PR https://github.com/triton-lang/triton-windows/pull/58 은 **리눅스 설치 지원을 자동 보장하지 않는다.** 실제 프로세스의 Triton / ROCm 빌드부터 식별해라.

**Sage는 벤치 숫자를 옮길 수 없다:** AMD 공식 2026년 측정(https://rocm.blogs.amd.com/software-tools-optimization/comfyui-fa-backends/README.html)은 최신 AMD RDNA에서 Sage가 SDPA보다 느릴 수도 있음을 보였다. RTX 카드의 2배 / 5배 수치를 gfx1030에서도 달성한다고 가정하지 마.

**모델별 sparse 품질:** Sparge는 training-free지만 근사고, Veda와 VSA는 별도 학습된 predictor/selector가 중요하다. Veda의 원 연구는 고해상도 장영상에서 5.1x end-to-end를 보고했다 (https://proceedings.mlr.press/v306/han26y.html). **그 수치를 16GB RX 6900 XT가 실행 가능한 모델·해상도·모드와 동일시하면 안 된다.**

## 4. 우선 연구 과제는 '코드를 이식'이 아니라 '체감 이득 확보'

1. 로컬 LLM 또는 디퓨전에서 실제 attention kernel/backend 식별. 기동 옵션만 보고 판단 금지.
2. 같은 워크로드에서 단계별 시간 측정: 텍스트 prefill/decode, 이미지 DiT/UNet vs VAE vs sampler, 영상 attention vs FFN vs data movement. 해당 shape에서 최적화 가능성이 높은 지배 병목 찾기.
3. **먼저 기존 RDNA2 특화 HIP exact attention을 안전한 대안과 비교**하고, 이미 최적이라면 새로운 sparse/Sage 구현으로 탐색.
4. Sparse/Veda 계열은 'sparsity 90%'가 전체 10배라는 뜻이 아님. mask 생성, gather, 재배열, 나머지 모델 연산을 포함해 실제 end-to-end 개선 측정.
5. 커널 이식 비용이 지나치면 다른 핵심 경로의 낮은 위험 최적화를 선택해도 된다.

## 5. 대표 기술 출처

- AMD 공식 ComfyUI Attention backend 평가: https://rocm.blogs.amd.com/software-tools-optimization/comfyui-fa-backends/README.html
- 실제 RDNA2 exact FlashAttention: https://github.com/2kiss/flash-attention-rdna2
- SageAttention: https://github.com/thu-ml/SageAttention
- SpargeAttn: https://github.com/thu-ml/SpargeAttn
- Veda 원문 및 ComfyUI 코드: https://veda-sparse.github.io/ / https://github.com/veda-sparse/Veda-on-ComfyUI
- VSA FastVideo: https://github.com/hao-ai-lab/FastVideo
- Radial Attention: https://github.com/mit-han-lab/radial-attention
- PyTorch/ROCm backend 실제 선택: https://github.com/Comfy-Org/ComfyUI
- Triton gfx1030 타깃 제거 PR: https://github.com/triton-lang/triton/pull/12042

제3자의 성능 수치와 지원 선언은 단지 **그 환경에서 보고한 사례**다. 원본 환경·commit·dtype·layout·shape·실제 백엔드를 기록하고, gfx1030에서 새로 측정하는 것만 새 결과로 취급한다.
