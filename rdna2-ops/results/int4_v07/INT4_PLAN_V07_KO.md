# INT4 가중치 GEMM v0.7 — 결과 전 고정한 판정

작성: 2026-11-07. GPU 결과를 보기 전에 기록한다.

## 계산 계약

- 대상: 기존 MOSS SFT INT8 ConvRot, Qwen 2.1 INT8 ConvRot, Anima base v1.0의 실제 Linear shape와 기존 저장 가중치. 새 모델·checkpoint를 받지 않는다.
- 기준선: MOSS는 v0.5의 INT8+FP32 dequant 융합 `(64,128,64), warps=4, stages=2, group_m=8`; Qwen은 설치된 patientx per-row kernel; Anima는 FP16 `F.linear`/rocBLAS. 이전 MOSS INT32 출력+별도 dequant를 새 기준으로 되돌리지 않는다.
- 가중치 INT4: K축 그룹 32/64/128, 대칭/비대칭. 대칭은 `round(w/s)`를 [-7,7]로 제한, `s=max(abs(w))/7`; 비대칭은 0을 포함한 [min,max], `s=(max-min)/15`, `z=clamp(round(-min/s),0,15)`, `q=clamp(round(w/s)+z,0,15)-8`, 복원 `(q-(z-8))*s`. round는 ties-to-even, all-zero 그룹은 s=1, q=0, z=0으로 복원한다.
- 저장 word는 8개의 signed nibble을 uint32에 담는다. low-first/high-first packing과 N×K/8 또는 K/8×N 연속 레이아웃을 구분한다. INT4 값은 표준 IQ 코드북 형식과 다르다.
- W4A8 dot8은 A8을 그대로 보존한다. `A=lo+16*hi+8`, `lo=(A&15)-8`, `hi=A>>4`; 결과는 `dot8(lo,W)+16*dot8(hi,W)+8*sum(W)`. 비대칭 zero-point 보정은 추가한다. A4로 다시 양자화한 결과를 이 경로로 부르지 않는다.
- Anima의 A16→A8은 activation 표현 변경이다. 해당 오차·양자화 비용을 W4 가중치 표현 손실과 분리한다.

## 결과를 보기 전 고정한 관문

1. packing/decoder: 16개 nibble 전 패턴, 양 끝 activation -128/127, 여러 packing/layout에서 CPU INT64 oracle과 정수 부분의 실제 byte 일치. NaN/Inf 0.
2. 같은 INT4 저장값의 계산 오차: 독립 CPU FP64 표본을 참조한다. FP32 정수 경로는 relative L2 ≤ 2e-5, 최대 오차 ≤ 1e-4 + 1e-4×참조 최대 절댓값. FP16 계산/출력 경로는 relative L2 ≤ 1e-3, 최대 오차 ≤ 2e-3 + 2e-3×참조 최대 절댓값. 출력 dtype별 gate를 바꾸지 않는다. FP16 dequant의 반올림도 같은 저장값 계산 오차에 포함한다.
3. 원본→INT4 표현 손실: 기존 저장 가중치와, 같은 activation을 사용하는 INT4 FP64 참조의 오차를 별도 보고한다. MOSS/Qwen의 원본은 이미 INT8인 기존 checkpoint 복원값이며 원래 학습 FP32 가중치가 아니다. 여기에 임의로 합격선을 만들지 않고 실제 모델 관문으로 판단한다.
4. 후보 승격: 준비 비용이 들어가는 실제 호출 경로를 포함해 기존 기준보다 중앙값 1.05배 이상 빠르고 계산 관문을 통과하면 실제 모델 A/B 후보로 삼는다. 준비된 weight-only 시간과 캐시 메모리는 따로 기록한다. 이 기준을 못 넘은 후보는 성능 실패로 남긴다.
5. 모델: 같은 workflow·seed·입력·step에서 최종 RGB max ≤ 2/255, RMSE ≤ 0.25/255, finite를 모두 만족해야 엄격 채택 가능. 비트 일치는 별도로 기록한다. 최종 출력 baseline 반복 불일치가 있으면 후보 품질 판단을 유보한다.
6. 시간: cold compile/autotune/weight preparation, warm GPU event, 동기화 포함 일반 호출 시간을 분리한다. 최소 3회 교대 round, 각 3회 이상 반복. GPU guard에서 foreign PID가 관측된 실행은 timing 증거로 폐기한다.

## 실제 shape

- MOSS: M1061, (N,K)=(3072,3072), (16384,3072), (3072,8192). 기존 v0.5 capture/실측과 layer0 checkpoint header.
- Qwen: M2400, (4096,4096), (24576,4096), (4096,12288). prefix 시 M4964는 필요 시 추가. 기존 v0.5 capture/실측과 layer0 header.
- Anima: M8192, (2048,2048), (8192,2048), (2048,8192). B2·S4096 self/MLP 실제 경로, 기존 v0.6 shape와 block0 header.

## 자료

- AMD RDNA2 ISA, V_DOT8_I32_I4: https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
- AMD/LLVM gfx1030 assembler syntax: https://rocm.docs.amd.com/projects/llvm-project/en/docs-6.2.4/LLVM/llvm/html/AMDGPU/AMDGPUAsmGFX1030.html
- LLVM intrinsic semantics: https://llvm.org/docs/AMDGPUUsage.html

실제 ISA와 실행 결과가 해당 PC의 A등급 근거다. 최신 일반 ROCm builtins 표의 RDNA2 signed/mixed-dot 분류는 RDNA2 ISA와 서로 맞지 않아 지원 판정 근거로 단독 사용하지 않는다.
