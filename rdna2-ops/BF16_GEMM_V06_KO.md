# BF16 GEMM — gfx1030 연산 카탈로그 실험 v06

측정일: 2026-11-06 UTC / PC 현지 2026-11-07. 대상: Radeon RX 6900 XT(gfx1030), 기존 MOSS venv의 PyTorch 2.12.0+rocm10.2.0a20261011, HIP 7.17.26405, Triton 3.8.0. 운영 환경·모델·설정 파일은 변경하지 않았다.

## 1. 카탈로그 판정

| 칸 | 판정 | 근거 등급 | 판단 |
|---|---|---|---|
| BF16 native GEMM | 지원하나 shape별 성능 주의 | A | 실제 rocBLAS BBS 커널이 실행된다. BF16 명령 부재가 PyTorch BF16 연산 불가를 뜻하지 않는다. |
| BF16→FP32 GEMM→BF16 | 연산 후보, 자동 적용 미채택 | A | 3 shape wall 1.022–1.618배. 실제 MOSS BF16 VAE RGB는 통과하지만 전체 decode 1.009/0.997배. Anima 실험 BF16 Linear 치환은 RGB 실패. |
| BF16→FP16 입력·출력→BF16 | 범위·오차 조건부 연구 후보 | A | 검사 제외 wall 1.215–2.022배. 입력 finite 검사가 별도로 0.157–0.478ms 들며, 입력 finite는 GEMM FP16 출력 overflow를 배제하지 않는다. 모델 채택 근거 없음. |
| BF16→FP16 입력·FP32 출력→BF16 | 이번 shape에서는 폐기 | A | HSS generic 커널 선택으로 native 대비 wall 0.914–0.961배. |
| BF16 load→FP16 dot2 직접 커널 | tall-skinny 한정 연산 후보 | A | 4096×4096 @ 4096×128에서 1.426배, 다른 두 shape 0.838–0.949배. 최종 모델 채택 아님. |
| BF16 load→FP32 SIMT 직접 커널 | 이번 타일 폐기 | A | wall 0.296–0.442배. 일반적인 직접 구현이 rocBLAS보다 빠르다는 근거 없음. |

현재 **자동 dispatch나 운영 반영은 하지 않는다.** 이 칸은 성공한 단위 경로와 실패한 통합 경로가 모두 채워진 상태다.

등급: **A=이 RX 6900 XT 실측**, **B=설치된 소스 확인**, **C=공식 문서**. 미측정 shape나 다른 모델로 위 배속을 확장하지 않는다.

## 2. 기준 커널과 하드웨어 계약

설치본에서 BF16 mm/bmm 기준선은 다음 이름 계열이었다(A).

- BF16 입력/출력: `Cijk_Alik_Bljk_BBS_BH_MT64x32x8_...ISA000...WS64...`
- 입력을 FP32로 바꾼 경로: `...SB_MT128x128x8_...ISA1030...WS32...`
- FP16 입력, FP32 출력: `...HSS_BH_MT64x32x8_...ISA000...WS64...`
- FP16 입력/출력: `...HHS_BH_MT128x128x16_...ISA1030...WS32...`

전체 커널 이름과 각 trace는 [단위 원자료](results/v06_bf16_gemm/results.json)에 있다. `ISA000`은 수집된 커널 이름의 일부이며, 그 문자열만으로 내부 명령을 단정하지 않는다. 이번 native rocBLAS는 profiler 커널 이름을 확인했고 native 바이너리의 전체 ISA 분해까지 수행한 것은 아니다.

직접 Triton dot2 후보는 생성 ISA에 `v_dot2c_f32_f16` 512개가 있고 matrix instruction은 없었다. 타일 64×64×32, warps 4, registers 186, spills 0, LDS 8192B. SIMT FP32 후보는 `v_fmac_f32_e32` 128개, spills 0이었다. 이는 각 컴파일된 커널의 정적 명령 수이며 실행 FLOP 수가 아니다(A).

AMD 공식 표에서 RDNA2는 FP16 fdot2를 지원하지만 BF16 fdot2와 FP8 dot4는 지원하지 않는다(C).

- [AMD floating dot builtins](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/docs-1.0.0/compiler-builtins/cross-arch/arithmetic-ref/dot-float-builtins.html)
- [HIP hardware features](https://rocm.docs.amd.com/projects/HIP/en/docs-6.3.2/reference/hardware_features.html)

## 3. 최소 3 shape 단위 실험

입력은 seed 20261108의 BF16 난수다. A는 표준편차 0.5, B는 0.5/√K로 생성했다. B는 물리적 [N,K] contiguous tensor의 transpose view다. 실제 VAE 출력값을 재사용한 단위 입력은 아니며 shape를 대표하도록 골랐다.

| ID | 연산 | B | M | N | K |
|---|---|---:|---:|---:|---:|
| square | mm, NT | 1 | 1024 | 1024 | 1024 |
| tall-skinny | mm, NT | 1 | 4096 | 128 | 4096 |
| VAE P·V 유사 | bmm, NT | 1 | 1024 | 384 | 1024 |

각 후보 2회 warm 후 순서를 회전·역전하며 3 round×3 call을 측정했다. 아래 **wall 중앙값에는 호출별 입력·출력 변환이 포함**된다. FP16 범위/정밀 재표현 검사는 별도이며 이 표에 들어 있지 않다. 준비·프로파일·CPU reference는 시간에서 제외했다.

### 동기화 wall 중앙값, ms

| 방법 | square | tall-skinny | VAE P·V 유사 |
|---|---:|---:|---:|
| native BF16 | 0.272655 | 0.938373 | 0.118016 |
| FP32 변환 | 0.177375 (1.537×) | 0.579946 (1.618×) | 0.115449 (1.022×) |
| FP16 입력·FP32 출력 | 0.283748 (0.961×) | 1.026729 (0.914×) | 0.129102 (0.914×) |
| FP16 입력·출력 | 0.134836 (2.022×) | 0.465563 (2.016×) | 0.097092 (1.215×) |
| Triton dot2 | 0.287385 (0.949×) | 0.657862 (1.426×) | 0.140822 (0.838×) |
| Triton FP32 SIMT | 0.921973 (0.296×) | 2.124725 (0.442×) | 0.376974 (0.313×) |

### GPU event 중앙값, ms — wall과 구분

| 방법 | square | tall-skinny | VAE P·V 유사 |
|---|---:|---:|---:|
| native BF16 | 0.266626 | 0.932750 | 0.112507 |
| FP32 변환 | 0.172123 | 0.574210 | 0.110237 |
| FP16 입력·FP32 출력 | 0.278487 | 1.021116 | 0.123603 |
| FP16 입력·출력 | 0.128040 | 0.460133 | 0.091610 |
| Triton dot2 | 0.281397 | 0.652383 | 0.135477 |
| Triton FP32 SIMT | 0.916320 | 2.119216 | 0.371667 |

### 정확도와 메모리

실행 전 단위 gate는 native 대비 normalized RMSE≤0.004, normalized max≤0.04로 정했다. 여기서 정규화 분모는 reference tensor RMS다. 최종 RGB gate와 다른 지표다. 모든 단위 후보가 finite이고 이 gate를 통과했다.

별도로 **CPU FP64에서 24×24 선택 출력 좌표를 전체 K에 대해 계산**했다. 독립 reference는 모든 출력 원소를 덮는 검증은 아니다. native의 해당 sample normalized RMSE는 0.001536–0.001722이고 FP16 출력 경로는 0.001574–0.001748이었다.

| 방법 | native 대비 최대 절대오차 범위 | native 대비 RMSE 범위 | 추가 peak MiB: square / tall / VAE |
|---|---:|---:|---:|
| native | 0 | 0 | 2 / 1 / 0.75 |
| FP32 변환 | 0.00390625 | 8.70e-6–1.04e-5 | 12 / 68 / 7 |
| FP16 입력·FP32 출력 | 0.00097656–0.00195313 | 1.17e-6–3.24e-6 | 8 / 35 / 4.25 |
| FP16 입력·출력 | 0.0078125 | 3.60e-4–3.63e-4 | 6 / 34 / 3.5 |
| Triton dot2 | 0.00390625 | 9.32e-6–1.30e-5 | 2 / 1 / 0.75 |
| Triton FP32 SIMT | 0.00390625 | 9.31e-6–1.17e-5 | 2 / 1 / 0.75 |

추가 peak는 호출 직전 PyTorch allocated memory 대비 증가량이며 **출력 tensor를 포함**한다. GPU 전체 사용량이나 external allocator를 뜻하지 않는다.

FP16로의 입력 재표현은 세 shape 모두 bit-exact하지 않았다. 작은 BF16 값의 FP16 underflow/반올림으로 최대 입력 차이 2.98e-8이 있었다. 따라서 이 표의 half 경로는 “손실 없는 BF16 대체”가 아니다.

| 별도 검사 | square | tall-skinny | VAE P·V 유사 |
|---|---:|---:|---:|
| 입력 FP16 finite 검사 wall, ms | 0.161546 | 0.477856 | 0.156642 |
| 같은 검사 GPU event, ms | 0.145087 | 0.466286 | 0.146500 |

검사 비용은 별도 측정값이며, 위 커널 시간과 단순 합산한 값을 실측 end-to-end로 표기하지 않는다. 특히 짧은 GEMM의 검사 비용이 커서 단위 가속이 사라질 수 있다.

기본 `bf16_gemm`는 엄격 FP16 roundtrip guard를 제공한다. 실제 2^80 overflow 및 2^-120 underflow 입력은 이 guard가 거부했다(A). 느슨한 finite guard를 쓰면 underflow/반올림을 허용하는 별도 계약이다. **입력이 유한하고 FP16로 정확히 표현돼도 누산 결과가 65504를 넘으면 FP16 출력은 overflow할 수 있고, FP16 출력 후 BF16 변환에는 이중 반올림이 있다.** 입력 검사와 출력 finite·오차 검사를 분리해야 한다.

## 4. 기존 PC 모델 최종 출력 A/B

### 4.1 실패한 연결·dtype 확인도 보존

- [MOSS VAE 첫 시도](results/v06_bf16_vae_fp16/results.json): `TorchDispatchMode`에서 aten.mm/bmm/addmm을 가로채려 했으나 **hit 0**. RGB 값 일치(torch.equal, signed zero 미구분)는 치환 검증 성공이 아니다. native profiler에는 BF16 BBS 커널과 bmm이 실제로 있었다. composite SDPA/MIOpen 내부 GEMM을 이 관측 위치에서 잡지 못한 **연결 범위 실패**다.
- [Anima --bf16-unet 첫 시도](results/v06_anima_bf16_2step/runs.json): storage는 BF16이어도 gfx1030의 `unet_manual_cast` 정책이 FP16 compute를 선택했다. F.linear 1,030회 모두 FP16, BF16 hit 0. CLI dtype 이름만 보고 활성 연산 dtype을 판단하면 안 된다(B+A).

### 4.2 Anima 요청 한정 BF16 compute: RGB 실패

[원자료](results/v06_anima_bf16_compute_2step/runs.json), [감시](results/v06_anima_bf16_compute_2step/gpu_audit.json).

기존 BF16 checkpoint를 쓰되 **연구 요청 동안만 `BaseModel.manual_cast_dtype`을 BF16로 변경하고 종료 시 원래 FP16로 복원**했다. 정상 운영 Anima는 FP16 compute다. 동일 seed 20261107, conditioning, 1024², 2 step의 완전 denoise+VAE로 진단했다. 30 step 성능/화질 결과가 아니다.

| 비교 | BF16 native control | FP32 GEMM 치환 |
|---|---:|---:|
| BF16 F.linear 실제 hit / 요청 | 1,030 | 1,030 |
| 후보 실제 호출 | 0 | 1,030 |
| denoise wall 중앙값 | 11.362760s (3회) | 6.834667s (2회) |
| BF16 control 대비 최종 RGB max | 0 | 0.78808594 |
| 같은 비교 RGB RMSE | 0 | 0.03694281 |
| strict gate | 기준 | 실패 |

후보의 계산 경로는 **F.linear→flatten mm→FP32 bias add→BF16 복귀**다. 입력 변환·GEMM 누산 순서뿐 아니라 fused addmm/별도 bias 반올림의 차이도 함께 바뀌었고, 이 Linear 재구성 자체가 원본 F.linear와 동등하다는 control을 통과한 실험은 아니다. 따라서 큰 RGB 오차를 “FP32가 BF16보다 부정확해서”라고 설명하면 안 된다. **복합 경로 치환이 최종 출력 계약을 못 지켰다는 실패**로 보존한다. 2 step에서 실패했으므로 30 step 반복은 하지 않았다.

`runs.json.image_error`는 처음 normal FP16 warm을 reference로 삼는다. 위 표는 반드시 **`comparison_errors.bf16_control.image`**를 사용한 값이다.

### 4.3 실제 MOSS BF16 VAE SDPA 경계: 정확도 통과, 가속 미채택

[원자료·profiler 커널 표](results/v06_bf16_vae_sdpa_fp32/results.json).

기존 MOSS 실제 denoising latent `latent_seed20261105_32x32.pt`와 기존 `flux2-vae_small.safetensors`를 썼다. 이 소유 프로세스의 BF16 VAE 정책에서 일반 VAEDecode와 VAEDecodeTiled(256, overlap64)를 실행했다. 운영 프로세스와 설정은 그대로다.

먼저 실제 `comfy.ops.scaled_dot_product_attention`를 관측했다. BF16 native의 Q/K sqrt-prescale, BF16 score 저장, BF16 softmax, BF16 P·V 출력 경계를 재구성한 **control은 두 디코딩 경로 모두 RGB 값 일치(torch.equal, signed zero 미구분)**였다. 당시 결과의 `bit_equal` 필드는 torch.equal을 사용했으며 실제 byte digest 검증을 뜻하지 않는다. reconstruction tensor는 저장하지 않아 사후 byte 비교도 할 수 없다. 이 제한을 보존하고 GPU를 재실행하지 않았다. 그 후 QK와 PV GEMM의 내부만 FP32로 계산하고 각 출력은 기존 위치에서 BF16로 돌렸다. masked/causal/dropout/GQA 등 지원하지 않는 계약은 fallback한다.

| 실제 경로 | eligible SDPA / 치환 GEMM | native→후보 wall 중앙값 | 배속 | RGB max / RMSE | strict |
|---|---:|---:|---:|---:|---|
| 일반 decode, Q/K/V [1,1,4096,384] | 1 / 2 | 182.286→180.660ms | 1.009× | 0.00390625 / 0.000721635 | 통과 |
| 분할 decode, S=1024×18회·512×4회·256×1회 | 23 / 46 | 915.351→917.806ms | 0.997× | 0.00312752 / 0.000440921 | 통과 |

RGB 기준은 사전 선언한 max≤2/255, RMSE≤0.25/255다. 255단계 단위로 일반 max/RMSE=0.9961/0.1840, 분할=0.7975/0.1124다.

각 경로 native/candidate 순서를 번갈아 3회씩 측정했다. 0.9% 수준의 일반 경로 차이만으로 안정적인 가속을 선언하지 않는다. 분할 경로는 오히려 소폭 느리다.

프로파일의 일반 decode native attention GEMM 합은 2.3421ms였고, FP32 후보는 1.3211ms였다. **이 operator attribution 숫자를 device kernel 표와 더하지 않는다.** 전체 native device activity는 165.7301ms라 attention 두 GEMM을 줄여도 전체 디코딩 개선 여지가 작다. BBS 64×32×8 kernel group은 약 134ms이고, 이 집계에는 convolution의 내부 GEMM과 PV 1회가 함께 들어 있다. kernel 이름만 보고 전부 attention이나 전부 convolution으로 분류하지 않는다.

이번 추가 BF16 경계 후보의 lowvram 검증은 하지 않았다. 기존 GN/VAE dtype 실험의 lowvram 검증 결과를 이 새 후보에 자동으로 물려주지 않는다. 가속 이득이 성립하지 않아 운영 채택 단계로 진행하지 않았다.

## 5. 새 모델에서 이 칸을 쓰는 절차

1. checkpoint dtype, CPU mmap dtype, GPU resident dtype, **실제 F.linear/mm/bmm/SDPA 입력 dtype**를 따로 기록한다. CLI dtype 옵션만으로 BF16 활성 경로를 확정하지 않는다.
2. native trace에서 BBS/HSS/HHS/SB 실제 kernel 이름과 shape/stride를 확인한다. 다른 layout의 결과를 가져다 쓰지 않는다.
3. FP32 변환 후보는 입력 범위를 보존하되 변환 비용과 추가 memory까지 측정한다. bias·flatten·softmax dtype 경계가 바뀌는지 기록한다.
4. FP16 재표현 후보는 BF16→FP16 roundtrip, 입력 nonfinite/overflow/underflow, GEMM 출력 finite, 오차를 별도 gate로 둔다. range scan을 사용하면 실제 호출 시간에 포함한다.
5. 3 shape 이상 단위 gate를 통과한 후보만 실제 모델 경계에 연결한다. hit 0이면 통합 미검증으로 돌려놓는다.
6. 원본 연산을 재구성하는 control부터 출력 동등성을 확인하고, 최종 RGB/오디오/logit 등 해당 모델 출력 계약을 비교한다. 이번 이미지의 strict 기준을 다른 modality에 그대로 쓰지 않는다.
7. 엄격 gate 실패나 실사용 속도 무이득은 실패/미채택으로 남긴다. 요청 한정 adapter를 벗어나 자동 적용하지 않는다.

## 6. 재사용 코드·실행·감시

- [BF16 연산 후보](rdna2_ops/bf16_gemm_v06.py): 2D mm/동일 batch 3D bmm, BF16 입력·출력. 기본 exact-half guard.
- [BF16 SDPA GEMM 경계](rdna2_ops/bf16_sdpa_gemm_v06.py): 기존 BF16 rounding 경계를 보존하는 reference/candidate.
- [단위 하네스](bench_dtype_gemm_v06.py), [VAE 연결](dtype_vae_v06.py), [요청 한정 Linear adapter](dtype_switches_v06.py).

기존 wrapper와 venv에서 실행:

```bash
scripts/with_local_storage.sh .venv-moss/bin/python -u \
  ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops/bench_dtype_gemm_v06.py \
  --kind bf16 --output /새로운/연구결과디렉터리
```

위 상대 wrapper/venv 경로의 기준 디렉터리는 MOSS 프로젝트다. VAE 재현 명령은 rdna2-ops 디렉터리에서 같은 wrapper/venv를 사용해 다음 인수를 준다:

```text
dtype_vae_v06.py
  --project ~/DevSpace/moss-image-rdna2-lab01
  --latent results/production_final_scope/shared_server/reference/latent_seed20261105_32x32.pt
  --method fp32 --boundary sdpa --output results/새결과디렉터리
```

| 실험 | durable job ID | GPU audit |
|---|---|---|
| BF16 3 shape | 72491549-5108-4b25-9c93-adb3f7d6a0ad | 88 samples, valid |
| VAE TorchDispatch no-hit | 3b04aae4-d7dc-4ad9-a396-408308744d05 | 119 samples, valid |
| Anima CLI BF16 no-hit | 80eb6a59-2423-4c1e-b447-2231427aa74a | 별도 gpu_audit.json |
| Anima BF16 compute 2 step | 3ac544c0-94de-45fe-a268-6283c867d01e | 261 samples, valid |
| VAE SDPA BF16/FP32 경계 | 178d66a3-1555-4719-8aa6-e09b859c627d | 121 samples, valid |

명시된 valid run에는 foreign KFD PID와 감시 오류가 없었다. 0.25초 KFD polling+연구 flock으로 겹침을 감시한다. 그래픽 작업이나 polling 사이에 끝난 극단적으로 짧은 외부 작업까지 배제하는 증거는 아니다.

주요 측정 소스 SHA-256:

| 소스 | SHA-256 |
|---|---|
| bf16_gemm_v06.py | 9d0797dbbb23af1e21cb281a916f7edab980eaedd03725e192cb7f7251c00227 |
| bench_dtype_gemm_v06.py | 3da14d6368e938d364702c5da27670256ceae2d4162d501b3398c5ba1c2b733f |
| bf16_sdpa_gemm_v06.py | 883d908a3c7451efb163ea45107bce3a97697e10823d09c4fca7770a6f76f4cf |
| dtype_vae_v06.py (SDPA run) | 5a6ba889eb649855fc707035cb0ca266cc872f4bfd54a7aeffca1c082dfe872a |
| dtype_switches_v06.py (BF16 Anima run) | 23aacbd4027745ba3f26116f969835def8b793246b60922e7460d7e4c4451fe6 |

이후 FP8 adapter 관측 기능이 추가되어 dtype_switches의 현재 hash는 다를 수 있다. 위 표는 측정 당시 hash이고, BF16 계산 함수는 동일하다.

측정 당시 VAE harness는 [source snapshot](results/v06_bf16_vae_sdpa_fp32/dtype_vae_v06.py)에 보존했다. 이후 현재 `dtype_vae_v06.py`(SHA `7bb93427c66829abb7e12d111fc9d058e60d9c6e091b2a3fa0ae321cf6043bd1`)는 `value_equal=torch.equal`과 `bit_equal=동일 dtype/shape의 uint8 view 비교`를 분리했다. 이 수정이 과거 raw 결과의 의미를 소급 변경하지는 않는다.

