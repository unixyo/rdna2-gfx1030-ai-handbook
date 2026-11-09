# 사용설명서 리허설: Qwen-Image-2.1 INT8 ConvRot (2026-11-06)

`rdna2-ops/RDNA2_OPTIMIZATION_GUIDE_KO.md`(v0.4)의 2절과 10절만 보고 처음 다루는 모델을 최적화할 수 있는지 시험한 기록이다. 수행자는 설명서 작성 맥락이 없는 다른 모델(Claude)이다. 문서 본문은 수정하지 않았고, 발견한 결함은 아래 3절에 모았다.

## 1. 대상과 조건

| 항목 | 값 |
|---|---|
| 모델 | `qwen_image_2.1_int8_convrot` + `qwen3vl_8b_int8_convrot` 인코더 + `qwen_image_2.1_vae_bf16` |
| 실행 | `~/DevSpace/qwen-image-21-rdna2-lab01` 기존 `.venv`, ComfyUI 0.37.2, Studio 운영 플래그(`--use-split-cross-attention --reserve-vram 1 --vram-headroom 3 --disable-triton-backend`, `Q21_RDNA2_INT8_PATCH=1`) |
| 작업 | 참조 이미지 1장(960×640 합성, 캐릭터 2명) 편집 생성, 출력 960×640, Euler/simple, CFG 1, seed 20261106 |
| 측정 | 디노이징(KSampler) + VAEDecodeTiled 구간, `torch.cuda.synchronize()` 경계, 같은 서버에서 conditioning 유지 |
| 격리 | 전용 포트 8461 서버를 하네스가 띄우고 종료. 입력·출력은 scratch 경로. lab 소스·설정·Studio는 변경하지 않음 |

## 2. 결과

### 2.1 프로파일 (A, 8단계, profiler 켬)

profiler를 켠 디노이징 wall은 20.86 s, VAE는 2.68 s였다. 아래는 `key_averages()`의 연산 단위 self GPU 시간이다(커널 행과 연산 행을 섞지 않고 연산 행만 골랐다).

| 연산 | self GPU ms | 호출 | 비고 |
|---|---:|---:|---|
| `comfy_kitchen::int8_linear` (patientx Triton per-row dequant GEMM) | 6,851 | 1,280 | 커널 `_int8_matmul_dequant_per_row_kernel` 8,818 ms |
| `aten::bmm` (Attention QK, PV) | 2,936 | 720 | `[32,2400,128]×[32,128,4964]` 등 |
| `aten::copy_` | 2,383 | 2,931 | Attention용 q/k/v contiguous 약 1.3 s, prefix KV cache stack 0.79 s |
| `aten::mul` | 1,870 | 500 | 이 중 score 행렬 `[32,2400,4964]` 스케일 1,759 ms |
| `aten::_softmax` | 1,845 | 352 | score 행렬 전체 |
| Host→Device memcpy | 886 | 380 | DynamicVRAM 가중치 이동 |

Attention 호출(하네스가 `comfy.ldm.qwen_image21.model.optimized_attention`을 감싸서 기록):

| q | k | heads | dtype | mask | 함수 | 8단계 호출 |
|---|---|---:|---|---|---|---:|
| [1,2400,4096] | [1,4964,4096] | 32 | FP32 | 없음 | `attention_split` | 512 |
| [1,2400,4096] | [1,2408,4096] | 32 | FP32 | 없음 | `attention_split` | 64 |
| [1,156,4096] | [1,2564,4096] | 32 | FP32 | [156,2564] | `attention_split` | 64 |
| [1,8,4096] | [1,8,4096] | 32 | FP32 | [8,8] | `attention_split` | 64 |

판단: INT8 GEMM이 1위, Attention의 score 행렬 처리(QK bmm, 스케일 mul, softmax, PV bmm, 복사)가 합쳐서 그다음 덩어리다. Attention은 SDPA가 아니라 ComfyUI `attention_split`이고, head 32·head_dim 128·K 길이 4964(prefix cache)라 설명서의 자동 적용 범위(`[1,24,S,128]`, S≤4096, SDPA)에 들지 않는다.

### 2.2 설명서 레시피 대응

| 설명서 레시피 | qwen 적용 | 근거 |
|---|---|---|
| A. GN+SiLU 융합 | 해당 없음 | 프로파일 상위에 GroupNorm 없음(VAE 2.7 s, 비중 작음) |
| B. Comfy VAE 연결 | 시도 안 함 | 위와 같음 |
| C. INT8 전치 복사 제거 | 해당 없음 | patientx 경로는 `weight.contiguous()`만 하고 전치 복사가 없음(B: `rocm_int8_kitchen_patch.py:137`) |
| D. FP32 math SDPA 안전 검사 축소 | 해당 없음 | SDPA를 타지 않음, shape 범위 밖 |
| 9.1 tiled Attention | 시도 안 함 | FP32 D128에서 native보다 느리다는 설명서 기록(A)을 따름 |
| (원리 적용) stride view로 복사 제거 | **실패: 비트 일치, 2배 느림** | 아래 2.3 |
| (신규) score 스케일을 GEMM epilogue로 접기 | **성공: 비트 일치, 전체 1.08~1.10배** | 아래 2.3·2.4 |

### 2.3 Attention 연산 단위 측정 (A, 랜덤 FP32 입력, 중앙값)

| 후보 | 2400×4964 | 2400×2408 | 출력 |
|---|---:|---:|---|
| native `attention_split` | 29.85 ms | 16.38 ms | 기준 |
| q/k/v `.contiguous()` 제거(strided einsum) | 60.82 ms (0.49배) | 30.57 ms (0.54배) | 비트 일치 → **폐기** |
| score `mul_` in-place | 29.49 ms | 16.10 ms | 비트 일치, 이득 미미 |
| `baddbmm(beta=0, alpha=scale)` | 22.71 ms (1.31배) | 12.98 ms (1.26배) | 비트 일치 → **채택 후보** |

원자료: `results/micro_split_strided_view_REJECTED.log`, `results/micro_scale_variants.log`, `results/micro_split_alpha_wrapper.log`.

비트 일치 이유(추론): FP32 GEMM 누산 결과를 FP32로 반올림한 뒤 alpha를 곱해 다시 반올림하므로, `einsum(...)`(1회 반올림) 뒤 `* scale`(1회 반올림)과 같은 두 번의 반올림이 된다. 이 설치의 rocBLAS 커널에서 실측으로 확인했고, 다른 BLAS 구현에서 같다고 단정하지 않는다.

### 2.4 전체 이미지 A/B (A)

후보 `split_alpha`(`scripts/candidates.py`)는 mask 없음·FP32·batch 1·메모리 여유가 있는 호출만 바꾸고 나머지는 원래 `attention_split`으로 넘긴다. 25단계 1회당 대체 832회, 원래 경로 64회(mask가 있는 호출)였다.

| 단계 | 기존(warm) | 후보 | 속도 | latent·RGB |
|---|---|---|---:|---|
| 8 | 22.410 / 22.415 / 22.446 s → **22.415 s** | 20.665 / 20.832 / 20.652 / 20.687 s → **20.676 s** | 1.084배 (−7.8%) | 모두 max_abs 0, RMSE 0 |
| 25 | 58.939 s (warm 1회) | 53.516 / 53.525 s | 1.101배 (−9.2%) | 모두 max_abs 0, RMSE 0 |

첫 요청(모델 로딩 포함, 8단계 32.4 s / 25단계 67.7 s)은 제외했다. 25단계 기존 warm 표본은 1개라 참고값이다. 결과 이미지: `images/sample_library_25step.png`.

## 3. 설명서 결함 (리허설에서 막히거나 틀린 곳)

1. **프로파일 진입점이 MOSS 전용이다.** 2.3절의 `rdna2_python`은 MOSS `.venv-moss`와 `with_local_storage.sh`를 고정하고, `profile_comfy_workflow.py`는 MOSS 워크플로 파일과 노드 번호(`"8"`, `"11"`)가 박혀 있다. ComfyUI 서버형 모델은 `run_existing_request()` 같은 함수가 없어서, 서버 프로세스 안에서 `nodes` 모듈 로딩을 가로채 KSampler/VAEDecode를 감싸는 하네스를 새로 써야 했다(`scripts/rehearsal.py`). 범용 하네스나 그 작성법이 필요하다.
2. **프로파일 예시 코드가 중복 합산을 부른다.** `prof.key_averages()`를 `self_device_time_total`로 정렬하면 연산 행(`comfy_kitchen::int8_linear`)과 커널 행(`_int8_matmul_dequant_per_row_kernel.kd`)이 함께 나온다. 합계가 38.4 s로 실제 디노이징 wall 20.9 s를 넘었다. 연산 행과 커널 행을 나눠 보라는 안내가 필요하다.
3. **Attention이 SDPA를 탄다고 전제한다.** ComfyUI는 실행 플래그(`--use-split-cross-attention`, `--use-pytorch-cross-attention` 등)에 따라 `optimized_attention`이 다르다. qwen은 `attention_split`이라 SDPA hook으로는 Attention이 보이지 않는다. "모델이 실제로 부르는 attention 함수를 먼저 확인한다"는 단계를 2.1절에 넣어야 한다.
4. **"stride를 받으면 복사를 빼라"는 일반화가 틀릴 수 있다.** INT8 Triton 커널(MOSS)에서는 맞았지만, rocBLAS FP32 bmm은 strided 입력에서 2배 느려졌다. 2.3절 표의 "반복 transpose/cast/복사 → view" 행에 "backend별로 실측, rocBLAS는 contiguous가 빠를 수 있음"을 덧붙여야 한다.
5. **수치 보존형 스케일 접기가 레시피에 없다.** `einsum*scale` → `baddbmm(alpha=scale)`은 이 설치에서 비트 일치였다. MOSS 7절의 "Q/K 선행 스케일"과 다른, 후행 스케일 구현(`attention_split`, `attention_basic` 계열)용 레시피로 추가할 가치가 있다.
6. **1.1절 표가 백엔드 차이를 구분하지 않는다.** 같은 "INT8 ConvRot"라도 MOSS는 `rdna2_int8_matmul`(stride-aware dot), qwen은 patientx `int8_fused_kernel`(per-row dequant)이다. 레시피 C는 백엔드 이름까지 확인해야 적용 여부가 갈린다.

## 4. GPT 연구 과제에 반영할 근거

- **INT8 GEMM이 두 모델 공통 1위 비용이다.** qwen에서는 patientx `_int8_matmul_dequant_per_row_kernel`이 8.8 s/8단계로 최대다. gfx1030 `v_dot4` 기반 INT8 GEMM 튜닝(타일·dequant 융합)은 두 모델 모두에 걸린다.
- **FP32 Attention K 길이가 4096을 넘는다.** qwen 960×640 편집에서 K=4964였다. score 행렬이 호출당 32×2400×4964×4 B ≈ 1.42 GiB다. FP32 tiled Attention 과제(D128 spill 해결)는 qwen에 바로 쓰인다.
- **score 행렬을 여러 번 읽는 구조가 남아 있다.** 스케일을 접은 뒤에도 softmax와 두 번의 bmm이 1.4 GiB 행렬을 읽고 쓴다. tiled로 없애는 것이 다음 단계다.

## 5. 재현

```bash
S=~/DevSpace/rdna2-pro-research-20261102/rehearsal_qwen_20261106/scripts
mkdir -p "$S/input" && cp "$S/../images/ref_image.png" "$S/input/"   # rehearsal.py는 스크립트 옆 input/을 읽는다
~/DevSpace/qwen-image-21-rdna2-lab01/.venv/bin/python -u "$S/rehearsal.py" \
  --out /tmp/qwen_ab --modes base,cand,base,cand,cand,base,base,cand --steps 8 --candidate split_alpha
```

GPU가 비어 있고 Studio(8451)가 돌지 않을 때만 실행한다. `REHEARSAL` 줄의 `latent_err`·`image_err`가 0이어야 통과다.

## 6. 범위와 미검증

- 운영 Studio에는 적용하지 않았다. 적용하려면 Studio 워커 시작 경로에 opt-in 스위치와 롤백을 따로 만들어야 한다.
- 측정은 960×640 편집 1종이다. 텍스트만 쓰는 생성, 다른 해상도, LoRA·Turbo 경로는 확인하지 않았다.
- 메모리 부족으로 `attention_split`이 여러 조각으로 나뉘는 상황에서는 후보가 원래 경로로 넘긴다. 그 분기의 실제 발생 여부는 기록하지 않았다.
- 서버 재시작 간 동일성은 확인하지 않았다. 모든 비교는 한 서버 안에서 했다.
