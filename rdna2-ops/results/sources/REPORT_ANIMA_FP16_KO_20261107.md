# Anima FP16 모델 단위 검증 (2026-11-07)

연산 카탈로그의 "FP16 tiled Attention — 모델 미검증" 칸을 채우려고 FP16 모델을 하나 받아 실제 생성으로 확인한 기록이다. 수행자는 Claude다.

## 1. 대상과 조건

| 항목 | 값 |
|---|---|
| 모델 | Anima base v1.0 (2B, CircleStone Labs·Comfy Org). `anima-base-v1.0.safetensors` + `qwen_3_06b_base.safetensors` + `qwen_image_vae.safetensors`, 공식 HF 저장소, SHA-256 일치 |
| 실행 | moss lab의 ComfyUI runtime과 `.venv-moss`(PyTorch 2.12 + ROCm 10.2)를 읽기 전용으로 사용. `--use-pytorch-cross-attention --disable-comfy-compiler`. 모델과 출력은 이 lab에만 둠 |
| 계산 dtype | **FP16**. ComfyUI는 RDNA2에서 BF16을 끄고(`should_use_bf16`), Anima는 FP16을 지원한다(B: `supported_models.py`) |
| 작업 | text-to-image 1024×1024, Euler/simple, 30단계, CFG 4, seed 20261107 |
| 측정 | KSampler와 VAEDecode 구간, `torch.cuda.synchronize()` 경계. 같은 서버에서 conditioning 유지, 기준과 후보를 교대 실행 |

## 2. 실제 Attention 호출 (A)

`comfy.ops.scaled_dot_product_attention` 진입점에서 기록했다.

| q | k | dtype | mask | 30단계 호출 수 |
|---|---|---|---|---:|
| [2,16,4096,128] | [2,16,4096,128] | FP16 | 없음 | 840 (self) |
| [2,16,4096,128] | [2,16,512,128] | FP16 | 없음 | 840 (cross) |

배치 2는 CFG(조건/무조건)다. q stride는 `[8388608,128,2048,1]`로, BSHD 배치를 전치한 view다.

## 3. 연산 단위 측정 (A, GPU 단독, 중앙값)

ComfyUI 시작 설정과 같게 `allow_fp16_bf16_reduction_math_sdp(True)`를 켰다. 오차는 FP32 SDPA 대비 최대 절대 오차다.

| 경로 [2,16,4096,128] self | 시간 | 오차 |
|---|---:|---:|
| PyTorch SDPA (math) | 67.3 ms | 3.08e-4 |
| 직접 경로: `baddbmm(alpha)` → softmax → `bmm` | 49.9 ms | 3.03e-4 |
| GPT tiled 16×32 / 32×32 / 32×64 | 70.7 / 58.7 / 54.0 ms | 1.90e-4 |

SDPA 한 번의 GPU 커널 분해: P·V GEMM 42.9 ms, 원소별 복사 12.0 ms(3회), softmax 4.8 ms, `where` 4.5 ms, `isneginf` 3.6 ms, Q·K GEMM 3.5 ms. **같은 계산량인데 P·V GEMM이 Q·K보다 12배 느렸다.**

P·V 레이아웃 비교([32,4096,4096]×[32,4096,128], FP16):

| 방식 | 시간 | FP32 대비 오차 |
|---|---:|---:|
| `bmm(P, V)` | 19.5 ms | 5.97e-5 |
| `bmm(V^T, P^T)` 후 전치 | 22.9 ms | 5.97e-5 (위와 비트 일치) |
| **V를 전치 복사해 `bmm(P, Vt.transpose)`** | **6.7 ms** | 5.97e-5 (비트 일치 아님) |

rocBLAS가 두 번째 행렬을 전치(NT) 형태로 받을 때 이 shape에서 훨씬 빠른 커널을 고른다. 커널 이름 비교는 하지 않았다(추론).

이 절의 측정은 두 번 했다. 첫 번째는 모델 A/B 실행과 GPU가 겹쳐서 폐기했고(SDPA 123 ms 등), 위 표는 GPU가 빈 상태에서 다시 잰 값이다. 같은 실행 안에서도 `bmm(P,V)`가 19.5 ms와 35.3 ms로 흔들렸으니 연산 단위 절대값은 참고용이다.

## 4. 모델 단위 A/B (A, 30단계)

| 후보 | 기준 디노이징(warm) | 후보 디노이징 | 속도 | 최종 RGB 차이 |
|---|---|---|---:|---|
| GPT tiled (D128 → 32×32, self·cross 모두) | 80.75 / 81.34 / 81.14 s | 95.78 / 95.22 / 99.62 s | **0.85배(느림)** | max 0.786, RMSE 0.02 |
| 직접 경로 | 81.17 / 80.91 / 80.87 s → 80.91 | 62.97 / 63.19 / 63.18 s → 63.18 | **1.28배** | max 0.820, RMSE 0.0248 |
| **직접 경로 + V 전치** | 81.01 / 81.07 / 80.86 s → **81.01** | 56.58 / 56.58 / 56.62 s → **56.58** | **1.43배 (−30%)** | max 0.825, RMSE 0.0243 |

- 후보는 매 요청 1,680회(self 840 + cross 840) 적용됐고 fallback은 0회였다.
- 후보끼리, 기준끼리는 반복 실행에서 비트 일치했다(결정적).
- tiled 마지막 실행(99.62 s)은 연산 벤치와 GPU가 겹친 오염 표본이다. 앞의 두 값으로도 결론은 같다.
- VAE 0.55 s는 모든 경로에서 같다.

## 5. 판정

- **엄격 기준(최대 2/255, RMSE 0.25/255): 세 후보 모두 실패.** 30단계 동안 반올림 차이가 누적되어 픽셀 값이 갈라진다.
- **지각 기준: 직접 경로는 통과로 본다(Claude 육안).** 구도·표정·소품·색이 같고 미세한 결만 다르다. 비교 그림 `results/ab_manual/side_by_side.png`, `results/ab_manual_vt/side_by_side.png`(왼쪽 기준, 오른쪽 후보). 다른 모델의 독립 판정은 받지 않았다.
- **카탈로그 반영:** "FP16 tiled"는 Anima에서 느렸으므로 ❔ → ⚠️. 새 칸 "FP16 SDPA math를 직접 GEMM+softmax로 교체 + P·V 전치 레이아웃"은 지각 기준 ✅, 엄격 기준 ⚠️.

## 6. 왜 이런 결과인가 (추론)

- Anima는 2B로 작아서 시간의 대부분이 Attention이다. 단계당 self 28회 × 약 50–70 ms가 2.7 s/단계 중 2 s 가까이 된다.
- PyTorch FP16 SDPA math는 실제 행렬곱보다 안전 검사·복사·느린 P·V 커널 선택에 시간을 더 쓴다. MOSS FP32에서 본 "안전 검사 낭비"(가이드 §7)와 같은 성격인데, FP16에서는 P·V 커널 선택 문제가 더해졌다.
- GPT tiled 커널은 score 행렬을 만들지 않지만, rocBLAS의 큰 타일 GEMM만큼 dot2를 효율적으로 쓰지 못한다. cross(K=512)에서는 native보다 느리다.

## 7. 하네스 기록 (문서 결함)

- GPT v0.5 범용 하네스(`research_comfy_v05.py`)는 기준 생성까지는 동작했지만 capture 모드에서 실패했다. 이 ComfyUI의 `optimized_attention`이 텐서 대신 `AttentionTensorContainer`를 받아서 `.shape` 접근이 깨진다. 이 lab의 `scripts/anima_lab.py`는 한 단계 아래인 `comfy.ops.scaled_dot_product_attention`에서 잡았다.

## 8. 재현

```bash
L=~/DevSpace/anima-rdna2-lab01; M=~/DevSpace/moss-image-rdna2-lab01
cd $L && HIP_VISIBLE_DEVICES=0 bash $M/scripts/with_local_storage.sh $M/.venv-moss/bin/python -u \
  scripts/anima_lab.py --out $L/results/repro --modes warm,base,cand,cand,base,base,cand --cand manual_vt
```

GPU에 다른 작업이 없을 때만 실행한다. `--cand`는 `manual`, `manual_vt`, `tiled`.

## 9. 범위와 미검증

- 1024² 한 해상도, 한 프롬프트, 한 seed다. 다른 해상도(토큰 수), Turbo 변형, LoRA는 확인하지 않았다.
- 직접 경로는 score 행렬을 FP16으로 통째로 만든다. 배치 2·4096 토큰에서 호출당 약 1 GiB다. 해상도를 올리면 메모리가 제곱으로 는다.
- 운영 경로에 넣지 않았다. ComfyUI 소스는 수정하지 않았고, 하네스 프로세스 안에서만 바꿨다.
