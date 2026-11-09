# rdna2-ops v0.6 검수·재현 기록 (2026-11-07)

GPT v0.6 보고(카탈로그 v0.2, 가이드 v0.6)의 핵심 숫자를 이 PC에서 다시 실행해 확인한 기록이다. 수행자는 Claude다. 재현 산출물은 `rdna2-ops/results/claude_repro_20261107/`와 세션 scratchpad에 두었다.

## 1. 재현 결과

| 항목 | GPT 값 | Claude 재현 | 판정 |
|---|---|---|---|
| BF16 Conv2D→FP32 계산, MOSS VAE normal regular | 2.449배, RGB max 1.494/255, RMSE 0.187/255 | **2.464배**, 오차 동일, 후보 출력 SHA `e7d671ec75be…` 동일 | 재현 |
| 같은 경로 tiled 256/overlap 64 | 2.430배, max 0.779/255, RMSE 0.106/255 | **2.451배**, 오차 동일, 후보 출력 SHA `8cda3bac8d17…` 동일 | 재현 |
| Anima FP16 4D native 보존(`prescale4d_native`), warm | 1.1906배, latent·RGB 바이트 일치 | 기준 81.094/81.528/80.673 s, 후보 warm 67.739/67.728 s → **1.197배**. 3회 모두 1,680 hit·fallback 0, RGB SHA `52ea16286731…`로 native와 GPT digest 모두와 같음 | 재현 |
| 같은 경로 첫 후보 요청 | 83.019 s | **83.097 s** | 재현(초회 비용 현상) |
| Applio period Conv2D FP16, MIOpen NORMAL forward | 3shape 중 2개 NRMSE 0.780/0.773 | 독립 난수 입력으로 **0.7727/0.7747**, 중간 shape 0.0002 | 재현 |

실행 명령:

```bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
M=~/DevSpace/moss-image-rdna2-lab01
HIP_VISIBLE_DEVICES=0 bash $M/scripts/with_local_storage.sh $M/.venv-moss/bin/python -u conv_vae_v06.py \
  --output results/claude_repro_20261107/vae_bf16_conv_normal --project $M \
  --latent results/production_final_scope/shared_server/reference/latent_seed20261105_32x32.pt
HIP_VISIBLE_DEVICES=0 bash $M/scripts/with_local_storage.sh $M/.venv-moss/bin/python -u research_comfy_v05.py \
  --spec research_configs/anima_v06.json --out results/claude_repro_20261107/anima_4d_native --steps 30 --seed 20261107 \
  --modes warm,baseline,attn_prescale4d_native,attn_prescale4d_native,baseline,attn_prescale4d_native --port 8474
```

period Conv2D 확인 스크립트는 입력 `[8,C,H,2]`, 가중치 `[O,C,5,1]`, stride (3,1), padding (2,0)을 FP16으로 계산하고 같은 입력의 FP32 결과와 비교한다. MIOpen user DB는 매번 빈 폴더로 시작했다.

## 2. 새로 확인한 사실

### 2.1 MIOpen `ConvHipImplicitGemmV4R1Fwd`의 오답은 ROCm 버전과 무관하고, 환경변수 하나로 막힌다

| 런타임 | NORMAL forward NRMSE (C32/H2134, C128/H712, C512/H238) |
|---|---|
| Applio `.venv` PyTorch 2.11 + ROCm 7.2 | 0.7727 / 0.0002 / 0.7747 |
| Applio `.venv-rocm10` PyTorch 2.11 + ROCm 10.0 | 0.7727 / 0.0002 / 0.7747 |
| moss `.venv-moss` PyTorch 2.12 + ROCm 10.2 | 0.7727 / 0.0002 / 0.7747 |
| ROCm 10.2 + `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0` | **0.0002 / 0.0002 / 0.0002** |
| FAST(참고, ROCm 10.0) | 0.0002 / 0.0002 / 0.0002 |

- 확인된 사실: 해당 solver만 끄면 NORMAL에서도 세 shape가 FAST와 같은 오차 수준으로 돌아온다. GPT의 "V4R1 Fwd entry 제거가 인과 근거"라는 판정과 일치한다.
- 1D에서도 같다. FP16 Conv1D NORMAL에서 C128/T3200/k7은 forward NRMSE 0.657, C64/T6400/k7은 0.375였고, 변수를 끄면 0.0004로 돌아왔다(FP64 CPU 참조).

#### 속도와 실제 학습 (후속 측정)

Applio dilation 1 FP16 12 shape(`bench_conv_catalog_v06.py --set all_d1`)를 V4R1 Fwd를 끈 NORMAL로 다시 쟀다.

| 경로 | 12 shape fwd+dX+dW 합 | y/dX 정확도 |
|---|---:|---|
| FAST(GPT `all_d1_fast`) | 31.740 ms | 통과 |
| GPT 필터 FindDb를 FAST로 재생 | 17.530 ms | 통과 |
| **NORMAL + V4R1 Fwd 끔(Claude)** | **17.860 ms** | 12/12 통과(최대 NRMSE 0.00046) |

dW는 후보 FP16 값이 모두 finite였고 FAST 대조와도 통과했다. 다만 이 측정의 FP32 참조 dW가 4 shape에서 NaN이었다. 원인은 §2.3에 적었다.

그다음 이 DB를 복사해 FAST로 재생하는 실제 Applio 4 epoch를 돌렸다(`applio_epoch_v06.py`). 같은 세션에서 FAST 대조를 바로 앞에 실행했다.

| 실행 | 정상 epoch (2–4) | 정상 epoch 중앙값 | GPT `epoch_fast_a` 대비 최종 파형 RMSE |
|---|---|---:|---:|
| FAST 대조(빈 DB) | 9.247 / 9.261 / 9.297 s | 9.261 s | 0.0802 |
| **V4R1 Fwd·WrW53 끄고 시드한 DB, FAST 재생** | 8.480 / 8.473 / 8.541 s | **8.480 s → 1.092배** | 0.0739 |
| 참고: GPT 필터 DB | 8.418 / 8.408 / 8.441 s | 8.418 s (1.106배) | 0.1066 |

- 확인된 사실: 환경변수 방식은 필터 DB와 같은 수준의 속도를 낸다. 최종 파형 차이는 FAST 반복끼리의 차이 안에 있다. 보호 파일 16개도 그대로였다.
- 한계: 4 epoch 한 번이다. 청감과 장기 학습 동등성은 확인하지 않았다(GPT 판정과 같음).
- 정정: 이전에 Claude가 제안했던 "shape별 NORMAL/FAST 혼합으로 conv 약 16% 단축"은 정확도 검사가 빠진 추론이었다. 필터 없이 적용하면 일부 forward가 틀린다.

### 2.2 Anima 첫 후보 요청의 지연은 1초 prewarm으로 없어진다

tqdm 진행 표시를 단계별로 보면 첫 후보 요청은 30단계 전체가 약 2.6 s/it로 고르게 느렸고, 이후 후보 요청은 약 2.1 s/it였다. 늘어난 약 15초를 호출 1,680회로 나누면 호출당 약 9 ms다. 컴파일 비용처럼 첫 단계에 몰리지 않는다.

`anima-rdna2-lab01/scripts/anima_lab.py`에 `--cand prescale4d`, 요청별 PyTorch allocator 통계, `--server-extra`, `--prewarm`을 추가해 가설을 하나씩 확인했다.

| 실험 | 첫 후보 | 둘째 후보 | 판정 |
|---|---:|---:|---|
| 기본 (warm, base, cand, cand, base) | 83.02 s | 67.79 s | 재현. allocator `num_device_alloc` 21/21, `num_alloc_retries` 0/0, 최대 예약 3.965/3.965 GiB로 두 요청이 같음 → **allocator 가설 기각** |
| `--disable-dynamic-vram` | 83.11 s | 67.61 s | **DynamicVRAM 가설 기각** |
| 서버 시작 시 후보 attention을 self·cross shape별 5회 실행(1.068 s) | **67.87 s** | 67.87 s | **첫 요청 지연 사라짐** |

모든 후보 요청은 1,680 hit였고 기준과 latent·RGB가 비트 일치했다.

- 확인된 사실: 같은 shape로 후보 경로를 몇 번 미리 돌리면 첫 요청부터 warm 속도가 난다. GPT의 "1.19배는 warm 서버 범위"라는 조건은 1초 prewarm으로 실제 첫 요청까지 넓힐 수 있다.
- 추론: 첫 사용 시 커널 준비(Triton launcher 또는 rocBLAS/Tensile 커널 적재)가 한 번에 끝나지 않고 여러 호출에 걸쳐 비용을 내는 것으로 보인다. 어느 쪽인지는 분리하지 않았다.
- 한계: prewarm은 1024² 한 해상도의 두 shape로만 확인했다. 해상도가 바뀌면 해당 shape로 다시 prewarm해야 하는지는 확인하지 않았다.

### 2.3 FP32 NORMAL에서 `ConvOclBwdWrW53`이 dW를 NaN으로 낸다

- 재현: FP32 Conv1D `[8,128,3200]`, `[128,128,3]`, padding 1, NORMAL, 빈 user DB에서 dW에 NaN이 3,000–9,000개 나왔다. ROCm 10.2에서는 3회 연속, ROCm 10.0에서는 find 직후 첫 호출에서 나왔다. FP16과 FAST는 정상이었다.
- 원인: MIOpen 로그에서 `BWrW Chosen Algorithm: ConvOclBwdWrW53`, 커널은 `MIOpenCvBwdWrW`였다.
- 차단: `MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53=0`. ROCm 10.0과 10.2 모두 3회 NaN 0으로 돌아왔다. 이 MIOpen에서는 변수 이름이 예전 `..._DIRECT_OCL_WRW53`에서 `..._DIRECT_HIP_WRW53`으로 바뀌었다. 옛 이름은 아무 효과가 없다(로그로 확인).
- 의미: Applio FP16 학습에 직접 닿는지는 확인하지 않았다. 다만 FP32로 계산하는 정확도 참조와 FP32 학습은 이 solver에 오염될 수 있다. NORMAL을 쓰거나 FP32 참조를 만들 때는 두 변수를 함께 지정하는 것을 기본으로 둔다.

## 3. 결함

| 항목 | 내용 |
|---|---|
| 패키지 SHA | `RDNA2_PACKAGE_V06_MANIFEST.json`과 `rdna2-ops/SHA256SUMS`의 375개 중 3개가 PC와 다르다. `rdna2-ops/kernels.hip`은 PC에 없고, `research_configs/anima_fp8_e4m3fn_v06.json`, `anima_fp8_e5m2_v06.json`은 SHA가 다르다(PC 파일 mtime 03:19, manifest 04:36). zip과 PC 사본의 동기화 차이로 보이며 결과 숫자에는 영향이 없다 |
| 카탈로그 §5 | 모델별 적용 예가 v0.1 그대로였다. 이번 갱신에서 Anima 4D, MOSS VAE, Applio 필터 DB를 추가했다 |

## 4. 운영 파일

검수 시점(2026-11-07)에 다시 계산한 SHA가 GPT 기록의 after 값과 같았다.

- `results/v06_operating_files_audit.json`의 MOSS·Qwen·Anima 15개: 15/15 일치
- Applio `protected_after.json`의 16개: 16/16 일치

이번 재현도 운영 파일을 수정하지 않았다. MIOpen user DB는 세션 scratchpad와 연구 결과 폴더에만 만들었다.

## 5. 재현하지 않은 범위

- GPT 필터 FindDb 자체의 epoch 재실행과 선택 fold 1.013배(환경변수 방식으로 같은 수준 1.092배는 확인)
- BF16 GEMM 단위 수치, FP8 weight GEMM 36조합, Norm 융합 12 case, rocBLAS 레이아웃 48조합
- 다른 Conv family(period 외 2D, 3D, transpose)에서 두 변수의 효과, dilated shape를 NORMAL로 시드했을 때의 효과
- Anima 4D 경로의 다른 해상도·seed, 독립 지각 평가
