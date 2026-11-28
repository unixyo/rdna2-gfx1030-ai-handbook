# RX 6900 XT (RDNA2, gfx1030) AI 최적화 핸드북

RDNA2(gfx1030)에는 WMMA·FP8·BF16 dot이 없고, Triton은 3.8이 마지막 지원 버전이다. 그래도 dot2·dot4·dot8은 있다. 같은 세대 GPU를 계속 쓰는 사람에게 참고가 되도록 측정 기록을 정리해 둔다.

*RDNA2 (gfx1030) has no WMMA, FP8 or BF16 dot, and Triton 3.8 is the last release that supports it, but dot2/dot4/dot8 are available. These measurement notes are shared for others still running this generation.*

RX 6900 XT 한 장에서 PyTorch/ROCm 기반 AI 작업(이미지 생성, 3DGS, 음성 학습, MoE LLM 프리필)을 **연산 단위로 실측하고**, 원래 계산 의미를 지키면서 비싼 연산만 바꾼 기록이다. 성공한 것, 실패한 것, 아직 모르는 것을 함께 남긴다.

## 어디서부터 읽나

1. [연산 카탈로그](RDNA2_OP_CATALOG_KO.md): 입구. "이 연산이 gfx1030에서 되나, 얼마나 빠르나, 원본과 얼마나 다른가"를 연산별로 정리했다.
2. [최적화 가이드](rdna2-ops/RDNA2_OPTIMIZATION_GUIDE_KO.md): 절차·레시피·판정 기준(§1.2)·새 모델 체크리스트(§10). §9.x가 실험별 기록이다.
3. `rdna2-ops/results/*/…_KO.md`: 실험별 정본 보고서와 원자료 요약.

문서는 한국어다. 아래 요약만 영어를 같이 둔다.

## 다른 gfx1030 사용자에게 도움이 될 수 있는 결과

| 발견 | 요지 | 위치 |
|---|---|---|
| spawn DataLoader 워커가 GPU 큐를 잡는 문제 | 학습 스크립트를 `__main__`으로 돌리면 spawn 워커가 다시 import하면서 각자 HIP 큐를 만든다. 학습 kernel이 전부 1.5–2배 느려졌다(22 → 15 s/epoch). 컨텍스트를 만든 뒤 `HIP_VISIBLE_DEVICES=-1`로 해결했다. `/sys/class/kfd/kfd/proc/*/queues`로 확인한다. | 가이드 §9.21 |
| MIOpen dilated GEMM workspace 0 | MIOpen `8d1ae90e`의 legacy 질의가 dilation>1에서 workspace 0을 돌려줘서, NORMAL 탐색이 후보를 조용히 빼 버린다. | 가이드 §9.21 |
| MIOpen 오답 solver 두 개 | `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0`, `MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53=0`로 끈다. NORMAL·FP32 Conv 참조에 필수다. | 카탈로그 §4.3 |
| TunableOp 비결정 | 선택된 FP32 GSU4 rocBLAS 커널의 atomic 누산 때문이다. 해당 행만 Default로 바꾸면 재현성이 돌아온다(Anima 1.79배 유지). | 가이드 §9.20 |
| 작은 batch Conv의 im2col+GEMM | period Conv2d·ConvTranspose dW에서 MIOpen이 batch별 작은 GEMM을 반복한다. Triton 직접 커널로 방향별 최대 27배 빨라졌다. 학습 전체로는 −8.8%였지만, 묶음 B′는 청취에서 보류됐다. | 가이드 §9.22 |
| Direct HIP Conv/AdamW + 가변 길이 Coset `F(4,3)`·입력 재사용·2-Tier 캐시 (Applio RVC `v6.2`→`v10`) | `v1.1 B+native` 위에 Direct HIP + C++ shim + zero-sync FP64 AdamW(`v6.2`)로 `11.50 → 7.11 s/epoch`(200ep 청취 통과). 이어서 `F(4,5)` Winograd(`v7`, `6.285 s/ep`) → `v8/v9`(`0-ULP` Wave32 `dW` LDS XOR 스쿌 + MPD 폴리페이즈 + 75행 CSV, **`11.55 → 5.86 s/ep` [`1.97배`] / `v8` 대조 `5.97 → 5.91 s/ep`**) → **`v10`**(`gfx1030/1031/1032` 213커널 0-spill, 가변 길이 Coset `F(4,3)` + 파형 오차 `0` 중첩 입력 재사용 `-2.02%` + 128슬롯 캐시 `eviction 0` + 청크 `empty_cache()` 제거)으로 273.9초 8청크 실제 변환을 **`10.03 → 5.62 s`(`v9` FP32 대비 `−44.0%`, 실시간 `48.7배`, SNR `52.37 dB`)** 까지 단축했다. | 카탈로그 §4.3·§4.7·§5·§6, [applio-rdna2-direct-hip](https://github.com/unixyo/applio-rdna2-direct-hip) |
| Direct HIP `D=128/64` Flash·Sage Attention + `256²`/LDS-only INT8 GEMM + Unit-Temporal `Conv3d→Conv2d` VAE (`v4.0`) | 330개 zero-spill Wave32 커널 + C++ shim(`v4.0`). Qwen VAE의 `T=1` `Conv3d` 716개를 `Conv2d`로 하향해 디코딩을 **`2.6463 → 0.9135 s`(`2.896배`, `616.6→419.7 MiB`, RGB PSNR `75.08 dB`)** 로 줄였고, `0-ULP` `CpuHalfPrefixCache`(`2.69→1.34 GB`) + `256×256` 타일(ID9)·LDS-only 배리어(ID10, `67.7 TOPS`) INT8 GEMM과 합쳐 웜 요청 전체를 **`12.42 → 10.12 s`(`−18.5%`)** 로 단축했다. 제로카피 strided 어텐션(`P2`)으로 MOSS `768×1152` **`3.67 s`**, Idol **`0.983 s/pass`** 달성(`Sage2++` `INT8 PV`는 세부 드리프트 검증 후 실험 opt-in 분리). | 카탈로그 §4.1·§4.2·§4.3·§4.5·§5, [rdna2-diffusion-sage-hip](https://github.com/unixyo/rdna2-diffusion-sage-hip) |
| INT8 dot4 융합 GEMM | MOSS INT8 ConvRot 생성이 2.28배 빨라졌고 출력은 바이트 일치했다. | 가이드 §9.6 |
| Linux의 "FSR 켜짐"은 FSR4가 아닐 수 있다 | Proton에서 무수정 FSR SDK 2.4 DLL(4.1.2)은 FSR 3.1.6을 골랐다. Navi2 지원 검사가 Wine이 처리하지 않는 AMD 전용 D3DKMT 조회에 걸린다. 커뮤니티 exact 셰이더는 4.1.2를 골랐고, 그 위에 clamp·shift 변환을 이식해 업스케일러 시간을 1.4–1.9% 더 줄였다(출력 바이트 동일, 1080p 정지는 미확정). | 가이드 §9.23 |
| Triton 3.8 고정 | 3.9부터 RDNA2 지원이 빠졌다. hipBLASLt는 gfx1030을 지원하지 않는다. | 카탈로그 §2–3 |

*English summary: worker-process GPU queues slowing training 1.5–2×; a MIOpen workspace-query bug for dilated GEMM; two MIOpen solvers that return wrong results; TunableOp nondeterminism traced to FP32 split-K atomics; Triton direct kernels replacing MIOpen's per-batch small GEMMs; a 213-kernel (`gfx1030;gfx1031;gfx1032` zero-spill) Direct HIP + FP64 zero-sync AdamW + arbitrary-length `F(4,3)` Coset Winograd (with `0`-diff overlapping input window reuse) + `F(4,5)` MPD Winograd + 2-Tier cache stack for Applio RVC (`11.50 → 7.11 s/ep` listening-verified `v6.2`, `5.861 / 5.910 s/ep` `v9`, and `10.03 → 5.62 s` 273.9s full-recording E2E in `v10`); a 330-kernel zero-spill Direct HIP Wave32 `D=128/64` Flash/Sage Attention + `256²`/LDS-only exact INT8 GEMM + Unit-Temporal `Conv3d→Conv2d` VAE lowering + `0-ULP` `CpuHalfPrefixCache` stack (`rdna2-diffusion-sage-hip` v4.0: Qwen VAE `2.6463 → 0.9135 s`, Qwen warm request `12.42 → 10.12 s`, MOSS `768×1152` `3.67 s`, Idol `0.983 s/pass`); an INT8 dot4 fused GEMM (2.28×, bit-exact); FSR 4.1.2 under Proton silently falling back to FSR 3.1.6 with the stock SDK DLL, plus a byte-exact 1.4–1.9% shader gain on top of the community exact build. Details are in the Korean docs above.*

## 관련 리포

이 핸드북은 연산 단위 연구 기록이다. 모델을 실제로 띄우는 방법은 아래 레시피 리포에 있다.

| 리포 | 내용 |
|---|---|
| [rdna2-gfx1030-imagegen-recipes](https://github.com/unixyo/rdna2-gfx1030-imagegen-recipes) | Qwen-Image 2.1, MOSS-Image, MiniMax H3의 ComfyUI 설정·워크플로·패치·시간 측정 |
| [strata-rdna2-gfx1030-recipe](https://github.com/unixyo/strata-rdna2-gfx1030-recipe) | Strata(Qwen3.8-Flash-Next) 엔진 설정·패치(`0001..0007`, `v4` `DFlash2` Track A `86.2 → 91.2 tok/s`·`EXACT_V3` `68.3 tok/s`·프리필 `combine -58%`)·측정 |
| [applio-rdna2-direct-hip](https://github.com/unixyo/applio-rdna2-direct-hip) | Applio RVC 학습·전체 녹음 추론용 Direct HIP 커널 213종(`gfx1030/31/32` 0-spill)·C++ shim·Coset `F(4,3)` 입력 재사용·`F(4,5)` Winograd·2-Tier 런타임 (`v6.2` 청취 완료 / `v9` `5.86~5.91 s/ep` / `v10` `5.62 s`) |
| [rdna2-diffusion-sage-hip](https://github.com/unixyo/rdna2-diffusion-sage-hip) | Qwen-Image-2.1·MOSS-Image·Idol 3DGS용 Direct HIP Wave32 Flash·Sage Attention, `256²`/LDS-only Exact INT8 GEMM, Unit-Temporal `Conv3d→Conv2d` VAE 하향 (`v4.0`, 330 커널 0-spill) |

- 이미지 레시피 리포에는 2026-11-10에 MOSS INT8 dot4 융합(가이드 §9.6)과 Qwen 후행 scale alpha 접기(가이드 §7.1)를 기본값으로 반영했다. 각 레시피의 "2026-11 update" 절에 있다.
- `rdna2-ops/production/MOSS_RDNA2_INT8/`는 연구 시점의 통합 스냅샷이다(fused 레이아웃 이전). 실행용으로는 레시피 리포의 `moss-image/custom_nodes/MOSS_RDNA2_INT8/`(fused 포함)를 기준으로 한다.
- Strata는 레시피 리포가 운영 설정을 다룬다. 이 핸드북의 §9.18은 그 위의 프리필 연구(J 타일, warm 병목)이며, 2026-11-26~28에 `RDNA2_OP_CATALOG_KO.md` §4.6·§5에 Strata `2af30ccaa` 기반 Phase 1·2·v3·v4(`P4+P5` `native_mmvq_il`, Zen4 `Q2_0` 원배치, `DFlash2` Track A `86.2 → 91.2 tok/s`, GPU `combine_weighted` `-58%` 및 PCIe 스테이저 fault 방어) 통합 검증 결과를 반영했다.
- Applio RVC 학습·추론은 2026-11-27~28에 `RDNA2_OP_CATALOG_KO.md` §4.3·§4.7·§5·§6에 [applio-rdna2-direct-hip](https://github.com/unixyo/applio-rdna2-direct-hip) 기반 Direct HIP `v6.2`(200 epoch 실전 청취 통과), `v7`(`F(4,5)` Winograd), `v9`(`5.861 / 5.910 s/ep`), `v10`(`gfx1030/31/32` 213커널 0-spill, 가변 길이 Coset 입력 재사용 `-2.02%`, 273.9초 녹음 `10.03 → 5.62 s`) 결과를 반영했다.
- Diffusion·3DGS 추론은 2026-11-27~28에 `RDNA2_OP_CATALOG_KO.md` §4.1·§4.2·§4.3·§4.5·§5에 [rdna2-diffusion-sage-hip](https://github.com/unixyo/rdna2-diffusion-sage-hip) `v3.0`→`v4.0`(330 커널 0-spill, Unit-Temporal `Conv3d→Conv2d` VAE `2.6463 → 0.9135 s`, `256²`/LDS-only INT8 GEMM, 제로카피 MOSS `3.67 s`·Idol `0.983 s/pass`) 실측 결과를 반영했다.

## 환경

- RX 6900 XT 16 GB(gfx1030), Ryzen 7 7800X3D, 64 GB RAM, Ubuntu 26.10
- PyTorch 2.11/2.12 + ROCm 7.2·10.0·10.2 wheel(모델별로 다름), MIOpen 3.5–3.6, Triton 3.8.0
- 측정값은 전부 이 PC 한 대에서 나왔다. 다른 RDNA2 카드(6800/6700/6600, 680M 등)에서는 CU 수·클럭·메모리가 달라 배속이 달라진다. 수치보다 **방법과 판정 기준**을 참고하는 편이 안전하다.

## 판정 기준

- 기본은 **지각 기준**이다. 사람이 보거나 들어서 구분되지 않으면 통과다. 바이트 일치나 허용 오차 같은 엄격 수치는 함께 재서 참고로 남긴다.
- 연산 검증, 실제 모델 경로 측정, 사용자 판정을 서로 다른 단계로 기록한다. 마이크로벤치 배속만으로 채택하지 않는다.
- 예를 들어 Applio v1.0 묶음(B′)은 8.8% 빨랐지만 익명 청취에서 기존 쪽이 선호돼 보류했다. 그런 기록도 그대로 둔다.

## 이 리포에 없는 것

- 연구 원본 폴더는 약 29 GB다. 여기에는 문서, 직접 작성한 코드, 결론을 받치는 작은 텍스트 원자료(약 30 MB)만 담았다.
- 컴파일 캐시, 모델 가중치, 학습 데이터, 생성 이미지·음성, 써드파티 코드 사본, 청취 대응표는 넣지 않았다.
- 원본에만 있는 파일을 가리키던 링크는 `` `경로`, 연구 폴더에만 있음 `` 형태의 텍스트로 바꿨다.
- 경로의 사용자 홈은 `~`로, 저장소 경로는 `<storage>`로, 사람 이름은 "사용자"로, 학습 모델 이름은 `voice_model_A`로 바꿨다. 이미지 생성 프롬프트는 지웠다. **그래서 문서에 적힌 SHA256 일부는 원본 파일 기준이고, 이 리포의 사본과는 맞지 않는다.**

## 기여한 것

- 실험과 커널 탐색·구현 대부분은 **GPT Pro**(장시간 연구 세션)가 했다.
- 초기 재현 확인, 운영 적용, 핸드북 초판 정리는 **Claude**가 했다.
- 후속 `Applio Direct HIP v6.2~v10`·`Diffusion v3.0~v4.0`·`Strata v2~v4 & DFlash2` 스웜·릴레이 오케스트레이션, 교차 검증, 공개 패키징 및 카탈로그 갱신은 **Gemini**가 했다.
- 방향 설정, 로컬 환경 운용, 지각 판정(청취·육안)은 리포 주인이 했다.
- 다른 RDNA2 환경의 재현 결과나 반례는 이슈로 알려 주면 반영한다.

## 라이선스

- 코드: MIT ([LICENSE](LICENSE))
- 문서(`*.md`)와 결과 요약: CC BY 4.0 ([LICENSE-docs](LICENSE-docs))
- `rdna2-ops/production/MOSS_RDNA2_INT8/backend.py`는 Comfy Kitchen 코드를 고친 것이라 Apache-2.0 고지를 유지한다(같은 폴더 `LICENSE.apache-2.0`).
