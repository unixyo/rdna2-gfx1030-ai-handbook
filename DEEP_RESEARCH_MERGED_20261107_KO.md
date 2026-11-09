# 딥리서치 3종 통합 (2026-11-07)

질문: 행렬 유닛이 없는 RDNA2(gfx1030)에서 없는 하드웨어 기능을 소프트웨어로 우회해 AI 추론을 빠르게 할 수 있는가.

| 출처 | 파일 |
|---|---|
| GPT(xhigh) 웹 딥리서치 | `~/다운로드/deep-research-report (1).md` |
| 아르곤쟝(Gemini) 웹 딥리서치 | `~/다운로드/AMD RDNA2 AI 추론 소프트웨어 우회 검증.txt` |
| Claude workerport deep_search(중국 웹) | `DEEP_SEARCH_CN_20261107_KO.md` |

세 보고 모두 외부 자료의 요약이다. 아래의 "이 PC 해당 여부"는 Claude가 이 PC의 소스·설정을 보고 판단했다. 외부 수치는 Claude가 재현하지 않았다.

## 1. 세 보고가 일치하는 결론

- **WMMA를 소프트웨어로 흉내 내서 큰 GEMM에서 rocBLAS를 이긴 근거는 없다.** Maxritz의 WMMA 에뮬레이션·FP8→INT8 층은 저자 스스로 "아직 전체 컴파일도 못 해 봤다"고 적었고, 성능 수치가 0건이다(GPT·아르곤쟝 모두 원문 확인).
- **소프트웨어 LUT(LUT-GEMM, T-MAC)는 GEMV(배치 1)에서만 이득**이고 GEMM에서는 dequant+GEMM보다 느리다. 아르곤쟝은 A100에서 LUT-GEMM이 BitBLAS dequant보다 W4A16 기준 2.34배 느렸다는 비교를 인용했다.
- **BF16 hi/lo 분할과 Ozaki scheme은 RDNA2에서 역효과다.** 이 기법들은 저정밀 행렬 유닛이 고정밀보다 8–32배 빠를 때만 이득인데, RDNA2의 FP16 dot2는 FP32의 2배뿐이다.
- **큰 이득은 라이브러리가 잘못된 커널을 고르는 곳을 피한 데서만 나왔다.** 이 연구의 결론(레이아웃, 안전 검사, MIOpen solver)과 같다.
- **디퓨전 GEMM에서 INT4/INT2 커스텀 커널이 dequant+FP16 rocBLAS를 이긴 gfx1030 사례는 없다.** comfy-kitchen도 RDNA2에서는 WMMA GEMM을 거절하고 GEMV·RoPE·AdaLN 같은 non-WMMA 커널만 쓴다.
- **INT dot 유닛의 가치는 속도가 아니라 여유**라는 평가(leapdragon V620 가이드)도 이 PC의 INT4 결과와 같다.

## 2. 새로 나온 후보와 이 PC 해당 여부

| 후보 | 근거 | 이 PC 해당 여부 (Claude 판단) |
|---|---|---|
| **PyTorch TunableOp로 rocBLAS solution을 shape별로 고르기** | GPT: RX 6900 XT FLUX 1024², `PYTORCH_TUNABLEOP_ENABLED=1`로 7.3 → 3.85 s/it(약 1.9배), 독립 사용자 실측(ComfyUI-GGUF issue) | **시험한 적 없음. 1순위 후보.** 이 연구의 "rocBLAS가 레이아웃에 따라 3배 느린 커널을 고른다" 결과와 같은 축이다. 다운로드 불필요 |
| attention_sub_quad의 `key.contiguous()` (ComfyUI #17500) | 아르곤쟝: RX 6800 XT, baddbmm 452.8 → 12.2 ms, Wan 2.2 스텝 7.5배 | **해당 없음.** 이 PC의 Qwen은 `attention_split`을 쓰고, 그 함수는 이미 q/k/v에 `.contiguous()`를 한다. MOSS·Anima는 PyTorch SDPA다. `--use-quad-cross-attention`을 쓰는 다른 워크플로가 생기면 그때 해당한다 |
| RDNA2 튜닝 FlashAttention (Aceeton, ROCm/flash-attention 2.8.0 Triton) | Claude 원문 확인: RX 6800 XT, triton 3.8, D128·4096에서 PyTorch 42.2 → 23.8 ms | **2순위 후보.** 다운로드 필요(승인 대기). 수학적 동일이라 엄격 기준은 비트 일치가 아니다 |
| RDNA2 네이티브 HIP SageAttention (LuXuxue v10, comfy-kitchen PR #206) | 아르곤쟝: QK^T를 v_dot4(INT8)로, PV를 v_dot2로 계산. gfx103x 빌드 확인, 배속 수치는 미확인 | 3순위. INT8 QK 근사라 v0.5 Sage 실패 이력과 같이 지각 평가가 필요하다. 소스 빌드 필요 |
| W8A8 dot4 projection (leapdragon vllm-rdna2-qwen) | V620 M=2048에서 GEMM 1.79–1.83배, 전체 prefill +3.7–7% | Anima FP16 Linear에 쓰면 근사가 된다. MOSS·Qwen은 이미 INT8이다. 우선순위 낮음 |
| Strata `STRATA_HIP_PROMPT_F16=1` | 두 보고 모두 2.1–2.25배 사례 인용 | **이미 적용 중**(사용자 recipe, 연구 실행 로그에서 확인) |
| Strata `STRATA_SH_STREAM`, `STRATA_HC_PERSIST` | 아르곤쟝 후보 2 | **이미 시험해 기각**(recipe `rejected.md`: 효과 없음 / 반복 이득 없음) |
| llama.cpp fattn-tile 128-스레드 패치, Vop3p INT8 QK^T FA | 아르곤쟝 후보 3 | Strata는 자체 QSA attention을 쓰고, 이 PC의 warm 프리필에서 attention은 약 15%다. 효과 상한이 작다 |
| comfy-kitchen non-WMMA HIP 커널(RoPE, AdaLN) | 아르곤쟝 후보 5 | **이미 기준선**(v0.6 Norm 연구에서 MOSS·Qwen의 기존 Kitchen 융합을 기준으로 씀) |
| MIOpen FAST + user DB | 아르곤쟝 후보 5 | **이미 적용**(Applio). VAE BF16 Conv는 v0.6에서 2.4배 레시피 확보 |
| `--disable-dynamic-vram` | ComfyUI #16878: ROCm 10 + RDNA2에서 Qwen-Image INT8 무한 멈춤 | 이 PC의 Anima 실험에서는 성능 차이 없음 확인(첫 요청 진단). 멈춤 증상이 생기면 참고 |

## 3. 세 보고가 서로 다른 말을 한 곳

- **llama.cpp RDNA2 FlashAttention의 효과 크기**: GPT은 V620에서 prefill +2–12%, 다른 포크는 e2e 무차이라고 정리했다. 아르곤쟝은 128-스레드 패치로 203 → 1,314 tok/s(6.47배)를 인용했다. 기준선이 다르다. 후자는 FA가 크래시해 꺼져 있던 상태와의 비교다. 둘은 모순이 아니다.
- **Q4_0 DOT8**: 아르곤쟝은 e2e 0 ~ −0.7%라고 했고, GPT은 "DOT8 사용은 확인, 고립 배속은 미확인"이라고 했다. 둘 다 큰 이득은 아니라는 방향이다.
- **vllm-rdna 수치**: 아르곤쟝이 독립 블로그(nyanko.win)를 인용했다. 저자 자칭 96.63 tok/s는 재현되지 않았고, 실측은 73–77 tok/s였다. 커뮤니티 수치를 그대로 믿으면 안 된다는 사례다.

## 4. 다음 실험 순서 (Claude 제안)

1. **TunableOp**: Anima FP16(SDPA math, 직접 GEMM 경로), Qwen FP32 attention bmm, MOSS에 `PYTORCH_TUNABLEOP_ENABLED=1`, `PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0`, 연구 폴더 전용 결과 파일로 튜닝한다. 튜닝한 뒤의 warm 속도와 최종 출력을 같은 seed로 비교한다. Anima·Qwen은 프로세스 간 결정적이라 프로세스를 나눠 비교할 수 있다. MOSS는 conditioning이 흔들리므로 같은 프로세스 비교가 필요하다.
2. **Aceeton FlashAttention**: 다운로드를 승인받은 뒤 Anima attention에 붙인다.
3. **LuXuxue HIP Sage**: 1·2 결과를 보고 결정한다.

## 5. 후속 실측: TunableOp on Anima (Claude, 2026-11-07)

`anima-rdna2-lab01/results/tunableop1/`. moss runtime·`.venv-moss`(PyTorch 2.12 + ROCm 10.2), `PYTORCH_TUNABLEOP_ENABLED=1`, `PYTORCH_TUNABLEOP_TUNING=1`, `PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0`, 결과 파일은 연구 폴더(`tunableop_results0.csv`, GEMM shape 61개). 1024², 30단계, seed 20261107, 한 프로세스에서 warm(튜닝) → 기준 2회 → 직접 GEMM+VT 2회 순서로 실행했다.

| 경로 | 튜닝 전(이전 실행, 같은 날) | TunableOp 튜닝 후 | native 대비 |
|---|---:|---:|---:|
| warm(튜닝 포함) | — | 261.78 s | 1회성 |
| 기준 SDPA math | 81.0 s | **60.84 / 60.86 s** | **1.33배** |
| 직접 GEMM + V 전치 | 56.6 s | 첫 요청 105.78 s(새 shape 튜닝 포함), 둘째 **45.18 s** | **1.79배** |

| 최종 이미지 비교 (8-bit PNG) | max | RMSE |
|---|---:|---:|
| 튜닝 전 native끼리(서로 다른 실행) | 0 | 0 |
| TunableOp 기준 ↔ 튜닝 전 native | 210 | 5.60/255 |
| TunableOp 직접 VT ↔ 튜닝 전 native | 212 | 5.79/255 |
| (참고) 튜닝 전 직접 VT ↔ native | 210 | 6.21/255 |

- 확인된 사실: TunableOp는 같은 Anima 요청에서 SDPA math를 1.33배, 직접 VT 경로를 추가로 1.25배 빠르게 했다. 튜닝 결과는 파일로 남아 재사용할 수 있다.
- 확인된 사실: rocBLAS가 다른 솔루션을 고르므로 최종 출력은 원본과 비트 일치가 아니다. 갈라지는 정도는 직접 VT 경로와 같은 수준이다. 엄격 기준은 실패, 수학적 동일 등급이다.
- 정리: Anima에서 원본 비트 일치를 지키는 최고 경로는 여전히 4D native(1.19배)다. 속도 우선 경로는 TunableOp + 직접 VT(1.79배)로 바뀐다.
- 미검증: 지각 평가, 다른 해상도(shape가 바뀌면 다시 튜닝), Qwen FP32·MOSS 적용, 튜닝 파일을 다른 프로세스에서 `TUNING=0`으로 재사용했을 때의 속도·결정성.

## 6. 후속 실측: Qwen TunableOp, TunableOp 재사용, Aceeton FlashAttention (Claude, 2026-11-07)

### 6.1 Qwen-Image-2.1 TunableOp

`rehearsal_qwen_20261106/results/tunableop25/`, 하네스 `scripts/rehearsal_tunableop.py`(기존 리허설 하네스에 모드별 `torch.cuda.tunable.enable()` 전환만 추가). 960×640 편집, 25단계, seed 20261106, 한 프로세스에서 TunableOp 꺼짐/켜짐을 번갈아 실행했다. 튜닝 파일은 첫 시도(`attempt1/`, 하네스 오류로 중단, 34행)에서 만든 것을 읽었다.

| 경로 | 디노이징 | 배속 | 최종 RGB(0–1) 기준 대비 |
|---|---:|---:|---|
| 기준(두 번째 warm) | 57.06 s | — | 0 |
| TunableOp | 56.50 / 56.40 / 56.31 s | 1.01배 | max 0.74–0.96, RMSE 0.017–0.023 |
| split_alpha(비트 일치 후보) | 51.65 s | 1.10배 | 0 |
| split_alpha + TunableOp | 50.87 / 50.68 / 50.61 s | 1.13배(split_alpha 대비 1.02배) | max 0.78–0.96, RMSE 0.014–0.018 |

- 확인된 사실: Qwen에서는 TunableOp 이득이 1–2%다. FP32 attention bmm과 INT8 GEMM(patientx Triton)은 TunableOp 대상이 아니거나 이미 좋은 커널을 쓰고 있다.
- 확인된 사실: TunableOp 실행끼리도 최종 이미지가 매번 다르다(8-bit PNG max 152–245, RMSE 3–7/255). 같은 프로세스·같은 파일·같은 seed다. 꺼진 실행끼리는 바이트 일치다.
- 추론: 고른 rocBLAS 솔루션 중 일부가 원자 연산을 쓰는 split-K 계열일 가능성이 있다. 확인하지 않았다.
- 판정: 속도 이득이 없어 기각했다. 지각 평가는 하지 않았다.

### 6.2 Anima TunableOp 파일 재사용

`anima-rdna2-lab01/results/tunableop_reuse/`. §5에서 만든 `tunableop_results0.csv`(md5 1568a62c…)를 새 프로세스에서 `PYTORCH_TUNABLEOP_TUNING=0`으로 읽었다.

| 경로 | §5(튜닝한 프로세스) | 재사용 프로세스 |
|---|---:|---:|
| 기준 SDPA math | 60.84 / 60.86 s | 61.78(첫 요청) / 60.87 s |
| 직접 GEMM + V 전치 | 105.78(튜닝 포함) / 45.18 s | **45.21 / 45.20 s** |

| 이미지 비교(8-bit PNG) | max | RMSE |
|---|---:|---:|
| 같은 프로세스 안 기준끼리, VT끼리 | 0 | 0 |
| 기준: §5 프로세스 ↔ 재사용 프로세스 | 207 | 5.48/255 |
| 기준: flash_tunableop 프로세스 ↔ 재사용 프로세스 | 214 | 6.92/255 |
| VT: §5 프로세스 ↔ 재사용 프로세스 | 185 | 4.08/255 |

- 확인된 사실: 튜닝 파일을 재사용하면 튜닝 비용 없이 첫 요청부터 1.79배가 난다.
- 확인된 사실: 같은 파일이라도 프로세스가 바뀌면 같은 seed의 그림이 미세하게 달라진다. 크기는 TunableOp ↔ native 차이와 같은 자릿수다. TunableOp 없는 Anima는 프로세스 간 결정적이었다.
- 미검증: 원인. 파일에는 솔루션 번호가 고정되어 있으므로, 같은 솔루션 안의 비결정 경로이거나 파일에 없는 shape의 기본 선택이 프로세스마다 다를 가능성이 있다.

### 6.3 Aceeton FlashAttention / SageAttention

받은 것(사용자 사전 허락: 100 GB 미만 다운로드):

| 이름 | 출처 | 크기 | 커밋 |
|---|---|---:|---|
| SageAttention_RDNA2_GFX1030 | github.com/Aceeton/SageAttention_RDNA2_GFX1030 | 106 MB | 7df2bb6e(2026-09-22) |
| flash-attention `v2.8.0-cktile` + `rocm/flash-attention-rdna.patch` | github.com/ROCm/flash-attention | 20 MB | b3ae4966 |

위치는 `~/DevSpace/rdna2-fa-lab01/src/`이다. 설치 스크립트(`rocm/install_comfyui_rocm.sh`)는 읽기만 하고 실행하지 않았다. pip 설치 없이 소스를 `sys.path`로 읽었고, 운영 venv(`.venv-moss`, Qwen `.venv`)에는 아무것도 설치하지 않았다. 패치 내용은 gfx103x에서 Triton prefill 블록 설정을 BLOCK_M 128·BLOCK_N 64·8 warps·2 stages 하나로 고정하는 것이다.

연산 단위(`rdna2-fa-lab01/results/bench_attn_20261107.json`, `scripts/bench_flash_layout.py`, 호출당 중앙값):

| shape [B,H,Sq,Sk,D] | native SDPA | Flash | Sage(INT8 QK) |
|---|---:|---:|---:|
| Aceeton 기준 [1,24,4096,4096,128] FP16 | 43.7 ms | **23.7 ms**(저자 23.8) | 26.1 ms |
| Anima self [2,16,4096,4096,128] FP16, 연속 | 57.7 ms | 33.4 ms | 34.6 ms |
| 〃 BSHD view(모델 실제 입력) | 64.6 ms | 33.5 ms | — |
| Anima cross [2,16,4096,512,128] BSHD view | 9.3 ms | 5.5 ms | — |
| Qwen joint [1,32,2400,4964,128] FP32 | 31.1 ms | 미지원 | 23.4 ms |

오차: Flash는 FP64 대비 RMSE 7.3e-6(native 5.4e-6)으로 같은 등급이다. Sage는 RMSE 3.4e-4, cos 0.99991로 근사다.

Anima 30단계 1024²(같은 프로세스, `results/flash1`, `flash_timed`, `flash_tunableop`, cuda event로 attention만 따로 측정):

| 경로 | 디노이징 | self 호출 중앙값 | cross 호출 중앙값 |
|---|---:|---:|---:|
| native | 80.6 s | 46.7–58.8 ms | 3.8 ms |
| Flash(self+cross) | 68.9 s(1.17배) | 33.1 ms | 5.45 ms |
| TunableOp | 60.95 s | 34.2 ms | 3.65 ms |
| TunableOp + Flash(self만) | 60.0 s | 33.2 ms | 3.9 ms |
| TunableOp + 직접 VT(§5, §6.2) | **45.2 s** | — | — |

- 확인된 사실: Aceeton Flash는 이 PC에서 저자 수치대로 재현된다. Flash 실행끼리는 바이트 일치이고, native 대비 최종 RGB RMSE 6.53/255다.
- 확인된 사실: 모델 안 native cross-attention(K=512)은 미세 벤치보다 빨라서 Flash가 오히려 느리다.
- 확인된 사실: TunableOp가 native SDPA math의 self-attention을 Flash와 같은 34 ms로 만든다. 두 기법을 겹쳐도 이득이 없고, 직접 VT 경로가 가장 빠르다.
- 판정: Anima 속도 최우선 경로는 TunableOp + 직접 VT(1.79배)로 유지한다. Flash는 TunableOp를 쓸 수 없는 환경(같은 seed 재현이 필요한데 TunableOp가 프로세스마다 달라지는 문제를 피하고 싶을 때)의 대안으로 남긴다. Flash 단독은 1.17배이고 프로세스 간 결정성은 확인하지 않았다.
- Sage는 모델에 붙이지 않았다. 연산 단위에서 Flash보다 빠르지 않고(Anima), 근사이며, Qwen FP32에서는 1.33배지만 v0.5 Sage식 엄격 실패 이력이 있다.


### 6.4 MOSS TunableOp

`rdna2-ops/results/claude_repro_20261107/moss_tunableop/`, 스위치 `rdna2-ops/tunableop_switches_v07c.py`, 사양 `research_configs/moss_tunableop_v07c.json`. 운영 워크플로 `sft_fast4_int8_512.json`(4단계, 512²), seed 20261105, 한 프로세스 안에서 꺼짐/켜짐을 번갈아 실행했다.

- 첫 시도(`attempt1_fault/`)는 첫 튜닝 요청에서 GPU 메모리 접근 오류(`HSA_STATUS_ERROR_MEMORY_FAULT`, 커널 `Cijk_Ailk_Bljk_SB_MT64x32x32_…_GSU1`, grid [256,2,24])로 프로세스가 죽었다. 직전에 기록된 shape는 head 24 strided-batched bmm이다. ROCr가 14.8 GB GPU 코어 덤프를 런타임 폴더에 남겨서 결과 폴더로 옮겼다(삭제하지 않음).
- 재시도는 `PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE=0`, `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1`을 더해 끝까지 돌았다. rotating buffer가 원인이라는 증거는 이 1회뿐이다.

| 경로 | 디노이징 | 배속 | 최종 RGB(0–1) 기준 대비 |
|---|---:|---:|---|
| 기준 | 4.77 / 4.76 s | — | 0 |
| TunableOp | 39.66(튜닝) / 4.84 / 4.84 s | 0.99배 | max 0.76–0.89, RMSE 0.085–0.107 |
| INT8 융합(운영 레시피) | 2.11 s | 2.25배 | 0(기준과 바이트 일치) |
| INT8 융합 + TunableOp | 2.08 / 2.09 / 2.08 s | INT8 융합 대비 1.02배 | max 0.84–0.90, RMSE 0.113–0.118 |

- 확인된 사실: MOSS에서 TunableOp 이득은 없다. 무거운 GEMM은 INT8 Triton이라 TunableOp 대상이 아니다.
- 확인된 사실: TunableOp 실행끼리도 이미지가 다르고, 차이가 RGB RMSE 22–30/255로 Anima·Qwen보다 크다. 4단계 모델이 작은 차이를 키우는 것으로 보인다(MOSS conditioning의 프로세스 간 흔들림과 같은 자릿수).
- 판정: 기각. 튜닝 중 GPU 오류 위험까지 있어 MOSS에는 쓰지 않는다.

### 6.5 Idol(image→3DGS) attention 교체

Idol은 11-01~03에 이미 자체 Triton fused attention(`triton_rdna2`, 기본 `sdpa_chunked` 대비 5.20배)으로 손본 모델이다. 그 상태에서도 forward 시간의 약 78%가 joint 12294 토큰 attention(B1·H16·D64 FP16)이라 Aceeton 커널을 대 봤다. Idol 소스는 수정하지 않았다. `layers.py`가 호출마다 `scripts.rdna2_flash_attention.sdpa`를 가져오므로 실행기(`~/DevSpace/rdna2-fa-lab01/scripts/idol_attn_variant.py`, `run_idol_variants.sh`)에서 그 속성만 바꿨다.

연산(`rdna2-fa-lab01/results/bench_idol_attn_20261107.json`, BSHD 저장·BHSD view, 호출당 중앙값):

| 커널 | 12294 토큰 | 4101 토큰 | FP32 참조 대비 RMSE |
|---|---:|---:|---|
| Idol `triton_rdna2`(M32·N64, 4 warps, 1 stage) | 86.3 ms | 9.48 ms | 4.3e-6 |
| 〃 타일 64×64 / 64×128 / 32×128 | 107 / 142 / 170 ms | — | — |
| Aceeton Flash | **69.2 ms(1.25배)** | 8.39 ms | 4.3e-6(같음) |
| Aceeton Sage(INT8 QK) | **54.7 ms(1.58배)** | 6.48 ms | 1.9e-4(근사) |

실제 공식 50단계·CFG3·262144 Gaussians·seed 42(`rdna2-fa-lab01/idol_runs/`, 순차 1회씩):

| 경로 | 샘플링 | 전체 | 배속 | PLY | 8방향 미리보기, 현재 커널 대비 |
|---|---:|---:|---:|---|---|
| Idol `triton_rdna2` | 261 s | 288.0 s | — | SHA가 11-03 결과와 같음(프로세스 간 결정적) | — |
| Aceeton Flash | 209 s | 230.5 s | 1.25배 | 검사 통과 | max 167, 평균 0.38/255, 10 넘게 다른 픽셀 0.97% |
| Aceeton Sage | 177 s | 193.0 s | **1.49배** | 검사 통과 | max 155, 평균 0.44/255, 1.22% |

- 확인된 사실: 미리보기 차이는 11-03의 `torch.compile` 실험(평균 0.41/255, 1.16%)과 같은 크기다. Claude 육안으로는 세 결과를 구분할 수 없었다.
- 판정: Sage는 지각 통과(2026-11-07 사용자, 비교 그림을 보고 "세이지도 괜찮네"), 채택 후보 1.49배. Flash는 수학적 동일을 원할 때의 후보(1.25배)다. 한 물체·한 seed·각 1회 측정이다.
- 의미: 토큰 수가 큰(12K) D64 attention에서는 rocBLAS math를 피한 fused 커널끼리도 블록 설정 차이로 1.25배가 갈리고, INT8 QK 근사가 실제 출력에서 보이지 않는 차이로 끝날 수 있다. Anima(4K 토큰, D128)에서 Flash가 TunableOp에 묻힌 것과 대비된다.

Sage 위 TunableOp(`idol_runs/sage_tune`, `sage_tuned`, `idol_runs/tunableop/tunableop_results0.csv` 151 shape, rotating buffer 0): 튜닝 실행 전체 877.1 s(샘플링 5분 17초), 파일 재사용 새 프로세스 191.2 s(샘플링 2분 57초)로 TunableOp 없는 Sage 193.0 s(2분 57초)와 같다. PLY SHA는 바뀐다. 이득이 없어 기각했다. Idol의 나머지 22%는 rocBLAS FP16 Linear가 이미 좋은 커널을 쓰고 있거나 GEMM이 아닌 연산이다(미분해).

### 6.6 남은 것
- TunableOp 프로세스 간 비결정의 원인(Anima), 프로세스 내 비결정의 원인(Qwen·MOSS).
- TunableOp 튜닝 중 GPU 메모리 오류가 rotating buffer 때문인지(1회 증거).
- Flash 단독 경로의 프로세스 간 결정성.
