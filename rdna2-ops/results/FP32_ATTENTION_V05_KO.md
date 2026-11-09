# FP32 tiled attention D128 재연구 — v0.5 실측 기록

실측일: 2026-11-06. 대상: Radeon RX 6900 XT, gfx1030, Triton 3.8.0. 이 문서는 주 사용설명서 §9.7에 넣을 결과와 재현 근거를 모은 별도 기록이다.

**판정: register spill 제거와 선형 메모리는 성공(A), 현재 최적 native 경로보다 빠르게 만드는 목표는 실패(A).** 실제 Qwen prefix-cache와 MOSS에서 FP32 정밀도를 유지한 채 검사한 모든 후보가 사전 연산 정확성 기준을 통과했다. 최선 pure Triton 후보도 Qwen `split_alpha`보다 약 2.40배, MOSS `exact_math`보다 약 2.08배 느렸다. 운영 자동 적용은 추가하지 않는다. 속도 탈락 때문에 최종 RGB A/B를 진행하지 않았으므로, RGB 허용 기준을 통과한 후보로 읽으면 안 된다.

## 1. 실제 입력과 비교 경로

| 실제 캡처 | BHLD Q / K,V | dtype와 layout | native 비교 경로 |
|---|---|---|---|
| Qwen prefix cache | `[1,32,2400,128]` / `[1,32,4964,128]` | FP32, 원래 `[B,S,H*D]` interleaved stride 복원 | 실제 ComfyUI `attention_split`, rehearsal `split_view`가 구현하는 `split_alpha` |
| Qwen 짧은 cache | `[1,32,2400,128]` / `[1,32,2408,128]` | FP32, 원래 stride 복원 | 위와 동일 |
| MOSS | `[1,24,1061,128]` / 동일 | FP32, Q/K stride `(3259392,128,3072,1)`, V stride `(9778176,128,9216,1)` | 강제 PyTorch math SDPA, 기존 `rdna2_ops.attention.scaled_dot_product_attention(backend="rdna2")` (벤치 alias `exact_math`) |

캡처 원본: `results/v05_qwen_capture/actual_qkv_2400_4964.pt`, `actual_qkv_2400_2408.pt`, `results/v05_moss_capture_fixed/actual_qkv_1061_1061.pt`. 각 디렉터리의 `actual_call_paths.json`은 함수 위치·dtype·stride 근거다. 원본 캡처는 연구 PC에 있으며 대용량 tensor를 배포 묶음에 중복하지 않았다.

Qwen 비교는 `comfy/ldm/modules/attention.py`의 `attention_split`을 직접 import했다. native와 alpha 후보 모두 head-contiguous rocBLAS 입력을 유지했다. `split_alpha`는 이 실제 입력들에서 native 출력과 비트가 일치했다. Qwen은 QK 이후 scale, MOSS math SDPA는 Q/K 각각에 `sqrt(scale)`을 곱한 뒤 dot하는 순서를 보존했다. softmax는 전 K 범위의 FP32 정규화다.

Comfy의 `@wrap_attn`은 `transformer_options.optimized_attention_override`와 `AttentionTensorContainer`도 처리한다. 따라서 함수 이름만 보고 모든 운영 호출을 이 경로로 판정하거나 wrapper를 무조건 바꾸면 안 된다. 이 benchmark는 캡처한 **mask 없는 실제 tensor**만 비교하며 운영 wrapper를 교체하지 않았다.

## 2. 측정 계약

- 환경: PyTorch `2.12.0+rocm10.2.0a20261011`, HIP `7.17.26405`, Triton `3.8.0`, 실제 arch `gfx1030`. 각 JSON에 PyTorch git hash와 파일 SHA256을 저장했다.
- 시간: GPU를 단독 사용하고 warmup 1회 후 3 round × 3회 일반 호출, 후보 순서를 round마다 앞/뒤로 바꾸어 측정했다. 표는 round 평균 시간의 중앙값이다. CUDA/HIP event 시간도 JSON에 있다. 컴파일·입력 생성은 제외하고 Python 계약 검사·출력 할당·Qwen 출력 layout 변환은 포함했다. 그래프 재생 결과가 아니다.
- 메모리: 입력과 참조 출력이 상주하는 상태에서 `torch.cuda.reset_peak_memory_stats`로 측정한 **추가 PyTorch allocated peak**다. 드라이버 scratch, 예약 allocator 전체, 모델 상주량은 포함하지 않는다. 표의 MiB는 2^20 byte다.
- 연산 정확성: 실행 전에 `torch.allclose(rtol=3e-4, atol=3e-5)`와 finite를 고정했다. 이는 원소별 `atol + rtol*abs(reference)` 판정이며 max-abs가 반드시 3e-5 이하여야 한다는 뜻은 아니다. 전체 native 출력 외에 두 head × 네 query row의 FP64 참조도 별도로 계산했다.
- 작은 입력 검사: 각 신규 tile에 rectangular tail을 적용하고 causal upper-left, D64, 음수/0 scale, split-K 등의 계약을 검사했다. initial 23건, rolled-D 15건, PV-chunk 12건이 통과했으며 공통 edge case가 중복되므로 50개의 고유 사례라고 세지 않는다. 실제 Qwen/MOSS 비교는 full tensor를 검사했다.
- 최종 영상 품질 gate는 여전히 RGB max `2/255`, RMSE `0.25/255`다. **이번 pure/hybrid 후보의 최종 RGB는 미측정**이다. 연산 allclose를 RGB 통과로 승격하지 않는다.

## 3. 대표 결과

각 pure 최종 행은 해당 최종 벤치의 native/alpha 기준과 비교한다. Hybrid는 별도 initial sweep의 같은 round 기준이다. native 기준의 run 간 작은 시간 차이를 섞어 속도비를 과장하지 않는다.

| 입력·방법 | 일반 호출 ms | 추가 peak MiB | native 대비 max abs / RMSE | 판정·근거 |
|---|---:|---:|---|---|
| Qwen K4964 `attention_split` (최종 PV run) | 29.753 | 3140.094 | 0 / 0 | 비교 기준 A |
| Qwen K4964 `split_alpha` (동일 run) | **22.787** | 3102.594 | 0 / 0 | 비트 일치 기준 A |
| Qwen K4964 pure `q32k64d16v128w8p16` | **54.597** | **75.500** | 2.754e-5 / 3.778e-7 | 정확성·선형 메모리 성공, 속도 실패 A |
| Qwen K4964 hybrid query128 (initial run) | 28.203 | 309.063 | 2.313e-5 / 3.179e-7 | 같은 run의 native 29.803보다 빠르나 alpha 22.927보다 느림; 비트 불일치 A |
| Qwen K2408 `attention_split` | 16.307 | 1563.500 | 0 / 0 | 비교 기준 A |
| Qwen K2408 `split_alpha` | **12.865** | 1526.000 | 0 / 0 | 비트 일치 기준 A |
| Qwen K2408 pure `q32k64d16v128w8p16` | 25.655 | **76.000** | 4.745e-5 / 8.506e-7 | 정확성·선형 메모리 성공, 속도 실패 A |
| Qwen K2408 hybrid query128 | 20.567 | 189.125 | 4.780e-5 / 8.284e-7 | 비트 불일치, 속도 실패 A |
| MOSS native math (최종 run) | 2.569 | 246.458 | 0 / 0 | 비교 기준 A |
| MOSS `exact_math` (동일 run) | **1.874** | 245.301 | 0 / 0 | 비트 일치 기준 A |
| MOSS pure `q32k64d16v128w8p16` | 3.889 | **12.434** | 1.907e-5 / 3.165e-7 | 정확성·선형 메모리 성공, 속도 실패 A |
| MOSS hybrid query128 (initial run) | 2.905 | 62.168 | 1.526e-5 / 3.284e-7 | 같은 run의 native 2.608와 exact 1.886 모두보다 느림 A |

Qwen K4964의 단일 full-score는 1.420 GiB다. pure 최선은 추가 allocated peak를 alpha의 3102.594 MiB에서 75.5 MiB로 약 41.1배 줄였다. 이 수치는 전체 모델 VRAM 41.1배 절감을 뜻하지 않는다. Qwen pure 출력의 BHLD→BSHD 변환이 동시에 보유하는 두 output 버퍼까지 시간·메모리에 포함되어 있다. MOSS는 BHLD 출력 하나만 필요하다.

Pure kernel은 global 메모리에 전체 score/probability나 query-block score를 만들지 않는다. 고정 tile의 score·online softmax 상태·출력 누산을 유지하고, split-K를 쓸 때만 고정 개수의 선형 partial을 만든다. 반면 **rocBLAS hybrid에는 `[BH, query_block, K]` global score tile이 있다**. query_block을 32/64/128로 고정하여 메모리가 선형인 별도의 방법이지 score-free/flash kernel은 아니다.

## 4. 전체 bounded 후보와 실패 보존

26개 신규 tile 후보와 v0.4 두 기준을 비교했다. 무제한 autotune은 하지 않았다. 아래 표는 모두 Qwen K4964이며 연산 정확성 기준을 통과했고, 모든 신규 후보가 alpha보다 느려 속도 개선 후보로는 탈락했다. `s4`는 split-K 4, `r`은 D inner loop unroll 금지, `p16`은 PV reduction을 16단위로 분리한 경우다. 전체 파라미터 tuple, 실제 stride, 시간 샘플, 메모리 byte, 오차, ISA resource는 원본 JSON에 있다.

| 후보 | 시간 ms | 추가 peak MiB | registers / spills | 증거 파일 |
|---|---:|---:|---|---|
| `v04_16x32` | 91.736 | 75.500 | 256 / 206 | initial A |
| `v04_32x64` | 77.739 | 75.500 | 256 / 926 | initial A |
| `q16k16v128w4` | 96.557 | 75.500 | 256 / 126 | initial A |
| `q16k32v128w8` | 150.566 | 75.500 | 256 / 184 | initial A |
| `q32k32v128w8` | 71.125 | 75.500 | 256 / 195 | initial A |
| `q32k64v128w8` | 102.645 | 75.500 | 256 / 615 | initial A |
| `q32k32v128w16` | 103.362 | 75.500 | 256 / 175 | initial A |
| `q16k32d64v128w4` | 94.103 | 75.500 | 256 / 214 | initial A |
| `q32k32d64v128w8` | 72.889 | 75.500 | 256 / 194 | initial A |
| `q32k32d64v64w4` | 123.747 | 75.500 | 256 / 462 | initial A |
| `q16k32d64v64w4` | 186.170 | 75.500 | 256 / 186 | initial A |
| `q32k64d64v64w8` | 197.786 | 75.500 | 256 / 583 | initial A |
| `q16k32d64v32w4` | 282.701 | 75.500 | 256 / 106 | initial A |
| `q32k32d64v128w8s4` | 72.978 | 190.344 | 256 / 193 | initial A |
| `q32k32d64v64w4s4` | 124.974 | 190.344 | 256 / 463 | initial A |
| `q64k32d64v128w8` | 78.106 | 75.500 | 256 / 611 | initial A |
| `q64k32d64v128w16` | 84.218 | 75.500 | 256 / 182 | initial A |
| `q16k16d16v128w4r` | 82.814 | 75.500 | 134 / 0 | rolled-D A |
| `q32k16d16v128w8r` | 55.832 | 75.500 | 127 / 0 | rolled-D A |
| `q64k16d16v128w16r` | 61.040 | 75.500 | 126 / 0 | rolled-D A |
| `q16k16d32v128w4r` | 93.968 | 75.500 | 133 / 0 | rolled-D A |
| `q32k16d32v128w8r` | 98.409 | 75.500 | 129 / 0 | rolled-D A |
| `q32k32d16v64w8r` | 168.120 | 75.500 | 184 / 0 | rolled-D A |
| `q32k16d16v64w4r` | 156.290 | 75.500 | 122 / 0 | rolled-D A |
| `q32k32d16v128w8p16` | 83.187 | 75.500 | 127 / 0 | PV16 A |
| `q32k64d16v128w8p16` | 54.597 | 75.500 | 186 / 0 | PV16 A |
| `q64k32d16v128w16p16` | 88.469 | 75.500 | 126 / 0 | PV16 A |
| `q64k64d16v128w16p16` | 61.254 | 75.500 | 186 / 0 | PV16 A |

Hybrid fixed query block 비교:

| query block | Qwen K4964 ms / MiB | Qwen K2408 ms / MiB | MOSS ms / MiB |
|---:|---|---|---|
| 32 | 56.681 / 250.891 | 33.455 / 160.906 | 4.298 / 52.843 |
| 64 | 30.566 / 270.281 | 23.324 / 170.313 | 3.677 / 55.951 |
| 128 | 28.203 / 309.063 | 20.567 / 189.125 | 2.905 / 62.168 |

MOSS initial 15개 신규 tile의 전체 결과도 `fp32_attention_moss_actual_sweep_v05.json`에 보존했다. 최종 두 no-spill 후보의 MOSS 시간은 `q32k16d16v128w8r` 4.358 ms, `q32k64d16v128w8p16` 3.889 ms다. Qwen K2408에서는 각각 27.322 / 25.655 ms였다.

## 5. D128 spill 원인: 확인한 것과 남은 추정

### 확인한 변화

1. **v0.4와 초기 신규 구현의 spill은 실제로 컸다(A).** Qwen에서 v0.4 16×32는 registers 256 / spills 206 / private segment 824 byte, 32×64는 256 / 926 / 3704 byte였다. 새 초기 `q32k32d64v128w8`도 256 / 194 / 776 byte였다. 추측으로 붙인 spill 설명이 아니다.
2. **단순 D64 분리는 operand 수명을 줄이지 못했다(B: 생성 IR).** 초기 TTGIR에서 두 `[32,64]` Q operand load가 K/V streaming loop 밖으로 이동되어 모두 살아 있었다. Python `tl.static_range`로 D128을 두 번 나누는 것만으로는 전체 D128 live state를 해제하지 못했다. 해당 `.ttgir`, `.amdgcn`, 초기 소스 snapshot을 저장했다.
3. **진짜 D inner loop + 작은 K tile은 spill을 제거했다(A/B).** `tl.range(0,D,BD,loop_unroll_factor=1)`와 BD16/BK16을 사용한 rolled-D 후보 7개 모두 spill 0, private segment 0이 됐다. 대표 `q32k16d16v128w8r`은 registers 127, Qwen 55.832 ms였다. static D 분리, 더 많은 warps, PV 출력 channel 분할만 했던 초기 후보들은 이 결과를 내지 못했다.
4. **PV reduction도 16씩 처리하면 BK64에서도 spill 0을 유지했다(A/B).** `tl.gather`로 현재 score tile의 P 16열을 가져와 V16×D128을 곱하는 실제 loop를 추가했다. 최선 `q32k64d16v128w8p16`은 Qwen K4964 registers 186 / spills 0, MOSS 187 / 0, K2408 191 / 0이었다. 모두 private segment 0이다. Qwen 최선 ISA에는 `v_fma_f32`가 있고, dot2/dot4/MFMA/WMMA와 scratch load/store는 없다.
5. **spill 0은 속도 성공 조건이 아니었다(A).** 가장 빨라진 pure 후보도 alpha/exact보다 느렸다. PV channel32/64 분리는 QK 반복을 늘렸고, split-K4는 partial write/reduce와 peak 증가가 생겼다. 해당 후보들은 실제로 개선되지 않았다.

### 아직 원인으로 확정하지 않은 것

작은 FP32 dot, D/PV loop와 layout conversion, P gather, 반복 Q/K/V load, online softmax reduction, partial merge의 비용이 남은 성능 차이에 기여할 가능성이 있다. 생성 코드에는 이 연산들이 존재하지만 하드웨어 counter로 각 비용의 비중을 분해하지 않았다. 따라서 “spill이 전부였다”, “Infinity Cache 때문에 hybrid가 빨랐다”, “gfx1030 FP32 flash는 원리적으로 rocBLAS를 이길 수 없다”는 결론은 근거가 부족하다. 이번 측정이 확인한 것은 **이 bounded 구현군에서는 메모리 절감과 spill 해결을 달성해도 native 최적 경로의 시간을 이기지 못했다**는 범위다.

## 6. 재사용 API와 적용 범위

```python
from rdna2_ops.fp32_attention import fp32_attention

# Qwen과 같은 FP32 post-scale, mask 없는 명시적 연구 호출
out = fp32_attention(
    q, k, v, scale_mode="post",
    block_q=32, block_k=64, block_d=16,
    block_v=128, num_warps=8, split_k=1,
    unroll_d=False, block_pv=16,
)
# MOSS math 의미론 비교에서는 scale_mode="sdpa".
```

입력은 BHLD FP32, D64/128, Q/K/V 같은 B/H/D, K/V 같은 길이, 마지막 stride 1이고 outer stride는 0 이상이어야 한다. 채널 외의 interleaved layout과 storage offset을 읽으며 외부 복사를 강제하지 않는다. 커널은 gfx1030, 양의 길이, 유한 입력과 overflow 없는 dot을 전제로 한 forward-only 연구 함수다. `is_causal=True`는 upper-left 정렬이며 prefix-cache 위치 offset을 자동 추론하지 않는다. arbitrary mask/dropout/GQA/backward/graph capture/compile은 지원 계약에 없다. overflow·NaN 탐지용 전체 GPU 입력 스캔은 하지 않는다. 배치/주소/출력 인덱스 범위와 dtype·autocast·autograd 등에 대한 host guard는 함수에 있다.

split-K1의 추가 tensor는 출력뿐이다. split-K>1은 고정 `split_k * B*H*L*(D+2)` FP32 partial/stat storage를 더한다. 따라서 고정 D/H/split_k에서 storage는 L과 S에 선형이다. split-K4는 실측 속도 이득이 없어서 기본 추천이 아니다.

`rocblas_query_tiled`는 `bench_fp32_attention.py`에 남긴 비교 helper다. Q/K/V head-contiguous packing, Qwen `baddbmm(alpha=scale)` 또는 MOSS pre-scale, 고정 query score buffer, native full-K softmax, PV `bmm`을 수행한다. 모델 채택용 dispatch를 추가하지 않았다.

## 7. 재현 명령과 원본

기존 Qwen venv를 그대로 사용한다. 연구 루트에서 실행하며 새 환경 생성·패키지 변경은 필요 없다. GPU를 다른 실험과 공유 실행하지 않는다.

```bash
~/DevSpace/qwen-image-21-rdna2-lab01/.venv/bin/python -u bench_fp32_attention.py \
  --models qwen \
  --captured results/v05_qwen_capture/actual_qkv_2400_4964.pt \
  --tiles q32k16d16v128w8r q32k64d16v128w8p16 \
  --output results/fp32_attention_recheck_qwen.json

~/DevSpace/qwen-image-21-rdna2-lab01/.venv/bin/python -u bench_fp32_attention.py \
  --models moss \
  --captured results/v05_moss_capture_fixed/actual_qkv_1061_1061.pt \
  --tiles q32k16d16v128w8r q32k64d16v128w8p16 \
  --skip-small --output results/fp32_attention_recheck_moss.json
```

K2408 확인은 Qwen capture 파일명만 `actual_qkv_2400_2408.pt`로 바꾼다. 결과 파일은 새 이름으로 남겨 기존 증거를 덮어쓰지 않는다. 모든 tile 이름은 benchmark `TILES`에 정의되어 있다. 초기 unroll 경로도 현재 소스에 남아 있지만 초기 원문 SHA까지 재현하려면 `results/fp32_attention_initial_source.py` snapshot을 참조한다.

원본 결과 파일(A):

- `results/fp32_attention_qwen_actual_sweep_v05.json`: initial 15개, v0.4 두 개, hybrid 3개, 작은 계약 검사.
- `results/fp32_attention_moss_actual_sweep_v05.json`: MOSS initial 15개, v0.4, hybrid.
- `results/fp32_attention_qwen_rolled_d_v05.json`: D 실제 loop 7개, 모든 spill 0.
- `results/fp32_attention_qwen_pv_chunks_v05.json`: PV16 chunk 4개, Qwen K4964 최선.
- `results/fp32_attention_moss_no_spill_v05.json`: MOSS 최종 두 개.
- `results/fp32_attention_qwen_k2408_v05.json`: 실제 짧은 cache, 최종 두 개와 hybrid.

소스/생성 코드(B), resource 측정(A):

- `rdna2_ops/fp32_attention.py` SHA256 `e64bf1223bd3ed91c8e89c4762b40d75e79a3878c5f2f6722d1542218dab1308`.
- `bench_fp32_attention.py` SHA256 `32bddf8aad31ff7c4334edc167aea3fa9f4a1064a30b7d63c35420009c80db74`.
- `results/fp32_attention_initial_source.py` SHA256 `7b845dc4b0d53adff93fdfde8c79c3aab330e3801617dd19c5fe137b0ad05665`.
- `results/fp32_attention_initial_d64.ttgir` / `.amdgcn`: Q operand hoist와 초기 spill이 있는 컴파일 결과.
- `results/fp32_attention_qwen_no_spill.ttgir` / `.amdgcn`: 최선 Qwen K4964 `q32k64d16v128w8p16` 생성 코드. ASM SHA256 `11937be724f1d416176fd7ba0cc776cb63184104fb75aa6fac6113dfcec1ac64`.
- `results/fp32_attention_isa_exports_v05.json`: 저장한 최종 ISA의 cache 원본 위치·SHA256. GPU를 다시 실행하지 않고 cache의 ASM hash를 결과 JSON과 일치시켜 추출했다. MOSS와 K2408의 ISA resource/hash는 해당 결과 JSON에 있으며 이 export는 Qwen K4964 파일을 담는다.

운영 MOSS·Qwen 설정, 시작 스크립트, custom_nodes 소스는 이 실험에서 수정하지 않았다. 코드·실패 기록은 재사용 연구용으로 남기고 자동 적용 정책에는 추가하지 않는다.
