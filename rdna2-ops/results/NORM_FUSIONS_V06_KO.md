# LayerNorm · RMSNorm · adaLN v0.6: RX 6900 XT 실측과 모델 판정

상태: **12개 연산 사례와 Anima 30-step 최종 A/B 완료. native LN을 보존한 modulation 융합은 두 번 모두 latent·RGB 바이트 일치, 모델 시간 감소는 약 0.39%로 작다. 통계 교체 LN/RMS/full adaLN은 Anima 엄격 RGB 실패.**

이 파일은 연산 카탈로그 7절의 정규화 칸을 채우기 위한 근거다. MOSS·Qwen·Anima의 운영 파일, 실행 스크립트, 모델 가중치, 기존 Python/ROCm 환경은 수정하지 않는다. 코드와 실험 자료는 연구 폴더에만 둔다.

## 1. 실제 기준 커널을 먼저 구분한다 — 소스 B + 연산 GPU A 근거

| 연산/호출 위치 | 현재 소스의 계산 | 보존할 계약 | GPU 기준 커널 이름 |
|---|---|---|---|
| Comfy `ops.LayerNorm` | `torch.nn.functional.layer_norm`, 또는 상속된 PyTorch forward | 마지막 D 차원, population variance, 해당 모듈 eps, weight/bias의 lowvram cast 수명 | `vectorized_layer_norm_kernel<..., false>` / RMS는 `<..., true>` |
| Comfy `ops.RMSNorm`, `comfy.rmsnorm.rms_norm` | `torch.nn.functional.rms_norm` | 마지막 D 차원, eps, weight cast, reduction 및 dtype 반올림 | `vectorized_layer_norm_kernel<..., true>` |
| Anima `cosmos.predict2.Block` | LN → `(1 + scale)` → 곱 → shift → compute dtype 변환 | residual 입력 FP32, modulation FP16, `1+scale`은 FP16 반올림, 곱·합 FP32, attention/MLP 입력 FP16 | FP32 LN + FP16 scale-add + FP32 mul/add + FP16 copy, 총 5 kernel |
| MOSS `modulated_rms_norm` | `ck.rms_adaln(x, weight * (1 + scale) - 1, shift, eps)` | 이미 융합된 경로. weight fold와 norm weight offload 수명 유지 | HIP `adaln_kernel<float, false>` 1회 |
| Qwen `_modulated_norm` | `ck.adaln` 전체 target 계산 후 prefix 부분을 재계산하여 덮어쓰기 | prefix/target scale 분리, shift=0, eps=1e-6 | HIP `adaln_kernel<float, true>` 2회 + prefix copy; target만이면 1회 |
| Qwen `ZeroCenteredRMSNorm` | weight를 FP32로 cast한 뒤 +1, 입력도 FP32, RMSNorm 뒤 원래 dtype 복귀 | 저장 weight가 scale−1이라는 계약, FP32 적용 순서 | 별도 모듈 경로이며 bare RMSNorm 표와 혼동 금지 |

여기서 `ck`는 **comfy_kitchen Python 패키지**다. AMD Composable Kernel이라는 이름만 보고 같은 구현으로 분류하면 안 된다. `comfy.quant_ops` 소스는 HIP backend를 우선하고 Triton은 opt-in이며 기본 비활성화라고 설명한다. 이번 벤치의 `registry.get_implementation(...)` 실측은 `comfy_kitchen.backends.hip.adaln/rms_adaln`을 선택했다. 구현 파일 SHA는 `26faf7b55ed050d90a08bbd9126a75d2bc2bbfdaaae384cb68b44cbb96ca1e13`이며, raw JSON에 실제 GPU kernel 전체 이름과 함께 보존했다.

소스 기준:

- `moss-image-rdna2-lab01/runtime/ComfyUI/comfy/ops.py`: LayerNorm/RMSNorm forward 및 CastBiasWeightContext.
- 같은 runtime의 `comfy/ldm/cosmos/predict2.py`: Block의 세 normalization 단계와 FP32 residual 보호. SHA256 `8f3427c1199ea0f2845d5934ca3929bc151178d60a9373e6a363b9e63b2fd3ce`.
- 같은 runtime의 `comfy/ldm/moss_image/model.py`: `modulated_rms_norm`, `gated_residual`, Q/K의 `rms_rope_split_half`.
- `qwen-image-21-rdna2-lab01/runtime/ComfyUI/comfy/ldm/qwen_image21/model.py`: `_modulated_norm`, `_gated_residual`, `ZeroCenteredRMSNorm`; SHA256 `81a43c55e2292889d8b77d79cf4cb56fd3dfb717baf5e40f1d31f7700100c039`.
- 실제 MOSS venv의 `comfy_kitchen/__init__.py`: 공개 adaLN 호출은 `torch.ops.comfy_kitchen.*` dispatch. package SHA256 `43464c123aa7fad61ff6e452df5ca6af018589ec960ed3423a90fc21682e0750`.

## 2. 후보 — 정확도 경계를 분리한다

`rdna2_ops/norm_fusions_v06.py`는 전역 dispatch를 설치하지 않는 명시적 연구 API다.

| 후보 | 구현 | 기대 효과 | 수치 위험 / 사전 판정 |
|---|---|---|---|
| `row_norm` | 한 행당 Triton 프로그램, FP32 통계, LN은 centered variance | bare LN/RMS launch 및 임시 tensor 감소 여부 조사 | reduction 순서가 달라질 수 있음. 연산 오차 및 모델 RGB 둘 다 필요 |
| `native_norm_fused_mod` | native LN 결과를 그대로 받아 scale-add·곱·shift·최종 cast만 Triton으로 융합 | Anima에서 큰 FP32 activation의 추가 읽기/쓰기와 launch 감소 | 중간 dtype cast를 명시하고 FP fusion을 끔. native 대비 **바이트 일치**를 연산 gate로 요구 |
| `triton_eager_rounding` | normalization 통계부터 modulation/cast까지 한 커널 | native LN 출력의 저장·재로드도 제거 | dtype 경계는 보존하지만 통계 reduction은 다름 |
| `triton_single_pass` | 현재 kitchen adaLN/RMS-adaLN과 비교; Qwen prefix scale을 행별 선택 | 이미 융합된 MOSS 경로보다 빠른지 확인, Qwen prefix 중복 계산 제거 | kitchen과 reduction/산술 순서 차이가 날 수 있음. 더 느리거나 RGB 실패면 채택하지 않음 |

MOSS·Qwen·Anima의 Q/K에는 이미 `rms_rope` 또는 `rms_rope_split_half`가 쓰이는 구간이 있다. bare RMSNorm에서 빠르다는 결과로 그 융합 경로까지 대체하지 않는다. gate/residual은 새 커널로 교체하지 않고 native `torch.addcmul`을 유지한다. MOSS의 inference in-place `addcmul_`과 Qwen의 prefix overwrite, Anima의 gate/result FP32 변환은 서로 다른 계약이다.

## 3. 최소 3-shape 연산 실험 — 12개 사례 완료

모든 입력값은 deterministic synthetic 값이며, shape 출처와 실제 모델 값의 차이를 결과에 명시한다. 이미 측정한 레이아웃 실험과 마찬가지로 실제 모델 값에 대한 판정은 4단계에서 한다.

| 연산군 | shape / dtype | 개수 |
|---|---|---:|
| LayerNorm | `[2,4096,2048]` FP32; `[1,2400,4096]` FP32; `[8,257,768]` FP16 | 3 |
| RMSNorm | `[2,512,16,128]` FP16; `[1,37,3072]` FP32; `[1,2564,4096]` FP32 | 3 |
| Anima-style adaLN | `[2,1,64,64,2048]`; `[1,1,32,32,2048]`; `[1,1,64,128,2048]`, 모두 FP32 residual / FP16 modulation·출력 | 3 |
| 기존 kitchen 융합 비교 | MOSS `[1,1061,3072]` RMS-adaLN; Qwen `[1,4964,4096]` prefix 2564; Qwen `[1,2400,4096]` target | 3 |

`bench_norm_fusions_v06.py`는 3회 warmup 이후 3라운드 순서를 회전·반전해서 측정한다. 기본 각 round 10호출이며, synchronized wall time과 GPU event time을 둘 다 남긴다. 할당 및 필요한 변환 비용을 포함하고, JIT·정확도 검사·프로파일 export는 timing에서 제외한다. profile의 연산 attribution과 실제 GPU kernel activity를 합산하지 않는다.

전체 tensor native 오차와 별도의 CPU FP64 통계 sample, peak allocated 증가량, 실제 kernel 명칭을 저장한다. FP16 tolerance는 rtol/atol 0.005/0.005, FP32는 0.0001/0.00003이다. `native_norm_fused_mod`에는 이 tolerance와 별도로 byte equality를 요구한다. **최종 RGB gate는 최대 2/255와 RMSE 0.25/255**이며 연산 tolerance가 이를 대신하지 않는다.

GpuAudit는 연구 lock과 `/sys/class/kfd/kfd/proc` 전후·0.25초 sampling을 사용한다. 외부 compute PID 또는 감시 불가가 발견되면 전체 timing을 폐기한다. graphics 활동과 sampling 사이의 극히 짧은 compute는 완전히 검출할 수 없다는 한계도 raw report에 남긴다.

## 4. 모델 A/B 연결과 CPU 검증

`norm_switches_v06.py`의 `activate()`를 generic harness의 `switch_module`로 지정한다.

- `norm_ln`, `norm_rms`: Comfy의 bare norm forward만 sampling context에서 교체한다. lowvram cast/uncast는 기존 CastBiasWeightContext로 유지한다.
- `norm_modulate`: Anima Block의 native LN을 유지하고 modulation+cast만 융합한다.
- `norm_adaln`: 같은 Anima Block에서 통계까지 포함한 융합을 실험한다.
- `norm_control`: 동일한 optional Attention control만 쓰는 비교군이다.
- `attn_...`: 기존 `attention_switches_v06.activate`에 그대로 위임한다. norm 실험 서버에 이미 로드된 Anima를 이용해 `attn_prescale4d_native`를 추가 2회 반복할 수 있으며, norm 기준선은 native 그대로 유지한다. 원본 Attention 연산 소스는 변경하지 않는다.

Anima의 세 stage 실행은 기존 module들을 호출한다. 측정한 graph의 `use_adaln_lora`, extra position, self/cross context 및 `torch.addcmul` 계산을 보존하고, source SHA가 다르면 적용을 거부한다. 현재 배포하는 연구 adapter는 custom `mlp_patch`가 하나라도 있으면 원본 Block 전체로 fallback한다. 다중 patch의 반환 dict 전달까지 새 orchestration이 일반적으로 보존한다고 주장하지 않는다. 모든 교체는 context exit에서 복구되며 운영 package 수정이나 bytecode 교체가 아니다.

CPU 실제 기존 venv 실행 `989806c8-da87-460e-9d90-b3253ce48a17`은 성공했다. LoRA 유무 × position 유무 × 단일 x-only MLP patch 유무 8가지 조합에서 설치된 Anima source forward와 새 orchestration의 출력이 값 단위로 정확히 같았다(max abs 0). 3개 compact broadcast/stride 계약이 같았고, 지원하지 않는 교대 broadcast는 거부했다. GPU initialization은 false다. 원자료는 `results/norm_cpu_contract_v06.json`이다. **이 CPU 검증은 GPU 커널 정확도·속도·모델 RGB 성공이 아니다.**

`research_configs/anima_norm_v06.json`은 기존 모델과 workflow를 재사용한다. 연산 gate를 통과한 후보를 seed 20261107 / 30 steps로 A/B했다. 실제 실행 job은 `9a5c21f9-f80e-41df-bd44-5748840c08f8`이며, 원자료는 `results/v06_anima_norm_ab/`에 보존했다.

```bash
# 기존 MOSS with_local_storage.sh / .venv-moss Python으로 실행
python bench_norm_fusions_v06.py --output results/norm_fusions_v06.json
python research_comfy_v05.py --spec research_configs/anima_norm_v06.json \
  --out results/v06_anima_norm_ab \
  --modes warm,norm_control,norm_ln,norm_rms,norm_modulate,norm_adaln,norm_control,norm_modulate,attn_prescale4d_native,attn_prescale4d_native \
  --seed 20261107 --steps 30 --port 8474
```

독립 Attention control을 쓰려면 연구 spec의 `norm_attention_mode`와 comparison reference를 명시해야 한다. 이 경우 원래 native 출력 대비 RGB와 해당 Attention control 대비 RGB를 둘 다 보고한다. norm만의 기여를 평가하면서 앞선 Attention의 오차를 숨기지 않는다.

## 5. 카탈로그 반영 상태

| 단계 | 상태 | 근거 등급 |
|---|---|---|
| 실제 함수 및 dtype/rounding 계약 확인 | 완료 | B 소스 |
| 후보 구현 및 CPU orchestration/broadcast 확인 | 완료 | CPU 검증, GPU A 등급 아님 |
| 3-shape GPU kernel/time/memory/error | 4개 연산군 × 3 shape, 총 12개 사례 완료 | A 연산 실측 |
| 기존 PC 모델 최종 RGB A/B | Anima 완료; MOSS/Qwen kitchen 후보는 모델 A/B 미실행 | A Anima 실측, 다른 모델로 확대하지 않음 |
| 채택 또는 실패 원인 확정 | 통계 보존 modulation은 bitexact·작은 모델 이득; LN/RMS/full adaLN은 Anima RGB 실패 | 본문 §9의 범위로 한정 |

이 보고서의 연산·모델 결과는 입구 카탈로그와 사용설명서에 연결한다. 새 모델의 출력 gate는 별도로 수행한다.


## 6. GPU 연산 실측 상세 — A, 모델 성능과 구별

실행 job `5c6827d9-78c5-410b-aa9c-a94ffc9cce28`, 2026-11-06 19:13:11–19:13:24 UTC, RX 6900 XT gfx1030, PyTorch 2.12.0+rocm10.2.0a20261011 / HIP 7.17.26405 / Triton 3.8.0. GPU audit 140 sample, 외부 compute PID 및 감시 오류 0, timing 유효. 원자료 `results/norm_fusions_v06.json` SHA256 `90bbdbaf93f1a7632c23a0f403e06dbf26cf56fba74e9eaec8d2afd413b06b4d`.

아래 시간은 **warm synchronized wall median µs**이며 괄호는 **GPU event median µs**다. 단일 profile trace의 kernel duration은 식별용으로 따로 남겼으며 이 반복 측정 숫자와 섞지 않았다. peak는 살아 있는 입력/준비 가중치를 제외한 additional peak allocated MiB다. MOSS의 사전 weight-fold는 두 경로에서 똑같이 제외하고, Qwen prefix 재계산·overwrite는 기준 경로에 포함했다.

| 사례 | 후보 | native wall (event) µs | 후보 wall (event) µs | wall 배속 | peak MiB native→후보 | native 대비 max abs | 실제 byte 일치 |
|---|---|---:|---:|---:|---:|---:|---|
| `ln_anima_residual` | `triton_centered` | 102.628 (100.150) | 91.503 (89.351) | 1.122× | 64.062→64.000 | 9.5367432e-07 | 불일치 |
| `ln_qwen` | `triton_centered` | 103.947 (101.764) | 68.917 (60.819) | 1.508× | 38.019→38.000 | 9.5367432e-07 | 불일치 |
| `ln_half_tail` | `triton_centered` | 16.935 (15.208) | 24.016 (21.866) | 0.705× | 3.028→3.012 | 0.001953125 | 불일치 |
| `rms_anima_cross` | `triton_centered` | 58.469 (56.733) | 24.899 (23.155) | 2.348× | 4.062→4.000 | 0.001953125 | 불일치 |
| `rms_moss_text` | `triton_centered` | 8.382 (6.634) | 24.771 (23.035) | 0.338× | 0.434→0.434 | 4.7683716e-07 | 불일치 |
| `rms_qwen_text` | `triton_centered` | 101.214 (99.370) | 67.926 (66.083) | 1.490× | 40.073→40.062 | 9.5367432e-07 | 불일치 |
| `adaln_anima` | `native_norm_fused_mod` | 742.023 (739.862) | 280.127 (277.777) | 2.649× | 192.000→96.000 | 0 | 일치 |
| `adaln_anima` | `triton_eager_rounding` | 742.023 (739.862) | 87.361 (85.390) | 8.494× | 192.000→32.000 | 0.001953125 | 불일치 |
| `adaln_anima_small` | `native_norm_fused_mod` | 61.316 (59.284) | 43.855 (41.475) | 1.398× | 24.000→12.000 | 0 | 일치 |
| `adaln_anima_small` | `triton_eager_rounding` | 61.316 (59.284) | 40.490 (38.617) | 1.514× | 24.000→4.000 | 0.001953125 | 불일치 |
| `adaln_anima_long` | `native_norm_fused_mod` | 698.631 (690.048) | 273.430 (271.285) | 2.555× | 192.000→96.000 | 0 | 일치 |
| `adaln_anima_long` | `triton_eager_rounding` | 698.631 (690.048) | 114.510 (112.335) | 6.101× | 192.000→32.000 | 0.001953125 | 불일치 |
| `kitchen_moss` | `triton_single_pass` | 46.007 (44.214) | 38.726 (36.950) | 1.188× | 12.434→12.434 | 9.5367432e-07 | 불일치 |
| `kitchen_qwen_prefix` | `triton_single_pass` | 603.027 (598.086) | 364.579 (362.502) | 1.654× | 118.062→78.000 | 9.5367432e-07 | 불일치 |
| `kitchen_qwen_target` | `triton_single_pass` | 133.255 (131.462) | 64.157 (62.071) | 2.077× | 38.000→38.000 | 9.5367432e-07 | 불일치 |

12개 native 대조와 15개 후보는 정해 둔 연산 tolerance를 모두 통과했다. **값이 비슷하다는 것과 바이트 일치는 다르다.** 3개 `native_norm_fused_mod`는 `view(torch.uint8)` 비교까지 통과했고, 나머지 후보는 native와 수치 차이가 남는다. FP64 비교는 전체 출력이 아니라 고정 4행의 full-D 통계 sample이다. sample이 같아도 전체 native 차이를 없었다고 쓰지 않는다.

### 6.1 성공과 실패의 원인

- 큰 FP32 Anima modulation은 5개 kernel을 native LN + Triton modulation 2개로 줄였다. 대표 shape 추가 peak가 192→96 MiB로 줄고, native 통계 및 FP16 `1+scale` 반올림을 유지했다. 통계까지 합치면 1개 kernel / 32 MiB가 되지만 reduction 차이가 생긴다.
- LN FP32 D2048/4096, RMS FP16 D128/FP32 D4096은 이 synthetic shape에서 1.12–2.35×였다. LN FP16 D768은 0.705×, 작은 `[1,37,3072]` FP32 RMS는 0.338×로 실패했다. 작은 RMS는 profile상 kernel 자체가 빠른데도 일반 호출의 wall/event가 길어 launch·Python 제출 비용이 지배한다. 이 결과로 전체 dtype/width에 자동 적용하지 않는다.
- MOSS의 기준선은 이미 **Comfy Kitchen HIP 융합**이며 eager RMS+modulation이 아니다. 새 single-pass의 이득은 이 기준선 대비 1.188×다. 실제 MOSS 모델 RGB A/B는 아직 없으므로 이 kitchen 후보를 채택하지 않는다.
- Qwen prefix는 전체 target norm 뒤 prefix를 다시 계산하고 복사했다. 행별로 scale을 고르는 후보는 이를 한 커널로 줄여 1.654×, peak 118.0625→78 MiB였다. target-only는 2.077×. 실제 Qwen 모델 RGB A/B는 아직 없으며 Anima 결과를 대신 쓰지 않는다.

### 6.2 실제 커널 식별

| 기준 또는 후보 | 실제 실행 kernel (긴 template signature는 raw JSON에 보존) |
|---|---|
| PyTorch LN FP32/FP16 | `at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float 또는 c10::Half, float, false>` |
| PyTorch RMS FP32/FP16 | 같은 이름의 `..., true>` specialization |
| Comfy Kitchen HIP RMS-adaLN | `comfy::hip_backend::adaln_kernel<float, false>` |
| Comfy Kitchen HIP adaLN | `comfy::hip_backend::adaln_kernel<float, true>` |
| Qwen prefix overwrite | `__amd_rocclr_copyBuffer.kd` |
| 새 norm/full adaLN/single-pass | `_row_norm_v06.kd` |
| native norm 유지 + modulation | native LN kernel + `_modulate_v06.kd` |

`device_activities`와 `operator_attribution`은 원자료에서 독립된 표다. 둘을 더해 GPU 시간을 중복 합산하지 않는다. profile의 단일 launch는 cache 상태·계측 비용이 반복 timing과 다를 수 있으므로 속도 순위는 위 median으로 판정한다.

## 7. 적용 범위와 롤백

연산 API는 독립 함수다. 운영 Comfy 파일·startup·모델 가중치에는 설치하지 않았다. 모델 검증 adapter는 측정한 dtype/width family만 받는다: bare LN은 FP32 D2048/4096, bare RMS는 FP16 D128 또는 FP32 D4096; Anima modulation은 FP32 residual D2048 + FP16 scale/shift/output. 이 필터는 **모든 행 수에서 이득을 보증하는 자동 dispatcher가 아니며**, 현재 모델 A/B용 경계다. BF16, 다른 stride, training 및 미측정 형태를 이번 결과로 채택하지 않는다.

다른 프로젝트에서 통계를 유지한 후단 융합만 시험할 때는 기존 normalization callable을 그대로 호출한다.

```python
import torch
from rdna2_ops.norm_fusions_v06 import modulate

with torch.inference_mode():
    normalized = native_norm(x)  # 기존 eps, weight, bias, 통계 구현 유지
    y = modulate(normalized, scale, shift, out_dtype=compute_dtype)
```

측정한 bitexact 계약은 normalized FP32 / scale·shift FP16 / output FP16이며, `scale`을 미리 FP32로 올려 `1+scale`을 계산하지 않는다. 지원한 broadcast는 `[B,T,1,1,D]`, `[B,1,D]`, 전역 `[D]`의 연속 행 그룹이다. 행 사이 stride가 있는 chunked scale은 복사 없이 받지만 교대 broadcast는 명시적으로 거부한다. 새 모델의 실제 shape와 dtype에서 native 표현식 및 최종 출력 gate를 다시 비교한다. `row_norm(...)`과 `adaln(...)` 통계 교체 API는 이번 Anima strict 실패가 있으므로 이 bitexact 레시피와 구분한다.

실험 중 교체는 sampling context 안에만 존재하며 context 종료 시 원래 Python callable로 복구한다. 롤백은 실험 서버 종료 또는 `norm_control` 사용이다. 기존 운영 서버를 다시 구성하거나 package를 재설치할 일이 없다. 새 모델에서는 해당 최종 출력 gate를 다시 통과해야 한다.


## 8. 소스 계보와 검증 범위

| 구분 | 실제 사용 또는 보존 SHA256 | 의미 |
|---|---|---|
| GPU 연산 kernel | `672d6c3171c04545ecf15c41b84875caba0cf5fa854724855171d49d9cbb5ee8` | `rdna2_ops/norm_fusions_v06.py`; operator와 모델 계산 코드 동일 |
| GPU operator bench | `f98c85186b85d8029a4befdab28b95c59134f90ef23c1044bcff7e6606ef51dc` | 12사례, 3 warmup/3라운드, 실측 당시 source |
| operator 실행 직전 읽은 adapter | `9d243e80dba387ae44d28c724b9a35d6586f6d350285eec86074fb1bd7e1868c` | `results/sources/norm_switches_v06_before_measured_filter.py`에 원문 보존; operator 자체는 이 adapter를 사용하지 않음 |
| 실제 모델 adapter | `2b8fcd431bd92a79badd52fa75a515e9396ef500854e8e4618bd2c0e1d98f9f2` | operator 결과 후 느린 dtype/width eligibility를 좁힘; `results/sources/norm_switches_v06_measured_model.py`에 보존 |
| 모델 완료 후 현재 adapter | `87a11ff0b8369bdebcd27931ea7c6d4a6bd592ccc1aaad0796716227c014f372` | custom MLP patch 존재 시 원본 Block fallback guard와 설명만 추가; 계산 본문 변경 없음, 추가 GPU 실행 없음 |
| 모델 harness | `f9c315caf7200e0ea664f491dc27b807e7b380f7d63d17a9ad9a962801136b30` | optional instrument_module 지원 버전; 이 Anima spec에는 instrument_module 없음 |
| GPU audit helper | `7409f98dbf2c1975f5727338346af248657d98229c4a92732809379207b4b329` | 다른 실험과 같은 KFD 감시/연구 lock |

operator 직전 adapter(9d243e80)와 실제 모델 측정 adapter(2b8fcd43)의 `anima_block` 함수 source는 완전히 동일하며 그 본문 SHA256은 `c3ec92e1d884dc8d0ecca9062c4e57a8f32b66a07613c86db05a9549c97f71c6`이다. 기존 CPU contract JSON은 설치된 Anima 원본 SHA만 기록했고 연구 adapter 전체 SHA는 기록하지 않았다. 따라서 현재 adapter SHA를 그 CPU 실행에 소급해서 붙이지 않는다. CPU JSON의 `bit_equal`은 당시 `torch.equal` 값 비교이며 signed-zero byte를 구분하지 않는다. 본문에서는 CPU 값 일치로만 표현한다. GPU operator의 `byte_equal`은 uint8 비교, 모델은 실제 tensor byte 비교와 SHA256 digest로 확인한다.

모델 첫 후보는 필요한 module 준비·커널 캐시/JIT 비용을 포함한 request 그대로 기록한다. operator 벤치는 warm 경로다. 첫 model sample만으로 warm 실행 시간을 확정하거나, 느린 첫 sample을 원자료에서 지우지 않는다.


## 9. 실제 Anima 최종 A/B — A, 엄격 RGB 판정 완료

기존 1024² Anima workflow, seed 20261107, 30 steps. 하나의 연구 서버에서 10 request를 순차 실행했다. job `9a5c21f9-f80e-41df-bd44-5748840c08f8`은 2026-11-06 19:16:51–19:30:20 UTC, exit 0. KFD GPU audit 3,178 sample에서 외부 compute PID/감시 오류 0으로 timing 유효다. 운영 서버·가중치·환경은 수정하지 않았다.

원자료: `v06_anima_norm_ab/runs.json`, `run_spec.json`, `runtime_metadata.json`, `submitted_workflow.json`, `gpu_audit.json`; 재계산 요약은 `summary.json`. 이미지 원본은 실제 PC의 같은 결과 폴더 `images/`에 보존했다. 아래는 모든 request를 그대로 나열한다. 시간은 synchronized denoise/VAEDecode stage wall이며, request wall에는 서버 처리·저장 등 추가 비용이 포함된다.

| 순서·경로 | denoise s | VAE s | request wall s | hit/fallback | RGB max / RMSE | latent·RGB byte | 엄격 RGB |
|---|---:|---:|---:|---:|---:|---|---|
| 0 `warm` | 81.04276 | 0.65260 | 83.42055 | 0/0 | 0 / 0 | 일치 | 통과 |
| 1 `norm_control` | 80.60703 | 0.55580 | 81.48646 | 0/0 | 0 / 0 | 일치 | 통과 |
| 2 `norm_ln` | 94.31176 | 0.55181 | 95.15586 | 2520/30 | 0.82470703 / 0.023685483 | 불일치 | 실패 |
| 3 `norm_rms` | 80.45289 | 0.55120 | 81.27377 | 1680/116 | 0.8203125 / 0.025157358 | 불일치 | 실패 |
| 4 `norm_modulate` | 80.24573 | 0.55323 | 81.09419 | 2520/0 | 0 / 0 | 일치 | 통과 |
| 5 `norm_adaln` | 79.37538 | 0.54616 | 80.21505 | 2520/0 | 0.80908203 / 0.024909076 | 불일치 | 실패 |
| 6 `norm_control` | 80.71983 | 0.55512 | 81.54889 | 0/0 | 0 / 0 | 일치 | 통과 |
| 7 `norm_modulate` | 80.45113 | 0.55356 | 81.30439 | 2520/0 | 0 / 0 | 일치 | 통과 |
| 8 `attn_prescale4d_native` | 67.77495 | 0.55348 | 68.61411 | 1680/0 | 0 / 0 | 일치 | 통과 |
| 9 `attn_prescale4d_native` | 67.72710 | 0.55433 | 68.57754 | 1680/0 | 0 / 0 | 일치 | 통과 |

### 9.1 정규화 후보 판정

- **통계 보존 modulation:** native controls 80.60703 / 80.71983 s, 후보 80.24573 / 80.45113 s. 중앙값 80.66343→80.34843 s, **1.00392× / 시간 약 0.391% 감소**. native span 0.11280 s, 후보 span 0.20540 s, 각 n=2다. 두 후보가 모두 native보다 짧았지만 작은 차이이며 장시간 반복이나 다른 모델의 성능 보증으로 일반화하지 않는다. 연산 1.40–2.65×와 전체 모델 약 1.004×는 별도 결과다. 확실한 품질 성과는 2,520회 적용/0 fallback에서 두 번 모두 latent·decoded RGB가 native와 바이트 일치한 것이다.
- **LN 통계 교체:** 2,520 hit / 30 fallback, RGB max 0.824707 / RMSE 0.0236855로 실패. 첫 후보 denoise 94.31176 s는 준비·캐시/JIT 비용을 포함하며 warm 반복을 측정하지 않았다. 실패를 숨기거나 그 시간을 삭제하지 않는다.
- **RMS 통계 교체:** 1,680 hit / 116 fallback. 실제 적용은 FP16 `[2,4096,16,128]` 840회 및 `[2,512,16,128]` 840회다. RGB max 0.8203125 / RMSE 0.0251574로 실패했고, 첫 model 시간 80.45289 s를 큰 개선으로 해석하지 않는다. 이미 융합된 self-attention RMS+RoPE를 bare RMS 후보로 교체한 결과가 아니다.
- **full adaLN:** 2,520 hit / 0 fallback, 79.37538 s이나 RGB max 0.809082 / RMSE 0.0249091로 실패. FP16 `1+scale` 경계와 FP32 residual을 보존해도 통계 reduction의 차이가 남는다. 연산 tolerance 통과가 최종 diffusion RGB 통과를 보장하지 않았다.

현재 엄격 기준에서 통계 교체 세 후보는 **Anima 자동 적용 대상에서 제외**한다. 범용 연산 구현은 연구 후보로 남기며, 다른 모델은 그 모델의 최종 출력 gate로 새로 판정한다. MOSS/Qwen kitchen 후보에는 이번 Anima A/B를 대신 붙이지 않는다.

### 9.2 4D Attention warm 추가 검증 — 정규화와 독립된 경로

Norm 후보를 모두 원복한 동일 서버에서 guarded `attn_prescale4d_native`만 두 번 적용했다. 두 요청 모두 self/cross 각 840회, 합계 1,680 hit / 0 fallback이다. startup `fp16_bf16_reduction_math_sdp_allowed=True`, Attention source `a207e62e2776163ad4b37119fcf76c5d32a495728ec8b382430d5125b97ba65a`, adapter `dad380e7d7c96a8021ab771128df68f8f10454743db61c73cf160046dfc9d2ca`를 실제 metadata에 기록했다.

| 별도 실행 | native denoise s | 4D Attention denoise s | 최종 판정 |
|---|---:|---:|---|
| 이전 Attention A/B | 80.79604 / 80.75076 | 첫 83.01858, 후속 67.72613 | 두 번 모두 byte 일치; 첫 느린 sample 보존 |
| 이번 Norm 서버의 추가 warm n=2 | 80.60703 / 80.71983 | 67.77495 / 67.72710 | 두 번 모두 byte 일치, native 중앙값 대비 1.19059× |

이번 warm 중앙값은 67.75103 s다. 이전의 83초 첫 요청을 평균에서 몰래 빼거나 파일에서 지우지 않았다. 4D 경로의 row repair는 Triton kernel이며 이번에는 앞서 Norm 후보가 Triton을 사용한 상태다. 첫 Triton runtime/launch 준비가 이전 느린 첫 요청의 원인일 가능성이 있지만 이를 분리 계측하지 않았으므로 **원인은 확정하지 않았다**. 이번 두 번은 warm 속도의 재현 근거이며 fresh-process 첫 사용 비용을 없앴다는 결과가 아니다.

### 9.3 출력과 conditioning 동일성

이번 native controls·modulation 두 번·4D Attention 두 번은 이전 `v06_anima_attention_ab`의 native와도 아래 digest가 모두 같았다. 비교는 CPU에서 raw JSON의 실제 digest를 대조했다. 핵심 gate는 각 서버 내부의 tensor byte 비교이며, digest는 별도 실행 간 동일성을 추가로 확인한다.

- latent SHA256: `cc054fd19dc8f334da8d9acaf3b0b53aab1165944a6023a38c6c1ee9a34c3a13`
- decoded RGB tensor SHA256: `52ea16286731f37e2760ef622b66446443e95e657aea57dc9b0dd511b853cb3a`
- positive conditioning 첫 embedding SHA256: `a6c15ee3d0208bb976cb338714960a180fd73170fa904280577da928000f4968`
- negative conditioning 첫 embedding SHA256: `f2199b5a6d7ba57a2d9d62675ea8d45c8d5da3f77d6e969fc1ee7f562cb0ec2e`

최종 RGB 기준은 max 2/255, RMSE 0.25/255를 유지했다. tensor byte/digest는 이미지 tensor의 값과 표현을 확인한 것이며 저장 PNG의 metadata·파일명까지 같은지 판정한 검사는 아니다.
