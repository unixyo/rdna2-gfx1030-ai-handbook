# MOSS / RX 6900 XT 운영 통합

기존 Ubuntu·Radeon RX 6900 XT (`gfx1030`) 환경의 MOSS ComfyUI에 적용하는 구현이다. 기존 `.venv-moss`, 모델, DynamicVRAM, 시작 스크립트를 사용한다. INT8 전치 복사 제거와 FP32 Attention 개선의 적용 범위, 실행 방법, 실제 서버 검증 결과를 기록한다.

## 정상 서버 적용 확인

2026-11-05 14:53 UTC에 기존 launcher로 정상 서버를 시작했다. `MOSS_RDNA2_MEASURE_DIR`는 비활성이고 선택적 INT8 통계도 껐다. 서버는 `127.0.0.1:8299`, PID `2015237`에서 `view:auto` 활성 로그를 남겼다. 원래 `sft_fast4_int8_512`, seed `20261105` 요청을 실제 실행했고 API 성공 상태와 322,210-byte PNG 저장을 확인했다. 작업 종료 시 이 서버를 실행 상태로 남겼다. 상세 기록: [normal_production_smoke.json](../results/normal_production_smoke.json).

이는 정상 설정의 전체 요청 실행 확인이다. 별도 프로세스이므로 cross-restart bit equality를 주장하지 않는다. 수치 동일성과 warm 성능은 아래의 paired A/B를 따른다.

## 실행과 롤백

대상 프로젝트는 `~/DevSpace/moss-image-rdna2-lab01`이다. 이 디렉터리의 `custom_nodes/MOSS_RDNA2_INT8/`에 이 패키지의 같은 이름 디렉터리를 배치한다. 기존 `scripts/rdna2_int8_matmul.py`가 필요하다. 설치된 PyTorch/Kitchen 파일이나 모델 가중치는 수정하지 않는다.

최신 [start_moss_comfy.sh](start_moss_comfy.sh)의 기본값은 다음과 같다.

| 환경변수 | 기본값 | 동작 |
|---|---|---|
| `MOSS_RDNA2_INT8_PATCH` | `1` | gfx1030용 기존 Triton INT8 GEMM 보정 |
| `MOSS_RDNA2_INT8_LAYOUT` | `fused`(2026-11-07부터, 이전 `view`) | `view`: FP32 입력·출력인 eager INT8 linear의 전치 복사 제거. `fused`: view 위에 INT8 GEMM+FP32 dequant 융합 커널(`fused_gemm.py`). 같은 프로세스 4.84 → 2.11 s, latent·PNG SHA 동일. 롤백은 `LAYOUT=view` |
| `MOSS_RDNA2_ATTENTION` | `auto` | 검증한 FP32 Attention 조건에만 개선 경로 적용 |

정상 실행에서는 `MOSS_RDNA2_MEASURE_DIR`를 설정하지 않는다. 프로젝트 디렉터리에서 서버를 시작하고, 다른 터미널에서 기존 워크플로를 제출한다.

```bash
cd ~/DevSpace/moss-image-rdna2-lab01
bash scripts/start_moss_comfy.sh
```

```bash
cd ~/DevSpace/moss-image-rdna2-lab01
python3 scripts/submit_workflow.py --workflow sft_fast4_int8_512 --seed 20261105
```

기존에 동작하던 경로로 되돌릴 때는 해당 MOSS 서버를 종료한 다음 다음 설정으로 시작한다.

```bash
bash scripts/stop_moss_comfy.sh
MOSS_RDNA2_INT8_PATCH=1 \
MOSS_RDNA2_INT8_LAYOUT=legacy \
MOSS_RDNA2_ATTENTION=native \
bash scripts/start_moss_comfy.sh
```

롤백에도 `PATCH=1`을 유지한다. `PATCH=0`은 gfx1030에서 필요했던 원래 GEMM 보정까지 끈다. 변경 전 파일은 [rollback/start_moss_comfy.sh](rollback/start_moss_comfy.sh)와 [rollback/MOSS_RDNA2_INT8_init.py](rollback/MOSS_RDNA2_INT8_init.py)에 보관돼 있다. 측정용 환경변수가 남아 있다면 정상 재시작 전에 해제한다.

## 실제 연결 지점

**INT8:** [backend.py](MOSS_RDNA2_INT8/backend.py)의 `install_backend(int8_matmul)`은 `comfy_kitchen.backends.eager.int8_linear`의 export만 교체한다. Kitchen의 public `int8_linear`와 Torch custom-op은 `registry.get_implementation()`에서 선택한 모듈의 속성을 매번 읽으므로 이 변경을 사용한다. 기존 `eager` 이름, 우선순위, constraints, capabilities는 보존한다. `quantization.int8_linear` 원본과 직접 import된 하위 alias도 유지한다. 운영 구현에는 `__code__` 변경이나 소스 재컴파일이 없다.

**INT8 view는 `x.dtype == torch.float32`이면서 `out_dtype == torch.float32`인 호출에만 적용한다.** 입력 또는 출력이 FP16/BF16이면 원래 Kitchen eager 함수를 그대로 호출한다. 이로써 외부 FP16 text encoder 등 미검증 precision 경로까지 변경하지 않는다. 기존 gfx1030 GEMM 보정은 유지한다.

이 FP32 경로의 전처리, ConvRot, rowwise quantization, 청크별 FP32 scale 계산, 출력 dtype 변환, bias, residual 순서는 기존 eager와 동일하다. GEMM의 두 번째 인자만 `weight.T.contiguous()`에서 `weight.T`로 바꾼다. 기존 gfx1030 커널은 실제 입력 stride를 사용한다. `disable_backend()`는 작업 사이에 원래 eager export를 복원한다.

**Attention:** [attention_dispatch.py](MOSS_RDNA2_INT8/attention_dispatch.py)가 `comfy.ops.scaled_dot_product_attention`에 연결된다. `auto`는 [math_attention.py](MOSS_RDNA2_INT8/math_attention.py)의 FP32 수학 경로를 선택한다. PyTorch와 같은 Q/K 선행 scaling, `torch.bmm`/rocBLAS, native `torch.softmax`를 사용하며, 작은 Triton 커널이 모든 score가 음의 무한대인 행을 보정한다. 기본 경로에서 FP16/BF16으로 낮추지 않는다.

자동 선택 조건은 다음을 모두 만족해야 한다.

- Q/K/V가 같은 `gfx1030` 장치의 FP32이고, 같은 shape `[1, 24, S, 128]`, `128 ≤ S ≤ 4096`이다.
- 마지막 차원 stride는 `1`, head stride는 `128`, Q/K의 sequence stride는 `3072`다. V의 sequence stride는 `3072` 또는 packed QKV의 `9216`이다.
- mask 없음, dropout `0`, non-causal, GQA 비활성화인 추론이다. autograd가 필요한 입력, forward AD, autocast, compile/trace/functorch 경로 등은 native fallback을 사용한다.

37-token Attention, VAE의 FP16 Attention 등 조건 밖 호출은 기존 경로를 사용한다. Seeded random FP32 입력으로 다음 두 layout을 검증했다. 모든 경우에서 finite 출력, `max_abs = RMSE = 0`, 전체 바이트 일치를 확인했다.

| V layout | 검증한 sequence length | Complete-call wall-time 개선 | 근거 |
|---|---|---:|---|
| Packed QKV, V sequence stride `9216` | 128, 256, 512, 768, 1024, 1061, 1536, 2048, 3072, 4096 | 1.18–1.41배 | [packed V](../results/attention_ranges_packed_v.json) |
| Interleaved, V sequence stride `3072` | 128, 129, 257, 513, 1025, 1537, 2049, 3073, 4095, 4096 | 1.17–1.38배 | [interleaved](../results/attention_ranges_interleaved.json) |

이 수치는 해당 operator microbenchmark의 결과다. 실제 이미지 검증은 아래 512×512 및 768×512 서버 결과로 구분한다.

### FP32가 선택되는 소스 경로

설치된 ComfyUI의 `comfy/` 아래 코드를 확인한 경로다. 상세 출처는 [SOURCES.md](../SOURCES.md)에 정리한다.

1. `supported_models.py`의 MOSS 지원 추론 dtype은 `[BF16, FP32]`이며 FP16은 없다. `sd.py`의 양자화 loader가 `unet_manual_cast(None, device, supported_dtypes)`를 호출한다.
2. `model_management.py`의 `should_use_bf16(..., manual_cast=False)`는 gfx1030을 거절하고, `unet_manual_cast`는 FP32를 선택한다. 실제 기록도 `model_dtype = bfloat16`, `manual_cast_dtype = float32`다.
3. `model_base.py`의 `get_dtype_inference()`와 모델 호출부가 이 manual-cast dtype으로 latent와 context를 변환한다. `ops.py`의 INT8 출력 dtype도 input/compute dtype을 따른다.
4. MOSS의 Q/K normalization과 RoPE가 그 dtype을 유지한다. `attention.py`는 reshape/transpose한 Q/K/V를 SDPA에 넘기며 여기서 별도의 FP32 cast를 추가하지 않는다. 실제 관측 Q/K/V는 FP32다.

또한 Comfy 시작 코드는 `allow_fp16_bf16_reduction_math_sdp(True)`를 설정한다. 이 환경에서 bare PyTorch의 기본값은 `False`였다. Precision 실험은 이 상태를 기록하고 각 후보의 flag를 명시적으로 설정한 뒤 원복한다.

## 한 번만 전치하는 방식의 검토

[int8_pretranspose_comparison.json](../results/int8_pretranspose_comparison.json)은 불변 INT8 weight에 대해 매번 복사, stride view, 사전 전치한 contiguous RHS를 비교한다. 다섯 case 모두 INT32 결과가 bitwise equal이었다. 아래는 `M=1061`인 네 DiT 행렬의 반복 GEMM 경로 시간이다. 사전 전치 준비 비용과 추가 상주 메모리는 시간 비율에 포함하지 않는다.

| `(N, K)` | View(ms) | 사전 전치(ms) | View 대비 | 추가 RHS(MiB) |
|---|---:|---:|---:|---:|
| `(9216, 3072)` | 5.592292 | 2.196446 | 2.546배 | 27 |
| `(3072, 3072)` | 1.873089 | 0.762261 | 2.457배 | 9 |
| `(16384, 3072)` | 9.686237 | 4.480008 | 2.162배 | 48 |
| `(3072, 8192)` | 5.100946 | 2.038990 | 2.502배 | 24 |

**2.16–2.55배는 GEMM 경로만의 이득이며 전체 이미지 생성의 개선율이 아니다.** 이 사전 전치 캐시는 운영에 넣지 않았다.

- Comfy의 gathered-buffer 복원은 shape/dtype을 기준으로 contiguous view를 재구성한다. 기존 weight를 전치 layout으로 바꾸는 것만으로는 offload 후 stride를 유지할 수 없다.
- 일반 `TensorWiseINT8Layout`의 Linear 경로와 eager linear의 weight 준비 단계가 `.contiguous()`를 요구한다. 저장 layout만 바꾸면 다시 복사하는 경로가 남는다.
- DynamicVRAM resident signature, cast-buffer 재사용, LoRA 적용 및 재양자화가 weight 수명을 바꾼다. 주소·shape만으로 캐시한 RHS는 최신 weight를 보장하지 못한다.
- 위 네 주요 행렬을 42개 DiT block마다 원본과 별도로 모두 상주시킨다면 추가 메모리는 **4.4296875 GiB, 약 4.43 GiB**다. Text encoder나 다른 buffer는 이 추정에 포함하지 않는다.

운영 경로는 포인터 기반 캐시 없이 매 호출 최신 weight의 view를 읽는다. 사전 전치를 통합하려면 별도의 storage/layout 계약과 offload·LoRA 수명 관리를 함께 구현해야 한다.

## 완료된 실제 서버 A/B

원본 `scripts/start_moss_comfy.sh`를 subprocess로 실행하고 원래 API graph `sft_fast4_int8_512`를 제출했다. 512×512, 4-step, seed `20261105`이며, **한 서버 안에서 같은 모델과 conditioning을 유지**해 비교했다. 근거: [production_server_shared.json](../results/production_server_shared.json).

| 모드 | Warm 3회, denoising + VAE 합계(초) | 중앙값(초) | legacy 대비 |
|---|---|---:|---:|
| `legacy:native` | 7.435194, 7.446393, 7.448174 | 7.446393 | 1.000배 |
| `view:native` | 4.994781, 4.994654, 4.999551 | 4.994781 | 1.491배 |
| `view:auto` | 4.751675, 4.808823, 4.762543 | 4.762543 | **1.564배** |

모든 warmup과 본 측정에서 최종 latent와 decoded RGB tensor의 dtype·shape·전체 바이트가 기준과 같았다. FP32 `max_abs = 0`, `RMSE = 0`이다. 기본값인 `view:auto`의 구간 시간은 legacy 대비 **36.04% 감소**했고, `view:native`에서 Attention `auto`를 더한 구간 시간 감소는 **4.65%**다.

시간은 GPU synchronize로 경계를 잡은 **KSampler denoising + VAEDecode wall time 합계**다. 모델 시작, text encoding, PNG 저장, reference 저장·비교 비용까지 포함한 전체 요청 시간으로 해석하지 않는다. Cold/warmup은 중앙값에서 제외했다.

최종 FP32 입력·출력 guard를 넣은 뒤에도 별도 warmup 후 한 번 더 실제 서버에서 확인했다. `legacy:native` 7.431528초 → `view:auto` 4.749214초였고 latent/RGB 전체 바이트가 같았다. 개선된 INT8 경로 672회, 개선된 Attention 168회와 native Attention 8회를 확인했다. 이 추가 1회는 위 3회 중앙값 표에 합치지 않았다. [production_final_scope.json](../results/production_final_scope.json)에 기록했다. 별도 실제 GPU dtype 검사에서도 FP32→FP32만 개선 경로를 사용했고 FP16→FP16, BF16→BF16, FP32→BF16은 원본 eager로 전달되어 네 경우 모두 기준과 바이트가 같았다. export 복원도 확인했다: [production_backend_scope.json](../results/production_backend_scope.json).

### 768×512 추가 검증

같은 기존 워크플로의 크기를 768×512로 바꿔 한 서버에서 비교했다. 출력은 `[1, 512, 768, 3]`이며 실제 Q/K/V shape `[1, 24, 1573, 128]`을 확인했다. `auto`는 개선 경로 168회와 37-token native 경로 8회를 선택했다. 근거: [production_range_768x512.json](../results/production_range_768x512.json).

| 모드 | Warm 3회, denoising + VAE 합계(초) | 중앙값(초) |
|---|---|---:|
| `view:native` | 8.347240, 8.358267, 8.355858 | 8.355858 |
| `view:auto` | 7.849930, 7.911292, 7.843361 | **7.849930** |

모든 기록에서 latent/RGB 전체 바이트가 같았으며 `max_abs = RMSE = 0`이었다. 이 크기에서 FP32 Attention `auto`를 추가한 구간 시간 개선은 **1.06445배, 6.05% 감소**다.

## 재현 방법과 측정 경계

`bench_server.py`를 실행하는 패키지 루트에서, 사용할 MOSS 포트가 비어 있을 때 다음과 같이 비교한다. 이 명령은 실제 모델과 GPU를 사용하며 원래 launcher로 서버를 시작한다.

```bash
python3 bench_server.py \
  --project ~/DevSpace/moss-image-rdna2-lab01 \
  --out results/production_server_repeat \
  --same-server \
  --modes legacy:native view:native view:auto \
  --seeds 20261105 --rounds 3
```

[measurement.py](MOSS_RDNA2_INT8/measurement.py)는 `MOSS_RDNA2_MEASURE_DIR`가 설정된 경우에만 설치된다. 원래 KSampler와 VAEDecode를 호출하며 실제 연산을 재실행한다. 같은 서버에서의 mode 전환은 측정된 KSampler 진입점에서만 수행한다. 정상 운영은 시작 시 환경변수로 정한 static mode를 사용한다. `MOSS_RDNA2_INT8_STATS=1`의 선택적 카운터는 tensor를 보관하거나 호출마다 로그를 쓰지 않는다.

**서버 재시작 간 동일성은 미해결 사항이다.** 변경하지 않은 `legacy:native`도 새 프로세스에서는 다른 출력을 냈고, 각 프로세스 안의 반복은 바이트 단위로 같았다. Denoiser 호출 **이전**에 수집한 text conditioning 해시도 달랐다. Shared-server positive 해시는 `7bc0b8e…`, 다른 precision 서버는 `85f58b6b…`였고 negative도 `10feac…`와 `938c22…`로 달랐다. 따라서 차이는 최적화한 이미지 커널보다 앞선 text-conditioning 경로까지 좁혀졌지만, 그 내부의 정확한 원인은 아직 확인하지 못했다. 시드 누락이라고 단정하지 않는다.

정확성 A/B에는 같은 입력 conditioning이 필요하다. 재시작한 프로세스의 출력만 비교해 개선 커널의 수치 오차로 판정하지 않는다. 현재 표는 같은 서버에서 conditioning 해시와 seed를 유지한 비교 결과다.

## FP16/BF16 Precision 실험과 거절 결과

`fp16_math`, `fp16_math_reduced`, `bf16_math`, `fp16_flash`, `fp16_efficient`는 측정 전용이며 `MOSS_RDNA2_MEASURE_DIR` 없이는 허용하지 않는다. 활성화되더라도 denoising 구간에서만 실험 경로를 선택한다. native 기준을 먼저 만들고 동일 서버/conditioning에서 비교한다.

실행 전에 고정한 RGB 허용 기준은 `[0, 1]` 범위에서 **최대 절대 오차 ≤ `2/255 ≈ 0.00784314`**, **RMSE ≤ `0.25/255 ≈ 0.000980392`**이며, 모든 latent/RGB 값이 finite여야 한다. 임계값은 결과를 본 뒤 완화하지 않았다.

별도의 동일 서버 실험에서 seed `20261105`, `20261106`, `20261107`을 각각 native 기준과 비교했다. 각 seed의 warmup 뒤 한 번씩 측정한 denoising+VAE 시간 중앙값이며, 오차 칸은 **RGB max_abs / RMSE**다. 근거: [production_precision.json](../results/production_precision.json).

| 모드 | 중앙값(초) | Seed 20261105 | Seed 20261106 | Seed 20261107 | 판정 |
|---|---:|---|---|---|---|
| `view:native` FP32 | 5.002610 | 0 / 0 | 0 / 0 | 0 / 0 | 기준, 3/3 통과 |
| `view:fp16_math` | 5.117927 | 0.770996 / 0.088461 | 0.988037 / 0.160306 | 0.838135 / 0.078135 | **거절, 0/3 통과** |
| `view:fp16_math_reduced` | 4.790884 | 0.904297 / 0.111170 | 0.939453 / 0.149440 | 0.870117 / 0.078296 | **거절, 0/3 통과** |
| `view:bf16_math` | 5.131151 | 0.930908 / 0.106474 | 0.945068 / 0.137118 | 0.930664 / 0.102344 | **거절, 0/3 통과** |

모든 출력은 finite였지만 세 precision 후보 모두 세 seed에서 오차 기준을 넘었다. `fp16_math_reduced`의 구간 시간 감소도 정확성 실패를 상쇄하지 못하므로 채택하지 않는다. `fp16_math`와 `bf16_math`는 math SDP의 reduced-precision reduction을 끄고, `fp16_math_reduced`는 켠 실험이다. 이 후보들은 SDPA 내부 Q/K/V 정밀도를 바꾸며, 기본 운영 경로는 검증된 FP32 `auto`를 유지한다.

## 라이선스

`backend.py`의 Kitchen 파생 부분은 Comfy Org의 Apache-2.0 attribution을 보존하며 [LICENSE.apache-2.0](MOSS_RDNA2_INT8/LICENSE.apache-2.0)을 함께 제공한다. 독립적으로 작성한 커널은 저장소 루트의 MIT 라이선스를 따른다.
