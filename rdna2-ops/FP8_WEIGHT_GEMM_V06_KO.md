# FP8 가중치 GEMM — gfx1030 연산 카탈로그 실험 v06

대상은 같은 RX 6900 XT와 기존 PyTorch 2.12.0+rocm10.2.0a20261011 / HIP 7.17 / Triton 3.8 환경이다. 새로운 모델은 다운로드하지 않았다. MOSS·Qwen·Anima·Applio 운영 설정은 그대로 두었다.

## 1. 판정과 적용 범위

| 칸 | 판정 | 등급 | 측정으로 확인된 범위 |
|---|---|---|---|
| raw E4M3fn/E5M2 → FP16 upcast + rocBLAS | 기준선 유지 | A | 두 형식 각각 3 shape, 기존 Comfy 수동 cast와 같은 수학적 계약 |
| Triton software decode → rocBLAS | 재사용 가능한 정확 decoder, 가속은 shape별 | A | 각 형식 256개 bit pattern 검증, 6 GEMM native 출력 값 일치(torch.equal). wall 0.968–1.091배 |
| 1회 FP16 준비 + rocBLAS | 단위 이득, 기존 adapter 모델 전체는 미채택 | A | 단위 1.015–1.196배·6 shape 값 일치. E4M3fn Anima 30 step은 0.9858배·최종 RGB byte 일치, +약3.83GiB cache |
| FP8 tile decode + FP16 dot2 융합 | 이번 타일 폐기 | A | wall 0.282–0.432배. spill 0이어도 native보다 느림 |
| FP8→INT8 + 활성화 INT8 + dot4/dequant | projection 단위 후보, 모델에서는 폐기 | A | [8192,2048,2048]만 1.205–1.280배. Anima projection 한정 연결도 더 느리고 RGB strict 실패 |
| E5M2 최종 모델 A/B | 미검증 | A 단위 / 모델 ❔ | 3 shape 및 decoder 검증까지. E4M3fn 모델 결과를 E5M2 모델 결과로 옮겨 적지 않음 |

**이 문서의 FP8는 plain unscaled `float8_e4m3fn` 또는 `float8_e5m2` tensor다.** FNUZ, tensor/row/block scale, QuantizedTensor, MXFP8/NVFP4와 같은 표현은 이 계약에 포함되지 않는다. scale을 가진 checkpoint에 raw byte decoder만 붙이면 다른 연산이 된다.

기존 10개 diffusion/text/VAE safetensors header 조사에서 original FP8 tensor는 없었다([inventory](results/dtype_model_inventory_v06.json)). 최종 모델 A/B는 사용자가 이미 보유한 Anima BF16 checkpoint를 **소유한 연구 요청에서 raw E4M3fn으로 재표현**하여 진행했다. native와 후보는 동일한 FP8 값을 사용한다. **원래 BF16 checkpoint를 FP8로 바꾼 오차는 별도**로 측정한다.

## 2. 기준 커널·후보

gfx1030은 FP8 dot4 하드웨어 연산을 지원하지 않는다(C: [AMD floating dot builtins](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/docs-1.0.0/compiler-builtins/cross-arch/arithmetic-ref/dot-float-builtins.html)). FP8은 여기서 저장 표현이다.

실제 기준선은 전체 weight를 FP16로 cast하고 `F.linear`를 호출한다. projection의 native profiler는 `Cijk_Alik_Bljk_HHS_BH_MT128x256x16_...ISA1030...WS32...` rocBLAS kernel과 FP8→FP16 copy kernel을 기록했다. 전체 이름과 각 shape의 trace는 [연산 원자료](results/v06_fp8_gemm/results.json)에 있다(A).

후보는 다음과 같다.

- `decode`: byte의 sign/exponent/mantissa를 Triton에서 복원해 FP16 weight를 만든 뒤 기존 rocBLAS GEMM.
- `prepared_fp16`: 현재 weight의 FP16 복사본을 명시적으로 준비해 재사용. 준비 비용과 상주 memory를 별도 기록.
- `fused_dot2`: 각 weight tile을 register에서 decode하고 FP16 dot2, FP32 accumulate, FP16 출력. 타일64×64×32, warps4, stage1.
- `int8`: FP8 weight row와 FP16 activation row를 maxabs/127로 양자화하고 기존 dot4/dequant kernel 사용. 호출마다 weight 준비까지 포함.
- `prepared_int8`: weight INT8+row scale만 준비하여 재사용. 활성화 quantization과 dequant·출력 cast는 호출별 시간에 포함.

fused dot2의 실제 ISA는 `v_dot2c_f32_f16` 512개, registers190–191, spills0, LDS8192B였다. INT8 경로는 `v_dot4c_i32_i8` 2048개, registers203, spills0, LDS12288B였다(A). 정적 명령 수를 실행 FLOP 수로 보지 않는다. native FP8 산술이나 matrix instruction이 생긴 것은 아니다.

## 3. 두 dtype 각각 3 shape: 속도·오차·메모리

A는 FP16 [M,K], weight는 FP8 [N,K], output은 FP16 [M,N]이다. seed 20261108 난수 BF16 weight를 한 번 FP8로 표현했다. 아래 값은 **synthetic 입력**이며, 실제 모델 tensor를 capture한 단위 입력은 아니다. 차원은 기존 모델의 projection/MLP/encoder 규모를 대표하도록 선택했다.

| 이름 | M | N | K |
|---|---:|---:|---:|
| projection | 8192 | 2048 | 2048 |
| MLP | 8192 | 8192 | 2048 |
| encoder projection 유사 | 512 | 3072 | 1024 |

각 후보를 compile/검증하고 2회 warm한 뒤, 순서를 회전·역전하여 3 round×3 call을 측정했다. **wall은 호출별 변환·quantization·출력 cast를 포함**하며, prepared 모드의 명시적 1회 weight 준비는 아래 별도 표에 있다. finite scan/CPU reference/profiler는 단위 timing에 포함하지 않았다. 실제 INT8 모델 adapter에서는 활성화 finite scan 비용이 포함된다.

### 동기화 wall 중앙값, ms

| dtype | 방법 | projection | MLP | encoder |
|---|---|---:|---:|---:|
| E4M3fn | upcast 기준 | 2.766797 | 8.415002 | 0.164029 |
| E4M3fn | software decode | 2.793510 | 8.608528 | 0.156362 |
| E4M3fn | prepared FP16 | 2.725654 | 8.156854 | 0.137486 |
| E4M3fn | fused dot2 | 7.672668 | 29.738531 | 0.384187 |
| E4M3fn | INT8, weight 준비 포함 | 2.295050 | 9.547644 | 0.197339 |
| E4M3fn | prepared INT8 | 2.272633 | 9.510434 | 0.177185 |
| E5M2 | upcast 기준 | 2.925879 | 8.380413 | 0.167645 |
| E5M2 | software decode | 2.772773 | 8.654347 | 0.153659 |
| E5M2 | prepared FP16 | 2.865133 | 8.178327 | 0.140132 |
| E5M2 | fused dot2 | 7.701021 | 29.749238 | 0.388024 |
| E5M2 | INT8, weight 준비 포함 | 2.288427 | 9.568764 | 0.200892 |
| E5M2 | prepared INT8 | 2.285330 | 9.552554 | 0.177765 |

### GPU event 중앙값, ms — wall과 별도

| dtype | 방법 | projection | MLP | encoder |
|---|---|---:|---:|---:|
| E4M3fn | upcast | 2.760486 | 8.324013 | 0.158603 |
| E4M3fn | decode | 2.786883 | 8.518478 | 0.150823 |
| E4M3fn | prepared FP16 | 2.718823 | 8.065076 | 0.131933 |
| E4M3fn | fused dot2 | 7.582536 | 29.648397 | 0.378973 |
| E4M3fn | INT8 | 2.287587 | 9.538585 | 0.191603 |
| E4M3fn | prepared INT8 | 2.267093 | 9.502965 | 0.171783 |
| E5M2 | upcast | 2.917396 | 8.288932 | 0.162166 |
| E5M2 | decode | 2.766756 | 8.561778 | 0.148273 |
| E5M2 | prepared FP16 | 2.857100 | 8.085492 | 0.134670 |
| E5M2 | fused dot2 | 7.610809 | 29.657392 | 0.382657 |
| E5M2 | INT8 | 2.282180 | 9.560961 | 0.195520 |
| E5M2 | prepared INT8 | 2.279910 | 9.537005 | 0.172163 |

### 1회 준비 비용

각 셀은 **wall / GPU event ms**다. weight가 바뀌면 다시 준비해야 한다. 아래 한 matrix 준비 시간을 전체 모델에 단순 곱해 모델 준비 실측값이라고 부르지 않는다.

| dtype·준비 | projection | MLP | encoder |
|---|---:|---:|---:|
| E4M3fn→FP16 | 0.059033 / 0.043733 | 0.156132 / 0.145587 | 0.044583 / 0.034760 |
| E4M3fn→INT8+scale | 0.047033 / 0.041053 | 0.073636 / 0.068403 | 0.034546 / 0.029497 |
| E5M2→FP16 | 0.051976 / 0.042520 | 0.153845 / 0.143427 | 0.044503 / 0.034547 |
| E5M2→INT8+scale | 0.034273 / 0.029193 | 0.075396 / 0.069840 | 0.033640 / 0.028660 |

### 메모리

추가 peak는 호출 시작 대비 PyTorch allocated 증가량으로 출력도 포함한다. prepared weight의 상주 bytes는 시작 시 이미 존재하므로 **추가 peak와 별도**다. 두 dtype의 같은 shape는 아래 footprint가 같다.

| 방법 | 호출 추가 peak MiB: projection / MLP / encoder | 별도 prepared 상주 MiB |
|---|---:|---:|
| native upcast / software decode | 40 / 160 / 9 | 0 |
| prepared FP16 | 32 / 128 / 3 | 8 / 32 / 6 |
| fused dot2 | 32 / 128 / 3 | 0 |
| INT8 | 116.04 / 416.06 / 12.51 | 0 |
| prepared INT8 | 112.03 / 400.03 / 9.50 | 4.008 / 16.031 / 3.012 |

INT8는 activation quantization buffer와 FP32 dequant output 후 FP16 cast가 있어 이 구현의 peak가 더 크다. “INT8이므로 항상 memory도 절약된다”는 규칙은 성립하지 않았다(B+A).

### 정확도

실행 전 gate는 exact계열 normalized RMSE≤0.002 / normalized max≤0.02, INT8계열≤0.03 / ≤0.15였다. 분모는 reference output RMS다. 모든 단위 후보는 finite이고 각 gate를 통과했다. CPU FP64에서 24×24 선택 출력 좌표를 전체 K에 대해 독립 계산했다. 이는 모든 output coordinate의 FP64 검증이 아니다.

- 각 dtype의 **256 byte pattern 전부**를 decoder에 넣었다. finite 값은 FP16 bit-exact이고 signed zero도 보존했다. NaN·±Inf 분류도 native와 같았다. NaN payload bit identity를 검증한 것은 아니다.
- software decode와 prepared FP16은 **6 GEMM 모두 native output 값 일치(torch.equal)**. 단위 harness의 `bit_equal` 이름은 이 값 비교를 뜻하며 signed zero 차이는 미구분한다. 위 256-pattern decoder 검증과 아래 모델 byte digest 검증은 실제 bit/byte 비교다.
- fused dot2는 두 projection에서 값 일치(torch.equal), 나머지는 max≤0.0009765625, RMSE≤3.01e-6.
- INT8은 normalized RMSE 0.010959–0.011809, max abs0.014161–0.018311. 이 단위 gate 통과가 최종 이미지 strict 통과를 보장하지는 않는다.

fused dot2의 실패는 spill로 설명할 수 없다(spills0). native가 더 큰 타일을 쓰는 반면 이 후보는 작은 타일에서 매번 bit decode를 반복하고 VGPR190개 안팎을 사용한다. 이 차이가 성능에 불리하다는 것은 코드·자원 지표에 근거한 해석이며, 비용을 개별 원인으로 분해한 profiler 실험은 아니다.

## 4. 실제 ComfyUI 연결에서 드러난 dtype 경계

첫 [2 step 연결](results/v06_anima_fp8_e4m3fn_2step/runs.json)은 hit0이라 실패로 남겼다. `module.weight.dtype`만 읽으면 CPU mmap의 원본 BF16이 나온다. 실제 DynamicVRAM은 `weight._model_dtype`를 사용한 TensorGeometry로 resident FP8을 준비한다(B).

설치 소스:
- `comfy/memory_management.py:tensors_to_geometries`: _model_dtype 우선.
- `comfy/model_patcher.py`: weight_comfy_model_dtype를 _model_dtype에 전달.
- `comfy/ops.py:resolve_cast_module_with_vbar`: resident `_v_weight`를 compute dtype으로 post-cast.
- `CastBiasWeightContext`: 기존 pin/unpin/stream lifetime 유지.

수정 adapter는 geometry dtype을 읽고 **원본 필드를 바꾸지 않은 채 기존 context에서 raw resident FP8**을 받는다. 30 step 후보 13,742회 모두 반환 weight가 `_v_weight`와 dtype·shape·stride·data_ptr가 동일했다. CPU BF16에서 별도로 새 양자화한 weight를 쓰는 실험이 아니다(A).

shape·dtype·반환 storage 확인은 `runs.json.candidate.actual_weight_contracts`, `context_weight_is_resident_storage`에 있다. 최초 no-hit의 비트일치는 성공으로 세지 않는다.

## 5. 실제 모델 최종 출력 A/B

기존 Anima, 1024², seed20261107, 동일 conditioning, Euler/simple, CFG4. 모델 weight는 연구 요청에서 raw E4M3fn, activation은 FP16이다. 최종 VAE 출력 기준은 max≤2/255, RMSE≤0.25/255로 미리 정했다.

### 5.1 projection INT8: 2 step에서 실패

[원자료](results/v06_anima_fp8_e4m3fn_geometry_2step/runs.json).

연산에서 이겼던 **M8192,N2048,K2048만** INT8로 치환했다. MLP/encoder의 느린 shape나 미측정 shape는 fallback했다. 336회 실제 후보가 실행됐고 168개 weight를 준비, 168회 재사용했다. cache706,019,328B. activation finite 검사도 모델 시간에 포함했다.

| 경로 | denoise wall | RGB max | RGB RMSE |
|---|---:|---:|---:|
| FP8 native control | 5.3865 / 5.4055s | 0 | 0 |
| projection prepared INT8 | 6.8429s | 0.65698242 | 0.02565801 |

정밀도와 속도 모두 실패했다. INT8 재양자화·activation 양자화·다른 GEMM 누산/bias 경로를 포함한 후보 전체가 실패한 결과다. 2 step에서도 strict 실패라 30 step 반복은 하지 않았다. 단위 배속만으로 “dot4면 빠르다”라고 일반화하지 않는다.

### 5.2 요청 캐시 + 매번 staging context: 30 step에서 무이득

[30 step 원자료](results/v06_anima_fp8_e4m3fn_30step/runs.json), [GPU 감시](results/v06_anima_fp8_e4m3fn_30step/gpu_audit.json).

첫 warm80.1979s는 비교 표에서 제외했다. 이후 native/candidate 두 쌍을 교대로 실행했다. 매 요청마다 cache를 새로 만들고, 준비·copy·실제 denoising을 포함한 stage wall을 측정했다. 모델 준비 시간을 독립 clock으로 분리한 측정은 없으며, **준비 포함 전체 시간**이 아래 값이다.

| 지표 | native control | prepared FP16 |
|---|---:|---:|
| denoise trial1 / trial2 | 79.602509 / 79.464816s | 80.585307 / 80.777661s |
| denoise 중앙값 | 79.533663s | 80.681484s |
| denoise+VAE stage 중앙값 | 80.084800s | 81.240406s |
| request wall 중앙값 | 80.479492s | 81.616071s |
| 요청당 eligible/candidate | 13,742 / 0 | 13,742 / 13,742 |
| 요청당 준비 / 재사용 | 0 / 0 | 515 / 13,227 |
| explicit cache | 0 | 4,116,201,472B (약3.83GiB) |
| PyTorch 요청 peak | 3,045,647,872B | 7,200,400,896B |
| 최종 latent / RGB | 기준 | 모든 요청 비트일치 |

배속 **0.9858배**, 약1.44% 느려졌다. PyTorch peak에는 external DynamicVRAM allocator가 빠져 있으므로 GPU 전체 resident memory로 읽으면 안 된다. cache는 요청 종료 시 해제했고 모델 원본·runtime 설정은 바꾸지 않았다.

2 step에서는 준비515/재사용515라 상각이 부족할 수 있었지만, 30 step에서 재사용13,227까지 늘려도 이 adapter는 이기지 못했다. 따라서 단위 prepared FP16의 1.015–1.196배를 실사용 배속으로 채택하지 않는다.

### 5.3 owned weight+bias, cache hit staging 제거

[원자료](results/v06_anima_fp8_owned_2step/runs.json), [감시](results/v06_anima_fp8_owned_2step/gpu_audit.json).

이전 adapter는 cache hit에서도 사용하지 않을 raw FP8을 얻기 위해 staging context에 재진입했다. 이를 제거하는 bounded 후보를 별도 파일에 구현했다. 첫 준비에서 기존 resident FP8을 통해 FP16 weight를 만들고, bias는 `clone()`하여 독립 저장소로 소유했다. 이후 hit는 owned weight+bias만 써서 `F.linear`를 호출한다.

cache key는 module identity, 원본 weight와 bias 각각의 identity/data_ptr/version/shape/stride/dtype/device, geometry dtype, 입력 dtype/device를 포함한다. 수명은 한 denoising 요청이며 종료 시 해제한다. 기존 immutable Anima에 한정하고 LoRA·weight patch·학습·모델 mutation으로 일반화하지 않는다. 외부 Comfy prefetch ring은 그대로 작동하고 기존 `cleanup_prefetched_modules`가 pin을 정리한다. **context 재진입은 제거했지만 외부 prefetch 자체를 없앤 실험은 아니다.**

추가 측정 전에 **최종 byte 일치와 최소 5% 속도 이득**을 30 step 재검증의 진행 기준으로 정했다. 2 step 교대 screen 한 번만 수행했다.

| 지표 | native control | owned FP16 cache |
|---|---:|---:|
| denoise trial1 / trial2 | 5.415627 / 5.388642s | 5.382311 / 5.453196s |
| denoise 중앙값 | 5.402135s | 5.417754s |
| 배속 | 1× | 0.9971× |
| 실제 candidate call / 요청 | 0 | 1,030 |
| 준비 / 재사용 | 0 / 0 | 515 / 515 |
| bias 독립 copy | 0 | 13 |
| explicit cache | 0 | 4,116,264,960B |
| PyTorch 요청 peak | 3,045,647,872B | 7,200,464,384B |
| 최종 latent / RGB | 기준 | 모든 요청 byte 일치 |

515개 준비 모두 context 반환 weight가 native resident와 동일 storage였다. cache hit에서는 context에 들어가지 않았고, fallback0이었다. audit155 samples, foreign/error 없음, valid.

**정확도는 통과했지만 사전 5% 이득 기준을 못 넘었다.** 이 후보도 미채택으로 남기고 30 step 추가 실행을 하지 않았다. 앞 30 step 결과는 staging context를 유지한 이전 후보의 결과이며, 이 개선 후보의 30 step 수치로 인용하지 않는다.


### 5.4 FP8 표현 자체의 오차 — 위 후보 A/B와 별도

[CPU 비교 원자료](results/v06_anima_fp8_e4m3fn_30step/representation_vs_original.json). 이미 저장된 정상 Anima reference와 FP8 native reference의 seed20261107·30 step·conditioning hash를 확인하고 CPU float64 차이를 계산했다. GPU는 초기화하지 않았다.

| 비교 | RGB max | RGB RMSE | strict |
|---|---:|---:|---|
| 원본 BF16 checkpoint, 정상 FP16 compute → raw E4M3fn storage, FP16 compute | 0.94775391 | 0.17677346 | 실패 |
| 같은 FP8 storage의 native → prepared FP16 후보 | 0 | 0 | 통과 |

원본 RGB SHA는 `52ea1628...853cb3a`, FP8 native RGB는 `3915987e...8f27f`다. 전체 hash는 원자료에 있다.

기존 Anima 실제 세 weight를 CPU에서 조사한 [별도 원자료](results/anima_fp8_weight_conversion_cpu_v06.json)는 다음과 같다.

| 실제 weight | RMS | raw E4M3fn normalized RMSE | raw E5M2 normalized RMSE |
|---|---:|---:|---:|
| cross_attn.output_proj [2048,2048] | 0.0068712 | 0.082421 | 0.053017 |
| cross_attn.k_proj [2048,1024] | 0.0061126 | 0.092402 | 0.052796 |
| adaln_modulation_cross_attn.1 [256,2048] | 0.0154631 | 0.038435 | 0.052422 |

스케일 없는 E4M3fn은 |x|<0.015625 구간이 subnormal이고 간격은 0.001953125다. RMS가 약0.006인 weight에는 이 간격이 거칠다. weight 변환 손실이 이미 관측됐으며, 이 원인을 확인하지 않고 RGB 차이를 compute kernel 탓으로 돌리면 안 된다. scaling된 FP8 표현의 재설계나 모델별 calibration은 이번 연구 범위에서 구현·채택하지 않았다.

## 6. 새로운 모델에서 쓰는 규칙

1. dtype 이름 외에 **raw/scale/FNUZ/QuantizedTensor 계약**을 확인한다. 이 구현은 plain unscaled E4M3fn/E5M2만 받는다.
2. checkpoint·CPU mmap·resident·compute dtype을 각각 기록한다. DynamicVRAM이면 geometry와 실제 반환 weight를 확인한다.
3. 최적화 A/B는 **동일한 FP8 값**을 기준으로 한다. 원본 BF16/FP16→FP8 표현 변경은 별도 화질 실험이다.
4. decode+rocBLAS, weight 준비 후 rocBLAS, INT8 경로를 같은 M/N/K와 stride에서 비교한다. weight 준비·activation quantization·출력 cast를 시간에서 빼지 않는다.
5. prepared 복사본은 weight/scale/LoRA/device 변경 시 폐기한다. 이번 request-local cache는 immutable 모델에서만 유효하고 운영 전역 pointer cache가 아니다.
6. 전체 모델에서 cache의 추가 VRAM과 offload/staging 비용을 확인한다. 단위 준비 생략 배속을 모델 성능으로 쓰지 않는다.
7. 근사 INT8는 operator gate와 최종 모델 출력 gate를 모두 통과해야 한다. 실패한 shape·모델은 fallback/미채택으로 기록한다.
8. E5M2는 여기서 decoder 및 3 shape A만 확보했다. E4M3fn 모델 output을 E5M2 검증으로 대체하지 않는다.

## 7. 재사용 파일·재현·증거

- [FP8 decode/GEMM 후보](rdna2_ops/fp8_weight_gemm_v06.py)
- [단위 하네스](bench_dtype_gemm_v06.py)
- [기존 Comfy staging 보존 adapter](dtype_switches_v06.py)
- [owned weight+bias 연구 adapter](fp8_owned_switches_v06.py)
- [E4M3fn request spec](research_configs/anima_fp8_e4m3fn_v06.json), [E5M2 spec](research_configs/anima_fp8_e5m2_v06.json), [owned cache spec](research_configs/anima_fp8_owned_v06.json)

기존 wrapper/venv에서 `bench_dtype_gemm_v06.py --kind fp8 --output 새결과디렉터리`를 실행한다. 모델 재현은 같은 venv에서 `research_comfy_v05.py`에 해당 연구 spec과 새 output 경로, seed20261107, steps2 또는30을 전달한다. 기존 workflow 파일은 고치지 않는다.

| 실험 | durable job ID |
|---|---|
| FP8 두 dtype ×3 shape | 38479086-c651-42fd-9ed6-b582522dd3dd |
| raw parameter dtype만 본 no-hit | aac83811-59a2-495f-b6ef-a5bb2e0c7527 |
| geometry/resident 수정 2 step | d833ff8b-24e4-46aa-9dbd-c802ca5c1fd9 |
| E4M3fn 30 step 두 쌍 | 06e609a7-96dc-4b80-9acc-d66e3f5d0f0a |
| 원본 대비 표현 오차, CPU only | 6f77abf3-48a8-45a0-bd7b-63dcba825c1e |
| owned cache 2 step | 6c128172-b488-47f9-85c2-d70715aed891 |

30 step audit는 1,645 samples, foreign clients/error 없음, valid다. 단위 GPU audit는184 samples, valid. 소유 GPU job은 하나씩 실행했고 CodexPro2 연결 장애 중에도 기존 durable job을 새로 띄우지 않았다. KFD 0.25초 polling의 한계는 [BF16 보고서](BF16_GEMM_V06_KO.md)와 같다.

| 소스 | SHA-256 |
|---|---|
| fp8_weight_gemm_v06.py | 6305048cda4e2dd61c1512d775cef700784de1aaf9831cae7d3eef1c12597223 |
| bench_dtype_gemm_v06.py | 3da14d6368e938d364702c5da27670256ceae2d4162d501b3398c5ba1c2b733f |
| dtype_switches_v06.py, 30 step | bb0a9fde7506c560b6a7187288adace6d8f9e9b8025c0826a07572aac6972f07 |
| fp8_owned_switches_v06.py | 5ac9b07ac4c2bf54249295ca4aea8acae25386251bf80a37680ac2e80c5da9e9 |
| comfy/ops.py | fa87356b1dd5aa8ecc5b1e71d5bcd43f5cd7e9eba4a469f72e4f099b00e6cb39 |
| comfy/memory_management.py | 8e8d5e961b78f3d8b6edac61405eacddc67ec4bfdbd2b9778d6d20c911ea5b6f |
| comfy/model_patcher.py | 88fe06e7dcc53a7a79df7be13044513e556927ecc0fa1fd8300803f75ce5e08b |

