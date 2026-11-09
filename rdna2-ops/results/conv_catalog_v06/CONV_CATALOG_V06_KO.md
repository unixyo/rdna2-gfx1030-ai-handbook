# Conv 카탈로그 v0.6 — gfx1030 실측과 재사용 레시피

측정 장치: 이 PC의 Radeon RX 6900 XT. 측정일: 2026-11-06 UTC(PC 로컬 로그는 2026-11-07). 운영 Applio/MOSS/Qwen 설정·학습 데이터·체크포인트를 수정하지 않았다. 새 환경·모델을 설치하거나 내려받지 않았다.

## 1. 판정부터

| 연산 칸 | 판정 | 연산 A | 실제 모델 A | 남은 범위 |
|---|---|---|---|---|
| BF16 Conv2D → FP32 계산 → BF16 복귀 | **성공, 검증 범위에서 opt-in** | MOSS 실제 대표 3 shape에서 3.10–3.59×, 변환 비용 포함 | MOSS BF16 VAE regular/tiled 및 실제 lowvram에서 2.30–2.63×, 엄격 RGB 통과 | 다른 VAE·latent·해상도는 다시 검증 |
| FP16 Conv1D dilation1 NORMAL seed → FAST replay | **무필터 후보 실패** | 18 shape 중 4개 forward 큰 오차; 분리 반복으로 재현 | 위험 후보는 학습에 적용하지 않음 | 낮은 수준의 kernel bug 메커니즘 미확정 |
| FP16 Conv1D 검증된 FindDb 필터 + FAST | **성능 성공, 학습 동등성 미확정** | d1 12 shape, y/dX/dW 통과, 합계 1.811× | 실제 Applio 4 epoch, steady 1.106× | native 반복도 파형 gate 실패; 장기·청각 품질 판정 미검증 |
| dilated Conv1D blanket fold | **일괄 적용 실패** | 18 shape 합계 1.004×; 개별 0.53–1.80× | 전체 fold epoch는 무이득으로 생략 | shape별 분리 필요 |
| C64/T6400/k7·11/d3·5만 selective fold | **조건부 성능 성공** | 네 winning shape 1.59–1.80× | filtered DB 대비 추가 1.013×, 각 shape 64 hit | 작은 추가 개선은 짧은 1회 A/B; 학습 품질 미확정 |
| FP16 period Conv2D NORMAL | **2/3 shape 정확도 실패** | FAST와 FP32 참조 대비 forward NRMSE 0.77–0.78 | 운영 적용 안 함 | 통과한 중간 shape만 일반화하지 않음 |
| grouped Conv1D NORMAL | **속도 이득 없음** | 3 shape, 0.96–1.00×, y/dX/dW 통과 | 최적화 후보의 모델 승격 안 함 | native 유지 |
| depthwise Conv1D NORMAL | **연산 성공 / 모델 ❔** | 3 shape, 2.03–5.61×, y/dX/dW 통과 | 이 PC 모델의 해당 경로 미연결 | 합성 유사 shape의 결과만 |
| ConvTranspose1D 검증된 FindDb 추가 | **연산 성공 / 모델 속도 무이득** | 실제 Applio 3 shape, FAST replay 1.09–1.30×, y/dX/dW 통과 | filtered 기준 8.41758→8.43412s, **0.99804×**, 각 shape 64 hit | 추가 DB 최적화는 채택하지 않음; 학습 품질 미확정 |
| 일반 Conv3D NORMAL | **속도 이득 작음 / 모델 ❔** | 합성 3 shape, 1.00–1.06× | 기존 모델 경로 미연결 | CK가 있었다면 얻을 성능은 미측정 |
| Qwen FP32 exact-tile Conv3D → Linear | **연산 성공 / 모델 판정 불가** | FP32 P128/512/2400, 1.98–3.34×, FP64 참조 통과 | 강제 재인코딩 native 반복이 RGB gate 실패, 후보 실행 전 중단 | 후보의 모델 품질·모델 배속 미측정 |

A는 이 PC 실측, B는 설치/해당 버전 소스, C는 공식 문서다. 연산 단위 gate를 통과해도 모델 A/B가 없으면 모델 검증 완료로 승격하지 않는다.

## 2. 공통 측정 계약

`../../bench_conv_catalog_v06.py`는 고정 CPU seed 63117, FP16로 반올림한 동일 input/weight/grad-output, `autograd.grad`를 쓴다. 반복마다 `.grad`를 누적하지 않는다. 일반 Conv 표 시간은 forward+dX+dW 전체, 3 warmup 후 3 round×7 반복 중앙값이다. GPU event와 동기화 wall을 모두 원자료에 남겼으며 표는 wall 중앙값이다. 추가 allocated peak는 별도로 기록한다.

FP16의 독립 참조는 동일하게 반올림된 입력을 FP32 native convolution으로 계산한다. gate는 y/dX/dW 모두 finite, NRMSE≤0.002, max error/reference RMS≤0.02다. FP32 exact-tile 비교는 CPU FP64 동치 Linear 참조를 사용한다. true byte equality는 dtype/shape 및 contiguous uint8 bytes로 검사하고 수치 `torch.equal`과 분리한다.

`gpu_guard_v06.GpuAudit`의 research flock 및 0.25초 KFD PID 감시를 사용했다. 성공 표의 모든 GPU job은 audit valid이며 외부 compute client가 관측되지 않았다. 그래픽 부하나 polling 사이의 짧은 client를 완벽히 검출하는 장치는 아니다. 실패·중단 job은 별도로 보존하며 성공 timing 분모에 넣지 않는다.

GPU 프로파일의 분모는 Chrome trace actual kernel/copy/memset 합계다. CPU operator attribution, annotation range, 같은 일을 가리키는 kernel 합계를 서로 더하지 않는다.

- Applio: 기존 `.venv-rocm10/bin/python`, Torch 2.11.0+rocm10.0.0, HIP 7.15.26333, MIOpen DB key 3_6_0_8d1ae90e.
- Qwen: 기존 `.venv/bin/python`, Torch 2.12.0+rocm10.2.0a20261011, HIP 7.17.26405, MIOpen DB key 3_6_2_3d8f334b.
- MOSS VAE: 기존 `.venv-moss/bin/python`와 기존 `scripts/with_local_storage.sh`; VAE의 dtype/weight casting/offload 정책은 기존 Comfy 호출자가 유지한다.

## 3. FindMode와 DB — 환경변수를 Conv마다 바꾸면 안 되는 이유

소스 B: MIOpen commit `8d1ae90eff7d022f26019ec55b2ec6a7674b3112`의 `find_controls.cpp`에서 `GetFindModeValue`는 function-template의 `static const` 값으로 환경변수를 최초 한 번 읽는다. process 내 Conv 호출마다 FAST/NORMAL 환경변수를 교체하는 방법을 사용하지 않는다. 해당 파일·`find_controls.hpp`·`env.hpp`는 `source/`에 보존했다.

설치 PyTorch `ATen/miopen/Descriptors.h`의 `ConvolutionDescriptor::set`은 `benchmark=True`일 때 descriptor를 NORMAL로 설정한다. 이 실험에서는 `benchmark=False`를 고정했다.

공식 문서 C: [FindDb](https://rocm.docs.amd.com/projects/MIOpen/en/latest/conceptual/finddb.html)와 [Find modes](https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html). User FindDb는 shape/dtype/방향별 solver 선택 결과다. PerfDb는 개별 solver tuning parameter다. 둘을 같은 DB라고 부르지 않는다. 공식 최신 문서가 설치 버전의 모든 세부를 보장하지 않으므로 소스·실측을 함께 남겼다.

**레시피:** 새 연구 DB에 관심 있는 shape만 NORMAL로 seed하고, 새 FAST process에서 그 DB를 읽어 전체 workload를 실행한다. 다른 shape는 FAST miss fallback으로 남긴다. GPU/라이브러리/shape/dtype/방향이 달라지면 새로 검증한다.

기존 `../../../conv_miopen_20261107/`의 NORMAL dilated 로그에는 workspace pointer/size 0인 GEMM 후보 탈락 경고가 있다. 그 결과를 깨끗한 최적 solver 간 비교로 해석하지 않는다. 기존 속도만으로 'dilation1이면 NORMAL'을 권장하면 안 된다. 이번 정확도 검사에서 실제 실패를 발견했다.

## 4. Conv1D: 무필터 FindDb 실패와 검증된 필터

기준: FAST의 im2col + Tensile HHS GEMM. 후보: 실제 HiFi-GAN d1 12 shape만 NORMAL seed 후 FAST replay. seed DB는 36개 F/B/W key, 12,630 bytes, SHA256 `91504c1d6458a342926180573a8978b68a40841f9dfd2c1676d47ba611ef2833`다.

18 shape replay 중 아래 네 shape에서 forward만 크게 틀렸다. dX/dW는 gate를 통과했다. 같은 C64/k7/d1을 별도 process에서 반복했을 때 max와 NRMSE가 동일해서 일시 현상으로 보기는 어렵다. 전체 profile에 Winograd backward kernel이 같이 등장한다고 Winograd forward가 원인이라고 단정하지 않았다.


| 원자료 행 index | shape ID | 오류 후보 ms | forward max | forward NRMSE |
|---|---|---|---|---|
| 0 | res_C64_T6400_k7_d1 | 1.9644 | 2.246003 | 0.384542 |
| 3 | res_C64_T6400_k11_d1 | 2.1753 | 1.705957 | 0.297546 |
| 9 | res_C128_T3200_k11_d1 | 2.3041 | 2.603542 | 0.517427 |
| 12 | res_C64_T6400_k3_d1 | 1.3584 | 3.561350 | 0.589294 |


원자료: `resblock_hybrid/result.json`, `resblock_hybrid_repeat/result.json`. FAST baseline의 18 shape는 모두 FP32 참조 gate를 통과했다. dilated 12 shape의 무필터 replay는 FAST와 y/dX/dW byte exact였다.

F 방향 DB의 `ConvHipImplicitGemmV4R1Fwd` 선택 경로가 문제와 연결됐다. **새 DB 사본에서 이 entry만 제거**하자 아래 12 shape 모두 통과했다. B/W 선택은 보존했다. 이는 해당 선택 경로의 인과 근거이며, 커널 내부 어느 instruction이 잘못됐는지까지 확정한 것은 아니다. 다른 shape와 역방향까지 solver 전체를 금지하는 규칙으로 확대하지 않는다.

`../../filter_miopen_finddb_v06.py`는 원본 SHA를 확인하고 기존 출력 폴더를 덮어쓰지 않는다. 대상은 연구 source DB의 NCHW FP16 1D forward key이며 제거 key/entry를 `hybrid_cache/filtered_db/filter_manifest.json`에 남긴다. 원본 DB는 그대로다. filtered DB SHA256: `d81ef3d8330bce40526dc885c744b8914de49d679cd33fd2493ef46ce1da1bbf`.


| B8 FP16 d1 shape | FAST ms | filtered FAST ms | 배속 | y/dX/dW 최대 NRMSE | gate |
|---|---|---|---|---|---|
| res_C256_T320_k3_d1 | 0.7922 | 0.3175 | 2.495× | 0.000460 | 통과 |
| res_C256_T320_k7_d1 | 0.8616 | 0.7542 | 1.142× | 0.000459 | 통과 |
| res_C256_T320_k11_d1 | 1.1949 | 0.9618 | 1.242× | 0.000459 | 통과 |
| res_C128_T3200_k3_d1 | 1.7770 | 1.1454 | 1.551× | 0.000459 | 통과 |
| res_C128_T3200_k7_d1 | 2.3482 | 1.7263 | 1.360× | 0.000460 | 통과 |
| res_C128_T3200_k11_d1 | 2.8748 | 2.3433 | 1.227× | 0.000451 | 통과 |
| res_C64_T6400_k3_d1 | 3.4154 | 1.3709 | 2.491× | 0.000459 | 통과 |
| res_C64_T6400_k7_d1 | 5.2140 | 2.0040 | 2.602× | 0.000457 | 통과 |
| res_C64_T6400_k11_d1 | 5.5512 | 2.2043 | 2.518× | 0.000461 | 통과 |
| res_C32_T12800_k3_d1 | 2.7119 | 1.2823 | 2.115× | 0.000459 | 통과 |
| res_C32_T12800_k7_d1 | 2.4149 | 1.5755 | 1.533× | 0.000413 | 통과 |
| res_C32_T12800_k11_d1 | 2.5835 | 1.8440 | 1.401× | 0.000453 | 통과 |


12개 합계 31.7396→17.5295ms. 합계는 동일 횟수의 연산 목록에 대한 요약이며 실제 모델 speedup은 아래 epoch로 별도 확인했다. 원자료: `all_d1_fast/result.json`, `all_d1_filtered/result.json`.


## 5. dilated fold: 전부 켜지 말고 실측한 shape를 구분

기준 native FAST, 후보는 dilation phase를 batch로 접어 d1 Conv를 수행하고 되돌리는 기존 Applio FoldConv1d와 같은 연산이다. padding/reshape/copy 및 backward 비용을 모두 포함했다. 18 shape 모두 y/dX/dW gate는 통과했지만 전체 합계는 69.7304→69.4557ms(1.004×)로 이득이 거의 없었다.


| B8 FP16 shape | native ms | fold ms | 배속 | y/dX/dW |
|---|---|---|---|---|
| res_C64_T6400_k7_d3 | 5.3884 | 2.9961 | 1.798× | 통과 |
| res_C64_T6400_k7_d5 | 5.4745 | 3.1659 | 1.729× | 통과 |
| res_C64_T6400_k11_d3 | 5.7770 | 3.5737 | 1.617× | 통과 |
| res_C64_T6400_k11_d5 | 5.9847 | 3.7577 | 1.593× | 통과 |
| res_C32_T12800_k3_d3 | 2.7349 | 4.4998 | 0.608× | 통과 |
| res_C32_T12800_k3_d5 | 2.7527 | 3.4918 | 0.788× | 통과 |
| res_C128_T3200_k11_d3 | 3.0650 | 4.3043 | 0.712× | 통과 |
| res_C128_T3200_k11_d5 | 3.3958 | 4.5789 | 0.742× | 통과 |
| res_C64_T6400_k3_d3 | 3.4425 | 3.1585 | 1.090× | 통과 |
| res_C64_T6400_k3_d5 | 3.5129 | 2.8552 | 1.230× | 통과 |
| res_C32_T12800_k11_d3 | 2.7872 | 5.2454 | 0.531× | 통과 |
| res_C32_T12800_k11_d5 | 3.0760 | 5.5722 | 0.552× | 통과 |


재사용 규칙은 우선 `B8,C64,T6400`, `OC64,k7 또는11,d3 또는5,stride1,groups1,zero padding=d*(k-1)/2`의 네 winning shape로 제한한다. `applio_epoch_v06.py --selective-fold`는 연구 process의 Conv1d `_conv_forward` 경계에서만 기존 FoldConv1d 구현을 호출한다. state_dict, weight_norm, autocast와 optimizer를 유지하고 종료 시 복원한다. 나머지 shape는 native다. 실제 학습에서 input FP16/weight FP32 경계가 각 64회, 총256회 선택됐다.

원자료: `resblock_fold/result.json`, `epoch_filtered_selective_fold/result.json`. 전역 `APPLIO_DILATION_FOLD=1`로 전체를 교체하는 방식은 이 결과의 채택안이 아니다.

## 6. Applio 실제 학습 epoch A/B와 파형 기준의 한계

기존 `rvc/train/train.py`의 실제 generator/discriminator/AdamW/AMP/DataLoader/손실을 재사용했다. `lab/train_bench.sh`는 운영 config를 쓰므로 실행하지 않았다. 기존 bench40k 111조각, batch8, 16step/epoch, 4epoch, seed1234, 기존 f0G40k/f0D40k pretrained다. 첫 epoch를 제외한 나머지 3개 중앙값이다. 출력·config 사본만 연구 경로로 redirect하고 원본 main/cleanup·checkpoint export를 호출하지 않았다.

모든 run의 초기 model state SHA와 최초 reference audio bytes가 일치했다. 첫 GradScaler65536은 원래 설정 그대로이며 초기 scale 감소와 nonfinite gradient를 기록했다. 이를 nonfinite loss와 혼동하지 않는다. 모든 run의 loss는 finite였다. native A의 초판 JSON에 있던 nonfinite gradient literal만 null+명시 문자열로 정상화했고 원본을 `result_original_with_nonfinite_literals.json`에 보존했다. 학습값/시간/출력은 변경하지 않았다.


| run | steady epoch s | FAST A 대비 | median D loss 변화 | median G loss 변화 | 최종 wave max | 최종 wave RMSE |
|---|---|---|---|---|---|---|
| FAST A | 9.30998 | 1.0000× | +0.000% | +0.000% | 0.000000 | 0.000000 |
| FAST B 반복 | 9.30841 | 1.0002× | -0.393% | -0.596% | 0.721924 | 0.134399 |
| filtered FAST | 8.41758 | 1.1060× | +0.311% | -0.952% | 0.721588 | 0.106584 |
| filtered+선택 fold | 8.30662 | 1.1208× | +0.350% | +0.445% | 0.538956 | 0.090849 |
| filtered+Transpose DB | 8.43412 | 1.1038× | +0.850% | -0.197% | 0.711426 | 0.061977 |


비교 대상은 FAST B/filtered는 FAST A, 선택 fold와 Transpose DB는 filtered다. 따라서 마지막 두 줄의 loss·파형 변화와 FAST A 기준 배속의 분모가 다르다. 선택 fold의 직접 추가 배속은 **1.01336×**, Transpose DB의 직접 추가 배속은 **0.99804×**다. 후자는 실제 모델에서 속도 이득이 없어 추가 적용을 채택하지 않는다.

사전 임시 audio gate는 max0.001/RMSE0.0001 full-scale와 median loss 변화2%였다. 사용자 합의의 지각 품질 기준이 아니다. **native A/B 자체가 max0.72192, RMSE0.13440으로 wave gate를 실패**했다. filtered 후보의 FAST A 대비 RMSE0.10658보다 native 반복 차이가 더 크므로 이 실험으로 청각 열화 또는 장기 학습 동등성을 판정할 수 없다. 최종 wave gate 실패와 성능 개선을 분리한다. 짧은4epoch의 loss 중앙값이 비슷하다는 것만으로 수렴·품질 동등성을 주장하지 않는다.

reference는 기존 train.py가 첫 DataLoader batch에서 택한 입력이고, eval의 RNG는 저장/복구했다. 초기 eval audio SHA는 모두 `61a882bdb48da6fb5f590d8e852f10ee43986bf7a97c4092a3e97cc37b2c7b81`이다. 선택 fold run부터 reference 다섯 tensor의 dtype/shape/byte SHA도 저장했다. 선택 fold와 Transpose DB run 각각 처음/끝이 일치하며 두 run 사이에도 동일하다. 앞선 세 run에는 입력 tensor digest를 직접 남기지 않았다는 한계는 보존한다. 초기 modelstate와 eval 출력 byte 일치가 이미 관측된 근거다.

운영 assets/config, bench config/filelist/model_info, train.py 및 관련 모듈·launcher, pretrained G/D의 before/after SHA는 모든 run에서 같았다. 각 run의 `protected_before.json`/`protected_after.json`과 result의 `protected_unchanged=true`에 남겼다.

## 7. Conv family survey와 CK 공백

같은 Applio Torch2.11 runtime에서 FAST→NORMAL을 비교했다. Conv2D/grouped/Transpose는 실제 Applio 모델의 shape, depthwise/일반 Conv3D는 그 범위를 바탕으로 만든 제한된 합성 shape다. 이 표 시간은 **forward+dX+dW 전체**다.


| FP16 shape ID | FAST ms | NORMAL ms | 배속 | forward NRMSE | y/dX/dW gate |
|---|---|---|---|---|---|
| period_C32_O128_H2134_P2 | 1.0569 | 0.8913 | 1.186× | 0.780137 | forward 실패 |
| period_C128_O512_H712_P2 | 1.3307 | 0.9215 | 1.444× | 0.000208 | 통과 |
| period_C512_O1024_H238_P2 | 2.0490 | 1.5557 | 1.317× | 0.773482 | forward 실패 |
| grouped_C16_O64_T12800_G4 | 1.4615 | 1.5241 | 0.959× | 0.000208 | 통과 |
| grouped_C64_O256_T3200_G16 | 0.9633 | 0.9618 | 1.002× | 0.000208 | 통과 |
| grouped_C256_O1024_T800_G64 | 0.9389 | 0.9371 | 1.002× | 0.000208 | 통과 |
| depthwise_C32_T12800_k3 | 5.8533 | 1.0442 | 5.605× | 0.000207 | 통과 |
| depthwise_C64_T6400_k7 | 2.8599 | 1.4078 | 2.031× | 0.000207 | 통과 |
| depthwise_C128_T3200_k11 | 4.3183 | 1.5062 | 2.867× | 0.000207 | 통과 |
| transpose_C512_O256_T32_k16 | 0.7442 | 0.6848 | 1.087× | 0.000284 | 통과 |
| transpose_C256_O128_T320_k16 | 1.2088 | 0.9214 | 1.312× | 0.000285 | 통과 |
| transpose_C128_O64_T3200_k4 | 1.2797 | 1.0297 | 1.243× | 0.000306 | 통과 |
| conv3d_C16_O32_D1_H32_W32 | 0.2187 | 0.2068 | 1.058× | 0.000208 | 통과 |
| conv3d_C32_O32_D4_H32_W32 | 0.3670 | 0.3680 | 0.997× | 0.000208 | 통과 |
| conv3d_C64_O32_D8_H16_W16 | 0.3957 | 0.3897 | 1.015× | 0.000208 | 통과 |


FAST는 주로 `Im2d2Col_v2`/`Im3d2Col` + Tensile `Cijk_*_HHS_*` + col2im을 썼다. NORMAL은 일부 shape에서 `gridwise_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw_lds_double_buffer`를 선택했다. 원자료에 전체 kernel 이름·count·GPU time·추가 메모리가 있으므로 위 표의 벽시계 시간을 특정 kernel 하나의 시간으로 읽지 않는다.

소스 B: `source/conv_hip_implicit_gemm_*grouped*_xdlops.cpp`의 `IsApplicable`은 CK loader가 없으면 false를 반환한다. 실측 A: CK grouped shared library 누락 경고가 나도 이15개 FAST는 모두 실행되고 FP32 reference gate를 통과했다. 빠진 CK 대신 fallback의 비용을 실제로 측정한 것이다. CK를 설치했을 때의 반사실 배속은 측정하지 않았고, 환경 재구축도 하지 않았다.

NORMAL의 period Conv2D 중 2개는 큰 forward 오차가 있었다. 그래서 NORMAL을 whole-model 기본값으로 권장하지 않는다. depthwise 3개의 개선은 연산 A지만 기존 PC 모델의 해당 경로가 연결되지 않아 모델 A/B는 ❔다. ConvTranspose는 아래처럼 검증된 키만 추가한 실제 epoch를 보완했고 모델 속도 이득이 없었다.

원자료: `family_fast/result.json`, `family_normal/result.json`, `family_normal_cache/miopen_db/`. MIOpen 버전·방향·shape key·workspace 조건까지 포함해 재현해야 한다.

### 7.1 ConvTranspose: 검증된 9개 키를 실제 학습까지 연결한 결과

최소 실험의 기준은 `family_fast`의 세 실제 Applio shape다. 후보는 기존 filtered FindDb의 36키를 보존하고 `family_normal`에서 y/dX/dW가 통과한 **해당 세 shape의 F/B/W 9키만** 새 연구 DB에 병합하는 것이다. ConvTranspose forward는 MIOpen B, dX는 F, dW는 W이므로 일반 Conv1D에서 제외한 V4R1 F entry를 역방향까지 일괄 제거하지 않았다. 병합 DB SHA256은 `b8bd35b3f4d39c44c67fa8b6c7073927cb5bd8b15ab7809947336b0ba2caf33b`이며 원본 두 DB와 운영 DB는 수정하지 않았다. 정확한 방향 키·원본 SHA·명령은 [레시피](CONVTRANSPOSE_DB_RECIPE_KO.md)와 [병합 manifest](hybrid_cache/filtered_transpose_db/merge_manifest.json)에 있다.

새 FAST process에서 세 shape를 다시 실행했다. 모두 독립 FP32 참조와 기존 FAST 대조에 대한 y/dX/dW gate를 통과했고 forward는 byte exact였다. 실제 프로파일은 forward의 Tensile HHS/Col2Im을 유지하고 선택된 dX/dW에 `gridwise_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw_lds_double_buffer`가 나타났다. 표의 시간은 forward+dX+dW 전체이며 개별 kernel 시간으로 해석하지 않는다.

| 실제 B8 FP16 shape | 원래 FAST ms | 병합 DB FAST ms | 배속 | forward 대조 | y/dX/dW 대조 최대 NRMSE | 전체 gate |
|---|---|---|---|---|---|---|
| transpose_C512_O256_T32_k16 | 0.744188 | 0.683457 | 1.0889× | byte exact | 0.00048325 | 통과 |
| transpose_C256_O128_T320_k16 | 1.208836 | 0.929619 | 1.3004× | byte exact | 0.00048530 | 통과 |
| transpose_C128_O64_T3200_k4 | 1.279697 | 1.035844 | 1.2354× | byte exact | 0.00001247 | 통과 |

그 뒤 같은 초기조건의 4epoch를 한 번 실행했다. `--observe-transpose`는 original F.conv_transpose1d를 호출하며 입력 메타데이터와 횟수만 기록한다. **대상 3shape가 각각 64회** 적중했고 input FP16/weight FP32인 autocast 경계를 관측했다. 네 번째 upsample `(8,64,6400)→OC32`와 B1 eval shape는 기존 선택 그대로다. wrapper 수정 전 소스는 `source/applio_epoch_v06_pre_transpose.py`에 SHA `aa91fe5c027fc0be8906b81eac9db458ec80643fdea27fbdfb504249ee4c312e`로 보존했다.

결과는 **filtered no-fold 8.417579208→8.434122308s, 0.99804×**다. 세 steady epoch는 8.432532/8.434122/8.444711s였다. 연산 3개 합계의 절감량은 0.583799ms이고 이를 단순히 16step에 곱하면 약 9.34ms/epoch, 기준 epoch의 약 0.11%다. 이는 실제 통합 프로파일의 대체값은 아니지만, 연산별 개선이 epoch에서 식별되지 않을 수 있는 크기임을 보여 준다. **실제 모델 속도 무이득으로 이 추가 DB 선택은 채택하지 않는다.**

초기 modelstate와 최초 eval audio는 filtered 대조와 byte exact였고, reference 5 tensor의 처음/끝 SHA가 일치했다. loss는 모두 finite, median D/G loss 변화는 +0.850%/−0.197%였다. 최종 waveform max 0.711426/RMSE 0.061977로 임시 수치 gate는 실패했지만, native A/B의 max 0.721924/RMSE 0.134399가 이미 더 크다. 따라서 청각 열화나 학습 동등성 어느 쪽도 확정하지 않는다. 보호 파일 16개의 before/after와 병합 DB SHA는 불변이고 GPU audit은 valid다.

원자료: [3shape replay](transpose_merged_replay/result.json), [실제 epoch](epoch_filtered_transpose/result.json), [보호 파일 before](epoch_filtered_transpose/protected_before.json), [after](epoch_filtered_transpose/protected_after.json). 파형은 같은 폴더의 `initial_audio.pt/.wav`, `final_audio.pt/.wav`에 보존했다.

## 8. BF16 Conv2D — 실제 VAE 성공 레시피

### 최소 실험: 기준 kernel → 후보 → 3 shape → 실제 decode

기준 actual MOSS BF16 VAE의 regular profile은 `Cijk_Ailk_Bljk_BBS_BH_MT64x32x8...ISA000...`에 39회/133.668ms를 썼다. 후보는 caller가 weight/bias를 cast하고 offload lifetime을 확보한 **F.conv2d 경계**에서 input/weight만 FP32로 올리고 계산 후 BF16로 되돌린다. 최종 dtype과 Comfy casting/offload 계약을 유지한다. weight cache는 없다. candidate profile은 FP32 `miopenSp3AsmConv_v30_3_1_gfx10_fp32_f2x3_stride1` 33회/31.117ms가 중심이었다. 전체 BF16 모델을 FP32로 바꾼 실험이 아니다.

중요한 반올림 경계: PyTorch MIOpen은 Conv 결과를 출력 dtype으로 만든 다음 bias를 더한다. 따라서 FP32 Conv(no bias)→BF16→BF16 bias add를 기본으로 둔다. bias까지 FP32에 합쳐 한 번만 BF16로 반올림하면 연산 NRMSE가 약0.00285로 더 커졌다. 그 후보도 실패 원자료로 남긴다.

코드: `../../rdna2_ops/bf16_conv_v06.py`. `BF16Conv2dMode`는 no-grad, GPU BF16 x/w/b, rank4, groups1만 대상으로 하고 다른 경우는 original로 돌아간다. outer autocast는 내부 FP32 계산 구간에서 끈다. context 종료 시 F.conv2d를 복원한다.


| 실제 VAE shape·합성 입력 | BF16 native ms | FP32→BF16+bias ms | 배속 | 보존 경계 NRMSE | FP32 bias NRMSE | 추가 peak MiB native/candidate |
|---|---|---|---|---|---|---|
| [1, 384, 128, 128] | 4.3357 | 1.2401 | 3.496× | 0.0000551 | 0.002851 | 120.00/53.47 |
| [1, 96, 512, 512] | 5.4961 | 1.7727 | 3.100× | 0.0000376 | 0.002849 | 480.00/192.32 |
| [1, 192, 256, 256] | 4.8207 | 1.3424 | 3.591× | 0.0000471 | 0.002841 | 240.00/97.27 |


각 시간에는 input/weight FP32 변환과 output BF16 변환·bias add를 모두 포함했다. native를 no-bias Conv+BF16 add로 재구성했을 때 세 shape 모두 byte exact였다. micro gate는 finite, NRMSE≤.002, max/reference RMS≤.05이며 모델 최종 RGB 기준과 별개다.

실제 기존 denoiser latent `results/production_final_scope/shared_server/reference/latent_seed20261105_32x32.pt`와 기존 flux2-vae_small을 사용했다. 최종512×512 RGB float32 오차의 기준은 **max≤2/255, RMSE≤0.25/255**다. regular/tiled 각각 native와 후보를 번갈아3회 측정했다. observer만 붙인 native는 원래 출력과 true byte exact다. `dtype_vae_v06.py`의 실제 decode 하네스를 별도 wrapper `../../conv_vae_v06.py`로 재사용했다.


| load | decode | native ms | candidate ms | 배속 | 실제 hit | RGB max | RGB RMSE | strict | actual lowvram/dynamic |
|---|---|---|---|---|---|---|---|---|---|
| normal | regular | 180.950 | 73.897 | 2.449× | 40 | 1.494141/255 | 0.186532/255 | 통과 | False/True |
| normal | tiled_256_overlap64 | 915.928 | 376.850 | 2.430× | 920 | 0.779004/255 | 0.106236/255 | 통과 | False/True |
| lowvram | regular | 172.969 | 65.872 | 2.626× | 40 | 1.494141/255 | 0.186532/255 | 통과 | True/False |
| lowvram | tiled_256_overlap64 | 940.545 | 409.179 | 2.299× | 920 | 0.779004/255 | 0.106236/255 | 통과 | True/False |


lowvram은 실제 `model_lowvram=True`, `dynamic_patcher=False`까지 확인했다. regular와 tiled의 내부 BF16 입력 shape/stride와 호출 횟수는 result의 `actual_convolution_inputs`에 있다. tiled256/overlap64 경로는 full512이미지에 920회 Conv를 수행하며 단일 고정 shape에만 연결된 시험이 아니다.

메모리 주의: 모델 result의 `peak_allocated_bytes`는 candidate 앞에서 reset하고 native/candidate A/B 전체 후 읽은 **A/B 구간 peak**다. candidate-only peak 또는 native 대비 절감으로 해석하지 않는다. 위 micro 표의 개별 추가 peak만 별도 후보 메모리 비교 근거다. 입력/weight FP32 임시 복사는 메모리를 늘릴 수 있으므로 큰 해상도에서는 재검증한다.

normal 원자료 `bf16_conv2d_vae/results.json`, lowvram `bf16_conv2d_vae_lowvram/results.json`, unit `bf16_conv2d_micro/results.json`. PNG는 나란히 볼 수 있지만 정량 gate는 원래 float tensor에서 계산했다. 실제 성공 범위는 이 latent·VAE·512출력·두 decode 정책이며 다른 모델의 모든 BF16 Conv를 자동으로 안전하다고 주장하지 않는다.

## 9. Qwen exact-tile Conv3D — 연산 성공, 모델 기준 비재현으로 판정 불가

### 기준과 후보

소스 B와 실제 hook A에서 Qwen3VL의 patch embed가 Qwen35VisionPatchEmbed임을 확인했다. 실제 입력 FP32 `[2400,3,2,16,16]`, weight FP32 `[1152,3,2,16,16]`, bias `[1152]`, stride2/16/16, padding0, dilation1, groups1, 모두 cuda:0 contiguous다. ref960×640을 patch16·merge2로 전처리한 P2400이다. 일반 Conv3D를 Linear로 바꿀 수 있다는 뜻이 아니다. 입력 공간 전체가 한 kernel이고 출력 공간이1인 **exact-tile 경계**에서만 flatten된 선형식과 같다.

같은 Qwen interpreter에서 FP32 P128/512/2400과 bias 포함, forward inference를 3round×7회 측정했다. native는 research DB의 NORMAL 선택이며 CPU FP64로 y/dX/dW도 별도 검증했다. 세 shape 모두 gate 통과다.

| P | NORMAL native forward ms | Linear forward ms | 배속 | native 대비 y max | native 대비 y NRMSE | 추가 allocated bytes(native/candidate) |
|---|---|---|---|---|---|---|
| 128 | 0.184955 | 0.090718 | 2.039× | 0 | 0 | 589824/589824 |
| 512 | 0.259092 | 0.130696 | 1.982× | 8.82148743e-06 | 8.19977208e-07 | 2359296/2359296 |
| 2400 | 1.582226 | 0.473349 | 3.343× | 1.16825104e-05 | 8.44998148e-07 | 11059200/11059200 |

P128의 y/dX/dW는 native와 byte exact였다. 다른 shape는 약1e-5 수준의 작은 최대오차가 있어 최종 모델 검증이 필요하다. raw: `qwen_patch_fp32_native_normal/result.json`, `qwen_patch_fp32_linear_vs_normal/result.json`. native 대표 kernel은 `gridwise_convolution_implicit_gemm_v4r4_ncdhw_kczyx_nkdhw`; Linear는 rocBLAS FP32 GEMM이다. candidate 입력/weight reshape는 contiguous view이며 bias 처리 비용을 포함했다.

### 실패도 함께: FAST 빈 DB와 CK/hipBLASLt

동일 Qwen Torch2.12/ROCm10.2 runtime에서 MIOpen FAST+빈 연구 User FindDb는 P128과 P2400의 **첫 forward**에서 실패했다. CK grouped library 누락 경고 다음에 `TensileLibrary_lazy_gfx1030.dat` 및 gfx1030 hsaco 부재, `hipBlasLt error encountered`, `miopenStatusInternalError`가 기록됐다. `qwen_patch_fp32_native/`, `qwen_patch_fp32_native_p2400/`의 JSON/`job_stderr.txt`에 보존했다.

NORMAL은 Lt 후보 실패를 건너뛰고 v4r4_ncdhw를 골라 P2400을 정상 실행했다. 이는 특정 FAST miss fallback의 호환성 실패이며 CK가 없으면 Conv3D 전체가 불가능하다는 결론이 아니다. 기존 model runtime의 FindDb/cache와 빈 연구 DB를 구분해야 한다.

cudnn-off fallback 대량 반복은 P128부터 매우 느려 owned job을 조기 중단했다. 이 partial 결과는 완료 audit가 없으므로 유효한 배속 표에 넣지 않는다. `qwen_patch_fp32_native_comfy`라는 폴더명은 초기 가설의 이름이며 실제 Comfy 기본 경로라는 근거가 아니다. 명칭 혼동 때문에 삭제하거나 성공한 baseline으로 재분류하지 않았다.

### 실제 모델 A/B 시도와 중단 이유

`../../qwen_conv3d_v06.py`는 별도 연구 process에서만 TextEncodeQwenImage21.execute의 진입/종료를 감싸고, 그 안의 해당 proj 인스턴스 `_conv_forward` 경계만 빌려 쓴다. 기존 CastBiasWeightContext가 준비한 tensor 수명과 weight casting/offload 계약을 유지하며 즉시 복원한다. node3 fingerprint를 바꾸어 conditioning을 실제 재실행한다. sampler 안에만 hook을 넣으면 이미 끝난 vision encoder를 건드리지 못한다.

기존 qwen_v05 graph/image/prompt, seed20261106, **2step bounded screen**으로 warm→baseline→Linear→baseline을 계획했다. 이것은 원래25step workflow의 품질 검증을 대체하는 완주 실험이 아니다. 첫 warm과 native baseline은 실제 생성·디코드에 성공했지만 **후보 실행 전에 native 재현 gate가 실패하여 종료**했다.

- actual backend: cudnn.enabled=True, benchmark=False. `MIOPEN_FIND_MODE`, `MIOPEN_USER_DB_PATH`, `MIOPEN_CUSTOM_CACHE_DIR`, `COMFYUI_ENABLE_MIOPEN`은 모두 unset. preferred BLAS getter는 `_BlasBackend.Cublas`(이 ROCm 환경의 PyTorch 표기).
- 각 요청의 actual native vision Conv는2회, FP32/P2400이며 original 함수는 `disable_weight_init.Conv3d._conv_forward`다.
- warm/native 요청에서 positive·negative conditioning embedding digest부터 달랐다. 최종 latent max0.790894/RMSE0.0780644, RGB max0.1381734(35.234/255), RMSE0.00375442(0.957377/255)로 엄격 gate 밖이다.
- 후보가 아직 실행되지 않았으므로 이 차이는 **Linear의 품질 실패가 아니다**. 모델 후보 품질·모델 배속은 판정할 수 없다. baseline-byte 요구를 느슨하게 바꿔 통과시키지 않았다.
- native 두 call의 GPU-event 구간 합은 warm3,547.08ms, baseline3,553.63ms였다. 이 event 범위는 host/find/compile/staging으로 GPU가 기다린 시간까지 포함할 수 있다. unit NORMAL kernel1.6ms와 차이가 크다는 사실만으로 실제 커널 자체가1000배 느리다고 단정하지 않는다. 실제 모델의 kernel 선택은 별도 profile 없이 미확정이다.
- job은 d878e9d7-d445-4785-865d-d67d46819960, GPU audit valid, 다른 GPU job과 겹치지 않았다. raw는 `qwen_model_screen2/runs.json`, `conditioning_000_warm.json`, `conditioning_001_baseline.json`, `gpu_audit.json`, `job_stderr.txt`다.

CPU 소스 조사에서는 Qwen21 VAE encode가 mu를 반환하고 noise를 샘플링하지 않으며 CLIP은 eval이다. qwen35의 RNG는 이번 경로가 아닌 generation에 있다. 그래서 근거 없는 global RNG 재설정을 넣지 않았다. 자세한 파일·함수 근거는 `QWEN_NATIVE_DRIFT_SOURCE_NOTE_KO.md`다. 다음 최소 확인은 동일 X/W/bias/Y byte digest로 최초 차이를 구분하는 것이다. 현재는 native 수치 결정성, load/offload 후 weight 값, 하류 encoder 중 어디가 원인인지 확정되지 않았다.

**채택 상태:** exact-tile Linear의3shape 연산 후보 A와 FAST compatibility 실패 A는 보존한다. 운영 flag `Q21_RDNA2_CONV3D_PATCH=0`은 그대로이고, 자동 적용 범위를 넓히거나 모델 성과로 승격하지 않는다.

## 10. 재사용과 롤백

모든 코드는 opt-in 연구 context/독립 process다. 운영 런처·설정 파일을 바꾸지 않았으므로 process 종료가 rollback이다. 연구 DB 경로를 넘기지 않으면 기존 운영 선택으로 돌아간다. 운영 cache를 삭제하거나 새 DB를 전역 기본값으로 설치하지 않는다.

새 모델의 Conv를 조사할 때는 다음 순서를 지킨다.

1. 실제 call의 dtype, shape, stride, groups, dilation, padding, bias, training/no-grad와 최종 출력 경계를 기록한다. model name만으로 선택하지 않는다.
2. 실제 caller의 `cudnn.enabled/benchmark`, FindMode, User FindDb와 라이브러리 버전을 확인한다. 운영 cache를 읽은 baseline과 빈 연구 cache baseline을 구분한다.
3. 같은 입력·weight·bias·grad-output으로 대표3개 이상 shape를 측정하고 실제 kernel/solver를 기록한다. dtype 변환·layout copy·bias/scale epilogue 비용을 포함한다.
4. 속도보다 먼저 finite, y/dX/dW 참조 오차를 확인한다. selected solver가 빨라도 오차가 크면 채택하지 않고 실패 DB/key를 보존한다.
5. 모델 실제 경계에서 hit을 확인하고 최종 RGB/audio 및 native-repeat floor를 측정한다. 학습은 초기state/optimizer/scaler/seed/data, inference는 latent/conditioning/input을 고정한다.
6. 검증한 shape·dtype·library 범위만 opt-in 레시피로 남긴다. 실제모델A가 없거나 기준이 식별력을 잃으면 ❔로 표시한다.
7. 새 모델 다운로드·환경 재구축 대신 부족한 모델 연결과 원인을 기록한다. 처리한 원자료는 이 results 폴더, 입구 표와 사용 가이드 갱신은 상위 문서에서 이어간다.

### 핵심 재현 명령

연구 루트 `rdna2-ops`에서 기존 interpreter로 실행한다. 각 출력 경로는 새 폴더여야 한다.

```bash
~/DevSpace/applio-lab/.venv-rocm10/bin/python bench_conv_catalog_v06.py --set seed --mode NORMAL --db results/new_conv_seed/miopen_db --out results/new_conv_seed/seed
~/DevSpace/applio-lab/.venv-rocm10/bin/python bench_conv_catalog_v06.py --set all_d1 --mode FAST --db results/new_conv_seed/miopen_db --out results/new_conv_replay
```

위 replay가 정확도 실패하면 원본 DB를 보존하고 solver/key를 조사한다. 이 기록의 필터는 위에 명시한 source SHA와 shape 범위에서만 검증됐다. 다른 source SHA에 filter를 무작정 적용하지 않는다.

```python
from rdna2_ops.bf16_conv_v06 import BF16Conv2dMode
# 기존 Comfy VAEDecode/VAEDecodeTiled 호출을 이 context 안에서 실행.
# 호출자가 원래 수행하던 dtype/weight/offload 경로를 그대로 둔다.
with BF16Conv2dMode('fp32', bias_rounding='bf16'):
    rgb = existing_decode()
```

현재 성공 범위 밖의 모델에는 위 예제를 자동 등록하지 말고 해당 모델 A/B를 먼저 수행한다.

