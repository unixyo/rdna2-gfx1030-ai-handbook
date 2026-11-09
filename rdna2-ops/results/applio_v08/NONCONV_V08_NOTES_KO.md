# Applio v0.8 — Conv 밖의 후보와 소스 근거

작성 시점: 2026-11-07. GPU 결과 갱신: 2026-11-08 KST. 담당 범위: optimizer, host 동기화, weight norm, 원소 연산.
본 문서는 주 보고서의 보조 근거다. 운영 소스는 수정하지 않았다.

## 현재 상태

| 항목 | 상태 | 근거 |
|---|---|---|
| 실제 AdamW 기본 선택 | GPU 확인 | 실제 G/D snapshot에서 native와 explicit foreach의 1·32회 update가 바이트 동일하고 같은 multi-tensor kernel을 실행했다. |
| fused AdamW | GPU 수치 gate 실패, 채택 제외 | G 2.527×, D 2.981× operator 속도였으나 moments와 일부 tensor의 update 오차가 고정 기준을 넘었다. §6 참조. |
| D만 single AdamW | GPU 연산 gate 통과, B_full 4 epoch 적용 | 실제 G flags=null/null, D=false/false. B_full 묶음의 eager 결과는 §10에 기록했다. |
| device gather로 segment slicing | GPU 8/8 y/dX 바이트 동일, B_full 적용 | 4 shape × FP16/FP32. forward warm wall 10.198–14.051×. backward는 정확도만 측정했다. |
| device/foreach grad norm | GPU gate 통과, B_full에 foreach 적용 | device 1.202×·바이트 동일, foreach 5.188×·NRMSE 6.345e-8. 로깅용 norm이며 gradient는 바꾸지 않는다. |
| D weight norm의 forward 범위 캐시 | 전체 gate 미통과, 채택 제외 | scale 1에서도 native D parameter gradient부터 nonfinite여서 유효한 D 기준을 확보하지 못했다. G 경로는 바이트 동일. §8 참조. |
| epoch마다 empty_cache 생략 | B_full 4 epoch에서 8회 생략 | peak allocated는 B_conv보다 1,182,720 B 증가했다. reserved peak는 이 결과에 없다. §10 참조. |
| leaky_relu·add와 Conv 융합 | 소스 경계 검토 완료, 구현·수치·성능 미검증 | forward/backward의 수학적 후보와 현재 Conv 단위 dispatch의 차이를 §11에 명시했다. 속도 이득 근거는 없다. |
| torch.compile | v0.8 복구 4 epoch 완료, 최종 B에서 제외 | Conv1d eager 경계 복구의 첫 18.608218 s·steady 12.583505 s는 eager B_full보다 빠르지 않았다. Conv 전후 fusion의 검증 자료는 아니다. §11 참조. |
| 100 epoch 최종 모델의 audio 평가 | native load·12 WAV·공개물 정합 확인, 청취 판정 대기 | 세 모델 모두 457/457 key, 누락/예상 밖 key 0. raw/WAV finite 및 보호 검사 통과. §12 참조. |

CPU 결과를 RX 6900 XT 성능이나 학습 품질 성공으로 세지 않는다.

## 1. 실제 소스에서 찾은 비용

### 1.1 AdamW와 상태 dtype

applio-lab/rvc/train/train.py 425–438행은 torch.optim.AdamW를 별도 flags 없이 만든다.
voice_model_A 설정은 learning rate 0.0001, betas (0.8, 0.99), eps 1e-9다.
weight decay는 호출 인자가 없으므로 이 버전 AdamW 기본 0.01이다.
모델은 .cuda()로 옮기고 autocast로 연산 dtype을 선택하므로, FP16 AMP에서도 모델과 Adam moments는 FP32다.

설치된 torch/optim/adam.py 934–969행은 기본에 foreach를 선택하고 fused는 기본으로 켜지 않는다.
동일 파일 104–113행의 fused 옵션은 AMP 지원 표시를 켠다.
878–896행은 torch._fused_adamw_를 부르고, found_inf를 장치 텐서로 넘긴다.

torch/amp/grad_scaler.py 363–373행의 일반 optimizer 경로는 found_inf의 .item()을 읽은 뒤 step 여부를 결정한다.
418–470행의 fused 지원 경로는 found_inf를 장치 텐서로 optimizer에 전달한다.
따라서 fused 후보는 여러 Adam elementwise launch와 optimizer 직전 host 동기화를 줄일 가능성이 있다.
실제 구현 유무와 속도는 GPU 벤치로 판정해야 한다.

기준/후보의 같은 입력 gradient에서 1회와 여러 회 update를 비교하고, parameter 자체뿐 아니라
parameter_after - parameter_before와 exp_avg, exp_avg_sq, step까지 비교한다.
FP16 또는 BF16 optimizer 상태로 줄이는 방법은 이 후보에 포함하지 않는다.

### 1.2 segment slicing

commons.slice_segments는 B=8의 각 행마다 ids_str[i].item()을 호출하고 Python slicing을 한다.
Synthesizer.forward의 latent/pitch, trainer의 waveform, epoch 로그의 mel에서 쓰인다.

후보는 동일한 시작 index와 길이를 torch.gather에 준다.
RNG를 새로 호출하지 않고, 자르는 위치·배치·학습 segment 길이도 바꾸지 않는다.
각 행의 index는 중복되지 않으므로 backward도 같은 위치에 원래 gradient를 되돌려 놓는다.
유효한 비음수 window라는 실제 Applio 호출 계약만 대상으로 한다.

Synthesizer가 함수를 직접 import한 별칭도 패치/복구해야 한다.
apply(train)은 별칭을 먼저 확보해서 restore() 때 commons와 Synthesizer를 모두 되돌린다.

### 1.3 grad norm, loss, scale 동기화

- commons.grad_norm은 parameter마다 norm kernel을 호출하고 마지막 scalar를 .item()으로 읽는다.
- train.py는 D와 G update 앞에서 각각 호출한다. 값은 로그에 쓰이며 gradient clipping에는 쓰이지 않는다.
- APPLIO_RDNA2_GRAD_NORM=device는 기존 reduction 순서를 유지하면서 scalar를 GPU에 남긴다.
- foreach는 parameter별 norm을 multi-tensor 경로로 모은다. gradient 내용은 수정하지 않는다.
- loss_gen_all.item()은 G backward 직전에 최저 loss 기록을 위해 읽는다.
- scaler.get_scale()은 매 step 호출된다. 설치된 GradScaler 소스가 이 호출의 CPU–GPU sync를 명시한다.
- APPLIO_AMP_LOG=1의 lab hook 역시 loss/grad float 변환으로 동기화를 추가한다.

loss와 scaler의 동기화는 이번 모듈에서 무작정 제거하지 않았다.
최저 loss의 step/epoch, overflow/scale 감소 기록을 보존하는 trainer 수준의 deferred 기록이 필요하다.
GPU trace 빈 구간을 보고 이득이 있는지 먼저 판단한다.

### 1.4 weight norm과 allocator

MultiPeriodDiscriminator.forward는 각 discriminator를 real/fake에 순서대로 호출한다.
같은 forward 안에서 파라미터는 바뀌지 않으므로 parametrization 결과를 공유할 수 있다.
후보는 torch.nn.utils.parametrize.cached() 범위를 하나의 D forward로 제한한다.
stateful spectral norm은 거부하고, gradient checkpointing을 쓰는 경우 native 경로로 돌아간다.
캐시를 optimizer.step 너머로 유지하면 오래된 weight를 써서 틀리므로 그렇게 하지 않는다.

train.py 819–820행과 994–995행은 매 epoch 끝에 torch.cuda.empty_cache()를 두 번 호출한다.
생략 후보는 계산을 바꾸지 않지만 VRAM 캐시 유지량을 늘린다.
프로파일/교대 epoch에서 별도 후보로 판정해야 한다.

## 2. opt-in 사용과 롤백

연구 하네스에서 원래 trainer를 import한 다음, train.run() 이전에 다음을 넣는다.

~~~python
import v08_nonconv_opt
handle = v08_nonconv_opt.apply(train)
try:
    train.run(...)
finally:
    result["nonconv"] = handle.report
    handle.restore()
~~~

| 환경변수 | 기본 | 후보 |
|---|---|---|
| APPLIO_RDNA2_ADAMW | native | single, foreach, fused |
| APPLIO_RDNA2_ADAMW_G | global ADAMW 값 상속 | G(index 0)에만 native/single/foreach/fused 지정 |
| APPLIO_RDNA2_ADAMW_D | global ADAMW 값 상속 | D(index 1)에만 native/single/foreach/fused 지정 |
| APPLIO_RDNA2_SLICE_GATHER | 0 | 1 |
| APPLIO_RDNA2_GRAD_NORM | native | device, foreach |
| APPLIO_RDNA2_D_WN_CACHE | 0 | 1 |
| APPLIO_RDNA2_SKIP_EMPTY_CACHE | 0 | 1 |

각 후보를 따로 측정한 뒤 성공한 것만 묶는다. 위 변수를 unset하고 새 연구 프로세스를 시작하면 롤백된다.
운영 launcher, assets/config, rvc 소스, 기존 model log와 pretrain은 그대로 둔다.
실제 trainer의 생성 순서 G=0, D=1을 사용하며, 그 뒤 optimizer가 생기면 global 값을 적용한다.
report의 optimizer_constructors에는 index·role·선택 mode·실제 foreach/fused flags를 기록한다.
혼합 후보는 `APPLIO_RDNA2_ADAMW_G=native APPLIO_RDNA2_ADAMW_D=single`이다.
dispatch의 stdlib 대역 검사에 이어 실제 B_full 4 epoch에서도 G native/D single의 생성 flags를 확인했다. §10 참조.

진단 snapshot은 APPLIO_RDNA2_OPT_SNAPSHOT_DIR를 rdna2-ops 내부 경로로 지정하고
APPLIO_RDNA2_OPT_SNAPSHOT_AT=3을 쓴다.
실제 unscale 뒤 optimizer update 직전의 FP32 params/grad/moments를 CPU로 저장한다.
G가 index 0, D가 index 1이며 actual shape와 group hyperparameter를 함께 보존한다.
이 복사는 수백 MiB 이상일 수 있고 시간 측정 구간에서 제외한다.
native AMP skip은 optimizer.step 호출 자체가 없고, fused의 step call은 overflow도 포함할 수 있으므로
snapshot의 optimizer_step_call을 실제 global training step으로 오해하지 않는다.

## 3. 사전 고정 판정과 재현

- slicing: y와 dX 바이트 동일.
- Adam parameter/moments/step: finite, NRMSE ≤ 2e-6, max/reference RMS ≤ 2e-5.
  aggregate와 parameter tensor별 기준을 모두 적용한다.
- parameter update 차이: NRMSE ≤ 0.002, max/reference RMS ≤ 0.02.
- AMP inf 삽입: params/moments/step 바이트 동일 유지, scale 정확히 절반.
- optimizer dtype: FP32 유지. 데이터·loss·LR·batch 의미 변경 없음.
- epoch와 100 epoch 음질 성공은 별도 판정.

~~~bash
~/DevSpace/applio-lab/.venv-rocm10/bin/python v08_nonconv_bench.py \
  --device cuda --only all --profile \
  --snapshots results/applio_v08/optimizer_actual_snapshot/optimizer_0_before_step3.pt \
              results/applio_v08/optimizer_actual_snapshot/optimizer_1_before_step3.pt \
  --out results/applio_v08/reproduce_nonconv_gpu_actual.json

~/DevSpace/applio-lab/.venv-rocm10/bin/python v08_nonconv_bench.py \
  --device cuda --only wn --profile --wn-grad-scale 1 \
  --wn-snapshot results/applio_v08/optimizer_actual_snapshot/optimizer_1_before_step3.pt \
  --db-seed results/applio_v08/db_fast_native --repeats 3 --trials 3 \
  --out results/applio_v08/reproduce_weight_norm_gpu_scale1.json
~~~

작업 디렉터리는 rdna2-ops다. GPU lease와 gpu_guard가 있는 순차 실행만 허용한다.
스크립트는 두 MIOpen 안전 변수를 지정하고 공통 GpuAudit을 사용한다.
출력 파일은 새 경로만 허용하며 기존 원자료를 덮어쓰지 않는다.

first_loaded_state_update_wall_ms는 해당 optimizer를 만든 뒤 첫 update다.
그 뒤 정확도 검사에 이어 시행한 timing의 first_timed_invocation_ms는 process cold가 아니다.
warm 중앙값과 profiler의 커널별 시간은 별도로 남긴다.

## 4. compile 평가 방법

첫 후보는 기존 APPLIO_COMPILE=dec 하나다. 운영/연구 lab_hooks.py는 SHA가 같고 module.compile(mode="default", dynamic=None)을 사용한다.
이전 2026-09-27의 5-step GPU 합계는 native 2.96 s, dec 2.74 s, dec,d 2.81 s였다.
12 epoch compile+fold 실험은 외부 GPU 점유 OOM으로 완료하지 못했으므로 유효한 epoch 개선 근거가 아니다.
기본 eager, eager+선택 후보, dec compile을 서로 분리해서 같은 데이터/seed/LR/epoch로 비교한다.
compile 비용 C와 epoch당 이득 ΔE가 있으면 손익분기 epoch 수는 대략 C/ΔE다.
100 epoch 판단에는 실제 첫 epoch와 이후 99 epoch, 동일한 저장/평가 비용을 합친다.
몇 step GPU 합계 감소만으로 100 epoch 이득을 선언하지 않는다.
연구용 TORCHINDUCTOR_CACHE_DIR/TRITON_CACHE_DIR을 써서 초회와 cache warm을 구분한다.
D weight-norm cache와 compile의 조합은 별도 검증 전에는 묶지 않는다.
실제 decoder는 HiFiGANNSFGenerator이며 고정 32-frame 학습 segment를 받는다.
NSF의 rand/randn에 대해 연구 하네스 `--compile-native-rng`는 inductor fallback_random=True를 지정한다.
하네스는 이 effective 값·cache 경로·dispatch manifest SHA를 결과에 남긴다.
direct hook의 Python 호출 카운터가 tracing 경계에 있으므로 graph_breaks/recompiles 로그로 실제 동작을 확인한다.
후속 v0.8 실행은 첫 FakeTensor AMP 오류와 Conv1d eager 경계 복구 결과를 남겼다. 실제 결과와 융합 검토의 관계는 §11에 기록했다.

## 5. CPU 확인과 측정 오염 기록

- nonconv_cpu.json: GPU를 숨긴 CPU device 실행, 4 slicing shape × 2 dtype,
  3 Adam parameter shape × 4 모드, 1·8 update와 AMP skip 검사 통과.
  이 초기 검사에서 bitwise 필드는 torch.equal 기반 값 동일이었다.
  이후 CUDA 실행용 코드는 uint8 view 비교로 강화했다. GPU 바이트 동일 결과와 구별한다.
- nonconv_install_cpu.json: 기본 no-op, 두 slice 별칭 패치/복구, fused flag,
  torch.load(weights_only=True) snapshot 로드, 전체 restore 확인.
- 중요한 측정 오염: CPU install job
  d15a3cd0-e57c-4ed7-86bd-8ec3d9629c15,
  PID 1530394, 2026-11-07 13:38:57.946–13:39:00.272 UTC.
  torch.cuda.is_initialized()==False와 GPU 숨김 변수에도 불구하고,
  root GpuAudit이 이 PID의 KFD client를 약 13:38:59 UTC에 관찰했다.
  해당 root epoch timing은 폐기한다.
  이후에는 torch import도 GPU lease 안에서만 한다. guard 기준을 완화하지 않는다.

## 6. 실제 G/D AdamW GPU 결과 — 2026-11-08

원자료: [nonconv_gpu_actual.json](nonconv_gpu_actual.json).
연구 GPU audit는 valid=true, foreign client 0이다. 실제 optimizer update 직전 snapshot에서
G 560개/36,431,170개 값, D 165개/71,410,594개 값을 복원했다.
같은 저장 gradient를 1회·32회 적용한 수치 검사와, 그 뒤 3 trial × 10회 warm update의 중앙값을 분리했다.

| 모드 | G wall ms | G 배속 | G 수치 gate | D wall ms | D 배속 | D 수치 gate |
|---|---:|---:|---|---:|---:|---|
| native | 7.718141 | 1.000000× | 통과 | 15.853552 | 1.000000× | 통과 |
| single | 17.018334 | 0.453519× | 통과 | 11.138054 | 1.423368× | 통과 |
| foreach | 7.717637 | 1.000065× | 통과·native와 바이트 동일 | 15.778000 | 1.004788× | 통과·native와 바이트 동일 |
| fused | 3.053804 | 2.527385× | 실패 | 5.317606 | 2.981333× | 실패 |

native와 explicit foreach는 parameters/moments/step/update까지 1·32회 모두 바이트 동일했다.
모든 모드의 저장 결과는 finite이고, 별도의 AMP inf 삽입에서는 parameter/moments/step 불변 및 scale 128→64를 통과했다.
fused의 탈락 원인은 AMP skip 실패나 NaN이 아니라 고정 수치 gate 초과다.

### fused의 실제 실패 위치

parameters/moments는 NRMSE ≤ 2e-6 및 max/reference RMS ≤ 2e-5를 aggregate와 tensor별로 모두 적용한다.
parameter update는 NRMSE ≤ 0.002 및 max/reference RMS ≤ 0.02를 같은 방식으로 적용한다.

| 모델·update 횟수 | parameter max/RMS | exp_avg max/RMS | exp_avg_sq max/RMS | 최악 tensor update NRMSE | update 실패 index |
|---|---:|---:|---:|---:|---|
| G·1 | 5.56315e-6 | 2.30046e-4 | 9.41259e-4 | 0.003034825 | 38 |
| G·32 | 1.77958e-4 | 2.08466e-5 | 0.002576179 | 0.002071226 | 38 |
| D·1 | 1.14677e-5 | 1.59543e-4 | 6.12137e-4 | 0.003656458 | 4, 22, 40, 58 |
| D·32 | 3.66721e-4 | 4.79846e-5 | 0.002928813 | 0.003198376 | 22, 40, 58 |

G index 38은 shape [192], D index 4는 [64,1,1], D index 22/40/58은 [32,1,1,1]이다.
index는 snapshot parameter list의 0-based 순서다. 원자료에는 모든 실패 index가 있다.
exp_avg_sq tensor별 실패 수는 G 73→202개, D 25→56개이고,
32회 뒤 parameter tensor별 실패도 G 143개, D 33개다.
일부 exp_avg 행은 tensor별 실패 index가 비어 있어도 aggregate max/RMS가 기준을 넘으므로 실패다.

오차는 FP32 fused update 경로에서 생긴 작은 절대차가 moments와 작은 tensor의 update 기준에 드러난 형태다.
1회 parameter 최대 절대차는 G 4.76837e-7, D 9.53674e-7이고 32회 뒤에는 각각 1.52588e-5, 3.05176e-5다.
연산 융합·반올림 경계 차이와 양립하지만, fused C++ 내부를 대조해 정확한 식 차이까지 확정한 결과는 아니다.
게이트를 완화하거나 일부 tensor를 제외하지 않고 **fused는 후보에서 제외한다**.

### D single을 남긴 이유

D single의 wall trial은 11.1812/11.1300/11.1381 ms이고 native는 15.3302/15.8536/17.0511 ms였다.
D의 exp_avg/exp_avg_sq/step은 native와 바이트 동일하며 parameter/update 차이도 모든 gate를 통과했다.
G에는 single을 쓰지 않고 native를 유지하는 혼합 후보를 실제 epoch에서 확인한다.
고립된 G+D update 합은 23.5717→18.8562 ms이지만 이 합을 전체 학습 배속으로 보고하지 않는다.

원자료 kernel_rows의 `Optimizer.step#AdamW.step`는 annotation이므로 실제 kernel 합계에서 제외했다.
그 행을 뺀 profile에서 G native/single은 각각 66/4,480 launch와 7.163/9.234 ms,
D native/single은 34/1,320 launch와 15.593/8.508 ms였다.
single의 많은 launch 부담이 G에서 크게 드러났고, D에서는 더 짧은 kernel 실행시간이 그 부담을 상쇄했다.
profile 합계와 warm event/wall은 서로 다른 측정이며 더하지 않는다.

## 7. Slicing·grad norm GPU 결과

같은 [nonconv_gpu_actual.json](nonconv_gpu_actual.json)에 담긴 결과다.
아래 slicing 시간은 forward 및 그 autograd graph 구성의 warm wall이다. backward는 정확도만 비교했다.

| 구간 | dtype | native ms | gather ms | 배속 | y/dX |
|---|---|---:|---:|---:|---|
| latent | FP16 | 0.192217 | 0.017536 | 10.961× | 바이트 동일 |
| latent | FP32 | 0.191895 | 0.017940 | 10.697× | 바이트 동일 |
| pitch | FP16 | 0.193358 | 0.013761 | 14.051× | 바이트 동일 |
| pitch | FP32 | 0.178840 | 0.015546 | 11.504× | 바이트 동일 |
| waveform | FP16 | 0.190452 | 0.016678 | 11.420× | 바이트 동일 |
| waveform | FP32 | 0.174483 | 0.017110 | 10.198× | 바이트 동일 |
| mel log | FP16 | 0.198683 | 0.016441 | 12.085× | 바이트 동일 |
| mel log | FP32 | 0.194404 | 0.016437 | 11.827× | 바이트 동일 |

norm 검사는 고정된 실제 계열 shape 17개를 사용한 합성 gradient다. 전체 G/D snapshot norm timing은 아니다.

| norm 경로 | wall ms | native_item 대비 | 오차 |
|---|---:|---:|---|
| native_item | 0.214707 | 1.000× | 기준 |
| device | 0.178633 | 1.202× | 바이트 동일 |
| foreach | 0.041388 | 5.188× | NRMSE 6.34476e-8, max abs 0.0001220703125, gate 통과 |

gather와 foreach norm을 epoch 후보로 넘긴다. norm은 로깅 값만 계산하며 gradient를 수정하지 않는다.
학습 루프의 다른 동기화·allocator·launch와 함께 실제 epoch에서 이득을 확인한다.

## 8. Weight-norm cache GPU 판정과 계측 복구

첫 [weight_norm_gpu_actual.json](weight_norm_gpu_actual.json)은 scale 128의 계산 뒤
nonfinite metric을 JSON에 기록하는 과정에서 allow_nan=False 오류로 종료됐다.
원래 console log (`weight_norm_gpu_actual.console.log`, 연구 폴더에만 있음)를 보존했다. 이 실행은 완성된 수치/시간 판정 자료가 아니다.
metrics는 이제 reference/candidate의 NaN·양의/음의 Inf 개수와 tensor index를 보존하고,
finite=false/pass_gate=false 및 수치 필드 null을 기록한다. 수치 gate는 바꾸지 않았다.
`--wn-grad-scale`은 기본 128을 유지하며, 실패 원인 분리를 위해 scale 1을 한 번 실행했다.

[scale 1 원자료](weight_norm_gpu_actual_scale1.json)는 JSON 저장 완료, GPU audit valid, foreign client 0이다.
실제 D snapshot FP32 weight와 고정 난수 B8×1×12800 waveform(FP32 real, FP16 fake), FP16 AMP를 사용했다.

| 단계 | native wall ms | cache wall ms | 관측 배속 | 수치 판정 |
|---|---:|---:|---:|---|
| D parameter gradients | 138.622208 | 135.572359 | 1.022496× | native부터 gradient nonfinite, 비교 기준 실패 |
| G input gradient | 63.793327 | 63.304648 | 1.007719× | 출력·loss·gradient 바이트 동일 |

D의 output/loss는 finite였으나 native parameter gradient index 0,10,11,13,14,16,17,19,20에 NaN/Inf가 있었다.
cache에서도 같은 index와 동일한 nonfinite 개수가 관측됐다. 따라서 cache만의 오답이라고 단정하지 않는다.
native reference가 실패한 D 시간은 채택 근거로 사용하지 않으며, **WN cache는 Bfull에서 제외한다**.
G 경로의 성공만으로 D 전체 backward나 최종 학습 품질이 통과한 것으로 표시하지 않는다.

## 9. 다음 단계에 넘기는 상태

- 연산 통과 후보 G native/D single AdamW, gather slicing, foreach grad norm을 B_full 4 epoch에서 함께 적용했다.
- empty_cache도 같은 실행에서 8회 생략했다. 묶음 시간·allocated peak는 §10에 기록했으며 개별 기여와 reserved peak는 미측정이다.
- 연산 gate 실패/기준 미확보로 제외: fused AdamW, D weight-norm cache.
- dec + native RNG는 Conv1d eager 경계 복구까지 측정했으나 추가 이득이 없어 최종 B에서 제외했다. §11 참조.
- 100 epoch 최종 모델 3개의 native load와 익명 12 WAV 생성을 완료했다. 청취 판정은 대기 중이며 §12와 [주 보고서](APPLIO_V08_KO.md)의 최신 상태를 따른다.

## 10. 실제 eager B_full 4 epoch — nonconv 묶음 적용

원자료: [B_full result](epoch_B_full/result.json), [B_conv result](epoch_B_conv/result.json).
둘 다 completed=true, valid_for_epoch_timing=true, valid_for_timing=true다.
GPU audit의 foreign client와 오류는 각각 0이고, 운영 보호 검사 16+987개 및 연구 copy 보호 항목의 변경은 0개다.
같은 초기 G/D SHA, 같은 직접 Conv manifest SHA `22e41075581e8b2d969388244bf063adba59579942c4959ae892e2aeb4fcfa59`,
같은 하네스 SHA, batch 8·31 step/epoch·4 epoch(총 124 step), 동일한 직접 Conv 호출 횟수를 확인했다.
FAST, FP16 AMP, fold=false, compile 비활성 및 두 MIOpen 안전 변수=0 조건이다.

| B_full 항목 | 완료 result의 실제 상태 |
|---|---|
| G AdamW | index 0, mode=native, foreach=null, fused=null |
| D AdamW | index 1, mode=single, foreach=false, fused=false |
| optimizer 공통 값 | lr=0.0001, betas=(0.8,0.99), eps=1e-9, weight_decay=0.01 |
| slicing / norm | slice_gather=true, grad_norm=foreach |
| empty_cache | skip_empty_cache=true, 실제 호출 생략 8회 |
| WN / snapshot | discriminator_weight_norm_cache=false, snapshot_dir="", 저장 snapshot 0개 |

gather와 norm은 process-local 교체의 활성 상태가 report에 남으며 별도 호출 횟수 counter는 없다.
optimizer는 생성된 실제 param group flags를, empty_cache는 실행된 wrapper 호출 횟수를 기록한다.
fused와 WN cache는 이 조합에서 제외했다.

| 시간·메모리 | B_conv | B_full | B_full의 변화 |
|---|---:|---:|---:|
| 첫 epoch | 16.575041 s | 16.536212 s | −0.234% |
| epoch 2 | 13.512757 s | 13.736979 s | — |
| epoch 3 | 13.314360 s | 12.556177 s | — |
| epoch 4 | 12.866639 s | 12.570070 s | — |
| steady 중앙값(epoch 2–4) | 13.314360 s | 12.570070 s | −5.590%, 1.059211× |
| 첫 epoch 포함 4 epoch 합 | 56.268797 s | 55.399439 s | −1.545%, 1.015693× |
| import 이후 process 시간 | 62.590894 s | 61.917993 s | −1.075% |
| 최대 peak allocated bytes | 4,473,159,680 | 4,474,342,400 | +1,182,720 B, +0.0264% |

시간 개선은 nonconv 묶음의 한 번의 실제 4 epoch 비교다. optimizer·gather·norm·allocator 각각의 학습 기여는 분해하지 않았다.
allocated peak는 활성 tensor 메모리 계측값이다. allocator reserved peak와 전체 장치 VRAM 최고치는 이 result에서 제공하지 않는다.

두 실행 모두 loss_d/loss_g/mel/fm/kl/adv 124 step 전체가 finite이고 nonfinite_losses=0이다.
B_conv의 AMP scale 감소는 11회(1–8, 32, 41, 51 step), 최종 32였다.
B_full은 10회(1–8, 24, 72 step), 최종 64였다. D grad norm의 nonfinite는 양쪽 모두 초기 1–5 step,
G grad norm의 nonfinite는 각 scale 감소 step에 기록됐다. scale 감소를 optimizer별 skip 횟수로 바꾸어 세지 않는다.
처음 6 step의 loss 값은 동일하고 step 7부터 학습 경로 차이가 관찰됐다.

| 마지막 31 step loss 중앙값 | B_conv | B_full | B_conv 대비 상대 변화 |
|---|---:|---:|---:|
| mel | 18.893467 | 18.547993 | −1.8285% |
| fm | 11.593750 | 11.656250 | +0.5391% |
| kl | 1.523440 | 1.535438 | +0.7876% |
| adv | 3.248934 | 3.232155 | −0.5164% |
| loss_d | 4.191768 | 4.137746 | −1.2888% |
| loss_g | 35.425400 | 35.448250 | +0.0645% |

이 차이는 B_conv에 대한 짧은 실행 관측이다. native A/A′ 반복 기준과 100 epoch·청취 판정은 아직 이 표로 충족되지 않는다.
최종 채택과 native 대비 이득은 주 보고서의 A/B 교대 실행 및 후속 평가에 연결한다.

## 11. leaky_relu·add와 Conv 전후 융합 — 원 지시 §2.4 검토

**현재는 소스와 연산식 검토이며, 추가 융합 kernel의 구현·수치 검증·GPU 성능 측정은 하지 않았다. 속도 이득을 주장하지 않는다.**

ResBlock.forward (`applio_copy/rvc/lib/algorithm/residuals.py`, 연구 폴더에만 있음)는 각 반복에서
`leaky_relu → conv1 → leaky_relu → conv2 → residual add`를 실행한다.
실제 HiFiGANNSFGenerator (`applio_copy/rvc/lib/algorithm/generators/hifigan_nsf.py`, 연구 폴더에만 있음)는 `resblock(x)`로 호출하므로 이 경로의 x_mask는 None이다.
ResBlock과 upsample 앞 activation slope는 0.1이고, 최종 conv_post 앞 `leaky_relu(x)`는 기본 slope를 쓰므로 하나의 slope로 통일하면 안 된다.
upsample 결과와 noise_conv 출력의 add, 여러 ResBlock 출력의 합/평균도 별도 Python tensor 연산으로 남아 있다.

| 실제 경계 | 수학적으로 가능한 forward 융합 | 필요한 backward 보존 |
|---|---|---|
| leaky_relu → Conv / Conv → leaky_relu | Conv 입력을 읽을 때 activation을 계산하거나, 출력 저장 전에 activation을 계산할 수 있다. activation을 Conv의 반대편으로 옮기는 식 변형은 하지 않는다. | 입력 쪽 융합은 dW에 활성화된 입력, dX에 activation 미분을 적용해야 한다. 출력 쪽 융합은 Conv에 들어가는 dY에 activation 미분을 적용해야 한다. 부호 정보와 0 경계의 native 미분 규칙을 보존해야 한다. |
| conv2 출력 + residual | Conv 출력 단계가 같은 위치의 residual을 함께 읽어 add할 수 있다. | residual 경로에는 upstream gradient가 그대로 전달되고 Conv 경로의 gradient와 합쳐진다. 해당 gradient들이 준비된 경우에만 합산 단계를 묶을 수 있으며 저장 tensor·dtype·합산 순서를 보존해야 한다. |
| upsample + noise / ResBlock 출력 합·평균 | 두 입력이 준비되면 add와 나눗셈을 같은 원소 연산 kernel에 넣거나 대응 Conv 출력 단계에 묶을 여지는 있다. | 각 분기의 gradient 분배와 평균 계수를 유지해야 한다. upsample은 ConvTranspose1d여서 현재 Conv1d dispatch의 적용 범위 밖이다. |

이는 실수 연산식 수준의 가능성이다. AMP에서는 중간 dtype과 반올림 위치도 맞아야 한다.
[현재 직접 Conv](../../v08_direct_conv.py)의 `_fwd`는 FP32 누산을 FP16으로 먼저 반올림한 뒤 bias를 더해 다시 FP16으로 저장한다.
FP32 누산에 activation/add까지 한꺼번에 적용하면 이 경계가 달라질 수 있으므로 기존 고정 수치 gate를 통과한 것으로 간주할 수 없다.
현재 `_MeasuredConv`는 x·w·bias와 Conv 방향만 받고 x·w만 저장한다. 앞의 activation 입력/부호나 residual tensor를 받지 않으며,
backward도 Conv dX/dW/db까지만 반환한다. 따라서 **선택 manifest의 Conv/bias 처리에는 추가 leaky_relu·residual add 융합이 포함되지 않는다.**
새로운 융합을 채택하려면 이 인접 연산과 필요한 backward 상태를 포함한 인터페이스와 별도 검증이 필요하다.

기존 compile 결과도 이 공백을 대신 검증하지 않는다.
[첫 dec compile](epoch_B_compile_dec/result.json)은 epoch 완료 전에 FakeTensor의 Half input/Float bias 오류로 실패했다.
[Conv1d eager 경계 복구](epoch_B_compile_boundary/result.json)는 4 epoch를 완료했지만,
[하네스](../../applio_epoch_v08.py)가 Conv1d dispatch 전체를 `torch.compiler.disable(..., recursive=True)`로 둘러싸므로
compile이 그 경계를 넘어 Conv와 activation/add를 합친 실행이 아니다. 주변 원소 연산이 정확히 얼마나 융합됐는지도 이 결과만으로 확정하지 않는다.
복구의 첫 epoch/steady/4 epoch 합은 18.608218/12.583505/56.760980 s였고, eager B_full은 16.536212/12.570070/55.399439 s였다.
최종 B는 검증한 직접 Conv 방향과 G native/D single·gather·foreach norm·allocator cache 유지의 eager 조합이며,
compile 및 미검증 leaky_relu/add 융합은 포함하지 않는다. 최종 100 epoch 상태는 주 보고서를 따른다.

## 12. 100 epoch 최종 모델의 실제 audio 결과 — 청취 판정 대기

원자료: [audio result](audio_eval_final/result.json). 실행 Core13505 exit 0,
state=audio_ready_user_judgment_pending, valid_for_listening=true, downloads=0이다.
연구 arm A/A′/B 모두 checkpoint epoch=100·step=3100이고, 실제 converter.net_g의 native load를 각 1회 관측했다.
입력/모델 key는 각각 457/457개, strict=false·assign=false, missing_keys=[]·unexpected_keys=[]다.
지정 export와 loaded model path·SHA도 일치하며 native_weight_load_complete=true다. 각 추론 모델 파일은 55,171,049 B다.

12개의 generator raw probe와 12개 저장 WAV 모두 finite다. raw는 [1,1,T], 40 kHz이고 abs≥1.0 비율은 전부 0이다.
WAV는 40 kHz mono PCM16이며 문장별 길이 4.86/4.30/5.66/5.38 s가 세 연구 arm에서 같다.
raw는 native padding 제거 전 출력으로 각 WAV보다 80,000 sample(2 s) 길다.
문장별 seed도 세 arm에서 동일하다. 아래 표는 연구 arm의 진단이며 익명 label 대응은 기록하지 않았다.

| 연구 arm | raw peak 범위 | WAV peak 범위 | peak97 구간 합계(raw/WAV) | 최장 구간 ms(raw/WAV) |
|---|---:|---:|---:|---:|
| A | 0.779565–0.912154 | 0.467560–0.638611 | 7 / 4 | 0.300 / 0.300 |
| A′ | 0.808241–0.880915 | 0.437866–0.615662 | 8 / 4 | 0.275 / 0.275 |
| B | 0.791130–0.918669 | 0.420532–0.641846 | 6 / 7 | 0.275 / 0.275 |

peak97은 각 측정 신호의 global peak 97% 이상이 연속 6 sample 이상인 구간이며, clipping/품질 실패 분류가 아니다.
비정렬 WAV의 B 관련 RMSE는 A/A′ 반복 RMSE의 1.570–1.866배였다. 이 지표는 청취 보조이며 음질 동등성 통과로 해석하지 않는다.
quality_judgment=pending_user_blind_listening 상태를 유지한다.

GPU audit 140 sample에서 foreign client/오류는 0이다. 운영 보호 16+987개는 전후 변경 0이고,
before/after hash map이 같으며 source_model_assets_unchanged=true, operating_unchanged=true다.
[공개 manifest](audio_eval_final/blind_manifest.json)는 허용 필드와 익명 WAV 경로만 포함하고,
청취 HTML (`audio_eval_final/LISTEN_BLIND.html`, 연구 폴더에만 있음)의 공개 텍스트에도 실제 연구 arm·model path 유출 표식이 없었다.
익명 WAV 12개와 HTML의 embedded WAV 12개는 각각 공개 manifest SHA와 일치한다.
HTML은 6,474,097 B, 공개 WAV 합계는 4,848,528 B다. blind_key를 읽거나 공개/연구 arm의 WAV hash를 대조하지 않았다.
