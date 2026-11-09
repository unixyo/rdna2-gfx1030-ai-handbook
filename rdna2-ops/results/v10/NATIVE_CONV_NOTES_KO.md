# v1.0 native 잔류 Conv: 구현과 검증 인계

작성: 2026-11-08. 이 문서는 `APPLIO_V10_KO.md`의 보조 자료다.

현재 상태: **전체60shape/169방향 GPU 검증과 CPU 병합 완료. 후보531개 수치 통과,31shape의68방향이 성능·AMP/weight_norm 계약을 통과했다. 운영 epoch 판정은 루트가 별도로 진행하며 이 결과로 대체하지 않는다.** GPU 실행은 루트가 직렬로 수행했고, 이 문서 담당자는 CPU 분석·구현만 했다. 운영 코드와 v0.8 소스는 수정하지 않았다.

## 1. 기존 증거를 다시 실행하지 않고 확인한 것

원자료:

- `../applio_v08/direct_initial9/result.json`
- `../applio_v08/direct_additional12/result.json`
- `../applio_v08/solver_selected_final_replay/result.json`
- `../applio_v08/conv_cases.json`
- `../applio_v08/direct_selected_final/dispatch.json`

v0.8의 직접 dW 세 설정은 split32/64였다. 다음은 당시 같은 실행의 native 대비 수치이며, 새로운 v1.0 측정값이 아니다.

| shape·방향 | 당시 native ms | 당시 가장 빠른 직접 후보 ms | 후보 scratch |
|---|---:|---:|---:|
| C128→128/T3200/K11/d1 dW | 1.05179 | 2.09880 | 23,068,672 B |
| C128→128/T3200/K11/d3 dW | 1.39321 | 2.09797 | 23,068,672 B |
| C128→128/T3200/K11/d5 dW | 1.49138 | 2.09372 | 23,068,672 B |
| C192→384/T74/K5 dW | 0.130336 | 0.483374 | 47,185,920 B |
| C192→384/T224/K5 dW | 0.164931 | 0.471379 | 47,185,920 B |
| C192→384/T326/K5 dW | 0.262250 | 0.704028 | 47,185,920 B |

C192 세 shape의 **논리적 학습 입력은 FP32**, 위 microbenchmark의 물리적 입력은 **AMP에서 반올림된 FP16**이다. 둘을 같은 dtype 실험으로 서술하지 않는다. 새 벤치는 operator FP32 참조와 FP32 논리 입력·FP32 weight_norm master 계약을 따로 검증한다.

기존 선택 DB replay에서 다음 잔류 경로도 확인했다.

| shape·방향 | v0.8 replay ms | 관측된 kernel 계열 |
|---|---:|---|
| ConvTranspose C64→32/T6400/K4/s2 dW | 2.4072 | GEMM + im2col + SubTensor 누적 |
| ConvTranspose C128→64/T3200/K4/s2 dW | 0.6626 | GEMM + im2col + SubTensor 누적 |
| period3 C1→32/H4267/K5/s3 dW | 0.9545 | 작은 GEMM + im2col + SubTensor 누적 |
| period3 C32→128/H1423/K5/s3 dW | 0.6473 | GEMM + im2col + SubTensor 누적 |
| period3 C1024→1024/H53/K5/s1 fwd/dX/dW | 0.4044 / 0.4059 / 0.6059 | Winograd f3x2 / f3x2 / f2x3 |

마지막 Winograd 행에 일반 직접 Conv의 FLOP 수를 나누어 ISA 이용률이라고 부르면 안 된다. 계산 알고리즘이 다르다.

## 2. 실제 운영 프로파일에서 추가로 확인한 shape

최신 `profile_B_target/profile_rows.json`에는 C192→384/K5의 **T=220, 298**도 들어 있다. 두 shape는 기존 v0.8 dispatch의 T74/224/326에 포함되지 않으므로 실제 운영 경로에서 forward/backward가 native로 남는다. 처음 읽었던 `profile_B/`는 metadata·계측 복구용 진단 자료로 분류됐으므로 다음 수치는 이를 대체하는 target 프로파일에서 읽었다.

- T220: `aten::convolution` 28회, inclusive GPU 4.235264 ms. `aten::convolution_backward` 28회, self GPU 9.100295 ms.
- T298: `aten::convolution` 28회, inclusive GPU 4.996821 ms. `aten::convolution_backward` 28회, self GPU 10.545539 ms.
- T74/224/326: `_MeasuredConv` 호출이 관측된다.

이는 운영 프로파일의 operator 귀속 수치다. 부모/자식 operator 시간을 더하거나 kernel 표와 합산하지 않는다. 벤치는 실제 `--baseline-dispatch`에서 shape별 기존 방향을 읽어 `v08_baseline_directions`에 기록한다. 기본 `auto`에서 C192의 fwd/dX가 기존 manifest에 모두 있으면 dW만, 둘 중 하나라도 없으면 fwd/dX/dW를 모두 잰다. 따라서 T220·T298은 기본 전체 실행에 세 방향이 포함된다. C128의 기존 방향 선택 규칙은 그대로다.

## 3. 새 구현의 범위

소스: `../../v10_native_conv.py`.

### 3.1 dW 분할과 출력 타일

v0.8은 kernel 위치 R마다 별도 program을 만들고 dW의 channel 축을 타일링했다. 새 커널은 `CI×R`을 하나의 출력 축으로 묶고, 출력 tile 수와 reduction 길이에 맞춰 split을 정한다. FP32 부분합을 고정 순서로 더하며 atomic은 쓰지 않는다. split1이면 FP32 scratch와 후처리 reduction kernel 없이 FP16 dW에 바로 저장한다.

확인된 구조적 낭비와 미검증 성능 가설을 구분한다.

- T74의 reduction 길이 B×T는 592다. 기존 BK64/split32에서는 32개 중 22개 분할에 실제 입력이 없다. FP32 scratch는 그래도 약47.2 MB를 만든다. 이는 코드와 shape로 계산한 사실이다.
- 이 낭비를 제거하면 더 빨라질 수 있다는 것은 **실행 전 가설**이다. 실제 native보다 빠른지는 GPU 결과로 판정한다.
- `--tile-count 4` 이상이면 새 flat dW 3개와 기존 v0.8 커널의 작은 split 후보도 함께 잰다. C192는 split1/2/4, C128은 split4/8/16이다. 이전의 split32/64 실험을 재실행하지 않는다.

### 3.2 ConvTranspose의 stride phase

ConvTranspose forward를 동치 normal-convolution의 backward-input으로 표현한다. 출력 높이를 stride로 나눈 phase마다 program을 나누고, 그 phase에 기여하는 filter tap만 적재한다. stride10/K16에서는 phase마다 유효 tap이 1개 또는2개다. 큰 zero-insert tensor나 im2col을 만들지 않는다.

ConvTranspose dX는 같은 normal-convolution의 forward, dW는 원래 입력과 grad-output의 역할을 바꾼 normal-convolution dW다. 실제 네 upsampler shape를 지원하며, 이번 구현은 groups1·dilation1·output_padding0으로 한정한다.

### 3.3 period Conv2d

실제 discriminator의 필터는 `(K,1)`이고 폭 방향 stride/dilation/padding이1/1/0이다. NCHW 저장을 유지한 채 높이와 period 위치를 함께 타일링한다. 별도 layout 변환 없이 forward/dX/dW를 계산한다. 폭 방향의 다른 geometry와 grouped convolution은 자동 적용하지 않는다.

### 3.4 AMP·weight_norm·선택 범위

`apply()`는 v0.8과 같은 `torch.amp.custom_fwd/custom_bwd` 경계를 쓴다. 계산 입력과 반환은 FP16이며 FP32 master로 돌아가는 autograd cast·weight_norm 미분은 원래 그래프가 처리한다. bias는 FP32 누산 결과를 먼저 FP16으로 반올림한 뒤 더해 native AMP 경계를 유지한다.

`measured_dispatch()`는 `APPLIO_RDNA2_V10_NATIVE=1`과 검증된 manifest가 모두 있어야 작동한다. 해당 kind의 메서드만 임시로 감싸고 모든 미일치 shape는 원래 메서드를 호출한다. Conv1d의 새 dW를 선택할 때 기존 v0.8 forward/dX 설정을 manifest에 함께 넣어 B 기준선의 다른 방향을 보존한다. lookup은 JSON 직렬화 없이 tuple로 한다.

## 4. 벤치마크와 판정

실행기: `../../bench_v10_native_conv.py`.

- 기본 seed20261108. `--seed`로 명시할 수 있다.
- 각 실행은 새 `results/v10/` 하위 out에 사전 gates와 FindDb 사본을 만든다.
- `--native-policy fastest-v08`는 정확도·GPU audit를 통과한 기존 single-solver rank 기록에서 가장 빠른 방향을 고른 뒤 **새 프로세스 FAST로 다시 측정**한다. 과거 기록은 선택 근거이고 새 성능 주장은 이번 replay에 근거한다.
- FP32 native reference는 FP16로 반올림된 동일 X/W/dY에서 계산한다. 두 MIOpen 안전 변수와 HSA coredump 방지 변수를 import 전에 설정한다.
- y/dX/dW 각각 finite, NRMSE≤0.002, max error/reference RMS≤0.02를 판정한다.
- 시간은 warm event와 wall을 따로 기록하며 할당·부분합·reduction을 포함한다. compile 비용은 별도다.
- 선택할 때 native 초기 측정과 recheck 중 더 빠른 값과 비교한다. event1.10배, wall1.05배를 모두 통과한 방향만 후보가 된다.
- 후보가 있으면 실제 관측된 input dtype별로 `parametrizations.weight_norm` 모듈을 만들어 FP16 AMP y, input gradient, FP32 bias/g/v master gradient를 비교한다. 계약을 통과해야 dispatch에 들어간다.
- dtype도 명시 gate다. 양쪽 y가 FP16인지, input gradient가 원래 input dtype인지, bias/g/v master 및 각 gradient가 FP32인지 모두 참이어야 한다. C192 다섯 shape의 원래 입력 메타데이터는 전부 FP32다.
- 상속된 TunableOp 설정이 baseline을 바꾸지 않도록 enabled/tuning/record-untuned를0으로 지정한다. MIOpen과 Triton cache는 v10 연구 경로를 명시한다.
- `--no-contract`로 생략하면 microbenchmark만 남고 dispatch에 들어가지 않는다.
- `GpuAudit` 실패·겹침이 있으면 전체 timing은 무효이고 dispatch entries도 비운다.
- 모델 채택에는 여전히 운영 경로 epoch 교대 비교가 필요하다. microbenchmark 통과를 학습 완료로 표현하지 않는다.

세 GPU 실행은 이미 completed=true·valid_for_timing=true로 끝났다. **현재 작업에서는 다시 실행하지 않고 기존 결과를 읽는다.** 아래는 Claude 등이 독립 재현할 필요가 생겼을 때만 쓸 명령이며, 완료된 out을 재사용하지 않는 별도 이름이다. 해당 새 out도 이미 존재하면 또 다른 새 이름을 고른다. 운영 venv는 실행 파일을 읽기만 하며 설치하지 않는다.

원자료의 `selection_args`, `gate.seed`, `rounds/reps/warmup`, 방향별 후보 수를 대조했다. C192 smoke는 dW **6후보**이므로 예전 탐색 예시의 `--tile-count 4`를 사용하면 이 결과를 재현하지 못한다. family smoke는2shape·6방향·각3후보(18후보)이고, full은 나머지57shape·162방향·507후보다. result.json에는 tile-count 값 자체가 저장되지 않으므로 아래 값은 **원자료와 같은 실제 후보 집합을 만드는 값**이다.

```bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops

# native_c192_smoke와 같은1shape/dW/6후보. 완료된 원본 out은 사용하지 않는다.
~/DevSpace/applio-lab/.venv-rocm10/bin/python -B bench_v10_native_conv.py \
  --family remaining1d --periods 3 \
  --case conv1d_B8_C192_T74_O384_K5_S1_D1_G1_P2 \
  --directions dw --tile-count 6 --seed 20261108 \
  --rounds 3 --reps 10 --warmup 3 --profile --contract \
  --native-policy fastest-v08 \
  --baseline-dispatch results/applio_v08/direct_selected_final/dispatch.json \
  --out results/v10/native_c192_smoke_reproduction_v1

# native_family_smoke와 같은2shape/6방향/18후보.
~/DevSpace/applio-lab/.venv-rocm10/bin/python -B bench_v10_native_conv.py \
  --family all --periods 3 \
  --case conv_transpose1d_B8_C64_T6400_O32_K4_S2_D1_G1_P1 \
  --case conv2d_B8_C1_H4267_W3_O32_K5x1_S3x1_D1x1_G1_P2x0 \
  --directions auto --tile-count 3 --seed 20261108 \
  --rounds 3 --reps 10 --warmup 3 --profile --contract \
  --native-policy fastest-v08 \
  --baseline-dispatch results/applio_v08/direct_selected_final/dispatch.json \
  --out results/v10/native_family_smoke_reproduction_v1

# native_full과 같은57shape/162방향/507후보. 위 세 shape는 재측정에서 제외한다.
~/DevSpace/applio-lab/.venv-rocm10/bin/python -B bench_v10_native_conv.py \
  --family all --periods all --directions auto --tile-count 6 --seed 20261108 \
  --exclude-case conv1d_B8_C192_T74_O384_K5_S1_D1_G1_P2 \
  --exclude-case conv_transpose1d_B8_C64_T6400_O32_K4_S2_D1_G1_P1 \
  --exclude-case conv2d_B8_C1_H4267_W3_O32_K5x1_S3x1_D1x1_G1_P2x0 \
  --rounds 3 --reps 10 --warmup 3 --profile --contract \
  --native-policy fastest-v08 \
  --baseline-dispatch results/applio_v08/direct_selected_final/dispatch.json \
  --out results/v10/native_full_reproduction_v1
```

각 명령은 root가 GPU 작업을 하나씩 실행한다. 원본 세 결과의 동결 source SHA가 맞는지는 §7.3의 CPU 병합 명령으로 확인할 수 있다. 기존 완료 결과에 대한 재실행이나 덮어쓰기는 하지 않는다.

## 5. roofline 수치의 해석

각 case에는 direct 알고리즘의 nominal FLOP, 실제 padding을 제외한 유효 FLOP, FP16 필수 read/write byte 하한과48.7 TFLOP/s·436.7 GB/s 기반 낙관적 시간 하한을 기록한다. 이는 모델이다.

- compulsory bytes는 실제 DRAM 트래픽 측정치가 아니다. 캐시 재사용·중복 읽기·임시 텐서를 포함하지 않는 하한이다.
- native Winograd는 계산량 자체가 달라 direct FLOP/s를 ISA 이용률로 해석하지 않는다.
- roofline보다 느리다는 사실만으로 나머지 시간을 모두 제거 가능한 소프트웨어 손실이라고 확정하지 않는다. tile/점유율·주소 계산·launch·캐시·전송과 실제 개선 여부를 함께 봐야 한다.
- 하드웨어 제약과 구현 제약의 분리는 새 커널의 kernel trace, scratch 및 epoch 개선과 함께 `GAP_5070_KO.md`에 반영한다.

## 6. CPU 검증과 소스 동결

CPU에서 확인한 사항:

- 두 파일 AST 파싱 성공.
- `bench_v10_native_conv` CPU import·case/config 열거 후 `torch`가 `sys.modules`에 없음.
- 기본 대표 목록은18shape/43방향이다. C128세shape, C192다섯shape, ConvTranspose네shape, period3여섯shape다. `--family remaining1d --directions auto`는8shape/13방향이며 T220·T298의 fwd/dX도 포함한다.
- 별도의 직접 convolution 인덱스 열거와 stride phase tap 열거가 일치한다. GPU 정확도를 대체하지 않는다.

동결 SHA:

```text
v10_native_conv.py
ab69ce5f24f3736fc755f894729325fa8766bb58f4f960f533e802ef1d24b169
bench_v10_native_conv.py
bf17394ad5aa5c8ab2e1c9981e2a604f56b321641192bdee281e356f6b7f2798
```

GPU를 쓰지 않는 확인 명령:

```bash
python3 bench_v10_native_conv.py --list --family remaining1d --tile-count 6
python3 - <<'PY'
import hashlib, json
from pathlib import Path
for name in ('v10_native_conv.py', 'bench_v10_native_conv.py'):
    p = Path(name)
    print(name, hashlib.sha256(p.read_bytes()).hexdigest())
for p in sorted(Path('results/v10').glob('native_*/result.json')):
    j = json.loads(p.read_text())
    print(p, j.get('completed'), j.get('valid_for_timing'),
          j.get('selected_entries'), j.get('selected_new_directions'))
PY
```

## 7. GPU 실행 결과

### 7.1 초기 smoke — 완료

원자료는 `native_c192_smoke/result.json`, `native_family_smoke/result.json`이다. 두 실행 모두 completed=true, valid_for_timing=true이며 외부 KFD client는0이다. seed20261108, 같은 동결 소스다. 아래 native 시간은 초기/recheck 중 더 빠른 값이다.

| shape·방향 | native ms | 가장 빠른 새 후보 ms | event 배율 | 방향 선택 |
|---|---:|---:|---:|---|
| C192→384/T74/K5 dW | 0.114738 | 0.104742 | 1.0954× | 1.10배 기준 미만으로 제외 |
| period3 C1→32/K5/s3 fwd | 0.100036 | 0.023653 | 4.2293× | 선택 |
| period3 C1→32/K5/s3 dX | 0.084941 | 0.072968 | 1.1641× | 선택 |
| period3 C1→32/K5/s3 dW | 0.906734 | 0.033158 | 27.3458× | 선택 |
| ConvTranspose C64→32/T6400/K4/s2 fwd | 0.202086 | 0.264295 | 0.7646× | native 유지 |
| ConvTranspose C64→32/T6400/K4/s2 dX | 0.332716 | 0.095061 | 3.5000× | 선택 |
| ConvTranspose C64→32/T6400/K4/s2 dW | 2.399234 | 0.110454 | 21.7216× | 선택 |

후보24개 모두 operator FP32 수치 gate를 통과했다. 가장 빠른 수치 통과 후보와 성능 선택은 구분했다. C192의 작은 split은 과거 직접 후보0.483ms보다 훨씬 짧아졌지만, 이번의 더 빠른 native0.115ms 대비 채택 문턱에 도달하지 않아 제외했다.

선택한 period shape는 원래 입력 FP16·FP32 두 경계를, ConvTranspose는 FP16 경계를 검증했다. AMP/weight_norm의 y/dX/bias/g/v 수치와 dtype gate는 모두 참이다. 계약의 가장 큰 NRMSE는 period0.000565735, ConvTranspose0.000480179이며 max/reference RMS는 각각0.00320052·0.00349122다. C192는 성능 선택이 없으므로 추가 AMP 계약을 수행하지 않았다.

관측한 ISA/자원도 함께 남긴다. 두 선택 dW는 dot2를 쓰고 spill0이다. period dX의 선택 tile은 spill93이 보고되지만, 실제 비교에서는 native보다1.1641배 빠르고 엄격 수치를 통과했다. 이를 hardware ceiling에 도달한 커널이라고 설명하지 않는다. ConvTranspose fwd는 spill0이어도 느렸으므로 제외했다. spill 유무만으로 성능을 판정하지 않는다.

이 결과로 `native_family_smoke/dispatch.json`에2shape/5방향이 들어갔다. **운영 epoch·100epoch·음성 판정은 아직 별도 단계다.** 전체 sweep에서는 완료한 세 shape를 제외해 같은 실험을 반복하지 않는다.

### 7.2 큰 dW 개선의 원인: batch마다 작은 GEMM 한 workgroup

두 dW의 raw trace를 추가로 읽었다. native는 batch별 Tensile GEMM8회·im2col8회·누적 초기화1회로17 kernel을 실행한다. 새 후보는 전체 batch를 reduction 차원에 합치고 split을 나눠2 kernel로 끝낸다.

| shape dW | native 프로파일 GEMM 합계 | native im2col 합계 | 새 partial/reduce 합계 | native GEMM launch당 workgroup | 새 partial workgroups |
|---|---:|---:|---:|---:|---:|
| period3 C1→32 | 1.043031 ms | 0.024272 ms | 0.024558 ms | 1 | 256 |
| ConvTranspose C64→32 | 2.080911 ms | 0.213192 ms | 0.100078 ms | 1 | 256 |

위 표는 **별도 프로파일의 summed kernel 시간**이다. §7.1의 unprofiled event 중앙값과 같은 분모가 아니며, 서로 합산하거나 이 표로 최종 speedup을 다시 고르지 않는다.

native runtime event 이름은 `hipExtModuleLaunchKernel`이고, 이 API의 원래 grid 인수는 work-item 수다. trace의 period grid/block은 `[64,1,1] / [64,1,1]`, ConvTranspose는 `[256,1,1] / [256,1,1]`이다. Kineto가 trace에 그 값을 어떤 단위로 저장하는지 설치 profiler 소스로 확정하지 않았으므로 이 숫자만 나눠 workgroup 수를 단정하지 않는다. 대신 실제 dW 출력 geometry와 kernel 이름을 교차 확인했다. period는 출력 matrix `32×5`, kernel `MT32x32x8`, `GSU1`, `WG8_8_1`이고, ConvTranspose는 normal-adjoint 출력 matrix `64×128`, kernel `MT128x128x16`, `GSU1`, `WG16_16_1`이다. 두 경우 모두 batch당 출력 tile이 1개이며 같은 kernel이 8회 호출된다. 따라서 batch별 1 workgroup이라는 해석은 이 독립적인 shape/MT/GSU 근거와 일치한다. 후보의 Triton program grid는 `[2,128,1]`, `[4,64,1]`로 각각256 workgroup이다. 서로 다른 HIP launch API에서 trace의 `grid`를 무조건 같은 단위로 읽으면 안 된다.

API 인수의 공식 정의: [AMD HIP module management](https://rocm.docs.amd.com/projects/HIP/en/docs-6.3.2/reference/hip_runtime_api/modules/module_management.html), `hipExtModuleLaunchKernel`의 `globalWorkSizeX` 설명. 현재 HIP header도 동일한 global/local work-size 인터페이스다: [AMD hip_ext.h](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/hip__ext_8h_source.html).

따라서 이 두 shape에서 확인한 개선은 im2col 제거만으로 설명되지 않는다. 큰 시간을 차지한 batch별 작은 GEMM의 병렬도가 부족했고, 전체 batch reduction을 분할해 더 많은 workgroup으로 실행한 구현이 이를 줄였다. 이는 이 두 shape의 소프트웨어 개선을 입증하며, 다른 Winograd shape나 전체 학습 시간이 같은 비율로 줄어든다는 뜻은 아니다.

원자료는 `native_family_smoke/`의 아래 trace다.

- `conv2d_B8_C1_H4267_W3_O32_K5x1_S3x1_D1x1_G1_P2x0_dw_native.trace.json`
- `conv2d_B8_C1_H4267_W3_O32_K5x1_S3x1_D1x1_G1_P2x0_dw_c0.trace.json`
- `conv_transpose1d_B8_C64_T6400_O32_K4_S2_D1_G1_P1_dw_native.trace.json`
- `conv_transpose1d_B8_C64_T6400_O32_K4_S2_D1_G1_P1_dw_c0.trace.json`

### 7.3 전체 sweep와 병합 — 완료

`native_full/result.json`은 초기3shape를 제외한57shape/162방향을 측정했다. 세 결과의 source SHA·benchmark SHA·seed·판정·runtime을 맞추고, 겹친 case0개·누락된 방향0개를 확인해 `native_selected/`에 병합했다. 모든 실행은 completed=true·valid_for_timing=true다.

| family | 측정 방향 | 새로 선택한 방향 | 선택 방향의 event speedup 범위 | 중앙값 |
|---|---:|---:|---:|---:|
| 잔류 Conv1d | 13 | 2 | 1.2482–1.2515× | 1.2499× |
| period Conv2d | 144 | 59 | 1.1111–28.2077× | 2.6816× |
| ConvTranspose1d | 12 | 7 | 1.1406–21.7216× | 2.1517× |
| 합계 | 169 | 68 | shape·방향별 수치 | 전체 학습 배율 아님 |

후보531/531개가 FP32 참조 수치를 통과했고 실행 오류0개, native 수치 실패0개다. 선택31shape의39개 논리 dtype 계약에서195개 y/dX/bias/g/v 비교와 모든 dtype 검사를 통과했다. operator 성능은 이겼으나 계약에서 탈락한 방향은0개다.

남은 방향의 결론도 보존한다.

- C128/K11 d1 dX와 d1/d3/d5 dW는 새 변형이 이기지 못했다. dW의 최선 새 후보/native 배율은 각각0.6215·0.8228·0.8785다. d1 dX의 이번 넓은 tile 변형은0.7382다. 과거 v0.8의 d1 dX1.049배도 당시1.10배 기준을 통과하지 못했으므로 기존 native 선택을 유지한다.
- C192/K5 dW는 다섯 shape 모두 native를 유지한다. T74의1.0954배는 선택 기준 미만이고 나머지 새 후보는 native보다 느렸다. T220·T298의 **forward만** 각각1.2515·1.2482배로 추가됐다. 두 shape의 dX는 native다.
- ConvTranspose forward는 네 shape 모두 native가 더 빨랐다(후보/native 배율0.756–0.804). C512→256의 dX도 native를 유지했다. 나머지 세 dX와 네 dW만 선택했다.
- period는144방향 중59방향을 선택했다. 이는 방향 수의 비율이며 학습 시간 비율이 아니다. 특히 큰 channel의 Winograd 경로에는 직접 후보가 이기지 못한 방향이 남는다.

이 결과는 이긴 shape에서 제거 가능한 구현 비용이 실측됐음을 보여준다. 이기지 못한 방향이 하드웨어의 절대 한계라는 증거는 아니다. `native_selected/direction_summary.json`은 Winograd/FFT/미귀속 native에 direct FLOP 기반 계산 이용률을 만들지 않는다. 나머지도 유효 수학 FLOP 기반 효율과 조건부 메모리 모델이며 실제 ISA/DRAM counter 이용률이 아니다. 캐시에 올라온 tensor를436.7 GB/s DRAM 상한에 강제로 귀속하지 않는다.

병합 산출물:

- `native_selected/dispatch.json`: 실제 주입할31shape/68방향 manifest.
- `native_selected/summary.json`: gate·runtime·coverage·입력 result/dispatch/policy SHA와 roofline 해석 조건.
- `native_selected/direction_summary.json`:169방향의 초기/recheck native, 최선 후보, 선택/제외 이유, kernel 이름, 원자료 SHA.
- `native_selected/direction_summary.csv`: 위 결과를 읽기 편하게 펼친 표.
- `../../v10_merge_native.py`: torch를 import하지 않는 재현 가능한 CPU 병합기.

병합 dispatch SHA256:

```text
37f1fede6701d2cc79efb60a1eb81afb7d568678df74cd11ba2a1fbd1d81a705
```

같은 원자료를 다시 병합하려면 기존 out을 덮어쓰지 말고 새 연구 out을 지정한다.

```bash
python3 -B v10_merge_native.py \
  results/v10/native_c192_smoke/result.json \
  results/v10/native_family_smoke/result.json \
  results/v10/native_full/result.json \
  --out results/v10/native_selected_reproduction
```

핵심 숫자5개를 GPU 없이 읽는 명령이다.60shape·169방향·531수치 통과 후보·68선택 방향·195계약 비교가 나와야 한다.

```bash
python3 - <<'PY'
import json
from pathlib import Path
j = json.loads(Path('results/v10/native_selected/summary.json').read_text())
for key in ('cases', 'directions', 'candidate_operator_pass',
            'selected_new_directions', 'contract_metrics'):
    print(key, j['counts'][key])
PY
```

## 8. 범위 갱신 이력

- 2026-11-08: 최초 기본 목록은 native 잔류 dW 위주39방향이었다. 운영 target 프로파일에서 T220·T298의 fwd/dX 미포함을 확인해 실제 baseline manifest 기반43방향으로 확대했다. 커널 소스·기존 v0.8 결과는 바꾸지 않았다.
- 같은 날: 근거 프로파일을 진단용 `profile_B/`에서 원래 대상 조건을 복원한 `profile_B_target/`로 교체했다. 이전 raw 파일은 그대로 보존한다.
- GPU 실행 전: 완료 smoke를 재실행하지 않도록 `--exclude-case` 반복 옵션을 추가했다. 전체는60shape/169방향이며 C192T74 dW·ConvTranspose C64 세 방향·period3 C1 세 방향을 빼면57shape/162방향이다.
