# RDNA2 연구 v1.0 지시문: 남은 열린 질문 네 개 (GPT Pro)

작성: 2026-11-08, Claude. 사용자 요청.

## 0. 무엇을 하는가

가이드 v0.9와 카탈로그 v0.5에 아직 "모름"으로 남은 칸 네 개를 채운다. 우선순위 순서다.

| # | 질문 | 왜 남았나 |
|---|---|---|
| A | Applio에서 더 줄일 수 있는 것: LeakyReLU·잔차 덧셈을 직접 Conv에 합치기, native로 남은 Conv | v0.8에서 검토만 하고 구현하지 않았다 |
| B | Applio의 RTX 5070 12분과 이 PC 약 22분의 차이에서 줄일 수 없는 몫 | v0.8 결론이 "줄인 몫은 실측, 줄일 수 없는 몫은 미식별"이었다 |
| C | TunableOp는 왜 프로세스마다(Qwen·MOSS는 프로세스 안에서도) 출력이 달라지나 | 원인 미확인 |
| D | Strata warm 프리필의 병목이 GPU 계산인가 PCIe·host인가 | v0.7에서 swap 보호 관문에 막혀 미판정 |

A와 B는 사용자가 가장 아끼는 Applio다. C와 D는 그다음이다. 시간이 모자라면 A, B만 끝내도 된다. 그 경우 C, D는 시작하지 않은 상태로 보고한다.

## 1. 이미 확인한 것 (다시 조사하지 않아도 됨)

### 1.1 공통 환경

- 장치는 RX 6900 XT(gfx1030)다. WMMA·FP8·BF16 dot이 없고 dot2/dot4/dot8은 있다. Triton은 3.8로 고정한다(3.9부터 RDNA2 제거). hipBLASLt는 gfx1030을 지원하지 않는다.
- 입구 문서는 [카탈로그 v0.5](RDNA2_OP_CATALOG_KO.md)와 [가이드 v0.9](rdna2-ops/RDNA2_OPTIMIZATION_GUIDE_KO.md)다. 특히 §1.2(판정 기준), §9.14, §9.18, §9.20, §9.21, §10(체크리스트)을 읽는다.
- 판정 기준은 **지각 기준이 기본**이다(가이드 §1.2). 엄격 기준(바이트·허용 오차)은 계속 재서 참고 수치로 남긴다.

### 1.2 Applio (A, B)

- 정본 보고서는 `rdna2-ops/results/applio_v08/APPLIO_V08_KO.md`이고, 끝의 §11이 Claude의 운영 롤아웃 기록이다. 가이드 §9.21이 요약이다.
- 운영 상태(2026-11-08)
  - `applio-lab/run-applio.sh`(GUI)와 `lab/train_bench.sh`가 `.venv-rocm10`, `APPLIO_RDNA2_V08=1`, `MIOPEN_FIND_MODE=FAST`, 안전 변수 두 개를 기본으로 쓴다.
  - 코드는 `applio-lab/rvc/train/rdna2_v08/`에 있다. 연구본 `v08_direct_conv.py`·`v08_nonconv_opt.py`·`dispatch.json`·선택 DB와 바이트가 같은 사본이다.
- **운영 경로 숫자가 하네스와 다르다.** 실제 `python core.py train → rvc/train/train.py`에서 잰 값이다.
  - 변경 전: 약 22 s/epoch
  - DataLoader 워커 GPU 숨김: 약 15.0 s/epoch
  - v0.8 B 전체: 약 12.6 s/epoch
  - 원인: spawn 워커 4개가 train.py를 다시 import하면서 각자 GPU 큐 3개를 잡았다. 학습 중 6 프로세스·19 큐였고, 학습 커널이 전부 1.5–2배 느려졌다.
  - GPT 하네스(`applio_epoch_v08.py`)는 진입점이 달라서 이 문제가 없었다. 하네스 A 15.0 s ≈ 운영 경로의 워커 숨김 상태다.
  - 2026-09-27의 옛 프로파일 `lab/results/profile_base_fp16_b8.txt`(5 step GPU 2.96 s)도 워커 문제가 있던 상태에서 잰 값이다.
- 운영 경로 100 epoch는 약 37분에서 약 21–22분으로 줄어든다. epoch 시간으로 계산한 추정이고, 100 epoch 전체 재실행은 하지 않았다.
- 하네스 100 epoch B는 1320 s였고, 익명 청취에서 사용자가 A/A′/B를 구분하지 못해 채택했다.
- v0.8 B에서 native로 남은 Conv는 다음과 같다. 직접 커널이 더 느렸다.
  - C128/K11 d1·d3·d5 dW, C128/K11 d1 dX
  - C192→384/K5 dW
  - discriminator period Conv2d 대부분, ConvTranspose1d 업샘플
  - 6방향만 solver를 다시 골랐다.
- GPT v0.8 메모에 "LeakyReLU·residual add를 Conv와 합치는 가능성은 검토만 하고 구현·수치·성능 미검증"이라고 남아 있다.

### 1.3 TunableOp (C)

가이드 §9.20, `DEEP_RESEARCH_MERGED_20261107_KO.md` §5·§6.1·§6.2·§6.4에 있다.

| 모델 | 출력 재현성 |
|---|---|
| Anima FP16 | 같은 프로세스 안은 바이트 일치. 같은 `tunableop_results0.csv`를 새 프로세스에서 `TUNING=0`으로 읽으면 같은 seed도 RGB RMSE 4–7/255로 달라진다. TunableOp를 끄면 프로세스 간 결정적이다 |
| Qwen INT8 | 같은 프로세스·같은 파일·같은 seed에서도 TunableOp 실행끼리 다르다(max 152–245, RMSE 3–7/255). 끄면 바이트 일치다 |
| MOSS INT8 | 프로세스 안에서도 다르다. 첫 튜닝에서 `Cijk_Ailk_Bljk_SB_MT64x32x32_…_GSU1` 메모리 오류가 났고, `ROTATING_BUFFER_SIZE=0`으로 다시 하니 끝까지 돌았다(1회) |

- 지금까지의 추론(미확인): split-K(GSU>1)나 atomic을 쓰는 솔루션이 뽑혔을 수 있다. 또는 파일에 없는 shape의 기본 선택이 프로세스마다 다를 수 있다.
- 이 PyTorch 빌드에는 `torch.cuda.tunable.write_file`이 없다. 파일은 종료할 때 자동으로 쓰이고, 실제 파일명에는 장치 번호가 붙는다(`results0.csv`).

### 1.4 Strata (D)

가이드 §9.18, `rdna2-ops/results/strata_v07/STRATA_PREFILL_V07_KO.md` §8(재개 조건과 명령), `PROTOCOL_KO.md`에 있다.

- 첫 fresh 9.7K 요청: 199 tok/s, 첫 요청 swap-out 6.1 GB. 첫 요청의 dequant 74.9%는 순수 GPU dequant가 아니라 gather·copy 대기와 host 지연이다.
- warm 1,200–1,300 tok/s는 사용자가 확인한 값이다. warm의 J off/on A/B는 추가 swap-out 256 MiB 관문에 두 번 걸려 중단됐다.
- 실제 라우팅 분포·J hit 비율·warm 병목은 모두 미측정이다.

## 2. 이번 범위

### 2.A Applio 남은 시간 해체

1. **운영 경로로 다시 분해한다.** v0.8 B 상태의 실제 `core.py train`으로 step 40–44를 프로파일한다. `APPLIO_PROFILE_AT=40`이 `lab_hooks`에 있다. 새 모델 이름을 쓰고 데이터는 `logs/voice_model_A`의 config·filelist만 복사한다.
   - native로 남은 Conv, ConvTranspose, discriminator Conv2d, 원소별 연산, optimizer, host 대기를 operator 표와 kernel 표로 따로 낸다.
   - 학습 중 KFD 클라이언트와 큐 수가 7 이하인지 같이 기록한다.
2. **prologue/epilogue 융합**: HiFi-GAN ResBlock의 `leaky_relu → conv → leaky_relu → conv → + x`를 직접 Conv 커널의 입력 적재(LeakyReLU)와 출력(잔차 덧셈)에 합친다. backward(LeakyReLU 기울기 mask, 잔차 gradient 분기)까지 포함한다.
   - AMP·weight_norm·FP32 master 경계는 v0.8 `direct_contract` 방식으로 검증한다.
   - 메모리 이동이 실제로 줄었는지 kernel 시간과 함께 보고한다.
3. **native로 남은 Conv**: C128/K11 dW, C192→384/K5 dW, ConvTranspose1d 업샘플(fwd/dX/dW), discriminator period Conv2d를 v0.8과 같은 방식(방향별, 가장 빠른 정확한 native 기준)으로 다시 본다. 이길 수 없는 이유가 하드웨어 한계인지 구현인지 roofline으로 구분한다.
4. 채택 후보는 v0.8 B 위에 얹는 B′로 만든다.

### 2.B 5070과의 차이 분해

- 2.A의 최종 분해에서 kernel마다 이 PC의 dot2 실측 상한(48.7 TFLOP/s)과 메모리 상한(436.7 GB/s) 대비 활용률을 계산한다.
- 공개된 RTX 5070 사양으로 같은 kernel 계열의 이론 하한을 추정한다. FP16 텐서 코어 처리량, 메모리 대역폭, cuDNN의 implicit GEMM 유무를 쓴다.
- "같은 알고리즘이면 5070이 빠를 수밖에 없는 몫"과 "이 PC의 소프트웨어가 아직 못 쓰는 몫"을 분 단위 표로 나눈다.
- 5070 12분은 외부 사용자 보고라 조건이 같다는 증거가 없다. 그 불확실성을 표에 적는다. 사양은 출처 URL을 붙인다.

### 2.C TunableOp 비결정 원인

1. 프로세스 안 비결정(Qwen 또는 MOSS 중 하나)을 먼저 재현한다. TunableOp를 켠 상태에서 같은 입력의 개별 GEMM 출력을 두 번 비교해, 다른 결과를 내는 GEMM shape를 찾는다.
2. 그 shape에서 선택된 솔루션 이름(`results0.csv`의 솔루션 번호 → rocBLAS/Tensile 커널 이름)이 split-K(GSU>1)나 atomic 계열인지 확인한다.
3. 프로세스 간 비결정(Anima)도 같은 방법으로 원인 shape를 찾는다. 파일에 없는 shape인지, 같은 솔루션 안의 비결정인지 가른다.
4. 해결책 후보를 세운다. 그런 솔루션을 파일에서 빼거나, 결정적 솔루션만 남기는 필터를 둔다. 필터 뒤의 속도(특히 Anima 1.79배가 유지되는지)와 재현성을 잰다.

### 2.D Strata warm 병목

- `STRATA_PREFILL_V07_KO.md` §8의 재개 조건(운영 서비스가 이미 꺼져 있음, 같은 swap 보호 관문)을 그대로 지킨다.
- 시작 전에 MemAvailable·swap 사용량·file cache를 기록한다. 관문에 다시 걸리면 그 자료를 남기고 멈춘다. 관문을 완화하지 않는다.
- 측정이 가능하면 warm 9.7K fresh 요청의 GPU 계산·H2D bytes·host stall 비율과 J off/on A/B(처리량·token ID 동일)를 낸다.

## 3. 판정 기준

- 연산: 고정 seed, FP32 참조, y/dX/dW finite, NRMSE ≤ 0.002, max error / reference RMS ≤ 0.02(v0.8 `PREDECLARED_GATES.json`과 같다). 기준은 실행 전에 파일로 저장한다.
- Applio epoch: **운영 경로(`core.py train`)** 에서 같은 세션 안에 B와 B′를 교대로 돌리고, steady epoch 중앙값을 낸다. 하네스 숫자만으로 판정하지 않는다.
- Applio 최종: B′가 B보다 의미 있게 빠르면 운영 경로 100 epoch B·B′를 학습한다. 같은 네 문장으로 익명 A/B 청취 파일(v0.8 `LISTEN_BLIND.html` 형식)을 만들어 사용자 판정을 받는다. 100 epoch는 시작 직전에 예상 시간을 알린다.
- TunableOp: 원인을 "확인된 사실 / 추론"으로 나눈다. 필터 후보는 속도와 같은 seed 재현성을 둘 다 잰다.
- Strata: token ID 동일, 처리량 중앙값. 관문 중단은 실패가 아니라 "미판정 + 자원 자료"로 보고한다.

## 4. 지켜야 할 것

- 운영 파일을 수정하지 않는다. 대상은 `applio-lab`, 각 랩의 런처·노드·설정, `~/.config/miopen`이다. 실험은 연구 폴더 사본이나 새 모델 이름에서 한다. 운영 경로 측정은 운영 코드를 읽기만 하고 새 모델 이름으로 실행한다.
- 기존 Applio 모델 이름(`voice_model_A` 등)으로 `--cleanup`을 쓰지 않는다. `lab/train_bench.sh`는 기본으로 `--cleanup`과 `MODEL=bench40k`를 쓰므로 그대로 실행하지 않는다.
- **Strata 서비스는 시작·중지·재시작하지 않는다.** 서비스가 꺼져 있을 때만 별도 worktree·build·port로 측정한다.
- 운영 venv에 패키지를 설치하지 않는다. 100 GB 미만 다운로드는 승인 없이 해도 되고, 파일명·출처·크기를 보고서에 적는다.
- GPU 작업은 하나씩 순서대로 한다. `gpu_guard_v06.GpuAudit`를 쓰고, 겹친 측정은 버린다. NORMAL 또는 FP32 Conv 참조에는 안전 변수 두 개를 함께 둔다. 튜닝·새 커널 실험에는 `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1`을 둔다.
- 비밀번호·키·토큰은 출력·저장하지 않는다.
- 세션이 길어질 수 있다. 단계 하나가 끝날 때마다 보고서를 갱신해서, 다음 세션이 그 파일만 읽고 이어 갈 수 있게 한다(v0.8에서 실제로 한 번 끊겼다).

## 5. 산출물

- `rdna2-ops/results/v10/`에 항목별 보고서를 둔다(`APPLIO_V10_KO.md`, `GAP_5070_KO.md`, `TUNABLEOP_DETERMINISM_KO.md`, `STRATA_WARM_KO.md`). 확인된 사실 / 추론 / 미검증을 나누고, 원자료 경로와 재현 명령을 붙인다.
- 카탈로그·가이드는 직접 고치지 않고 `V10_CATALOG_GUIDE_PROPOSAL_KO.md`에 반영 문안을 둔다. Claude가 재현한 뒤 반영한다.
- Claude가 재현할 핵심 숫자 5개와 그 숫자를 읽는 명령(GPU를 쓰지 않는 것)을 보고서 끝에 둔다.
- 운영 파일 보호 검사의 전후 SHA를 남긴다. 범위는 v0.8 `protected_before.json`과, 2026-11-08에 바뀐 `applio-lab`의 `train.py`·`lab_hooks.py`·`rdna2_v08/`·두 런처다.

같이 힘내자 GPT.
