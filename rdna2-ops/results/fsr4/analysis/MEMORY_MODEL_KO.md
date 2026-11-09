# FSR4 exact Native1080 / Native4K 메모리 모델

작성: 2026-11-09. GPU 측정 없이, 실제 exact profile의 PSO/DXIL·dispatch 기록과 고정된 상류 소스를 대조한 분석이다. 대상은 이 연구의 커뮤니티 exact DLL이다. 다른 DLL이나 게임의 트래픽을 대신 측정한 결과가 아니다.

## 결론과 적용 범위

12개 신경망 pass의 **논리 INT8 활성값 입력+출력 payload 합계**는 Native1080에서 **132,710,400 B**, Native4K에서 **530,841,600 B**다. 같은 값을 다른 pass가 읽으면 다시 센 값이다. 이것은 DRAM 전송량이나 하나의 전체 버퍼 크기가 아니다. 각 pass 내부에서 동일 입력을 convolution과 residual이 재사용하는 경우에는 같은 논리 입력으로 한 번만 센다.

실제 DRAM read/write byte는 현재 **미측정**이다. 따라서 이 수치나 ISA의 load/store 바이트를 기존 copy 측정값 436.7 GB/s로 나눈 수치를 DRAM 시간 하한으로 제시하지 않는다. 같은 입력을 읽는 많은 명령은 캐시에서 처리될 수 있고, pass 사이의 중간 텐서도 캐시에 남을 수 있다. 반대로 cache-line 과잉 읽기, 쓰기 정책, 반복 miss, 페이지/텍스처 배치 때문에 실제 전송량이 커질 수도 있다. 현재 자료만으로 그 양을 정할 수 없다.

이 문서는 계산 가능한 activation payload를 제공하고, 원래 요청의 `bytes / bandwidth` 하한 칸을 **측정 불가**로 남기는 근거다. dot4 작업량은 별도의 연산 하한에 사용할 수 있으며, 유효 invocation과 loop 횟수도 아래에서 확인한다.

## 근거와 재현

- 실제 PSO→DXIL 연결: `analysis/exact_profile_pso_map.json`. 관측된 PSO DXBC 전체 SHA256과 vkd3d dump 전체 SHA256 일치에 기반한다.
- 실제 셰이더: `pipelines/exact_profile/*.dxbc`, `dumps/exact_profile/*.dxil`.
- 실제 dispatch와 시간: `measurements/exact_profile_passes.csv`. 여기의 pass 시간은 계측 실행의 값이며 최종 전체 시간 판정용 교대 표본과 구분한다.
- 실제 ACO 통계: `analysis/exact_profile_isa.json`. raw instruction 경계를 복원해 ACO code size/instruction 수와 대조한 자료다.
- 상류 구조: [every-pass.md](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/blob/d1c3b30db656bf6a38634642542ee7736127d211/research/wmma/every-pass.md), [gen_down.py](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/blob/d1c3b30db656bf6a38634642542ee7736127d211/research/wmma/gen_down.py), [gen_deep.py](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/blob/d1c3b30db656bf6a38634642542ee7736127d211/research/wmma/gen_deep.py).
- 외부 텍스처 형식: `harness/fsr4time_local.c`의 Texture 생성과 FFX resource 선언. color/output RGBA16_FLOAT, depth R32_FLOAT, motion RG16_FLOAT다.

상류 WMMA 연구의 구조 설명은 4.1.1에서 얻은 것이다. 현재 exact 24개 network PSO에서 픽셀 범위, store 폭과 반복 수, 실제 7/8/9 loop를 다시 확인하여 현재 tensor shape에 적용했다. 상류의 다른 GPU 시간이나 WMMA 성능 수치를 이 PC 실측으로 사용하지 않았다. 현재 exact의 pass11은 SDK2.4 RX6000 셰이더로 바뀌었으므로 옛 pass11의 명령별 메모리 행동까지 같다고 가정하지 않는다.

재현 명령(기존 도구만 이용, CPU 전용):

```sh
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops/results/fsr4
python3 analysis/build_memory_model.py
```

결과 `analysis/MEMORY_MODEL.json`에는 각 PSO SHA, early guard, mask별 raw store 개수, 논리 I/O, loop 근거, 물리 traffic의 null 상태가 들어 있다. 생성 스크립트는 PSO·bitcode SHA와 예상 shape/store 바이트를 검사하고, 다르면 중단한다.

## 텐서와 바이트의 정의

출력 W×H일 때 level1은 (W/2)×(H/2)×16, level2는 (W/4)×(H/4)×32, level3은 (W/8)×(H/8)×64의 INT8 텐서다. 각 원소는 1 byte다. 채널은 16개 단위 plane으로 나뉘고, plane의 행 주소 간격에는 좌우 1픽셀 border가 포함된다. 아래 payload는 내부 유효 픽셀만 센다.

| 텐서 | Native1080 shape | 내부 payload B | Native4K shape | 내부 payload B |
|---|---|---:|---|---:|
| L1 | 960×540×16 | 8,294,400 | 1920×1080×16 | 33,177,600 |
| L2 | 480×270×32 | 4,147,200 | 960×540×32 | 16,588,800 |
| L3 | 240×135×64 | 2,073,600 | 480×270×64 | 8,294,400 |

1픽셀 border를 포함한 기하학적 크기는 `(w+2)(h+2)C`다. 실제 allocation 크기에는 별도 정렬과 버퍼 재사용이 있으므로 이 식도 allocation 측정값으로 쓰지 않는다. border clear, weights/bias, 상수, 텍스처, LDS, scratch, instruction fetch, cache-line overfetch는 아래 표에 포함하지 않았다. weight가 DXIL/ISA의 상수로 들어가는 경우도 있어서 네트워크의 weight 원소 수를 그대로 VRAM read byte에 더할 수 없다.

## Pass별 논리 activation I/O

표의 `입력 → 출력`에서 `+skip`은 별도 논리 입력 텐서다. 단위는 **byte**이며, 각 해상도의 숫자는 `고유 입력 내부 payload / 출력 내부 payload`다. 같은 pass의 반복 tap 읽기와 동일 입력을 사용하는 residual 읽기는 중복 합산하지 않는다.

| Pass | 입력 → 출력 | Native1080 read / write B | Native4K read / write B | dot4 / 유효 invocation |
|---|---|---:|---:|---:|
| 1 | L1 → L1 | 8,294,400 / 8,294,400 | 33,177,600 / 33,177,600 | 832 |
| 2 | L1 → L1 | 8,294,400 / 8,294,400 | 33,177,600 / 33,177,600 | 832 |
| 3 | L1 → L2 | 8,294,400 / 4,147,200 | 33,177,600 / 16,588,800 | 512 |
| 4 | L2 → L2 | 4,147,200 / 4,147,200 | 16,588,800 / 16,588,800 | 1,600 |
| 5 | L2 → L2 | 4,147,200 / 4,147,200 | 16,588,800 / 16,588,800 | 1,600 |
| 6 | L2 → L3 | 4,147,200 / 2,073,600 | 16,588,800 / 8,294,400 | 2,048 |
| 7 | L3 → L3 | 2,073,600 / 2,073,600 | 8,294,400 / 8,294,400 | 5,248 |
| 8 | L3 → L3 | 2,073,600 / 2,073,600 | 8,294,400 / 8,294,400 | 5,248 |
| 9 | L3 + skip L2 → L2 | 6,220,800 / 4,147,200 | 24,883,200 / 16,588,800 | 7,296 |
| 10 | L2 → L2 | 4,147,200 / 4,147,200 | 16,588,800 / 16,588,800 | 1,600 |
| 11 | L2 + skip L1 → L1 | 12,441,600 / 8,294,400 | 49,766,400 / 33,177,600 | 2,112 |
| 12 | L1 → L1 | 8,294,400 / 8,294,400 | 33,177,600 / 33,177,600 | 832 |
| 합계 | 각 pass의 read+write 합계 | **132,710,400** | **530,841,600** | 해상도별 총합은 JSON |

이 표는 신경망의 의미상 데이터 경계를 설명한다. 패스들이 같은 임시 버퍼 영역을 번갈아 사용하더라도, 서로 다른 시점의 텐서들은 별도 payload로 센다. pass3/6은 stride2 2×2 downsample, pass9/11은 한 invocation이 2×2 출력 위치를 만드는 upsample이므로 출력 payload를 입력 grid와 같은 크기로 계산하면 틀린다.

## 실제 DXIL과 대조한 부분

Network의 첫 early guard가 정한 유효 좌표는 아래와 같다. dispatch의 남는 lane은 이 guard로 neural work를 건너뛴다.

| Pass 묶음 | Native1080 valid x,y | Native4K valid x,y | 유효 invocation 수 1080 / 4K |
|---|---|---|---:|
| 1,2,12 | x≤959, y≤539 | x≤1919, y≤1079 | 518,400 / 2,073,600 |
| 3,4,5,10,11 | x≤479, y≤269 | x≤959, y≤539 | 129,600 / 518,400 |
| 6,7,8,9 | x≤239, y≤134 | x≤479, y≤269 | 32,400 / 129,600 |

각 valid invocation의 출력은 pass1/2/12가 16 B, pass3/4/5/10이 32 B, pass6/7/8이 64 B다. pass9는 4개의 상위해상도 위치×32 B=128 B, pass11은 4×16 B=64 B다. 실제 `rawBufferStore.i32` mask와 loop 반복 수를 곱한 바이트가 표의 출력 텐서 크기와 모든 24개 PSO에서 일치했다. 이 확인은 shape·store 폭에 대한 정적 확인이며, 실제 store transaction이나 버스 전송량 계측은 아니다.

### 7/8/9의 동적 dot4

실제 dispatch의 groups_z=1이므로 `groupId.z=0`이다. 7/8의 DXIL은 groupId.z+7, +31, +15까지 inclusive로 반복한다. 이는 8,32,16회다. 첫 grouped3×3 구간은 inner4 반복이 ACO에서 전개되어 outer 반복당 dot4 144개, 이어지는 1×1 두 구간은 64개와 128개다.

`pass7=pass8=144×8 + 64×32 + 128×16 = 5,248 dot4 / valid pixel`.

pass9는 같은 세 구간에 upsample 구간 `64×2×2×8=2,048`을 더한다.

`pass9=5,248+2,048=7,296 dot4 / valid pixel`.

따라서 ISA의 static dot4 336/336/400을 단순히 invocation 수와 곱하면 크게 과소계산한다. 상류의 MAC 설명 20,992/20,992/29,184와도 일치한다(한 dot4가 INT8 곱셈-누산 4개).

### 최종 postpass의 grid

postpass는 group마다 반해상도 16×16 좌표를 만들고, cbuffer1 row4의 x/y를 각각 2로 나눈 한계를 먼저 검사한다. 그 구간의 valid invocation당 dot4 call은 960개다. Native case의 출력 W,H가 이 상수에 대응한다는 전제에서 유효 invocation은 L1과 같은 518,400 / 2,073,600이다. CB payload 자체는 별도로 캡처하지 않았으므로 이 전제를 JSON에 남겼다. 실제 row 주소 stride는 1080=15392 B, 4K=30752 B로 L1 layout과 맞는다.

추가 `bb.*` 코드는 커뮤니티의 store 재배열 부분이다. 이에 따라 전체 dispatch의 256 thread×group 수가 곧 모든 thread의 neural dot4 실행 수인 것은 아니다. 후속 store와 neural 계산의 guard를 구분해야 한다.

## 반복 명령 바이트와 DRAM을 구분하는 예

pass1의 3×3×16채널 입력은 유효 위치당 144개의 논리 tap byte를 요구한다. 1080 L1의 전체 tap 요구량은 `960×540×144=74,649,600 B`지만 입력 tensor 내부의 고유 payload는 8,294,400 B다. 인접 위치가 같은 입력을 공유하므로 이 둘부터 다르다. halo·residual·weights·compiler 최적화·cache의 동작까지 합치면 실제 명령 요청량과 물리 DRAM byte는 다시 달라진다.

`MEMORY_MODEL.json`의 static raw load/store mask는 그 차이를 추적할 출발점이다. loop 횟수를 반영하지 않은 DXIL call-site 개수이고, ACO가 제거하거나 scalar load로 바꿀 수 있어 실행 바이트로 간주하지 않는다. ISA 통계 역시 scalar broadcast, active EXEC mask, loop, texture sampling, cache 재사용을 반영하기 전에는 물리 traffic이 아니다.

## 현재 숫자를 채울 수 없는 칸

| 항목 | 현재 상태 | 이유 |
|---|---|---|
| Auto exposure 전체 read/write | 미측정 | 이미지 샘플링, mip/중간값, atomic 등 전체 resource·동적 요청량 미계측 |
| Prepass 전체 read/write | 미측정 | color/depth/motion/history와 임시 출력의 descriptor·sample 경로 전체를 복원하지 않음 |
| Postpass 전체 read/write | 미측정 | 네트워크 L1 외 temporal/resource·texture 접근이 있어 public output만으로 전체량을 정할 수 없음 |
| 13개 `_post` 경계 처리 | 미측정 | 논리 interior 계산에서 제외. 실제 active mask·주소·쓰기를 전부 세지 않음 |
| 각 pass의 cache hit/miss·물리 DRAM byte | 미측정 | 해당 하드웨어 traffic counter나 동등한 측정 없음 |
| `bytes / 436.7 GB/s` 시간 하한 | 산출하지 않음 | 기존 copy 실측과 다른 경로이며, 분자에 사용할 실제 DRAM byte가 없음 |

알고 있는 public resource payload는 Native1080 color16,588,800 B, depth8,294,400 B, motion8,294,400 B, output16,588,800 B다. Native4K는 각각 4배다. 이는 한 현재 프레임의 texture 내부 원소 크기다. 여러 입력 phase, 이전 프레임의 history, 메모리 배치와 실제 읽기 횟수를 포함한 allocation/traffic 합계가 아니다. 출력은 RGBA16F이므로 raw 비교의 public 출력 payload도 8 B/pixel이다.

## 병목 판단에 사용할 수 있는 범위

두 해상도 사이에서 논리 activation I/O와 유효 neural 작업량은 정확히 4배다. 계측 시간의 배율이 4와 다르면 고정 dispatch 비용, 실행 점유율, cache 재사용, 실제 클럭 등과 함께 해석해야 한다. 그 배율만으로 DRAM 포화 또는 cache 용량 한계를 확정할 수 없다.

현재 자료로는 **메모리 하드웨어 상한 근접 여부를 정량 판정할 수 없다**. dot4 처리량 대비 연산 하한, 실제 pass 시간, occupancy·spill, 최종 전체 시간 변화를 합쳐 후보를 평가하는 것은 가능하다. 추가 counter가 없는 상태에서 메모리 시간 하한을 꾸며 채우는 것보다 이 미측정을 보존하는 쪽이 재현 가능한 결론이다.
