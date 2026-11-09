# FSR4 연구의 카탈로그·가이드 반영 제안

작성: 2026-11-10 KST. 이 문서는 반영 제안이며 기존 카탈로그와 가이드는 수정하지 않았다.

검토한 현재 정본은 `~/DevSpace/rdna2-pro-research-20261102/RDNA2_OP_CATALOG_KO.md` v0.7과 `~/DevSpace/rdna2-pro-research-20261102/rdna2-ops/RDNA2_OPTIMIZATION_GUIDE_KO.md` v1.1이다. 요청서의 v0.6/v1.0보다 최신판이었다. 아래는 해당 구조에 추가할 문안이며 기존 Applio·MOSS·Strata 결과를 대체하지 않는다.

## 1. 카탈로그에 추가할 행

| 연산/경로 | dtype·backend·shape | 확인된 결과 | 판정·범위 | 근거 |
|---|---|---|---|---|
| 실시간 업스케일러 신경망 | Full INT8 FSR4.1.2, Proton→vkd3d-proton3.1.0→RADV/ACO Mesa26.2.4, RX6900XT | SDK 원본 셰이더 대비 community exact 전체시간6.074~7.424% 감소, 28쌍 최종 raw frame byte동일 | SDK 합성 Native1080/4K 정지·움직임. 실제 게임 provider/FPS 미검증 | A+B, `rdna2-ops/results/fsr4/FSR4_RDNA2_KO.md` |
| INT8 round/clamp/pack 표현 변환 | 현재 exact SPIR-V20개, Native1080/4K | exact 대비1080moving1.9134%,4Kstill1.3735%,4Kmoving1.4334% 감소. Probe4+primary28+temporal64 byte비교 일치 | 세 조건의 exact 후보 통과. 1080still은CI에0 포함, 미채택. 운영 적용 없음 | A+B, `analysis/clamp_all_primary_analysis.json`, `CANDIDATE_ISA_COMPARISON.json` |
| Vulkan signed dot4 관측 기준 | wave32,4/8/16/32 accumulator | best-variant median91.679754437TOPS,8integer ops/dot4 | 해당 프로브 기준. wave64 FSR의 보편적 하드웨어 상한으로 일반화 금지 | A+C, `runs/dot4_steady/`, `ceiling/ISA_VALIDATION.json` |
| 재학습 lossy 업스케일러 | current community lossy | exact 대비14.899~23.096% 시간 감소, 출력은다름 | 익명 정지·움직임·밝기 A/B 제공. 사용자 지각 판정 전, 미채택 | A, `captures/ab_review/index.html` |

수치의 %는 같은 round의 상대 감소율 중앙값이다. FPS 개선율이 아니며, 서로 다른 warmup 집합의 절대 시간을 교차 비교하지 않는다.

## 2. 가이드 신규 절 제안: §9.23 Vulkan 그래픽 셰이더의 INT8 추론

### 9.23.1 진입 경로와 provider를 먼저 확인한다

FSR SDK 2.4의 DLL 파일 버전이 4.1.2.4555여도, 이 RX6900XT/Proton 경로의 무수정 DLL은 자동으로 3.1.6을 선택했다. Navi2 IsSupported의 Windows private adapter 질의가 Wine에서 미구현인 것이 실제 trace와 정적 분석으로 연결됐다. 연구 사본의 CPU predicate 6바이트만 바꾸고 모든 1778개 DXBC와 나머지 바이트를 보존했을 때 실제 Navi2 provider 4.1.2가 선택됐다.

이 결과를 무수정 SDK의 FSR4 성능 또는 실제 Fallout4 provider 확인으로 표현하지 않는다. 실제 게임은 D3D11과 플러그인의 DX12 context를 함께 쓰므로, 운영 context를 직접 조회하거나 동일 경로의 shader evidence가 있어야 적용 효과를 주장할 수 있다.

### 9.23.2 기존 실행 경로를 재사용하고 전체 구간을 잰다

커뮤니티 timing-kit child의 RunOne·입력·자원·FFX dispatch·타임스탬프를 보존하고 Native AA와 raw 증거를 추가했다. pass 계측은 timestamp를 더하므로 후보 채택은 별도 일반 totals 모드로 판단한다. shader 시간 합, dispatch 사이 gap, 전체 GPU span을 구분한다. gap 전체가 barrier 단독 비용인 것은 아니다.

원본/exact/lossy 기준선은 warm40/timed200×7 round이다. 이후 pass 진단에서 초기 시간 하강이 남아, 첫 정식 후보 pair 전에 양쪽 warmup 400을 고정했다. 원인이 clock·cache 중 무엇인지는 확정하지 않았다. 이 두 집합의 절대 시간을 직접 비교하지 않는다.

### 9.23.3 정적 ISA 개수에서 동적 작업량으로

실제 기계어는 signed v_dot4_i32_i8/v_dot4c_i32_i8이고 dot와 누산은 이미 결합돼 있었다. pass7/8의 static 336개는 loop에 의해 유효 thread당 5248개, pass9의 400개는 7296개가 된다. inactive border lane과 up/downsample의 서로 다른 grid도 반영한다.

network12+post는 1080p 27.1724544 GOP, 4K 108.6898176 GOP다. Vulkan probe 91.679754437 TOPS로 나눈 조건부 dot 기준 시간은 0.296384/1.185538 ms이며, probe와 shader의 wave·memory·packing·clock 조건 차이를 명시한다. 물리적 절대 하한이라고 단정하지 않는다.

동시에 logical tensor payload, 반복 tap 요청, 실제 DRAM transfer를 구분한다. network 12 pass의 activation I/O 합 132,710,400/530,841,600 B는 실제 DRAM bytes가 아니다. physical traffic counter가 없으면 기존 copy 436.7 GB/s로 나눈 수치를 실측 하한으로 제시하지 않는다.

### 9.23.4 register와 occupancy 숫자는 정의부터 확인한다

community post는 SDK 원본보다 VGPR 72→128, LDS 0→16 KiB로 늘고 RADV 정적 자원 계산 14→8로 낮아졌지만 전체 시간은 개선됐다. 공동 store 설계가 자원 숫자만으로 드러나지 않는 사례다.

RADV 26.2.4의 Subgroups per SIMD는 32-lane 환산값이다. wave64에서 8은 내부 4 wave 자원 계산에 해당하며 실제 달성 occupancy가 아니다. 함수는 환산 후 16으로 다시 상한 처리하지 않으므로 18/20/24를 관측 resident wave로 읽지 않는다. pass6은 scratch 0이어도 SGPR 4개를 VGPR lane으로 spill한다.

일부 SDWA/SMEM 역어셈블 오류가 다음 word를 가짜 명령으로 보이게 했다. raw machine word 경계를 복원해 Code size·ACO instruction 수와 대조했다. 단순 텍스트 줄 수는 dot4와 packing count를 틀리게 할 수 있다.

### 9.23.5 이번 추가 후보와 최종 판정

현재 exact ISA에는 shift→signed-byte clamp 패턴이 남았다. 기존 community model_clamp 변환을 현재 dump에 포팅하면 20개 shader에서 명령이 줄고 dot4·register/LDS 할당은 유지됐다. 의도한 20개 기계어만 바뀌고 36개는 byte 동일해서 actual override도 확인했다.

| Native AA 조건 | paired gain | 95% CI | 판정 |
|---|---:|---|---|
| 1080p 정지 | 1.9396% | [−0.4577,1.9825]% | 미채택: 측정 분포에서 이득 미확정 |
| 1080p 움직임 | 1.9134% | [1.8141,2.0009]% | 조건부 exact 성공 |
| 4K 정지 | 1.3735% | [1.3423,1.5065]% | 조건부 exact 성공 |
| 4K 움직임 | 1.4334% | [1.3251,1.6454]% | 조건부 exact 성공 |

이는 현재 SDK 합성 입력에서의 결론이다. 게임 통합·임의 입력의 보편적 동일성·지각 판정까지 확장하지 않는다. 1080 still의 큰 음수 표본은 삭제하지 않았다. 외부 Steam engine 활동이 관측된 r05는 실행창 오염 의심으로 양쪽 전체 pair를 분리했고, 동일 순서 한 번의 보충만 사전 선언했다. 원본 7쌍·제외 6쌍·보충 7쌍 민감도를 모두 남겼다.

## 3. 기존 §1.2 / §9.7 / §10에 보충할 판단 문안

- **§1.2:** exact라도 shader 단독·마이크로벤치 개선만으로 채택하지 않는다. 같은 history의 최종 raw 출력과 일반 전체 구간을 비교한다. lossy는 수치 오차가 작아도 사용자 지각 판정을 유지한다.
- **§9.7:** register 감소, scratch 0, 정규화 subgroup 수를 각각 별도로 읽는다. SGPR→VGPR lane spill과 정적 자원 계산의 환산 정의를 확인한다.
- **§10:** KFD client 0은 Vulkan 작업 부재의 증거가 아니다. 실제 DRM client와 engine 변화·kernel fault를 함께 기록하되 표본 간격과 counter 없는 client 한계를 명시한다.
- **§10:** override 환경 변수를 설정한 것만으로 적용을 인정하지 않는다. 실제 PSO/DXIL/SPIR-V hash·ABI와 바뀐 기계어의 대상을 대조한다.
- **§10:** 큰 성능 이상값과 독립적으로 관측된 외부 GPU 활동을 구분한다. 제외·보충 규칙은 결과를 보고 계속 바꾸지 않고 원본·민감도를 보존한다.

## 4. 반영 전 상태와 원자료 입구

기존 두 정본은 수정하지 않았다. 제안 문안의 수치는 `FSR4_RDNA2_KO.md`와 `analysis/KEY_NUMBERS.json`, `clamp_all_primary_analysis.json`을 따른다. 실제 셰이더는 `analysis/candidate_clamp_shift_all/override/`, 품질 비교는 `captures/ab_review/index.html`에 있다. 게임 적용, lossy 채택, 광범위한 새 모델 일반화는 이 제안에 포함하지 않는다.
