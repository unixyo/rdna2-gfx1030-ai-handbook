# FSR 4.1.2 INT8 — RX 6900 XT / Linux 연구

작성·최종 갱신: 2026-11-10 KST / 2026-11-09 UTC. **상태: 이번 연구 범위 완료. SDK 직접 경로의 조건부 exact 이득 확인. 게임 적용·게임 품질 판정은 미수행.**

정본 폴더: `~/DevSpace/rdna2-pro-research-20261102/rdna2-ops/results/fsr4/`. 이 보고서의 상대 경로는 이 폴더 기준이다. 기존 카탈로그 v0.7·가이드 v1.1은 수정하지 않았고, 반영 문안은 `FSR4_CATALOG_GUIDE_PROPOSAL_KO.md`에 분리했다.

## 1. 결론과 채택 범위

커뮤니티 최신 exact보다 더 줄일 여지가 있었다. 현재 4.1.2 exact의 실제 SPIR-V에 커뮤니티의 기존 clamp/shift 변환을 포팅한 `clamp_all`은 **1080p Native AA 움직임에서 전체 업스케일러 시간 1.9134%, 4K 정지에서 1.3735%, 4K 움직임에서 1.4334% 감소**를 사전 수치 gate로 확인했다. 새 최적화 원리를 발명했다는 주장은 아니다. 기존 4.1.1용 변환을 현재 4.1.2/RADV 경로에서 추가 적용하고, 실제 기계어·출력·전체 시간을 검증한 결과다.

**1080p 정지는 미채택이다.** paired gain 중앙값은 +1.9396%였지만 95% CI가 [−0.4577%, +1.9825%]라서 이득 확정 기준을 통과하지 못했다. 큰 시간 변동을 보인 r04 표본을 제거하지 않았다. 이 조건의 실패 분류는 **측정 불가: 현재 반복 분포에서 작은 이득을 확정할 수 없음**이다. 0% 효과 또는 하드웨어 한계가 증명된 것은 아니다.

후보의 원본 exact 대비 출력은 probe 최종 프레임 4쌍, 주분석 최종 프레임 28쌍, 움직임·밝기 변화 연속 프레임 64쌍에서 모두 전체 바이트 일치했다. 총 96회의 raw frame 비교이며, 반복 round의 동일 입력·history 비교도 포함된다. 임의 게임 장면 전체에 대한 보증은 아니다.

실용적으로는 **SDK 합성 입력의 4K 두 장면과 1080p 움직임에서 조건부 성공한 연구 후보**로 남긴다. 실제 Fallout4 provider·성능·화질은 미검증이고, 운영 설치나 자동 적용은 하지 않았다.

## 2. 실제 실행 경로와 중요한 호환성 발견

| 항목 | 직접 확인한 값 | 근거 |
|---|---|---|
| GPU | RX 6900 XT, Navi21/gfx1030, PCI 1002:73af, 0000:03:00.0 | `runs/device_query_02/`, timing-kit G 행 |
| Vulkan 드라이버 | RADV NAVI21, Mesa 26.2.4-1ubuntu1 | `runs/device_query_02/stdout.log` |
| Proton 경로 | Proton Experimental → vkd3d-proton 3.1.0, build cae2e003bb82516 → RADV/ACO | `runs/proton_probe/stderr.log`, 각 run stderr |
| D3D12 | Shader Model 6.8, 별도 연구 prefix | timing-kit G 행, `runs/*/command.json` |
| 주요 입력 | 1920×1080→1920×1080 / 3840×2160→3840×2160 Native AA, jitter 8 phases | 각 `measurements/*.txt` CASE 행 |

### 2.1 무수정 SDK는 실제로 3.1.6을 선택했다

공식 SDK 2.4 upscaler 사본의 파일 버전은 4.1.2.4555이지만, 자동 context query는 **3.1.6**을 선택했다. 제공 목록도 3.1.6과 2.3.4뿐이었다. DLL 파일 이름·버전 문자열만으로 실제 FSR4 사용을 판정하면 틀리는 사례다. 근거: `bench/original/query_confirm.txt`, `runs/sdk_auto_confirm/`.

Navi2 provider의 `IsSupported`는 AMD Windows private adapter 정보에 의존한다. 원본 SDK 실행의 `runs/sdk_original_d3dkmt/stderr.log`에는 `NtGdiDdDDIQueryAdapterInfo type 0 not handled`가 두 번 있다. Type 0은 UMDRIVERPRIVATE이다. 원본 DLL의 호출 구조와 고정 Wine 소스도 이 연결을 뒷받침한다. [Wine D3DKMT 소스](https://github.com/ValveSoftware/wine/blob/7b3ceaf8644bb5d57140903e12897efac69be16a/dlls/win32u/d3dkmt.c), 상세 분석 `analysis/SDK_PROVIDER_GATE_KO.md`.

### 2.2 원본 셰이더 비교에 사용한 연구 사본

`tools/force_navi2_compat.py`는 연구 폴더의 SDK 사본에서 Navi2 CPU 지원 판정 함수 시작 6바이트만 `mov eax,1; ret`로 바꾼다. 원본 SHA·75바이트 함수·RTTI/vtable slot·패치 위치를 검사하고, **1778개 DXBC occurrence / 1389개 고유 SHA와 수정 구간 밖 모든 바이트가 그대로임을 검증**했다. `sdk/navi2_compat/PROVIDER_COMPAT_MANIFEST.json`에 전후 증거가 있다.

이 사본은 실제 provider **4.1.2 / f5a5ca1e01001002**를 선택하고 두 Native 해상도의 전체 dispatch를 완료했다. 표에서 ‘SDK 원본 셰이더’는 이 **Navi2 호환 사본**을 뜻한다. 무수정 공식 runtime이 이 Linux 경로에서 FSR4를 자동 선택했다는 뜻이 아니다.

| 대상 | SHA256 | 역할 |
|---|---|---|
| SDK 4.1.2.4555 원본 DLL, 57,664,272 B | `aad4fccccb29084a655f9a4f9f613bd22c8c03fb4a0f8840148cc6543b4ccace` | 보존 사본, 무수정 자동 선택 3.1.6 |
| SDK Navi2 호환 사본 | `c3fc8ce90d6294b68fd5845e8867be797048c459b4fecb307c85c15fcfac7d69` | 원본 셰이더 FSR4 기준선 |
| 커뮤니티 exact DLL | `08dfca86c15bb1d4e8d12740e68df6f054f12434f0d9d254a76387b86a621ea0` | 추가 최적화의 기준, provider 4.1.2-exact-r2-cyboman |
| 커뮤니티 lossy DLL | `2ae6d5c6ce6eded5bd78e1f9813dd0eefb34690d213abef311258ba632d9d432` | 별도 품질·성능 비교 |
| 동결 측정 EXE | `bc4e94a781ee9b1670fb192bbdf2e7e58170f6c8f7bf2d54439789aacecf3f92` | 기존 kit RunOne/리소스/dispatch 경로 재사용 |

Fallout4는 기록상 D3D11이고 Upscaling 플러그인은 DX12 FSR context를 만든다. 플러그인의 d3d12/dxgi import와 기존 LINUX_NOTES는 이 구조를 뒷받침한다. **이번에 게임을 실행해 DXVK↔vkd3d interop와 선택 provider를 직접 추적하지는 않았다.** 따라서 이 연구의 SDK 경로 발견을 그대로 게임의 현재 provider에 대입하지 않는다.

## 3. 측정 방법·판정·실패 자료

`PREDECLARED_GATES.json`은 GPU FSR 측정 전에 저장했다. 출력 전체 바이트 일치와 함께, 같은 round의 전체 업스케일러 중앙값에서 계산한 7개 paired 상대 이득의 중앙값이 양수이고 그 분포의 2×비보정 MAD보다 크며, paired bootstrap 95% 하한이 0보다 커야 한다. bootstrap은 **프레임 200개가 아니라 round 7개**를 10,000회 재표집하며 seed는 20261110이다. paired gain은 `100×(reference−candidate)/reference`이다.

배포 kit에 Native AA가 없어, pinned community child를 측정 전용으로 기계적으로 파생했다. 기존 D3D12 자원 생성·합성 입력·FFX dispatch·타임스탬프 루프를 유지하고 Native 크기, 실제 provider query, 정렬 전 frame CSV, PSO dump, raw 출력, 진단 밝기 장면을 추가했다. 새 FSR 실행기를 만들지 않았다. parent/updater/network 제출 코드는 파생본에 없다. `harness/README.md`, `fsr4time_local.diff`, `BUILD.json`에 구현·SHA·빌드 근거가 있다.

| 측정 집합 | 준비 / 표본 | 반복 | 판정 용도 |
|---|---|---|---|
| SDK / exact / lossy 기준선 | warm40 / timed200 | 7round×3빌드×4조건, 16,800 frame | 같은 집합 안의 전체 시간 비교 |
| clamp_all 주 비교 | warm400 / timed200 | reference/candidate 교대 7pair×4조건, 주분석 11,200 frame | 추가 exact 이득 판정 |
| pass 진단 | warm40 / timed200 | 5빌드·후보, 각 2해상도 moving | dispatch·gap 분해, 전체 이득 gate에는 사용하지 않음 |
| 움직임·밝기 캡처 | warm40 / timed16 | 두 해상도, frame40..55 | 전체 raw byte 및 품질 비교, 성능 표본 아님 |

초기 warm4/timed8 진단의 4K 시간은 첫→끝 8.93% 내려갔다. warm40 pass 진단의 1080p에서도 초기 시간 하강이 남았다. 이에 **첫 정식 후보 교대 측정 전에 양쪽 warmup을 400으로 고정**했다. 원인이 clock인지 cache인지 계측 영향인지는 확정하지 않았다. 선언의 ‘clock ramp’ 문구는 관측된 시간 하강에 대한 가설로 읽어야 한다. baseline 40와 후보 400의 절대 ms를 서로 교차 비교하지 않는다.

### 3.1 외부 GPU 활동 한 쌍의 보충

원래 후보 r05 reference 실행창에서 steamwebhelper의 gfx 78.914 ms, dma 48.756 ms, compute 5.129 ms 증가를 관측했다. 이들은 각 engine의 누적 차이이며 합쳐 overlap duration으로 쓰지 않는다. 어느 timed case와 겹쳤는지는 알 수 없고 startup/capture 때였을 수도 있다.

사전 overlap 원칙에 따라 **실행창 외부 활동을 보수적으로 오염 의심으로 분류**하고, r05의 reference/candidate 네 조건을 모두 주분석에서 분리했다. 세부 counter 수치 규칙까지 처음부터 선언돼 있었다고 주장하지 않는다. 보충 전에 `analysis/clamp_all_SUPPLEMENTAL_DECLARATION.json`으로 동일 candidate→reference 순서·warm400/timed200·전체 pair 1회만 고정했고, 배경 engine 증가 시 연구 프로세스만 중단하도록 guard를 보강했다. 보충 두 실행은 정상 종료·관측 배경 증가 0이었다.

r04 candidate의 1080p 정지는 처음 50 frame 중앙값 0.72508 ms→마지막 50 frame 0.96318 ms, 전체 중앙값 0.98344 ms였다. 해당 paired gain −33.016% 표본은 별도 오염 근거가 없어 **그대로 포함**했다. 불리한 시간을 버리는 보정은 하지 않았다. 원본 14 run 및 원본 분석의 SHA도 보존했다.

### 3.2 GPU 감시 범위

연구 lock, 현재 게임 프로세스, KFD client, 대상 DRM fd와 engine counter, kernel journal cursor를 감시했다. 새 GPU reset/hang/ring timeout은 없었고, 관측 game/KFD client는 0이었다. KFD만으로 Vulkan 독점을 주장하지 않는다. 기본 desktop 클라이언트의 존재를 기록했고, Xwayland에는 engine counter가 없었다. 약 1초 간격 `/proc` 표본은 짧게 생겼다 사라진 클라이언트를 놓칠 수 있으며 일부 idle sysfs clock 조회는 null이었다.

첫 SDK query가 Wine의 분리 자식을 외부 작업으로 잘못 분류해 중단된 기록은 `runs/sdk_auto_query_1080/`에 남아 있다. 소유 표식·연구 prefix로 식별을 고친 뒤 재실행했으며 GPU reset이 아니었다. 운영 파일, 게임 prefix, MO2, mods/Data, linux-play.sh, fo4_probe.py는 수정하지 않았다. Strata 서비스도 시작·중지·재시작하지 않았고 운영 venv·시스템 패키지 설치·clock/power tuning을 하지 않았다.

## 4. 전체 업스케일러 시간

### 4.1 원본 셰이더·커뮤니티 기준선

ms는 round별 frame 중앙값 7개의 중앙값이다. %는 같은 round끼리의 상대 감소율 중앙값이므로 표의 두 ms를 나눈 값과 조금 다를 수 있다. 모두 합성 Native AA의 업스케일러 GPU 시간이며 게임 FPS가 아니다.

| 조건 | SDK 원본 셰이더 ms | 커뮤니티 exact ms | 커뮤니티 lossy ms | SDK→exact 감소 | exact→lossy 감소 |
|---|---:|---:|---:|---:|---:|
| 1080p 정지 | 0.75096 | 0.70152 | 0.59278 | 6.584% | 15.552% |
| 1080p 움직임 | 0.76162 | 0.70570 | 0.59834 | 7.424% | 14.899% |
| 4K 정지 | 3.38648 | 3.17892 | 2.45636 | 6.074% | 23.096% |
| 4K 움직임 | 3.39698 | 3.17038 | 2.45776 | 6.618% | 22.561% |

SDK→exact의 paired bootstrap 95% CI는 위 순서로 [6.533,6.735], [7.288,8.043], [6.006,6.291], [6.400,6.665]%다. SDK↔exact 최종 frame 239는 **28쌍 모두 전체 바이트 동일**했고 각 빌드의 7회 반복 출력도 동일했다. `analysis/baseline_analysis.json`, `baseline_rounds.csv`, `BASELINE_ANALYSIS_KO.md`가 분포·MAD·jitter군·감시·SHA 정본이다.

### 4.2 추가 후보: 현재 exact와 같은 조건의 교대 비교

| 조건 | exact reference ms | clamp_all ms | paired 감소 중앙값 | 95% CI | gate |
|---|---:|---:|---:|---|---|
| 1080p 정지 | 0.737440 | 0.723100 | 1.9396% | [−0.4577, 1.9825]% | 미통과 |
| 1080p 움직임 | 0.741860 | 0.727640 | 1.9134% | [1.8141, 2.0009]% | 통과 |
| 4K 정지 | 3.200760 | 3.152540 | 1.3735% | [1.3423, 1.5065]% | 통과 |
| 4K 움직임 | 3.201120 | 3.151340 | 1.4334% | [1.3251, 1.6454]% | 통과 |

짝별 절감 시간의 중앙값은 0.014340 / 0.014200 / 0.044040 / 0.045920 ms다. 이 값은 ‘두 중앙값끼리의 차이’와 다르다. 4K 움직임의 두 중앙값 차이는 0.049780 ms지만 paired 절감 중앙값은 0.045920 ms다.

!후보의 paired 이득과 95% CI (`analysis/candidate_gain_ci.png`, 연구 폴더에만 있음)

원본·분리·보충에 대한 민감도는 다음과 같다. 6 pair 집합은 계획한 7 pair를 충족하지 않아 gate가 false이다.

| case | 원본 7쌍 sensitivity | r05 제외 6쌍 | 보충 포함 7쌍 primary |
|---|---|---|---|
| 1920x1080/still | n=7; +1.9446% [-0.4577, 1.9825]; gate=False | n=6; +1.9421% [-16.7368, 1.9851]; gate=False | n=7; +1.9396% [-0.4577, 1.9825]; gate=False |
| 1920x1080/moving | n=7; +1.9134% [1.8490, 2.0009]; gate=True | n=6; +1.9151% [1.7430, 2.0024]; gate=False | n=7; +1.9134% [1.8141, 2.0009]; gate=True |
| 3840x2160/still | n=7; +1.3735% [1.2591, 1.5065]; gate=True | n=6; +1.3878% [1.3007, 1.5471]; gate=False | n=7; +1.3735% [1.3423, 1.5065]; gate=True |
| 3840x2160/moving | n=7; +1.4334% [1.3052, 1.6454]; gate=True | n=6; +1.4471% [1.3493, 1.6608]; gate=False | n=7; +1.4334% [1.3251, 1.6454]; gate=True |

주분석 정본: `analysis/clamp_all_primary_analysis.json`, `clamp_all_primary_rounds.csv`, `CLAMP_ALL_PRIMARY_ANALYSIS_KO.md`. raw는 `measurements/clamp_all_*_frames.csv`. percentile·MAD·IQR·표본표준편차·시간순 사분구간·jitter 부호별 분포를 보존했다.

## 5. Pass별 GPU 시간과 구간 정의

아래는 warm40/timed200의 **단일 pass 계측 실행 평균 µs**다. 입력은 moving이다. 추가 timestamp가 들어가므로 전체 시간 gate는 앞 절의 일반 totals 모드를 따른다. 열의 SDK는 원본 셰이더 호환 사본, exact/lossy는 커뮤니티, post/all은 이번 후보다. dispatch 0은 자동 노출 SPD, 1은 prepass, 홀수 3..25는 신경망 1..12, 27은 최종 postpass다. 짝수 2..26은 실제 경계 zero store가 있는 padding이다.

### 1080p Native

| dispatch | 역할 | SDK | exact | lossy | post 후보 | all 후보 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | SPD/노출 | 40.464 | 40.806 | 41.548 | 40.905 | 40.817 |
| 1 | prepass | 97.565 | 103.564 | 125.751 | 105.127 | 105.786 |
| 2 | pass0_post | 0.062 | 0.061 | 0.072 | 0.062 | 0.061 |
| 3 | main/pass1 | 55.885 | 52.902 | 35.005 | 53.733 | 51.354 |
| 4 | pass1_post | 0.060 | 0.059 | 0.460 | 0.059 | 0.063 |
| 5 | main/pass2 | 55.562 | 52.690 | 36.156 | 53.514 | 51.938 |
| 6 | pass2_post | 0.062 | 0.059 | 0.459 | 0.062 | 0.061 |
| 7 | pass3 | 11.959 | 11.519 | 7.397 | 11.619 | 11.655 |
| 8 | pass3_post | 0.059 | 0.061 | 0.477 | 0.059 | 0.060 |
| 9 | main/pass4 | 29.385 | 28.320 | 18.202 | 28.742 | 27.491 |
| 10 | pass4_post | 0.063 | 0.064 | 0.487 | 0.060 | 0.062 |
| 11 | main/pass5 | 29.359 | 28.317 | 18.165 | 28.725 | 27.128 |
| 12 | pass5_post | 0.063 | 0.059 | 0.472 | 0.060 | 0.061 |
| 13 | pass6 | 16.650 | 15.835 | 9.524 | 15.982 | 15.967 |
| 14 | pass6_post | 0.062 | 0.062 | 0.363 | 0.061 | 0.060 |
| 15 | pass7 | 31.325 | 30.098 | 18.780 | 30.530 | 30.176 |
| 16 | pass7_post | 0.060 | 0.061 | 0.352 | 0.065 | 0.065 |
| 17 | pass8 | 30.582 | 29.894 | 18.748 | 30.344 | 29.988 |
| 18 | pass8_post | 0.064 | 0.059 | 0.376 | 0.060 | 0.061 |
| 19 | pass9 | 49.313 | 46.912 | 28.950 | 47.572 | 47.561 |
| 20 | pass9_post | 0.060 | 0.061 | 0.263 | 0.061 | 0.063 |
| 21 | main/pass10 | 29.495 | 28.006 | 16.984 | 28.417 | 27.511 |
| 22 | pass10_post | 0.062 | 0.060 | 0.472 | 0.061 | 0.064 |
| 23 | pass11 | 40.479 | 43.353 | 28.821 | 44.009 | 44.273 |
| 24 | pass11_post | 0.061 | 0.063 | 0.561 | 0.063 | 0.064 |
| 25 | main/pass12 | 55.565 | 52.878 | 36.194 | 53.686 | 51.402 |
| 26 | pass12_post | 0.061 | 0.062 | 0.451 | 0.064 | 0.063 |
| 27 | postpass | 178.203 | 127.762 | 205.575 | 127.746 | 128.445 |

### 4K Native

| dispatch | 역할 | SDK | exact | lossy | post 후보 | all 후보 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | SPD/노출 | 207.427 | 241.368 | 232.294 | 235.075 | 234.641 |
| 1 | prepass | 716.448 | 708.912 | 624.042 | 704.075 | 704.738 |
| 2 | pass0_post | 0.065 | 0.062 | 0.059 | 0.063 | 0.063 |
| 3 | main/pass1 | 232.843 | 220.265 | 115.928 | 220.047 | 208.777 |
| 4 | pass1_post | 0.063 | 0.063 | 0.457 | 0.058 | 0.063 |
| 5 | main/pass2 | 228.088 | 215.163 | 118.218 | 215.100 | 207.309 |
| 6 | pass2_post | 0.064 | 0.063 | 0.461 | 0.059 | 0.064 |
| 7 | pass3 | 39.062 | 38.547 | 21.084 | 38.385 | 38.547 |
| 8 | pass3_post | 0.064 | 0.062 | 0.470 | 0.060 | 0.062 |
| 9 | main/pass4 | 108.513 | 103.784 | 53.857 | 103.829 | 98.547 |
| 10 | pass4_post | 0.064 | 0.062 | 0.472 | 0.061 | 0.064 |
| 11 | main/pass5 | 108.368 | 103.714 | 53.832 | 103.738 | 97.526 |
| 12 | pass5_post | 0.065 | 0.063 | 0.478 | 0.062 | 0.060 |
| 13 | pass6 | 43.217 | 43.417 | 22.257 | 43.288 | 43.410 |
| 14 | pass6_post | 0.062 | 0.059 | 0.506 | 0.060 | 0.061 |
| 15 | pass7 | 93.959 | 92.369 | 46.125 | 92.438 | 90.784 |
| 16 | pass7_post | 0.064 | 0.065 | 0.505 | 0.063 | 0.063 |
| 17 | pass8 | 93.704 | 92.265 | 46.169 | 92.222 | 90.596 |
| 18 | pass8_post | 0.064 | 0.067 | 0.508 | 0.065 | 0.064 |
| 19 | pass9 | 167.191 | 166.639 | 84.351 | 167.339 | 169.258 |
| 20 | pass9_post | 0.064 | 0.064 | 0.463 | 0.065 | 0.063 |
| 21 | main/pass10 | 109.684 | 103.710 | 51.085 | 103.665 | 99.828 |
| 22 | pass10_post | 0.063 | 0.063 | 0.468 | 0.063 | 0.064 |
| 23 | pass11 | 160.065 | 158.516 | 78.955 | 158.493 | 158.814 |
| 24 | pass11_post | 0.064 | 0.062 | 0.611 | 0.066 | 0.065 |
| 25 | main/pass12 | 228.509 | 215.548 | 119.009 | 215.632 | 205.539 |
| 26 | pass12_post | 0.064 | 0.062 | 0.463 | 0.063 | 0.064 |
| 27 | postpass | 808.783 | 624.089 | 800.862 | 632.047 | 630.829 |

exact의 평균 전체 구간은 1080p **770.706 µs = dispatch 합 693.649 + gap 합 75.982 + 선두 1.076 µs**, 4K **3217.743 µs = 3129.124 + 87.507 + 1.113 µs**다(표시 반올림 차이 가능). gap은 shader 끝→다음 shader 시작과 마지막 꼬리를 포함한다. 전부 barrier 비용이라고 부르지 않는다. 합의 항등식은 평균끼리 계산하며 pass 중앙값 합과 전체 중앙값을 혼동하지 않는다.

1080p에서는 gap 등 고정 비용 비중이 상대적으로 크고, 4K에서는 prepass·노출·postpass가 더 큰 비중을 차지한다. 논리 neural 작업량은 4배지만 warm40 baseline moving의 전체 exact 시간은 약 4.49배였다(3.17038 / 0.70570 ms). cache/메모리·실행 점유·clock·고정 비용이 함께 영향을 줄 수 있으며, 이 비율만으로 특정 cache 용량이나 DRAM 포화를 확정하지 않는다. `analysis/WARM_PROFILE_TIMING_KO.md`, `warm_profiles_passes.csv`, `warm_profiles_summary.json`에 모든 분해와 jitter군이 있다.

## 6. 실제 gfx1030 ISA·자원·packing

PSO DXBC 전체 SHA와 vkd3d DXIL dump SHA가 대응한다. 원래 ISA block→PSO 연결에는 생성 순서 추론이 들어갔으나, 후보에서 **post 2개 또는 all 20개의 의도한 코드만 변하고 나머지 54/36개 기계어가 동일**한 것을 확인해 해당 대응을 추가로 뒷받침했다. source PSO 56개의 DXBC는 후보에서도 모두 그대로다.

### 6.1 자원 표

표는 1080p이며 대응 4K도 동일한 총 instruction·자원·dot4 수다. SGPR allocation은 모두 108이다. ‘정규화 subgroup/SIMD’는 **RADV 자원 계산값의 32-lane 환산**이며 실제 resident/achieved occupancy가 아니다. 함수는 시작값 16에 자원 제한을 적용한 뒤 wave64면 2배로 환산하고 최종 16 재상한 처리를 하지 않는다. 따라서 18/20/24를 물리 resident wave 관측으로 읽지 않는다. network/post의 64-bit EXEC/VCC 경로는 wave64와 일치한다. post의 8은 내부 4 wave/SIMD 자원 계산에 대응한다. [Mesa 26.2.4 소스 정의](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/vulkan/radv_shader.c), `analysis/OCCUPANCY_DEFINITION_KO.md`.

| pass | dispatch | 명령 수 | static dot4 | VGPR | 정규화 subgroup/SIMD | LDS bytes | scratch bytes | SGPR spill | med3_i32 | SDWA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exposure/SPD | 0 | 1036 | 0 | 40 | 24 | 5120 | 0 | 0 | 0 | 44 |
| pre | 1 | 983 | 0 | 48 | 20 | 4096 | 0 | 0 | 16 | 16 |
| 1 | 3 | 1317 | 832 | 56 | 18 | 0 | 0 | 0 | 64 | 80 |
| 2 | 5 | 1318 | 832 | 56 | 18 | 0 | 0 | 0 | 64 | 80 |
| 3 | 7 | 1076 | 512 | 48 | 20 | 0 | 0 | 0 | 32 | 24 |
| 4 | 9 | 2470 | 1600 | 80 | 12 | 0 | 0 | 0 | 112 | 132 |
| 5 | 11 | 2469 | 1600 | 80 | 12 | 0 | 0 | 0 | 112 | 132 |
| 6 | 13 | 3475 | 2048 | 80 | 12 | 0 | 0 | 4 | 64 | 48 |
| 7 | 15 | 963 | 336 | 56 | 18 | 0 | 0 | 0 | 12 | 13 |
| 8 | 17 | 964 | 336 | 64 | 16 | 0 | 0 | 0 | 12 | 13 |
| 9 | 19 | 1227 | 400 | 72 | 14 | 0 | 0 | 0 | 12 | 28 |
| 10 | 21 | 2430 | 1600 | 80 | 12 | 0 | 0 | 0 | 112 | 148 |
| 11 | 23 | 4300 | 2112 | 48 | 20 | 0 | 0 | 0 | 176 | 228 |
| 12 | 25 | 1318 | 832 | 56 | 18 | 0 | 0 | 0 | 64 | 80 |
| post | 27 | 2922 | 960 | 128 | 8 | 16384 | 0 | 0 | 64 | 64 |

VGPR spill과 scratch는 0이지만, pass6은 **SGPR 4개를 VGPR lane으로 spill**하며 v_writelane/v_readlane 각각 4개가 있다. scratch 0을 모든 spill 0으로 읽으면 틀린다. post는 128 VGPR·16 KiB LDS이고 현재 group에서 VGPR에 따른 4 wave 계산이 LDS만의 8 wave 계산보다 작다.

### 6.2 packing 관련 opcode 계열

이 표는 shader 전체의 정적 계열 개수다. 주소 계산·unpack도 포함되며 ‘packing만의 비용’이 아니다. SDWA는 다른 opcode 열과 겹치는 인코딩 속성이므로 열을 합산하지 않는다. 논리 shift에는 v_add_lshl_u32도 포함하고 v_lshl_or는 shift에만, v_and_or는 AND/OR/XOR에 한 번 센다. 자세한 key와 집계식은 `analysis/PACKING_COUNTS_KO.md`에 있다.

| Pass | dispatch | med3_i32 | ashrrev_i32 | 논리 shift 계열 | AND/OR/XOR 계열 | SDWA | perm | cvt_pk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exposure/SPD | 0 | 0 | 0 | 56 | 28 | 44 | 0 | 0 |
| pre | 1 | 16 | 0 | 21 | 6 | 16 | 0 | 0 |
| network 1 | 3 | 64 | 64 | 18 | 0 | 80 | 0 | 0 |
| network 2 | 5 | 64 | 64 | 18 | 0 | 80 | 0 | 0 |
| network 3 | 7 | 32 | 0 | 3 | 0 | 24 | 0 | 0 |
| network 4 | 9 | 112 | 96 | 18 | 0 | 132 | 0 | 0 |
| network 5 | 11 | 112 | 96 | 18 | 0 | 132 | 0 | 0 |
| network 6 | 13 | 64 | 0 | 3 | 0 | 48 | 0 | 0 |
| network 7 | 15 | 12 | 8 | 3 | 0 | 13 | 0 | 0 |
| network 8 | 17 | 12 | 8 | 3 | 0 | 13 | 0 | 0 |
| network 9 | 19 | 12 | 16 | 13 | 0 | 28 | 0 | 0 |
| network 10 | 21 | 112 | 112 | 34 | 0 | 148 | 0 | 0 |
| network 11 | 23 | 176 | 80 | 49 | 98 | 228 | 0 | 0 |
| network 12 | 25 | 64 | 64 | 18 | 0 | 80 | 0 | 0 |
| post | 27 | 64 | 64 | 79 | 10 | 64 | 0 | 0 |

ACO는 이미 min/max clamp를 **v_med3_i32**로 합친다. v_perm_b32와 v_cvt_pk 계열은 현재 15개 주요 shader에서 0이다. signed16 포화 두 쌍과 high-byte permutation을 이용하는 아이디어는 ISA 의미상 가능성을 검토했지만, 현재 HLSL/SPIR-V→ACO에서 원하는 명령으로 선택되는 표현은 검증하지 못했다. 실행 후보나 이득으로 계산하지 않는다.

실제 dot는 signed v_dot4_i32_i8 / v_dot4c_i32_i8 계열이고 dot와 누산은 이미 한 명령이다. network1은 64 explicit dot4+768 dot4c, network11은 176+1936이다. 32-dot window의 목적 accumulator register 중앙값은 network1/4/5/10에서 8, 7/8/9에서 4였다. 같은 accumulator 재사용 간격과 긴 chain도 관측했지만, 이는 scheduling proxy이며 실제 stall/IPC나 수학적 독립 accumulator 생존 수를 측정한 결과가 아니다.

### 6.3 덤프의 역어셈블 오류

일부 SDWA/SMEM encoding이 LLVM disassembler에서 invalid로 표시되어 다음 word를 가짜 fmac 등으로 세는 문제가 있었다. raw instruction 경계를 복원하고 Code size와 ACO instruction 수를 대조했다. 초기 exact 56개 모두 일치했다. 예를 들어 network10 dot4는 text 1599가 아니라 1600, network11은 2109가 아니라 2112다. 원본/복원 assembly와 raw bin을 모두 보존했고 복원하지 않은 operand를 지어내지 않았다. [AMD RDNA2 ISA](https://docs.amd.com/api/khub/documents/Et~wpu9g~Ffl7d9q0QZ~Og/content), `analysis/parse_isa.py`, `analysis/*_profile_isa/`.

## 7. 연산 기준·메모리 모델·남은 한계

Vulkan dot4 프로브는 실제 RADV/Navi21에서 wave32, 4/8/16/32 독립 accumulator, warm50·timed31을 사용했다. dot4 한 번을 곱 4+누산 add 4=8 integer ops로 정의했다. 가장 빠른 변형의 중앙값은 **91.679754437 TOPS**였다. 각 변형의 실제 128/256/512/1024 dot4c 명령 수와 scratch 0을 확인했다. 마지막 output buffer의 2048개 참조값은 일치했지만 각 변형의 모든 출력을 별도로 readback한 검사는 아니다. `ceiling/ISA_VALIDATION.json`, `runs/dot4_steady/stdout.log`, `ceiling/README.md`.

아래 dot 기준 시간은 실제 유효 thread·loop를 반영한 연산량을 이 관측 처리량으로 나눈 값이다. wave32 synthetic probe와 wave64 FSR은 조건이 달라 물리적 최저 시간을 보증하지 않는다. pass7/8은 static 336이 반복되어 5248 dot4/thread, pass9는 static 400이 7296 dot4/thread가 된다. 아래 ‘진단 median’은 community exact의 warm40/timed200 moving pass 계측 중앙값이다.

| 출력 | pass | static dot4 | 유효 thread당 dot4 | 유효 영역 | INT GOP | dot 기준 ms | 진단 median ms | 기준 TOPS 대비 |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| 1920x1080 | 1 | 832 | 832 | 960×540 | 3.450470 | 0.037636 | 0.051400 | 73.2% |
| 1920x1080 | 2 | 832 | 832 | 960×540 | 3.450470 | 0.037636 | 0.051200 | 73.5% |
| 1920x1080 | 3 | 512 | 512 | 480×270 | 0.530842 | 0.005790 | 0.011440 | 50.6% |
| 1920x1080 | 4 | 1600 | 1600 | 480×270 | 1.658880 | 0.018094 | 0.027560 | 65.7% |
| 1920x1080 | 5 | 1600 | 1600 | 480×270 | 1.658880 | 0.018094 | 0.027560 | 65.7% |
| 1920x1080 | 6 | 2048 | 2048 | 240×135 | 0.530842 | 0.005790 | 0.015600 | 37.1% |
| 1920x1080 | 7 | 336 | 5248 | 240×135 | 1.360282 | 0.014837 | 0.029360 | 50.5% |
| 1920x1080 | 8 | 336 | 5248 | 240×135 | 1.360282 | 0.014837 | 0.029160 | 50.9% |
| 1920x1080 | 9 | 400 | 7296 | 240×135 | 1.891123 | 0.020627 | 0.045840 | 45.0% |
| 1920x1080 | 10 | 1600 | 1600 | 480×270 | 1.658880 | 0.018094 | 0.027280 | 66.3% |
| 1920x1080 | 11 | 2112 | 2112 | 480×270 | 2.189722 | 0.023884 | 0.042160 | 56.7% |
| 1920x1080 | 12 | 832 | 832 | 960×540 | 3.450470 | 0.037636 | 0.051420 | 73.2% |
| 1920x1080 | post | 960 | 960 | 960×540 | 3.981312 | 0.043426 | 0.124280 | 34.9% |
| 3840x2160 | 1 | 832 | 832 | 1920×1080 | 13.801882 | 0.150544 | 0.221020 | 68.1% |
| 3840x2160 | 2 | 832 | 832 | 1920×1080 | 13.801882 | 0.150544 | 0.216200 | 69.6% |
| 3840x2160 | 3 | 512 | 512 | 960×540 | 2.123366 | 0.023161 | 0.038520 | 60.1% |
| 3840x2160 | 4 | 1600 | 1600 | 960×540 | 6.635520 | 0.072377 | 0.104280 | 69.4% |
| 3840x2160 | 5 | 1600 | 1600 | 960×540 | 6.635520 | 0.072377 | 0.104200 | 69.5% |
| 3840x2160 | 6 | 2048 | 2048 | 480×270 | 2.123366 | 0.023161 | 0.043480 | 53.3% |
| 3840x2160 | 7 | 336 | 5248 | 480×270 | 5.441126 | 0.059349 | 0.092760 | 64.0% |
| 3840x2160 | 8 | 336 | 5248 | 480×270 | 5.441126 | 0.059349 | 0.092680 | 64.0% |
| 3840x2160 | 9 | 400 | 7296 | 480×270 | 7.564493 | 0.082510 | 0.166800 | 49.5% |
| 3840x2160 | 10 | 1600 | 1600 | 960×540 | 6.635520 | 0.072377 | 0.104160 | 69.5% |
| 3840x2160 | 11 | 2112 | 2112 | 960×540 | 8.758886 | 0.095538 | 0.159180 | 60.0% |
| 3840x2160 | 12 | 832 | 832 | 1920×1080 | 13.801882 | 0.150544 | 0.216480 | 69.5% |
| 3840x2160 | post | 960 | 960 | 1920×1080 | 15.925248 | 0.173705 | 0.623760 | 27.8% |

network12+post 합계는 1080p **27.1724544 GOP / dot 기준 0.296384 ms**, 4K **108.6898176 GOP / 1.185538 ms**다. 같은 community exact warm40/timed200 moving pass들의 진단 중앙값 합은 0.534260 / 2.183520 ms였다. prepass·노출·padding·gap은 이 합 밖이다. 이 차이를 곧바로 ‘남은 전부 compiler 낭비’ 또는 ‘하드웨어 한계 근접’으로 해석하지 않는다.

메모리 모델은 실제 24개 network PSO의 SHA·guard·store 폭·loop를 대조했다. 12 pass의 **논리 activation 입력+출력 payload 합**은 1080p 132,710,400 B, 4K 530,841,600 B다. pass 경계마다 같은 데이터가 다시 사용되면 다시 센 합이며 물리 DRAM traffic이나 단일 allocation 크기가 아니다. L1/L2/L3 내부 payload는 1080p 8,294,400 / 4,147,200 / 2,073,600 B이고 4K는 각 4배다.

아래 값은 각 pass의 고유 입력 내부 payload와 출력 내부 payload다. 반복 tap 읽기·border·weight·캐시 전송을 합친 물리 바이트가 아니다. 별도 skip 텐서는 입력에 포함하고, 같은 pass에서 재사용하는 동일 입력은 한 번만 센다.

| Network pass | 입력 → 출력 | 1080p 논리 read / write B | 4K 논리 read / write B |
|---|---|---:|---:|
| 1 | L1 → L1 | 8,294,400 / 8,294,400 | 33,177,600 / 33,177,600 |
| 2 | L1 → L1 | 8,294,400 / 8,294,400 | 33,177,600 / 33,177,600 |
| 3 | L1 → L2 | 8,294,400 / 4,147,200 | 33,177,600 / 16,588,800 |
| 4 | L2 → L2 | 4,147,200 / 4,147,200 | 16,588,800 / 16,588,800 |
| 5 | L2 → L2 | 4,147,200 / 4,147,200 | 16,588,800 / 16,588,800 |
| 6 | L2 → L3 | 4,147,200 / 2,073,600 | 16,588,800 / 8,294,400 |
| 7 | L3 → L3 | 2,073,600 / 2,073,600 | 8,294,400 / 8,294,400 |
| 8 | L3 → L3 | 2,073,600 / 2,073,600 | 8,294,400 / 8,294,400 |
| 9 | L3 + skip L2 → L2 | 6,220,800 / 4,147,200 | 24,883,200 / 16,588,800 |
| 10 | L2 → L2 | 4,147,200 / 4,147,200 | 16,588,800 / 16,588,800 |
| 11 | L2 + skip L1 → L1 | 12,441,600 / 8,294,400 | 49,766,400 / 33,177,600 |
| 12 | L1 → L1 | 8,294,400 / 8,294,400 | 33,177,600 / 33,177,600 |
| 합계 | 각 pass의 read+write 합 | **132,710,400** | **530,841,600** |


**실제 DRAM read/write byte는 미측정이므로 `bytes / bandwidth` 물리 시간 하한은 산출하지 않았다.** 기존 copy 436.7 GB/s는 다른 경로의 실측이고, 위 payload를 그 값으로 나누면 cache 재사용·write 정책·overfetch를 무시하게 된다. pre/post·노출의 모든 resource 접근, cache hit/miss, runtime achieved occupancy, stall도 미측정이다. `analysis/MEMORY_MODEL_KO.md`와 `analysis/MEMORY_MODEL.json`에 pass별 논리 read/write와 미측정 칸을 명시했다.

## 8. 구현과 원인 분류

`analysis/build_clamp_candidate.py`는 관측한 exact SPIR-V에 pinned upstream [model_clamp.py](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/blob/d1c3b30db656bf6a38634642542ee7736127d211/tools/model_clamp.py)를 적용한다. 원래 SPIR-V 1.3을 유지했고 Vulkan 1.3 spirv-val을 통과했다. 모델 가중치·tensor layout·dot4 수를 바꾸지 않는다.

예를 들어 이미 반올림된 signed 정수 R에서 shift 8 후 INT8 포화는, R을 [−32768,32767]로 clamp한 뒤 byte 1을 꺼내는 표현으로 바꿀 수 있다. 원래 round-to-even과 해당 ReLU 의미를 유지하면서 shift/별도 max를 합친다. 현재 ACO가 이미 수행하는 med3 결합에 더해, source 표현을 바꿔 남은 명령을 줄인 것이다.

| 후보 | 실제 변경 | 정확성 | 전체 시간 판정 | 분류 |
|---|---|---|---|---|
| post만 | 두 hash, 해상도당12 vec4 clamp그룹; post 2922→2877명령, dot4 960개 유지 | 최종frame239 네 조건 byte동일 | 단일 pass 선별에서 뚜렷한 post 이득 없음; 별도7pair gate 미수행 | 선별 후보·미채택, 전체 이득 미검증 |
| clamp_all | 두 해상도×10pass=20 hash, 해상도당114 vec4그룹 | probe4+primary28+temporal64 비교 모두byte동일 | 1080moving·4Kstill/moving gate통과 | compiler 입력 표현의 추가 최적화 여지 확인 |
| clamp_all 1080still | 위와 동일 | byte동일 | CI에0 포함, 큰 변동 표본 유지 | 측정 불가: 이득 미확정 |
| pack16+perm 유도 | ISA 후보 검토만 | 미실행 | 미실행 | 표현 선택·compiler 경로 미검증 |

all 후보의 ACO instruction 감소는 pass1/2/4/5/7/8/9/10/12/post 순서로 61/45/124/141/5/5/8/76/61/45개다. pass7/8/9에는 padding NOP 4/3/6개가 새로 생겨 raw instruction 실제 감소는 1/2/2개다. 모든 변형에서 dot4 수·VGPR/SGPR allocation·LDS·scratch·spill·RADV 자원 계산값이 유지됐다. `analysis/CANDIDATE_ISA_COMPARISON_KO.md`와 `analysis/CANDIDATE_ISA_COMPARISON.json`.

커뮤니티가 이미 해결한 대표 부분은 post의 공동 store 구조다. 원본→exact에서 post는 3007→2922명령, VGPR 72→128, LDS 0→16 KiB, 정규화 자원 계산 14→8인데도 전체 기준선은 빨라졌다. register가 늘거나 정적 점유 계산이 낮아졌다고 자동으로 퇴보 판정하면 안 된다. 반대로 pass11은 4045→4300명령, VGPR 64→48이며 단일 1080 진단에서 항상 이득을 보이지 않았다. 적용 판단은 전체 교대 시간이다. `analysis/VARIANT_ISA_COMPARISON_KO.md`와 `analysis/VARIANT_ISA_COMPARISON.json`.

## 9. Lossy 익명 A/B

커뮤니티 lossy는 exact보다 15~23% 정도 빠른 조건이 있었지만 재학습으로 출력이 다르다. 최종 frame 239의 선형 RGB MAE는 1080p 정지 0.003288/움직임 0.005919, 4K 0.003027/0.005615다. 이 숫자는 사람의 품질 판단을 대신하지 않는다. exact의 dynamic dot4 작업량 공식을 구조가 다른 lossy에 그대로 적용하지 않았다.

익명 보기 입구는 **`captures/ab_review/index.html`**이다. 정지, 움직임, 밝기 전환을 두 해상도로 제공한다. 132개 raw 입력에서 전체 PNG 132개와 동일 좌표 512×512 crop PNG 132개, 합계 PNG 264개 및 RGB lossless MP4 8개를 생성했고 영상 전부를 다시 디코드해 전체 RGB24 길이·SHA 일치를 확인했다. 공개 묶음은 275파일, 1,241,377,493 B다. 독립 키 `analysis/ab_review_key.json`은 공개 묶음에 넣지 않았다.

동일 linear RGB→sRGB 표시 변환과 [0,1] clip, 8 bit 양자화를 적용했다. RGB/alpha NaN·Inf는 0이었다. 밝기 입력의 최대 1.2138671875와 약 3.17%의 1 초과 component는 표시에서 clipping된다. HDR 차이는 원래 RGBA16F로 확인해야 한다. 영상은 16 frame·30 fps이며 합성 입력 8-phase wrap이 있어 실제 게임의 연속 카메라 움직임과 다르다. RGB 4:4:4 lossless 영상 재생을 지원하지 않는 뷰어에서는 전체 PNG/frame 링크를 이용한다. **사용자의 지각 판정은 아직 받지 않았고 lossy를 채택하지 않았다.**

## 10. 요청서의 외부 주장 네 가지 확인

| 주장 | 원문 확인 | 이번 결과와의 구분 |
|---|---|---|
| 26.10.1 RDNA2 pack/unpack 우회·dot4 수 유지 | [issue39 분석 댓글](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/issues/39#issuecomment-6330523795) 확인 | 당시 정적DXIL family count이며 이번gfx1030 ISA 실측과 단위가 다름 |
| 6900XT Windows exact−4%, lossy−19% | [pinned README](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/blob/d1c3b30db656bf6a38634642542ee7736127d211/README.md) 확인 | Windows1건,4K moving Balanced, 업스케일러 시간; 이번Native Linux 수치 아님 |
| Linux override 어두운 화면 사례 | [Linux guide](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/blob/d1c3b30db656bf6a38634642542ee7736127d211/docs/linux.md) 확인 | FFVII Rebirth의 root constants layout 차이; game dump 재생성으로 정상이라는 보고 |
| INT8 역공학 저장소 | [고정 README](https://github.com/int3rrobang/fsr4-int8-reverse-engineering/blob/739e667454ffaba78a985412443d5df73602f7e2/README.md) 확인 | 4.1.1 개별kernel replay·GTX1650 중심, 현재4.1.2 RDNA2 전체이득을 뜻하지 않음 |

[AMD 공식 SDK 문서](https://gpuopen.com/amd-fsr-sdk/)는 4.1.2의 RX6000 full INT8와 DX12/Vulkan 지원을 명시한다. Linux prebuilt는 4.1.1.2740 hash/layout 대상이며 현재 exact DLL과 동일하게 취급하지 않았다. 현재 prebuilt는 1080을 포함한 크기를 지원한다. 이번 override는 실제 실행기의 dump로 만들었다. `SOURCES_KO.md`, `SOURCE_EVIDENCE.json`에 날짜·고정 commit·원문 한계가 있다.

## 11. 원자료와 재현

| 위치 | 내용 |
|---|---|
| `PREDECLARED_GATES.json`, `MEASUREMENT_MAPPING.json` | 최초 gate·baseline 순서·DLL 매핑 |
| `analysis/clamp_all_*DECLARATION.json` | 후보 순서·SHA·보충범위와 제외 근거 |
| `measurements/` | 전체frame/pass CSV, query·PSO·capture 로그 |
| `runs/` | 실행명령·환경·GPU감시·kernel fault·stdout/stderr·종료상태 |
| `dumps/`, `pipelines/`, `analysis/*_isa/` | DXIL/SPIR-V/PSO SHA·실제ISA raw/original/normalized |
| `analysis/candidate_clamp_shift_all/` | 실제 수정한20개SPIR-V와원본/후보assembly·manifest |
| `harness/`, `ceiling/`, `tools/` | 기존kit측정파생본·Vulkandot4프로브·지원판정사본생성기 |
| `captures/` | raw RGBA16F,익명A/B실물및manifest |

GPU 없이 숫자·표를 다시 만드는 명령:

```sh
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops/results/fsr4
python3 analysis/analyze_baselines.py
python3 analysis/analyze_baselines.py --trial clamp_all --supplement analysis/clamp_all_SUPPLEMENTAL_DECLARATION.json
python3 analysis/summarize_profiles.py
python3 analysis/parse_isa.py exact_warm_profile
python3 analysis/compare_candidate_isa.py
python3 analysis/build_dot_workload.py
python3 analysis/build_memory_model.py
python3 analysis/read_key_numbers.py
```

후보 재생성은 `python3 analysis/build_clamp_candidate.py all`이다. 원본 소스·SPIR-V version·모든 SHA는 manifest를 따른다. 기존 결과를 보존하고 재현은 새 연구 디렉터리에서 한다. GPU를 다시 실행할 경우 현재 게임/다른 작업/stop marker를 확인하는 동일 guard를 사용하고 새 run 이름을 지정한다. 예:

```sh
env FSR4TIME_WARM=400 FSR4TIME_TIMED=200 bash run_case.sh repro_reference exact 0 totals
env FSR4TIME_WARM=400 FSR4TIME_TIMED=200 \
  FSR4_OVERRIDE_PATH="$PWD/analysis/candidate_clamp_shift_all/override" \
  bash run_case.sh repro_candidate exact 0 totals
```

이 두 번만으로 채택 판정을 갱신하지 않는다. 정식 반복 순서는 `run_pair_trials.py`와 선언을 따른다. 기존 동명 run/선언은 덮어쓰지 않도록 거부한다. 게임·운영 DLL에 대한 설치 명령은 제공하지 않았다.

### 다운로드 기록

작성 UTC: 2026-11-09T18:54:38.841670+00:00

실제 보존 다운로드 **15개, 420,802,912 bytes (0.420802912 GB)**를 다시 읽어 크기와 SHA256을 검증했다. 그 밖에 압축본과 중복되는 로컬 소스 2개도 다시 해시했다. 불일치는 없었다. SDK 헤더는 `downloads/sdk_headers.json`에 기록된 `sdk/` 아래 5개 파일이며, 별도 `sdk_headers/` 폴더는 없다.

미보존 Mesa archive의 기록값 **68,613,948 B**, 추가 원문 fetch 5개의 기록값 **272,261 B**, 추가 확인된 동일 원문 재조회 2회의 **267,164 B**를 합친 명시 항목 payload는 **489,956,285 B**다. 개별 전송 여부가 기록되지 않은 원문 11개 **1,028,339 B**를 모두 별도 fetch로 보수적으로 합해도 **490,984,624 B (0.490984624 GB)**다. 요청한 **100 GB**보다 작은 범위다. 이 계산은 지정된 기록의 파일 payload 합계이고, 전송 재시도·중복 열람·HTTP/TLS·목록 밖 웹/API 응답을 포함한 회선 총량 계측은 아니다.

이번 정리는 새 다운로드·설치 없이 수행했다. deb 2개는 `apt-get --print-uris download`로 현재 저장소 URL만 조회했고, 저장소의 byte 수와 SHA512가 실물과 일치했다. 원래 다운로드 당시 mirror URL은 기록되어 있지 않아, URL 출처를 JSON에 별도로 표시했다.

정본 JSON: `analysis/DOWNLOAD_MANIFEST_ALL.json`. 기존 다운로드 기록과 보고서는 수정하지 않았다.

#### 1. 보존된 원본 다운로드: 실물 크기·해시 검증

| 파일 / 저장 위치 | 출처 URL | Bytes | SHA256 | 상태 |
|---|---|---:|---|---|
| `downloads/fsr4-timing-kit-linux-2026-11-06.tar.gz` | [원문](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/releases/download/timing-kit-2026-11-06/fsr4-timing-kit-linux-2026-11-06.tar.gz) | 54,611,189 | `fe99cd28db375447d971732c1114fe29546311ee95513312df23fe90f49725f3` | 실물 재해시 일치 |
| `downloads/overrides-d1c3b30.tar.gz` | [원문](https://api.github.com/repos/zgauthier2000/fsr4-shader-performance-overrides/tarball/d1c3b30db656bf6a38634642542ee7736127d211) | 63,530,999 | `5624395ada3f5a60911b22628ff5d5766945a56350abc23bc5e4739e9fb1c1f5` | 실물 재해시 일치 |
| `downloads/fsr4.1.2-cyboman-rx6000-2026-11-05.zip` | [원문](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/releases/download/dll-2026-11-05/fsr4.1.2-cyboman-rx6000-2026-11-05.zip) | 104,375,648 | `771dce4019f2afa832ef87f0d8f3747696efef74d58f665bd4e584b5e2590aac` | 실물 재해시 일치 |
| `downloads/fsr4-timing-kit-windows-2026-11-02.14.zip` | [원문](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/releases/download/timing-kit-2026-11-02.14/fsr4-timing-kit-windows-2026-11-02.14.zip) | 108,131,138 | `b9ea1f25c259356531c0491c7339e60b01f410fad179b120c7ddde34de579134` | 실물 재해시 일치 |
| `downloads/llvm-mingw-20261104-ucrt-ubuntu-22.04-x86_64.tar.xz` | [원문](https://github.com/mstorsjo/llvm-mingw/releases/download/20261104/llvm-mingw-20261104-ucrt-ubuntu-22.04-x86_64.tar.xz) | 84,104,244 | `919380403de87dca3c581c2053230cd298af5e891286628ba0613b7fed370926` | 실물 재해시 일치 |
| `sdk/api/include/ffx_api.h` | [원문](https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/v2.4.0/Kits/FidelityFX/api/include/ffx_api.h) | 10,848 | `4d9dcf243ef5efbf03340a02f16b51a22053243a22e8a9a0b3ce91c26982b4bf` | 실물 재해시 일치 |
| `sdk/api/include/ffx_api_loader.h` | [원문](https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/v2.4.0/Kits/FidelityFX/api/include/ffx_api_loader.h) | 2,977 | `18d1901dbbc881e8131b91c65ad44bb3b4a1de6c31435d5b75236f16727ba80e` | 실물 재해시 일치 |
| `sdk/api/include/ffx_api_types.h` | [원문](https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/v2.4.0/Kits/FidelityFX/api/include/ffx_api_types.h) | 13,637 | `53f07c113a9ba493f297656e4481b6539aed0afc9f3cc8da682aa477d270c46c` | 실물 재해시 일치 |
| `sdk/api/include/dx12/ffx_api_dx12.h` | [원문](https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/v2.4.0/Kits/FidelityFX/api/include/dx12/ffx_api_dx12.h) | 11,323 | `c9afbc3c4c673723e9e5369a666f0ff54560543b5a93537f8ff453bf1c3bf530` | 실물 재해시 일치 |
| `sdk/upscalers/include/ffx_upscale.h` | [원문](https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/v2.4.0/Kits/FidelityFX/upscalers/include/ffx_upscale.h) | 19,576 | `13091e7fe5fd9c82b1fb7c044b9f08f664d9653f599d2ea500738709236fca4d` | 실물 재해시 일치 |
| `analysis/source_evidence/wine_d3dkmt.c` | [원문](https://raw.githubusercontent.com/ValveSoftware/wine/7b3ceaf8644bb5d57140903e12897efac69be16a/dlls/win32u/d3dkmt.c) | 74,582 | `da3a87c7a4da8efde479218c22983a8e44ca9da1ce4208aae974ed957a0521cc` | 실물 재해시 일치 |
| `analysis/source_evidence/wine_d3dkmthk.h` | [원문](https://raw.githubusercontent.com/ValveSoftware/wine/7b3ceaf8644bb5d57140903e12897efac69be16a/include/ddk/d3dkmthk.h) | 40,801 | `639ba21e4fe0f8260e973b9ef2f0abb6a0d47ad226613ca394b35489c936c6c1` | 실물 재해시 일치 |
| `analysis/source_evidence/wine_amdxc64_main.c` | [원문](https://raw.githubusercontent.com/ValveSoftware/wine/7b3ceaf8644bb5d57140903e12897efac69be16a/dlls/amdxc64/main.c) | 8,472 | `fb02aec3a71f61627438d9a45b3c9fe8bf2a039fa215837b93f79fd2ac2ba4a6` | 실물 재해시 일치 |
| `downloads/debs/glslang-tools_16.4.0-1_amd64.deb` | [원문](http://kr.archive.ubuntu.com/ubuntu/pool/universe/g/glslang/glslang-tools_16.4.0-1_amd64.deb) | 2,750,708 | `b4bef960f3deae8825462da1c4d7eb284affec6a764a4e16e609d8caede6204c` | 실물 재해시 일치 · apt SHA512도 일치 |
| `downloads/debs/spirv-tools_2026.3~rc1-1ubuntu1_amd64.deb` | [원문](http://kr.archive.ubuntu.com/ubuntu/pool/universe/s/spirv-tools/spirv-tools_2026.3%7erc1-1ubuntu1_amd64.deb) | 3,116,770 | `4d99ec9f6f4ee7e7af461d1dfe07a8b5bbba608a251b58314914f5b41c338ca1` | 실물 재해시 일치 · apt SHA512도 일치 |

#### 2. 압축본과 중복되는 보존 소스: 총량에 다시 더하지 않음

| 파일 / 저장 위치 | 출처 URL | Bytes | SHA256 | 상태 |
|---|---|---:|---|---|
| `source/zgauthier2000-fsr4-shader-performance-overrides-d1c3b30/tools/model_clamp.py` | [원문](https://raw.githubusercontent.com/zgauthier2000/fsr4-shader-performance-overrides/d1c3b30db656bf6a38634642542ee7736127d211/tools/model_clamp.py) | 5,988 | `5baffb9b88088a4c03f63919ba6703a371e1945c72edde71aaed65f09a547c69` | 로컬 코드 일치 · archive와 중복 |
| `source/zgauthier2000-fsr4-shader-performance-overrides-d1c3b30/tools/model_tail.py` | [원문](https://raw.githubusercontent.com/zgauthier2000/fsr4-shader-performance-overrides/d1c3b30db656bf6a38634642542ee7736127d211/tools/model_tail.py) | 4,429 | `817ef9a929a81d47cdf2984bc904a1f37cd712046ecc2a420d3da5c52afaca61` | 로컬 코드 일치 · archive와 중복 |

#### 3. 메모리에서 읽은 원문: 보존된 식별 기록

| 파일 / 저장 위치 | 출처 URL | Bytes | SHA256 | 상태 |
|---|---|---:|---|---|
| `mesa-26.2.4.tar.xz` | [원문](https://archive.mesa3d.org/mesa-26.2.4.tar.xz) | 68,613,948 | `bce5f7fbebb934373b86c999a064d52fb5065878dc57f287f95346648ec832e9` | 기록만 보존 · 실물 미보존 |
| `aco_select_nir_alu.cpp` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/compiler/instruction_selection/aco_select_nir_alu.cpp) | 177,011 | `1c3e092dd6d82c27e08fca55b11f15d89724a0509c7044b2c441351afd4b4a5b` | 식별값 보존 · 개별 전송 미기록 |
| `aco_optimizer.cpp` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/compiler/aco_optimizer.cpp) | 223,082 | `88da817fea2011af66f44c439ca5ced65c33524f2d8c313a08b81ec1eab97242` | 식별값 보존 · 개별 전송 미기록 |
| `radv_instance.c` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/vulkan/radv_instance.c) | 15,848 | `9d6db5588cdffd670fbbb58b75ccd448249c01052100a895b4d9a9fbd63c6ea8` | 식별값 보존 · 개별 전송 미기록 |
| `radv_shader.c` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/vulkan/radv_shader.c) | 156,593 | `48da66a47a6cc66da6ecb998cd203bc940988adf1db465bde060addffb99d415` | 식별값 보존 · 개별 전송 미기록 |
| `ac_shader_util.h` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/common/ac_shader_util.h) | 13,198 | `c911fc892f9a995570ba14287f98eb3c9ef3d5348130221054cd5b71382191ba` | 식별값 보존 · 개별 전송 미기록 |
| `ac_shader_util.c` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/common/ac_shader_util.c) | 64,333 | `91b28d0dc39b95b8fa737ab2a3bd3732518c0641481cb8168af908289e8dcc64` | 식별값 보존 · 개별 전송 미기록 |
| `aco_statistics.cpp` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/compiler/aco_statistics.cpp) | 25,709 | `e67cfaea74c07d595ebe8892a364e707030737b7ed07e263e08b4a9c9bb18868` | 식별값 보존 · 개별 전송 미기록 |
| `nir_opcodes.py` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/compiler/nir/nir_opcodes.py) | 87,677 | `c8ed942f9c130627f79109228980f232ffc0259ae6d1f6a26a17443fe1f7bfd8` | 식별값 보존 · 개별 전송 미기록 |
| `nir_opt_algebraic.py` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/compiler/nir/nir_opt_algebraic.py) | 254,471 | `f63c37ce8072425e8f490546b9800699363be01fda0431ec950c1686b605adbf` | 식별값 보존 · 개별 전송 미기록 |
| `ac_gpu_info.c` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/common/ac_gpu_info.c) | 110,571 | `0cede344881cdac51dcdc2f3ca5ba1ffc7bdf6ee1a89d5eed4c90b0b37fc15e1` | 별도 원문 fetch 기록 · 미보존 |
| `radv_pipeline.c` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/vulkan/radv_pipeline.c) | 50,118 | `b28475667a6b751f875d69a014237442d365b8d38eca672a8ee34e2ea027a06c` | 별도 원문 fetch 기록 · 미보존 |
| `radv_shader_object.c` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/vulkan/radv_shader_object.c) | 29,851 | `e38b43993b4582254a03b1378edc609e26d9f91938db7c31c77c5d0229d59497` | 별도 원문 fetch 기록 · 미보존 |
| `radv_shader_info.c` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/vulkan/radv_shader_info.c) | 68,236 | `d614c9ff88a7160ea944f65a58e15ad7e5186fa5333756293f76e6dd1ec41398` | 별도 원문 fetch 기록 · 미보존 |
| `radv_pipeline_compute.c` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/vulkan/radv_pipeline_compute.c) | 13,485 | `0c36fb5dfcf1c20099e2137c7368b5199816006a87352763c79de6b4eaf1096f` | 별도 원문 fetch 기록 · 미보존 |

#### 총량과 확인 범위

| 구분 | Bytes | 판정 |
|---|---:|---|
| 실물 재검증한 원본 다운로드 15개 | 420,802,912 | 보존 파일 payload 확정 |
| 압축본과 중복되는 로컬 source 2개 | 10,417 | 재해시했지만 다운로드 합계에 중복 가산하지 않음 |
| 미보존 Mesa archive 기록 | 68,613,948 | 기록값; 현재 재해시 불가 |
| 별도 추가 원문 fetch 5개 | 272,261 | 기록값; 현재 재해시 불가 |
| 보존 원본 + 명시된 archive/추가 fetch | 489,689,121 | 지정된 기록의 payload 합계 |
| 추가 확인된 같은 원문 재조회 2회 | 267,164 | 새 고유 파일은 아니지만 추가 요청 byte로 기록 |
| 명시된 payload + 알려진 재조회 | 489,956,285 | 재조회 2회를 포함한 기록 범위 합계 |
| 불명확한 개별 fetch 11개도 전부 추가한 값 | 490,984,624 | 알려진 재조회와 중복 가능성을 모두 가산한 보수적 합계 |
| 전체 회선 전송량 | 미계측 | 반복 fetch·재시도·웹/API 응답을 포함하는 정확한 총량은 알 수 없음 |

ISA_METHOD의 초기 Mesa source 9개와 model_clamp/model_tail 2개는 archive 추출인지 별도 raw 요청인지 구분이 남아 있지 않다. 따라서 독립 다운로드로 확정 합산하지 않는다. 추가 source fetch 5개는 별도 요청이었다는 기록이 있다. 모든 미보존 항목의 SHA는 기존 원문 식별 기록이며 이번에 원문을 다시 내려받아 검증했다는 뜻이 아니다.

기존 vendor SDK DLL·Proton/Wine·드라이버 사본, 압축 해제 결과 전체, 생성한 harness/shader·GPU 캡처·A/B 산출물은 새 다운로드 합계에서 제외했다. 이 표는 DOWNLOADS.json, downloads의 지정 JSON/deb, SDK 헤더 5개, Wine 원문 3개, ISA_METHOD/ISA_ADDITIONAL_SOURCE_FETCHES 식별 기록을 대상으로 한다.

#### 추가 확인된 동일 원문 재조회

ISA 담당이 subgroup 각주를 확인하며 아래 두 파일을 같은 raw URL에서 다시 읽었다고 보고했다. SHA는 기존 기록과 같고 메모리에서만 읽었으므로 새 고유 파일로 추가하지 않았다. 알려진 추가 요청 payload 267,164 B를 위 총량에 별도 반영했다. 다른 반복 요청은 전송 이력이 없어 계량하지 않았다.

| 파일 | 출처 URL | Bytes | SHA256 | 상태 |
|---|---|---:|---|---|
| `radv_shader.c` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/vulkan/radv_shader.c) | 156,593 | `48da66a47a6cc66da6ecb998cd203bc940988adf1db465bde060addffb99d415` | 동일 원문 재조회 · 메모리에서만 읽음 |
| `ac_gpu_info.c` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/common/ac_gpu_info.c) | 110,571 | `0cede344881cdac51dcdc2f3ca5ba1ffc7bdf6ee1a89d5eed4c90b0b37fc15e1` | 동일 원문 재조회 · 메모리에서만 읽음 |

## 12. Claude가 GPU 없이 확인할 핵심 숫자 5개

아래 명령은 저장된 원자료만 읽는다. `analysis/read_key_numbers.py`는 dot4 JSONL, 실제 frame CSV, 보충 선언에 해당하는 pair, temporal 비교 기록에서 재계산한다. `analysis/KEY_NUMBERS.json`은 그 출력 사본이다.

| 번호 | 핵심 숫자 | 읽는 명령 |
|---|---|---|
| 1 | Vulkan dot4 관측기준 **91.679754437 TOPS** | `python3 analysis/read_key_numbers.py --key vulkan_dot4_reference_tops` |
| 2 | community exact 1080p moving **0.70570ms** (warm40 baseline) | `python3 analysis/read_key_numbers.py --key exact_1080_moving_ms` |
| 3 | community exact 4K moving **3.17038ms** (warm40 baseline) | `python3 analysis/read_key_numbers.py --key exact_4k_moving_ms` |
| 4 | 추가candidate의4K moving paired 감소 **1.4334053366%** (warm400 primary) | `python3 analysis/read_key_numbers.py --key candidate_4k_moving_gain_pct` |
| 5 | 움직임·밝기64frame의불일치 **0 bytes** | `python3 analysis/read_key_numbers.py --key candidate_temporal_mismatched_bytes` |
