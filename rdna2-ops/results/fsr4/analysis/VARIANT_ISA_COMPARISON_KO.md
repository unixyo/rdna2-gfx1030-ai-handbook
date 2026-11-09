# AMD SDK·community exact·lossy의 실제 ISA 비교

모두 같은 RX6900XT/RADV 경로의 warm40/timed200 pass 프로파일이다. SDK412는 주 작업의 provider 호환 선택 수정을 거친 원본 셰이더 경로이며, 셰이더 byte 보존 증거는 주 보고서에 있다. 아래 정적 자원과 명령은 실제 컴파일된 ISA다. 단일 pass 프로파일 시간은 진단용이며 일반 total 교대 측정이 최종 성능 판정이다.

표는 1080p의 SDK / exact / lossy 순서다. JSON에는 두 해상도 전체 28 dispatch와 각 opcode가 있다. 서로 다른 lossy 모델에 exact의 loop/유효 영역 공식을 적용하지 않는다.

| pass | 명령 수 SDK / exact / lossy | dot4 SDK / exact / lossy | VGPR SDK / exact / lossy | LDS bytes SDK / exact / lossy |
|---|---|---|---|---|
| main | 1036 / 1036 / 1036 | 0 / 0 / 0 | 40 / 40 / 40 | 5120 / 5120 / 5120 |
| prepass | 2355 / 983 / 1730 | 0 / 0 / 0 | 64 / 48 / 48 | 4096 / 4096 / 4096 |
| pass1 | 1384 / 1317 / 1346 | 832 / 832 / 796 | 56 / 56 / 64 | 0 / 0 / 0 |
| pass2 | 1385 / 1318 / 1381 | 832 / 832 / 832 | 56 / 56 / 64 | 0 / 0 / 0 |
| pass3 | 1092 / 1076 / 1134 | 512 / 512 / 512 | 48 / 48 / 48 | 0 / 0 / 0 |
| pass4 | 2557 / 2470 / 2516 | 1600 / 1600 / 1600 | 80 / 80 / 80 | 0 / 0 / 0 |
| pass5 | 2557 / 2469 / 2513 | 1600 / 1600 / 1600 | 80 / 80 / 80 | 0 / 0 / 0 |
| pass6 | 3507 / 3475 / 3484 | 2048 / 2048 / 2048 | 80 / 80 / 80 | 0 / 0 / 0 |
| pass7 | 963 / 963 / 972 | 336 / 336 / 336 | 56 / 56 / 56 | 0 / 0 / 0 |
| pass8 | 964 / 964 / 973 | 336 / 336 / 336 | 64 / 64 / 64 | 0 / 0 / 0 |
| pass9 | 1244 / 1227 / 1232 | 400 / 400 / 400 | 72 / 72 / 72 | 0 / 0 / 0 |
| pass10 | 2554 / 2430 / 2358 | 1600 / 1600 / 1446 | 80 / 80 / 80 | 0 / 0 / 0 |
| pass11 | 4045 / 4300 / 4180 | 2112 / 2112 / 2112 | 64 / 48 / 56 | 0 / 0 / 0 |
| pass12 | 1385 / 1318 / 1388 | 832 / 832 / 832 | 56 / 56 / 64 | 0 / 0 / 0 |
| postpass | 3007 / 2922 / 9523 | 960 / 960 / 3264 | 72 / 128 / 128 | 0 / 16384 / 16384 |

## SDK→exact에서 직접 관찰한 변화

post는 dot4 960개와 image_store 12개가 유지되면서 3007→2922명령이 됐다. VGPR 72→128, LDS 0→16384bytes, 정규화 subgroup 14→8로 자원 점유 상한은 낮아졌다. 동시에 shared memory로 데이터를 모은 뒤 출력하는 명령이 생겼다. upstream의 coalesced store 설명과 일치하는 컴파일 관찰이며, image_store 개수 감소로 설명하면 틀리다.

pass11은 dot4 2112개, vector buffer load 14개, scalar buffer load 142개가 유지됐다. 명령 4045→4300(+255), VGPR 64→48, 정규화 subgroup 16→20이다. 상류의 메모리 접근 변경이 이 ACO 경로에서 단순 명령 감소로 나타난 것은 아니다. 실제 주소/데이터 배치의 효과와 정적 opcode 개수는 구분해야 한다.

| 출력 | pass | SDK 진단median µs | exact 진단median µs |
|---|---|---:|---:|
| 1920x1080 Native | pass11 | 39.320 | 42.160 |
| 1920x1080 Native | post | 173.580 | 124.280 |
| 3840x2160 Native | pass11 | 160.600 | 159.180 |
| 3840x2160 Native | post | 809.540 | 623.760 |

post는 정적 occupancy 상한이 낮아지면서도 진단 시간이 줄었다. 따라서 VGPR·occupancy 한 지표만 좋아지면 전체가 빨라진다는 판정은 성립하지 않는다. pass11의 개별 진단 시간이 악화되더라도 전체 exact의 효과와 동일한 주장은 아니다. 전체 time은 다른 pass의 이득과 dispatch 간 비용을 함께 포함한다.

lossy post는 9523명령·3264 static dot4로 다르고, 일부 network의 static dot4와 dispatch 형태도 다르다. 이 표의 목적은 실제 차이를 보존하는 것이다. 모델 재학습의 품질 판정과 동적 연산량은 이 표만으로 결론 내리지 않는다.

원자료: *_warm_profile_isa.json/.csv와 각 *_warm_profile_isa/의 bin/assembly. 재현: `python3 analysis/compare_variant_isa.py`. 알려진 upstream 변경 설명은 [고정 commit의 문서](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/blob/d1c3b30db656bf6a38634642542ee7736127d211/docs/how-it-works.md)이고 실제 ISA/시간과 구분해 읽는다.
