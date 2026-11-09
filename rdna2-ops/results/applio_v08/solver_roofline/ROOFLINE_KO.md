# v0.8 현재 FAST: shape·방향별 시간과 낙관적 하한 비교

- 144 shape, 432 방향, strict gate 432/432.
- dot2 register calibration 48.725 TFLOP/s; streaming copy 436.745 GB/s; cached copy 801.970 GB/s.
- 아래는 현재 시간 상위 20방향이다. 전체 432방향은 shape_direction.csv와 roofline.json.
- dense FLOP 기반 수치는 직접 convolution 가정이다. Winograd는 연산량이 줄어 이 비율을 하한으로 해석하지 않는다.
- DRAM bandwidth 기반 최소시간은 streaming 가정이다. 캐시 내 재사용·추가 im2col traffic·LDS·점유율·launch 비용을 분리해서 해석한다.
- 이 표의 kernel 이름은 실제 trace다. solver ID는 `observed_solver`/`observed_solver_status`, DB 후보는 `compatible_db_solver_ids`/`solver_candidate_status`로 구분한다. singleton 요청을 관측값으로 쓰지 않는다.
- 기존 증거에서 kernel 관측 432행, 호환 DB 후보 1개 183행·복수 126행·기록 없음 123행이다. 같은 kernel을 낸 singleton rank 141방향 중 7방향은 서로 다른 f2x3/g1 요청 ID가 같은 kernel을 냈다.
- native ID metadata probe를 B100 뒤 완료했다. **432/432행 `confirmed_native_metadata_replay`, 미확정·복수·누락 0**이다. 각 방향에 최종 Chosen ID 1개와 실제 API 호출 1개를 대응했고 출력 finite·audit clean·baseline DB 전후 동일성을 확인했다. 이번 probe는 timing false이며 기존 유효한 시간·정확도 값은 그대로다.
- 관측은 동일 FP16 descriptor·seed·FAST DB를 사용한 별도 metadata replay 범위다. 실행 계획은 `baseline_metadata_probe_plan.json`, CPU parser·집계는 `update_solver_attribution.py`·`solver_attribution.json`, 상세 설명은 `../../../v08_solver_notes_ko.md` §9를 따른다.
- 실제 ID 9종의 방향 수: Winograd g1 69, f3x2 37, f2x3 20; GemmFwdRest 73, GemmBwdRest 75, GemmWrwUniversal 83, GemmFwd1x1_0_1 28, GemmBwd1x1_stride1 29, GemmWrw1x1_stride1 18. 기존 rank kernel만으로 모호했던 7방향도 f2x3 5개 / g1 2개로 구별했다.

| shape | 방향 | 현재 event ms | direct compute ms | streaming traffic ms | cached traffic ms | direct 가정 대비 | im2col/col2im 비중 |
|---|---|---:|---:|---:|---:|---:|---:|
| conv1d_B8_C64_T6400_O64_K11_S1_D3_G1_P15 | dw | 4.126205 | 0.094689 | 0.030217 | 0.016456 | 43.6× | 10.8% |
| conv1d_B8_C64_T6400_O64_K11_S1_D5_G1_P25 | dw | 4.120712 | 0.094689 | 0.030217 | 0.016456 | 43.5× | 12.7% |
| conv1d_B8_C64_T6400_O64_K7_S1_D5_G1_P15 | dw | 4.010777 | 0.060257 | 0.030142 | 0.016415 | 66.6× | 10.2% |
| conv1d_B8_C64_T6400_O64_K7_S1_D3_G1_P9 | dw | 4.009180 | 0.060257 | 0.030142 | 0.016415 | 66.5× | 10.2% |
| conv1d_B8_C64_T6400_O64_K3_S1_D3_G1_P3 | dw | 2.535757 | 0.025824 | 0.030067 | 0.016374 | 84.3× | 14.0% |
| conv1d_B8_C64_T6400_O64_K3_S1_D5_G1_P5 | dw | 2.534849 | 0.025824 | 0.030067 | 0.016374 | 84.3× | 14.0% |
| conv_transpose1d_B8_C64_T6400_O32_K4_S2_D1_G1_P1 | dw | 2.411054 | 0.017216 | 0.030049 | 0.016364 | 80.2× | 9.2% |
| conv1d_B8_C32_T12800_O32_K3_S1_D3_G1_P3 | dw | 1.921585 | 0.012912 | 0.030025 | 0.016351 | 64.0× | 18.9% |
| conv1d_B8_C32_T12800_O32_K3_S1_D5_G1_P5 | dw | 1.921567 | 0.012912 | 0.030025 | 0.016351 | 64.0× | 18.9% |
| conv1d_B8_C32_T12800_O32_K3_S1_D1_G1_P1 | dw | 1.921486 | 0.012912 | 0.030025 | 0.016351 | 64.0× | 18.8% |
| conv1d_B8_C1_T12800_O16_K15_S1_D1_G1_P7 | dw | 1.571708 | 0.001009 | 0.007973 | 0.004342 | 197.1× | 1.9% |
| conv1d_B8_C32_T12800_O32_K11_S1_D5_G1_P25 | dw | 1.530760 | 0.047344 | 0.030063 | 0.016372 | 32.3× | 35.4% |
| conv1d_B8_C1_T12800_O32_K1_S1_D1_G1_P0 | dw | 1.517568 | 0.000135 | 0.015475 | 0.008427 | 98.1× | 0.0% |
| conv1d_B8_C128_T3200_O128_K11_S1_D5_G1_P25 | dw | 1.495403 | 0.189378 | 0.030836 | 0.016793 | 7.9× | 36.5% |
| conv1d_B8_C32_T12800_O32_K11_S1_D3_G1_P15 | dw | 1.466578 | 0.047344 | 0.030063 | 0.016372 | 31.0× | 32.3% |
| conv1d_B8_C64_T6400_O64_K7_S1_D1_G1_P3 | dw | 1.452073 | 0.060257 | 0.030142 | 0.016415 | Winograd: 적용 안 함 | 0.0% |
| conv1d_B8_C64_T6400_O64_K3_S1_D1_G1_P1 | dw | 1.449994 | 0.025824 | 0.030067 | 0.016374 | Winograd: 적용 안 함 | 0.0% |
| conv1d_B8_C64_T6400_O64_K11_S1_D1_G1_P5 | dw | 1.447857 | 0.094689 | 0.030217 | 0.016456 | Winograd: 적용 안 함 | 0.0% |
| conv1d_B8_C32_T12800_O32_K11_S1_D1_G1_P5 | dw | 1.435387 | 0.047344 | 0.030063 | 0.016372 | 30.3× | 30.2% |
| conv1d_B8_C128_T3200_O128_K11_S1_D3_G1_P15 | dw | 1.420881 | 0.189378 | 0.030836 | 0.016793 | 7.5× | 33.1% |
