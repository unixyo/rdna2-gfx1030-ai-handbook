# v0.7 Claude 재현용 핵심 숫자 5개

작성: 2026-11-07 KST. 모든 명령은 **이 연구 PC의 기존 환경**에서 실행한다. GPU 작업은 하나씩 실행하며 각 하네스의 `gpu_guard_v06` 관문을 유지한다. 운영 서비스를 시작·중지·재시작하는 명령은 없다. 출력 폴더가 이미 있으면 덮어쓰지 말고 새 이름을 쓴다.

## 공통 준비

~~~bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
moss_runtime=~/DevSpace/moss-image-rdna2-lab01
qwen_python=~/DevSpace/qwen-image-21-rdna2-lab01/.venv/bin/python
export HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1=0
export MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53=0
~~~

이 두 MIOpen 변수는 각각 FP16 forward 오답 solver와 FP32 dW NaN solver를 막는다. `OCL_WRW53`라는 옛 이름을 쓰지 않는다. 시간은 같은 서버의 초회와 warm을 분리한다. 통상적인 실행 변동이 있으므로 아래 소수점의 완전 일치가 아니라 경로·정확도·반복 분포를 함께 확인한다.

## 1. INT4 G128의 BK64 분리: Qwen 2.032배, 기존 native 대비 0.663배

Qwen projection의 이전 W4 7.165080 → 3.525428 ms, Triton `n_spills` 137 → 18이다. MOSS/Anima projection도 각각 1.819/1.940배 개선됐다. 대칭·비대칭 여섯 비교에서 **이전 W4 출력 전체 byte 일치**다. 이것은 이전 W4 구현의 개선이며, 기존 모델의 INT8/FP16 native보다 빠르다는 결과가 아니다.

~~~bash
bash "$moss_runtime/scripts/with_local_storage.sh" "$moss_runtime/.venv-moss/bin/python" -u bench_int4_v07.py --mode split_smoke --out results/int4_v07/repro_subk_smoke
bash "$moss_runtime/scripts/with_local_storage.sh" "$moss_runtime/.venv-moss/bin/python" -u bench_int4_v07.py --mode sweep --cases moss_proj,qwen_proj,anima_proj --groups 128 --quant sym,aff --layouts kn --dot4-tile 64,64 --methods dot4,dot4_split --out results/int4_v07/repro_subk_screen
~~~

원자료: [subk_screen01](int4_v07/subk_screen01/results.json), [전체 보고서](int4_v07/INT4_GEMM_V07_KO.md).

## 2. MOSS W4 계산 정확성과 표현 변경 경로를 분리

512²·4-step·seed 20261105에서 W4 독립 참조와 custom dot4의 **latent/RGB byte 일치, max/RMSE 0**를 확인한다. 원래 INT8 fused와 W4 경로의 최종 RGB는 **최대 240.370/255, RMSE 45.288/255**여서 엄격 실패다. 실제 치환 672회, fallback 0이다. W4 모델 경로에는 매 호출 pack이 포함되므로 이것을 성능 채택 비교로 쓰지 않는다.

~~~bash
bash "$moss_runtime/scripts/with_local_storage.sh" "$moss_runtime/.venv-moss/bin/python" -u research_comfy_v05.py --spec research_configs/moss_int4_v07.json --out results/int4_v07/repro_moss_model --seed 20261105 --steps 4 --modes warm,int8_fused,int4_reference,int4_dot4,int8_fused --port 18671
~~~

원자료: [moss_model01/runs.json](int4_v07/moss_model01/runs.json). PNG 파일 metadata SHA 대신 decoded RGB pixel SHA와 저장 tensor byte 비교를 확인한다.

## 3. Qwen r4 희소 attention 연산: 1.219배, 추가 allocator peak 3,102.594 → 38.000 MiB

실제 FP32 B1/H32/Q2400/K4964/D128의 step 0 / layer 0 캡처다. `split_alpha` 22.665616 ms → `radial_r4_prepared` 18.599989 ms다. 동일 mask oracle의 계산 관문은 이 early case에서 통과했다. 이는 전체 5-case 통과를 뜻하지 않는다. late case는 고정 native 허용 오차 관문에 실패했다. mask는 사전 준비한 정적 2D image r4이며, 원래 temporal Radial의 이득으로 일반화하지 않는다.

~~~bash
"$qwen_python" bench_sparse_attention_v07.py --out results/sparse_v07/repro_target --captures results/sparse_v07/capture25_seed20261107_retry/sparse_capture/actual_qkv_2400_4964.pt
~~~

QKV `.pt`는 PC의 원자료 폴더에 남아 있다. 배포 묶음에는 대용량 tensor를 넣지 않는다. 재캡처 절차는 [Sparse 보고서](sparse_v07/SPARSE_ATTENTION_V07_KO.md)를 따른다. memory는 입력·prepared mask를 제외한 PyTorch allocator 추가 peak이며, 전체 모델 VRAM이 아니다. 원자료: [5-case 요약](sparse_v07/OPERATOR5_SUMMARY_KO.md).

## 4. 같은 Qwen 실제 25-step 모델: warm 1.09657배지만 엄격 RGB 실패

seed 20261107의 alpha 전후 중앙값 50.874496 s → 두 번째 r4 46.394129 s, **1.09657185배**다. 첫 native 66.895564 s와 초회 r4 46.431769 s는 warm 중앙값에 넣지 않는다. r4 처음/두 번째의 latent/RGB byte는 같다. native↔alpha도 byte 일치다.

r4의 최종 RGB는 **최대 243.441/255, RMSE 49.800/255**로 엄격 기준 최대 2/255·RMSE 0.25/255를 크게 넘는다. 지각 판정은 원응답과 별도이며 이 수치 실패를 없애지 않는다.

~~~bash
"$qwen_python" research_comfy_v05.py --spec research_configs/qwen_sparse_radial4_v07.json --out results/sparse_v07/repro_model20261107 --steps 25 --seed 20261107 --modes baseline,dense_alpha,sparse_radial,sparse_radial,dense_alpha --port 8487
~~~

비율은 `dense_alpha` index 1/4의 denoising 중앙값 ÷ `sparse_radial` index 3의 denoising이다. API wall에는 원자료 저장 비용도 있으므로 섞지 않는다. 원자료: [seed07](sparse_v07/model_seed20261107/runs.json).

## 5. Strata 실제 IQ pack 연산: GU 1.090배 / down 1.080배, 54행 byte 일치

16 expert의 세 가지 합성 행 분포에 실제 layer 35 가중치를 쓴다. skew_tail의 평균 행 힌트는 실제 J64 → J48을 선택하며, GU 440.676 → 404.290 μs, down 197.342 → 182.684 μs다. medium/large는 두 경로가 모두 J64라서 그 비율은 가속으로 세지 않는다. 탐색 최저 J40/J32는 별도 탐색값이며 자동 정책이 아니다.

~~~bash
/usr/bin/python3 -u strata_operator_v07.py --out results/strata_v07/reproduce_operator_01
~~~

연산 자료: [operator_summary.json](strata_v07/operator_summary.json), [raw JSONL](strata_v07/operator_screen/operator.jsonl).

**9.7K/19.8K/119K warm fresh 모델 A/B, 모델 token ID 일치, warm GPU 계산/PCIe 병목 판정은 미완료다.** 첫 fresh 요청에 대량 파일 읽기·swap이 섞였고, 후속 두 번은 동일 자원 보호 조건으로 종료했다. 이 명령의 연산 성공을 모델 성공으로 옮기지 않는다. 서비스가 이미 꺼져 있고 자원 조건이 달라졌을 때만 [Strata 보고서 §8.2](strata_v07/STRATA_PREFILL_V07_KO.md)의 연구 서버 절차를 재개한다.
