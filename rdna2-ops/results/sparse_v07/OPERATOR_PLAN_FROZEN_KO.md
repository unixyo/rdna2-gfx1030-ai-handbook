# Sparse v0.7 operator 실행 계획 확정

확정 시점: 9개 native map·all-true model control 이후, 실제 5-case operator benchmark 이전.

- 실제 서로 다른 shape 3개: Q2400/K2408, Q156/K2564(masked prefix), Q2400/K4964; 모두 B1/H32/D128 FP32.
- 같은 long shape의 실제 representative QKV 2개 추가: native step 12/layer 15, step 24/layer 31. layer 0에서 Sparge 보호가 dense로 돌아가는 자료만으로 성능을 판정하지 않기 위함이다.
- 타일은 정확히 두 개: q32/k64/d16/pv16/w8과 q16/k64/d16/pv16/w4. 이 결과 뒤 tile/shape/selector 격자를 더 늘리지 않는다.
- 사전 정확도 gate atol=3e-5, rtol=3e-4는 유지한다. 독립 FP64 3문제를 두 tile 모두 검증한다. 같은 mask gate 실패는 채택 후보에서 제외한다. 실패한 경우 가장 큰 차이의 3개 row를 CPU FP64로 진단하며, 선택 row 결과를 전체 정확도 통과로 바꾸지 않는다.
- static mask 준비 시간은 별도, 동적 Sparge selector는 매 호출 전체 시간에 포함한다. model은 mask 고정 비용뿐 아니라 density 계측·분기·서버 통합 비용까지 포함한다.
- 최초 native/candidate 실제 호출과 준비/JIT 가능 시간은 따로 기록한다. correctness 이후 첫 timing 호출은 cold JIT가 아니며 그렇게 표시한다. 2 warmup 후 순서 회전/반전 3×2 call 중앙값, allocator incremental peak, regs/spill/ISA와 native kernel trace를 남긴다.
- 선택: 같은-mask 계산을 통과한 후보 중 실제 long-sequence 대표 자료의 selector 포함 속도, 9개 map 보존 질량을 함께 보고 가장 유망한 하나를 model 3 seeds로 비교한다. 속도 이득이 없거나 strict 실패면 자동 적용하지 않는다. RGB를 보고 threshold를 느슨하게 바꾸지 않는다.
- 원 temporal Radial f1/f2가 dense인 사실은 CPU/source 근거로 별도 기록한다. image-only 2D radius는 다른 inspired 후보다.

소스:
- bench_sparse_attention_v07.py: c46c44edb4d892eee06bf583c0e75db3cf1d2cb7b11b186b16bf708a91e062c3
- rdna2_ops/sparse_attention_v07.py: f5d556a26151abb0e6501999c1125269e6e7df05836d50cebd0186291d7d3039
- sparse_switches_v07.py: 601ea53d6586b54a2c42c0a4bd351c14dae0a3bdcc10b7076fecd998464c736b
- instrument_sparse_capture_v07.py: f176735171f714d634f7a0eeca7a63c5ff8763196e4bd919afec4d0811e9d1a7

모델 검증 전 계측 수정: prefix dense 32회는 이미 hits에 포함되므로 fallback에 다시 더하지 않고 prefix_dense_calls로 별도 기록한다. 이전 all-true control의 fallback=96은 masked64+prefix32를 뜻하며 총 호출에 단순 합산하면 안 된다. 커널/계산식에는 변화가 없다. 현재 sparse_switches_v07.py SHA: 60a59bee493a5609b49b5308b3b07bcaa16e284b214e27273b4ce05de69a6fc3
