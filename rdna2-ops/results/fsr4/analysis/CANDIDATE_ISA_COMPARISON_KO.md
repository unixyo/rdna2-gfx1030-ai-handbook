# clamp 후보의 실제 ISA 비교

기준은 warm40/timed200의 exact_warm_profile이다. 별도 프로파일에서 PSO 원본 DXBC 56개가 모두 기준과 SHA256 일치한다. post 후보는 의도한 2개만 machine code가 바뀌고 나머지 54개가 byte 동일하다. all 후보는 의도한 20개만 바뀌고 나머지 36개가 byte 동일하다. 원본 DXIL/SPIR-V dump의 2개/20개 누락은 override를 탄 경로와 일치하며, 실제 코드 변경도 같은 대상을 가리킨다. 따라서 override 적용과 PSO 순서 대응은 실제 코드 변화로 뒷받침된다.

dot4 수, VGPR/SGPR allocation, LDS, scratch, spill, 정적 subgroup 상한은 모두 유지됐다. 새 ISA에는 v_perm_b32나 v_cvt_pk_i16_i32가 추가되지 않았다. 얻은 변화는 shift와 ReLU max를 줄여 clamp/byte 선택에 합친 것이다.

아래는 all 후보의 1080p 표다. 4K에서도 동일한 instruction/resource 변화가 관찰됐다. post 단독 후보의 post 변화는 all의 post와 같다.

| pass | ACO 명령 전→후 | raw 명령 전→후 | 새 padding NOP | ashr 감소 | max 감소 | VGPR | 정규화 subgroup/SIMD |
|---|---:|---:|---:|---:|---:|---:|---:|
| pass1 | 1317→1256 | 1317→1256 | 0 | 32 | 32 | 56 | 18 |
| pass2 | 1318→1273 | 1318→1273 | 0 | 16 | 32 | 56 | 18 |
| pass4 | 2470→2346 | 2470→2346 | 0 | 64 | 64 | 80 | 12 |
| pass5 | 2469→2328 | 2469→2328 | 0 | 80 | 64 | 80 | 12 |
| pass7 | 963→958 | 963→962 | 4 | 0 | 4 | 56 | 18 |
| pass8 | 964→959 | 964→962 | 3 | 0 | 4 | 64 | 16 |
| pass9 | 1227→1219 | 1227→1225 | 6 | 4 | 4 | 72 | 14 |
| pass10 | 2430→2354 | 2430→2354 | 0 | 16 | 64 | 80 | 12 |
| pass12 | 1318→1257 | 1318→1257 | 0 | 32 | 32 | 56 | 18 |
| postpass | 2922→2877 | 2922→2877 | 0 | 16 | 32 | 128 | 8 |

post는 2922→2877명령, code 18588→18408bytes, dot4는 960개다. VGPR 128·LDS 16384bytes·정규화 subgroup 8이 유지돼 allocation 경계 개선은 없다.

pass7/8/9는 ACO 명령 수가 5/5/8개 줄지만 assembler의 s_nop 4/3/6개가 새로 들어가 raw 명령 수는 1/2/2개만 줄었다. dump의 `(then repeated N times)`를 실제 machine word로 확장해 Code size와 대조했고, 전체 raw instruction 수와 ACO 수의 차이는 이 NOP 개수와 일치한다. original/normalized assembly와 bin을 모두 보존했다.

이 결과는 컴파일된 코드와 override 적용 확인이다. 출력 byte 동일성과 일반 total timing 교대 비교는 주 보고서의 별도 판정으로 연결한다. pass 프로파일의 단일 비교를 최종 성능 이득으로 판정하지 않는다.

재현: `python3 analysis/compare_candidate_isa.py`. 상세 opcode 변화와 SHA는 CANDIDATE_ISA_COMPARISON.json.
