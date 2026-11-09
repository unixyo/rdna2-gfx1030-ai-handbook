# FP16 Attention: 연산 단위와 실제 Anima A/B — v0.6

2026-11-07, RX 6900 XT/gfx1030. 실제 Anima 30step A/B 10요청과 4D 비트 보존 경로의 후속 warm2회를 완료했다. warm 중앙값은1.1906배·latent/RGB 비트 일치이며, 느렸던 최초83.018576초도 그대로 보존한다. 초회 비용의 구체 원인은 아직 분리하지 않았다.

## 1. 기준선·입력·판정 계약

- 성능 기준은 Anima 기존 `manual_vt`: contiguous Q/K → FP16 baddbmm(alpha=D^-0.5) → native softmax → 전치 저장 V와 bmm. 정확도 비교 기준은 설치된 PyTorch reduced-precision math SDPA이며 독립 CPU FP64 표본도 따로 계산했다.
- 실제 container 입력에서 캡처한 self `[2,16,4096,128]`, cross Q 동일/KV `[2,16,512,128]`를 사용했다. 세 번째 `[2,16,2048,128]`은 실제 self 캡처의 prefix slice다. 세 번째를 새 모델의 실제 호출로 표기하지 않는다.
- 3 warmup 후 3 round×2회, 순서를 forward/reverse로 교대했다. 전치·packing·할당·host dispatch를 wall 시간에 포함한다. 별도 CUDA event 시간과 모든 반복도 JSON에 있다.
- 연산 gate: max abs ≤0.02, relative RMSE ≤0.005. 최종 float RGB gate: max ≤2/255, RMSE ≤0.25/255. 이 둘을 혼동하지 않는다.
- CPU FP64는 두 head의 네 행에서 모든 key를 사용한 독립 표본이다. 전체 native 비교와 함께 사용하며 전체 FP64 검증이라고 부르지 않는다.
- 추가 메모리는 입력이 존재하는 상태에서 측정한 PyTorch allocated peak다. allocator reserved나 드라이버 내부 할당까지 포함하지 않는다.
- 전체 33개 결과가 native와 FP64 표본 수치 gate를 통과했다. GPU audit 207 samples, foreign client 0, 오류 0으로 이 측정은 유효하다.

## 2. 세 shape의 속도·메모리·오차

단위: ms, GiB. 원자료 `byte_equal`은 이 연산 벤치에서는 `torch.equal`을 사용한다. 즉 dtype·값 완전 일치이며 signed-zero의 바이트 차이까지 구분한 검사는 아니다. 모델 latent/RGB는 별도 digest로 실제 바이트를 비교했다.

| shape | 경로 | wall ms | manual VT 대비 | 추가 GiB | native max | native RMSE | 값 일치 |
|---|---|---:|---:|---:|---:|---:|---|
| self4096 | `native_math` | 67.6897 | 0.333× | 2.5314 | 0 | 0 | 예 |
| self4096 | `manual_nn` | 51.7738 | 0.436× | 2.1250 | 0.0009765625 | 2.350295e-05 | 아니오 |
| self4096 | `manual_vt` | 22.5611 | 1.000× | 2.1250 | 0.0009765625 | 2.35417e-05 | 아니오 |
| self4096 | `post_out_vt` | 22.4312 | 1.006× | 2.1250 | 0.0009765625 | 2.35417e-05 | 아니오 |
| self4096 | `post_inplace_vt` | 23.0865 | 0.977× | 1.1250 | 0.0009765625 | 2.353526e-05 | 아니오 |
| self4096 | `prescale_native` | 56.0667 | 0.402× | 2.1250 | 0.0004882812 | 8.514866e-07 | 아니오 |
| self4096 | `prescale_vt` | 21.8603 | 1.032× | 2.1250 | 0.0009765625 | 4.008783e-06 | 아니오 |
| self4096 | `fused_prescale_vt` | 22.3588 | 1.009× | 2.1250 | 0.0009765625 | 4.008783e-06 | 아니오 |
| self4096 | `fused_prescale_inplace_vt` | 22.7569 | 0.991× | 1.1250 | 0.0009765625 | 4.398479e-06 | 아니오 |
| self4096 | `prescale4d_native` | 54.3322 | 0.415× | 2.0625 | 0 | 0 | 예 |
| self4096 | `prescale4d_vt` | 32.9271 | 0.685× | 2.0625 | 0.0009765625 | 3.937509e-06 | 아니오 |
| cross512 | `native_math` | 3.8428 | 0.768× | 0.3439 | 0 | 0 | 예 |
| cross512 | `manual_nn` | 3.1625 | 0.933× | 0.3203 | 3.051758e-05 | 1.609173e-06 | 아니오 |
| cross512 | `manual_vt` | 2.9514 | 1.000× | 0.3203 | 3.051758e-05 | 1.609177e-06 | 아니오 |
| cross512 | `post_out_vt` | 2.9587 | 0.998× | 0.3203 | 3.051758e-05 | 1.609177e-06 | 아니오 |
| cross512 | `post_inplace_vt` | 2.9235 | 1.010× | 0.1953 | 3.051758e-05 | 1.609123e-06 | 아니오 |
| cross512 | `prescale_native` | 2.6084 | 1.132× | 0.3203 | 3.051758e-05 | 3.174533e-08 | 아니오 |
| cross512 | `prescale_vt` | 2.3063 | 1.280× | 0.3203 | 3.051758e-05 | 3.852275e-08 | 아니오 |
| cross512 | `fused_prescale_vt` | 2.8229 | 1.046× | 0.3203 | 3.051758e-05 | 3.852275e-08 | 아니오 |
| cross512 | `fused_prescale_inplace_vt` | 2.7557 | 1.071× | 0.1953 | 3.051758e-05 | 6.234875e-08 | 아니오 |
| cross512 | `prescale4d_native` | 2.7155 | 1.087× | 0.2852 | 0 | 0 | 예 |
| cross512 | `prescale4d_vt` | 2.4730 | 1.193× | 0.2852 | 3.051758e-05 | 2.184006e-08 | 아니오 |
| cropped_self2048 | `native_math` | 10.4147 | 0.595× | 0.6407 | 0 | 0 | 예 |
| cropped_self2048 | `manual_nn` | 5.8549 | 1.058× | 0.5625 | 0.0009765625 | 2.495738e-05 | 아니오 |
| cropped_self2048 | `manual_vt` | 6.1941 | 1.000× | 0.5625 | 0.0009765625 | 2.495902e-05 | 아니오 |
| cropped_self2048 | `post_out_vt` | 6.1675 | 1.004× | 0.5625 | 0.0009765625 | 2.495902e-05 | 아니오 |
| cropped_self2048 | `post_inplace_vt` | 6.1563 | 1.006× | 0.3125 | 0.0009765625 | 2.495419e-05 | 아니오 |
| cropped_self2048 | `prescale_native` | 6.0038 | 1.032× | 0.5625 | 0 | 0 | 예 |
| cropped_self2048 | `prescale_vt` | 5.9903 | 1.034× | 0.5625 | 0.0009765625 | 2.99215e-06 | 아니오 |
| cropped_self2048 | `fused_prescale_vt` | 6.1400 | 1.009× | 0.5625 | 0.0009765625 | 2.99215e-06 | 아니오 |
| cropped_self2048 | `fused_prescale_inplace_vt` | 6.0123 | 1.030× | 0.3125 | 0.0009765625 | 3.490971e-06 | 아니오 |
| cropped_self2048 | `prescale4d_native` | 7.9540 | 0.779× | 0.5312 | 0 | 0 | 예 |
| cropped_self2048 | `prescale4d_vt` | 9.4186 | 0.658× | 0.5312 | 0.0009765625 | 2.99215e-06 | 아니오 |

## 3. 구현별 성공·실패와 원인

| 후보 | 관측 | 연산 단계 판정 |
|---|---|---|
| `manual_nn` → `manual_vt` | self4096의 느린 NN PV 커널을 NT로 바꾼다. crop2048에서는 오히려 NN이 더 빠르다. | shape 조건 필요. 최종 출력은 별도 A/B |
| `post_out_vt` | score baddbmm의 출력 buffer를 재사용하지만 큰 peak는 score+P가 지배한다. | 의미 있는 일관된 추가 가속 없음 |
| `post_inplace_vt` | owned score에 Triton row-softmax를 덮어써 self 추가 peak 2.125→1.125 GiB. self 시간은 22.561→23.087 ms로 퇴보. | 속도 교체 거절, 메모리 후보만 보류 |
| `prescale_vt` | native와 같은 Q/K sqrt-scale 순서를 사용하고 flatten bmm+VT PV를 쓴다. self 1.032×, cross 1.280×. native 대비 RMSE는 manual보다 크게 작아짐. | 모델 A/B 대상으로 선택 |
| `fused_prescale_vt` | Triton scale+pack은 동작하나 이 shape에서는 Torch 경로보다 느리다. | 추가 융합 채택 거절 |
| `fused_prescale_inplace_vt` | 낮은 peak와 작은 오차를 함께 얻지만 빠른 prescale_vt보다 느리다. | 메모리 전용 후보, 기본 제외 |
| `prescale_native` | sqrt-scale은 같아도 reshape/bmm의 QK 커널이 native와 달라 self/cross에서 비트가 다르다. | 비트 보존 경로로 채택하지 않음 |
| `prescale4d_native` | q×sqrt(scale), k.transpose×sqrt(scale), 4D matmul, native softmax, row repair, 원래 V의 4D matmul을 유지. 세 shape 모두 dtype·값 완전 일치. | 모델 비트 보존 A/B 대상으로 선택 |
| `prescale4d_vt` | QK는 native와 같은 선택 커널이지만 V 전치 후 4D matmul이 추가 pack을 만들 수 있다. 빠른 flattened VT보다 느리다. | 속도 후보에서 제외 |

## 4. 실제 선택 커널과 정밀도 원인

self4096 native/prescale4d_native QK는 `Cijk_Ailk_Bljk_HHS_BH_MT128x128x16...SU32...WGM4.kd`, flatten prescale의 QK는 `Cijk_Alik_Bljk_HHS_BH_MT128x64x8...SU0...WGM8.kd`로 달랐다. native의 NN PV는 `...Ailk_Bljk...MT128x128x16...EPS1...`, VT PV는 `...Alik_Bljk...MT128x256x16...EPS0...`를 선택했다. 전체 이름·개별 시간은 원자료 `kernel_profile.rows`에 있다. 커널 귀속 시간과 CPU operator 시간을 더하지 않는다.

native self profile 한 회의 isneginf/reduction/where는 각각 3.611/1.526/4.497 ms였다. row repair는 별도 측정에서 약 0.73 ms다. profile 호출은 진단이고 warm wall 중앙값의 대체가 아니다.

이번 실제 Q/K/V 캡처는 모두 finite였다. unscaled QK의 보수적 절대 상한 D·max|Q|·max|K|는 self 2366.26, cross 2260.60, crop 2180.06으로 FP16 최대 65504보다 작다. 따라서 **이 캡처에서 시작된 오차를 QK overflow로 설명할 수 없다**. 이후 모든 step의 범위를 증명한 것은 아니다. scale을 alpha로 옮기는 중간 FP16 반올림 변화와 layout에 따른 실제 GEMM 선택 변화가 관측된 차이다.

native에 가까운 것과 실수 연산에 더 가까운 것은 구분한다. CPU FP64 표본에서 manual과 native는 둘 다 작은 오차였고, cross에서는 manual의 RMSE가 오히려 더 작았다. 최종 RGB 기준은 native 재현성을 묻는 기준이며 FP64 정확도 순위를 뜻하지 않는다.

## 5. 실제 모델 최종 출력 A/B

동일 프로세스·conditioning·seed20261107·1024²·30step에서 10요청을 수행했다. warm/중간 baseline/마지막 baseline의 latent 및 RGB는 비트 일치다. 모든 후보는 self840+cross840=1680회 적중, fallback0이다. denoise와 VAE 시간을 따로 재었고 VAE는 후보 적용 밖이다. GPU audit2761 samples, foreign0/error0, 유효.

| 경로 | denoise 초(실행 순서) | RGB max | RGB RMSE | 판정 |
|---|---|---:|---:|---|
| native 대조 | 80.796039, 80.750761 | 0 | 0 | latent/RGB 비트 일치 |
| 직접 NN | 63.723788 | 0.8203125 | 0.024798754 | 엄격 실패 |
| 기존 직접 VT | 56.952527, 57.022677 | 0.824707031 | 0.0243068023 | 엄격 실패 |
| 새 prescale VT | 57.148174, 57.178169 | 0.805664062 | 0.024888045 | 엄격 실패 |
| 4D native 보존 | 83.018576, 67.726131 | 0 | 0 | latent/RGB 비트 일치 |

기존 직접 VT의 중앙값은 56.987602초로 native 80.773400초 대비 1.417배다. 새 prescale VT는 57.163172초로 기존 직접 VT보다 약 0.3% 느렸고, RGB max는 조금 줄었지만 RMSE는 증가했다. 따라서 **추가 속도 또는 두 오차 기준의 동시 개선 후보로 채택하지 않는다.**

NN→VT만 분리해도 최종 RGB max0.69140625, RMSE0.0146536561로 strict 실패다. layout은 입력 값을 바꾸지 않지만 다른 GEMM 누산·반올림 경로를 선택했고 그 차이가 최종 출력까지 전파됐다. 단계별 증폭 지점은 이번 결과만으로 확정하지 않는다.

4D native 보존은 위의 두 회 모두 latent/RGB byte gate를 통과했다. 첫83.018576초와 후속67.726131초의 차이는 크며 이 첫 실험의 두 값을 포함한 중앙값은75.372354초다. 초기 JIT/준비 비용은 가능한 설명이지만 로그로 원인을 확정하지 않았다. **느린 첫 결과를 버려 warm 수치로 바꾸지 않고**, 아래 후속 실험을 별도 기록한다.

### 5.1 후속 warm 재현 — A

정규화 모델 A/B와 같은 서버에서 norm 후보를 끈 뒤 `attn_prescale4d_native`만 두 번 실행했다. 먼저 여러 Triton norm을 실행한 상태이므로 이 결과는 warm 범위다.

| 구분 | denoise 초 | 최종 출력 |
|---|---:|---|
| 같은 서버 native control | 80.60703 / 80.71983 | 이전 실험 native와 latent/RGB byte digest 동일 |
| 4D native 보존 warm | 67.77495 / 67.72710 | 두 회 모두 latent/RGB bytes 일치, RGB 오차0 |

중앙값80.66343→67.75103초, **1.1906배**다. 각 후보1680hit/fallback0, positive/negative conditioning digest도 이전 실험과 같다. native 및 후보 RGB SHA는 `52ea16286731f37e2760ef622b66446443e95e657aea57dc9b0dd511b853cb3a`, latent는 `cc054fd19dc8f334da8d9acaf3b0b53aab1165944a6023a38c6c1ee9a34c3a13`이다. GPU audit3178 samples, 외부 compute/오류0, valid.

현재 guarded kernel source `a207e62e...`와 adapter `dad380e7...`, reduced math flag=True를 runtime/candidate metadata에서 확인했다. [후속 요청 원자료](v06_anima_norm_ab/runs.json), [GPU audit](v06_anima_norm_ab/gpu_audit.json), [정규화와 함께 실행한 기록](NORM_FUSIONS_V06_KO.md).

첫83.018576초는 당시 서버의 첫 Triton row-repair 후보였을 가능성이 있다. 새 서버에서도 첫 Triton LN 후보만94.312초로 느린 패턴을 보였으나 runtime 초기화·JIT·host 대기를 분리 측정하지 않아 원인을 확정하지 않는다. 새 서버에서 단 한 번 생성하는 경우의 가속을 warm1.1906배로 보증하지 않는다.

엄격 기준이 구조적으로 불가능하다는 주장은 이 결과와 맞지 않는다. 정확한 native 경로를 유지하면 비트가 일치한다. 아직 찾지 못한 것은 약57초인 빠른 VT 경로의 속도와 strict 출력 보존을 동시에 달성하는 후보다.

[10요청 원자료](v06_anima_attention_ab/runs.json), [요약](v06_anima_attention_ab/summary.json), [GPU audit](v06_anima_attention_ab/gpu_audit.json).

## 6. 범위와 재사용

- 공개 API `rdna2_ops.fp16_attention_v06.fp16_attention`는 gfx1030, FP16 BHLD, D64/128, inference, unmasked/no-dropout/no-GQA, finite 입력/score 계약으로 명시 호출한다. 이번 v0.6 세 shape는 D128이다. D64까지 이번 모델 실측이 확장된 것은 아니다.
- sampler 중에만 `attention_switches_v06.activate`로 연결하고 `finally`에서 원래 함수를 복원한다. 설치된 ComfyUI나 custom node의 기본 dispatch를 수정하지 않는다.
- backward·mask·dropout·GQA·compile·graph 재생은 새 경로의 검증 범위가 아니다. 후보 이름이 지원 범위를 넓혀 주지 않는다.
- `prescale4d_native`는 `fp16_bf16_reduction_math_sdp_allowed()==True`인 현재 FP16 math 계약을 대상으로 한다. 현재 API는 이 flag를 검사하고 adapter는 flag=False일 때 원본에 맡긴다. 이 guard는 최초 실측 뒤 추가했으며 계산식은 바꾸지 않았다. [실측 당시 확장 소스](sources/fp16_attention_v06_expanded.py)를 현재 guarded 소스와 구분해 보존했다.
- 엄격 기준의 구조적 불가능성을 입증한 결과는 없다. 동일 값 경로가 연산 단계에 있고 모델 바이트까지 확인됐으므로 빠른 layout과 동일 속도에서 strict를 통과할 수 있는지는 별도 연구 질문이다.

## 7. 지각 기준을 사용할 때의 판정 제안

엄격 RGB 통과 여부와 지각 판단을 한 칸에 섞지 않는다. 원본/후보를 A/B로 익명화하고 좌우 위치를 바꿔 같은 크기의 전체 이미지와 고정 위치 crop을 나란히 비교한다. 얼굴·손·텍스트·경계·미세 질감·색을 공통 항목으로 삼고 원본보다 열화/동등/개선으로 기록한다. 후보를 만든 연구자가 아닌 독립 판정자 또는 별도 vision model에 두 방향 순서로 제시하고, 모델명·버전·프롬프트·원응답을 보존한다. 다른 seed와 프롬프트에서도 같은 절차를 반복한 뒤 합격 비율과 불확실성을 정해야 한다. 한 장의 생성 결과나 자기 평가만으로 지각 동등을 승인하지 않는다. 이번 연구에서 새 판정 모델은 다운로드하지 않는다. 독립 모델의 실제 판정이 없으면 `지각 미판정`이라고 쓴다.

## 원자료와 재현

- [확장 33조합 결과](fp16_attention_v06_expanded.json), SHA-256 `82dca9c1241348bc16aff012f816152666dd2da6366485dd26d5f582a2b20249`.
- 초기 24조합은 [fp16_attention_v06.json](fp16_attention_v06.json)에 별도 보존. 초기 kernel module은 [해당 SHA의 소스](sources/fp16_attention_v06_initial.py)에 보존.
- 실행: 기존 `rdna2_python bench_fp16_attention_v06.py --capture-dir results/v06_anima_capture --out results/새_결과.json`. 출력 파일이 있으면 덮어쓰지 않고 실패한다.
- A/B: `rdna2_python research_comfy_v05.py --spec research_configs/anima_v06.json --out results/새_Anima_AB --steps 30 --seed 20261107 --modes warm,attn_manual_nn,attn_manual_vt,attn_prescale_vt,attn_prescale4d_native,baseline --port 8473`. GPU guard로 다른 compute job이 감지되면 timing을 폐기한다.
