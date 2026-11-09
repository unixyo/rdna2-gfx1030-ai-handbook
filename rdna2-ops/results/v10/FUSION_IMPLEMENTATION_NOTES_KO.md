# Applio v1.0 ResBlock 융합 구현 메모

## 상태와 범위

- 작성: 2026-11-08. bias reduction 경계를 복구한 최종 source의 18쌍 검증을 완료했고 12쌍을 선택했다. FP32 참조 metric 108/108과 AMP·weight_norm 계약 metric 144/144가 통과했다. 구형 source의 첫 smoke는 맨 아래 이력으로 분리했다.
- 구현: [v10_fused_resblock.py](../../v10_fused_resblock.py).
- 검증 도구: [bench_v10_fusion.py](../../bench_v10_fusion.py).
- 운영 파일과 기존 v0.8 연구 파일은 수정하지 않는다. Python 프로세스 내부 context에서 `ResBlock.forward`만 임시 연결하고 종료 시 원래 메서드를 복원한다.
- B는 v0.8의 실제 선택 manifest이다. 융합 B′도 그 방향 선택을 유지한다. 둘 모두 forward가 선택된 ResBlock Conv 쌍만 대상이다. 초기 manifest에서 대상은 18개 쌍이다.
- 융합 12쌍과 새 native 대체 68방향을 함께 넣은 실제 운영 short 비교는 12.5421→11.4475 s/epoch, 1.09561배였다. 이 수치를 융합 단독의 epoch 효과로 해석하지 않는다. 100 epoch B/B′ 전체 결과와 사용자 청취 판정은 대기 중이며 [주 보고서](APPLIO_V10_KO.md)를 따른다.

## 최종 source의 전체 GPU 검증

정본은 [fusion_full/result.json](fusion_full/result.json)과 [selection.json](fusion_full/selection.json)이다. `completed`, `valid_for_timing`, `valid_for_contract`는 모두 true다. 고정 seed 20261108, FP16 입력, 5 round × 10 rep으로 18쌍을 측정했다. FP32는 연산 참조와 master·gradient dtype이며, 이번 전체 실행에서 FP32 입력까지 측정했다는 뜻은 아니다.

| 항목 | 확정 결과 |
|---|---|
| 검증·선택 쌍 | 18쌍 / C32·C64 12쌍 |
| 개별 연산 FP32 참조 metric | 108/108 통과; 최대 NRMSE 0.000438903, 최대 error/reference RMS 0.00442590 |
| 실제 ResBlock AMP·weight_norm 계약 metric | 144/144 통과; 이 고정 입력의 B/B′ 호출 경계는 모두 바이트 일치 |
| 선택된 쌍의 event 배속 | 1.039106–1.134607배 |
| 선택된 쌍의 wall 배속 | 1.039146–1.140568배 |
| 제외한 C64/K11 d1·d3·d5 | event 1.02289–1.02344배, wall 1.02337–1.02976배; 1.03배 관문 미달 |
| 제외한 C128/K11 d1·d3·d5 | event 0.99193–0.99652배, wall 0.99195–0.99667배 |

선택 파일이 고정하는 source SHA256은 `c824ee32ccaf69d6b17b605f68687bc556b4958c2d951663e0563b481ee780b6`이다. benchmark SHA256은 `d3f6c6f7d5a26e601c6d0780a8510e5e5e4dbe9aeff59c5fc771f1041569068c`, v0.8 manifest SHA256은 `22e41075581e8b2d969388244bf063adba59579942c4959ae892e2aeb4fcfa59`다. 전체 학습이나 모든 가능한 입력의 바이트 동일성을 뜻하지 않는다.

대표 B8/C32/T12800/K3/d1의 현재 source 결과는 다음과 같다. event/wall은 반복 측정의 중앙값이며 profiler 합은 별도 계측 실행이다.

| 항목 | B | B′ |
|---|---:|---:|
| pair forward+backward event 중앙값 | 0.785987616 ms | 0.705911589 ms |
| pair forward+backward wall 중앙값 | 0.788610199 ms | 0.708414798 ms |
| profiler kernel 시간 합 | 0.716972 ms | 0.614658 ms |
| kernel launch 수 | 28 | 22 |
| allocator peak 증가량 | 33,180,672 bytes | 26,633,728 bytes |
| `aten::leaky_relu` 호출 | 2 | 0 |
| `aten::leaky_relu_backward` 호출 | 2 | 0 |
| `aten::add` 호출 | 2 | 0 |

이 쌍의 event 배속은 1.113436, wall 배속은 1.113204다. allocator peak는 6,546,944 bytes 줄었다. 제거된 tensor read/write의 정적 모델은 순전파·역전파 각각 39,321,600 bytes, 순전파 activation 할당은 13,107,200 bytes 감소로 계산한다. 이 쌍은 native backward의 activation 재생성이 없으며, 정적 read/write 감소량은 실제 DRAM counter가 아니다. 원 trace는 [B](fusion_full/B8_C32_T12800_K3_D1_float16_B.trace.json)와 [B′](fusion_full/B8_C32_T12800_K3_D1_float16_B_prime.trace.json)에 있다.

## 확인한 실제 호출

운영 `rvc/lib/algorithm/residuals.py`의 `ResBlock.forward`는 각 쌍에서 다음 순서로 실행한다.

1. 현재 입력 `x`를 skip으로 보관한다.
2. `leaky_relu(x, 0.1)`와 Conv1을 실행한다.
3. `leaky_relu(h, 0.1)`와 Conv2를 실행한다.
4. Conv2 결과와 skip을 더한다.

HiFi-GAN NSF의 현재 일반 학습 호출은 `resblock(x)`로 들어온다. `x_mask`가 주어지거나 dilation-fold가 켜진 경로에는 이 구현을 적용하지 않는다. weight_norm은 `torch.nn.utils.parametrizations` 방식이며 master parameter가 FP32이다.

## 계산 경계

### 순전파

- Conv1과 Conv2는 각각 별도의 dot2 kernel이다. 두 Conv 전체를 한 번의 kernel로 계산한다고 주장하지 않는다.
- 입력 load 뒤 LeakyReLU를 적용한다. 입력이 FP16이면 LeakyReLU 결과도 FP16으로 먼저 반올림한다. FP32 입력이면 FP32 LeakyReLU 뒤 Conv autocast의 FP16 변환을 적용한다.
- dot2 누산은 FP32이며, v0.8과 동일하게 Conv 결과를 FP16으로 반올림한 뒤 bias를 더하고 다시 FP16으로 반올림한다.
- Conv2의 마지막 store는 이 FP16 결과에 원래 skip을 더한다. skip이 FP32인 경계에서는 최종 출력도 FP32이다.
- custom autograd에 들어가기 전에 `conv.weight`를 각각 한 번 읽는다. weight_norm의 계산과 원래 FP32 parameter graph는 custom 함수 밖에 남는다.

### 역전파

- Conv2 dX의 store에서 두 번째 LeakyReLU mask를 적용한다.
- Conv1 dX의 store에서는 첫 LeakyReLU mask를 적용한 뒤 skip gradient를 더한다.
- `input > 0`이면 기울기 1, 0을 포함한 나머지는 0.1이다.
- Conv dX의 FP16 반올림, 입력 dtype 변환, LeakyReLU gradient의 입력 dtype 반올림, skip gradient 합산 순서를 보존한다. FP32 연산이 의도치 않게 FMA로 합쳐지는 것을 막도록 kernel의 `enable_fp_fusion=False`를 지정한다.
- dW는 저장된 활성값 대신 원래 preactivation을 읽어 LeakyReLU를 다시 계산한다. v0.8의 결정적 FP32 split partial과 고정 순서 reduction을 유지한다. atomic은 추가하지 않았다.
- v0.8에서 native가 선택된 방향은 native로 유지한다. native dW/dX 호출에 필요한 활성값은 backward에서만 다시 만든다. native dX 뒤에도 별도 mask kernel에서 skip gradient 합산을 함께 처리한다.
- FP32 master weight/bias로 반환하는 gradient는 원래 dtype으로 돌린다. 고차 gradient는 이 학습용 실험 범위에 포함하지 않는다.

## 검증과 선택

`bench_v10_fusion.py`는 GPU를 사용하기 전에 실행 폴더에 `PREDECLARED_GATES.json`을 저장한다. `GpuAudit`에 진입하고 다른 KFD client가 없음을 확인한 다음에만 torch를 import한다.

검증 층은 다음과 같다.

| 층 | 기준 | 의미 |
|---|---|---|
| 개별 Conv | 같은 고정 preactivation·upstream gradient를 사용한 native FP32 참조 | y/dX/dW/db, residual을 더한 y/dX를 각각 검사 |
| AMP·weight_norm | 같은 master parameter를 가진 실제 `ResBlock`의 B와 B′ | 전체 쌍의 y/dX와 각 FP32 master gradient 및 dtype을 검사 |
| native AMP 진단 | 같은 `ResBlock`의 native AMP와 B/B′ | 기존 B부터 존재하는 수치 차이와 새 융합 차이를 구분 |
| 쌍 성능 | 같은 프로세스에서 B/B′ 교대 | event와 wall 중앙값 모두 1.03배 이상일 때 micro 후보 |
| 최종 성능 | 주 작업의 실제 `core.py train` B/B′ | 이 microbench 결과만으로 운영 채택하지 않음 |

연산 gate는 finite, NRMSE ≤ 0.002, max error/reference RMS ≤ 0.02이다. FP32 Conv 참조의 입력은 원래 AMP Conv가 실제로 받는 반올림된 activation을 FP32로 확장한 것이다. LeakyReLU/두 Conv 전체를 FP32로 실행한 뒤 그 결과와 비교하는 것과 범위가 다르다. 후자는 중간값의 아주 작은 차이가 activation mask의 부호 선택을 바꾸는 효과까지 포함한다.

결과의 `selection.json`은 현재 source와 v0.8 manifest의 SHA, 통과한 pair key 및 검증된 input dtype을 묶는다. 해당 dtype으로 검증되지 않은 입력은 원래 B로 간다. 선택에 실패한 쌍을 더 느린 candidate로 강제하지 않는다.

## 메모리 수치 읽는 법

- `peak_extra_allocated_bytes`는 각 arm의 실제 PyTorch allocator peak 증가량이다.
- `profile.kernels`, `profile.operators`는 실제 profiler trace에서 얻은 launch 수와 시간이다. LeakyReLU와 gradient 중간 kernel이 없어졌는지 여기서 확인한다.
- `logical_traffic_model`은 제거된 tensor read/write에 대한 정적 계산이다. GPU DRAM hardware counter를 측정한 값이 아니다.
- 양쪽 입력과 출력이 FP16인 완전 직접 경로에서는 두 LeakyReLU와 Conv2→add 중간값 때문에 생기던 순전파 read/write `6 × N × 2 bytes`가 사라진다. 역전파는 mask와 skip 누산 중간값 제거분에서 native fallback의 activation 재생성 비용을 뺀다.
- load마다 activation을 다시 계산하므로 산술 명령은 늘어난다. 따라서 tensor materialization 감소만으로 시간이 줄었다고 판정할 수 없다. event/wall 및 실제 epoch를 같이 읽어야 한다.

## 재현 명령

첫 명령은 GPU를 사용하지 않고 대상 목록만 읽는다. 두 번째는 `fusion_full/result.json`에 기록된 seed·dtype·round·rep·manifest·연구 DB를 그대로 쓰는 GPU 재현 명령이다. 다른 GPU 작업이 없는 상태에서 한 번씩 실행한다.

```bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
/usr/bin/python3 -B bench_v10_fusion.py \
  --manifest results/applio_v08/direct_selected_final/dispatch.json --list

~/DevSpace/applio-lab/.venv-rocm10/bin/python -B bench_v10_fusion.py \
  --manifest results/applio_v08/direct_selected_final/dispatch.json \
  --db results/v10/fusion_bench_db \
  --out results/v10/fusion_full_reproduction_01 \
  --seed 20261108 --input-dtypes float16 --rounds 5 --reps 10 \
  --min-speedup 1.03 --profile
```

원 실행의 출력 폴더는 `fusion_full`이며 위 명령은 재현용 새 이름을 사용한다. 출력 폴더가 이미 존재하면 덮어쓰지 않고 종료하므로 추가 재현에는 다시 새 이름을 사용한다. 운영 DB는 사용하지 않는다. 이 문서를 갱신한 워커는 위 GPU 명령을 재실행하지 않았다.

## 운영 경로 측정 연결 API

v0.8 B 설치 직후, 연구용 런처의 같은 프로세스에서 context를 유지한다.

```python
from v10_fused_resblock import fused_resblocks
fusion_hits = {}
fusion_context = fused_resblocks(
    manifest_path,
    selection=validated_selection_json,
    hits=fusion_hits,
)
fusion_context.__enter__()
# 기존 core.py → train.py 경로. 운영 파일 수정 없음.
# 종료 시 fusion_context.__exit__(...)로 원래 메서드 복원.
```

Profiler 귀속은 `_backward_conv(dy, x, w, info, need_x, need_w, need_b, *, residual_grad=None, slope=.1)`를 임시 래핑해 기록할 수 있다. `info=(dilation,padding,groups,directions)`이고, residual_grad가 있으면 pair의 첫 Conv이다. Forward는 `conv_act_fwd`의 x/w geometry와 residual 유무로 식별한다. 이 계측 시간은 일반 epoch 시간과 분리한다.

## CPU 확인 기록

- `ast.parse`로 두 소스 전체 파싱: 통과.
- 시스템 Python으로 `bench_v10_fusion.py --list --case B8_C32_T12800_K3_D1`: 실제 manifest에서 올바른 두 방향 구성을 읽었다.
- 본 메모를 작성한 워커는 GPU 실행 및 torch/ROCm import를 하지 않았다. GPU 결과는 메인 작업이 직렬로 생성한다.

## 첫 GPU smoke — 구형 source의 유효한 중간 결과

메인 작업이 `fusion_smoke/result.json`에 기록했다. 대상은 B8/C32/T12800/K3/d1, FP16 입력, seed 20261108, 3 round × 5 rep이다. `completed`, `valid_for_timing`, `valid_for_contract` 모두 true이고 다른 KFD 작업과 겹치지 않았다.

| 항목 | B | B′ |
|---|---:|---:|
| pair forward+backward event 중앙값 | 0.811647987 ms | 0.717425776 ms |
| pair forward+backward wall 중앙값 | 0.820484001 ms | 0.721758604 ms |
| profiler kernel 합 | 0.703452 ms | 0.609740 ms |
| kernel launch 수 | 28 | 20 |
| allocator peak 증가량 | 33,180,672 bytes | 26,633,728 bytes |
| `aten::leaky_relu` 호출 | 2 | 0 |
| `aten::leaky_relu_backward` 호출 | 2 | 0 |
| `aten::add` 호출 | 2 | 0 |

event 1.13133배, wall 1.13678배였다. y/dX와 weight_norm g/v gradient는 B와 바이트가 같았고, bias gradient만 NRMSE 약 0.0002의 차이가 있었다. 별도 FP32 Conv 참조의 6개 metric도 모두 고정 gate를 통과했다. 6,546,944 bytes의 allocator peak 감소는 실제 측정이며, DRAM 이동량을 직접 잰 값은 아니다.

이 smoke의 source SHA256은 `ea92d5836665b87a1c356ba01e732f7f8c35a83e32574a4b8ef3d00492854f6b`이다. 후속 소스 리뷰에서 custom backward의 autocast가 bias `sum`을 FP32로 승격시키는 것을 확인했다. v0.8 B의 FP16 bias reduction 경계를 그대로 유지하도록 `dtype=torch.float16`을 명시했다.

수정된 source SHA256은 `c824ee32ccaf69d6b17b605f68687bc556b4958c2d951663e0563b481ee780b6`이다. 위 smoke의 속도·launch 수·selection은 수정 전 source에만 해당한다. 수정된 source의 전체 18개 쌍 검증에 첫 shape를 다시 포함했고, 현재 선택·성능은 문서 앞의 `fusion_full` 결과로 확정했다. `fusion_smoke/selection.json`을 수정된 source에 재사용하면 SHA 검사에서 거부된다.
