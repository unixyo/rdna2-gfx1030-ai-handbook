# ConvTranspose1D의 검증된 9개 FindDb key 병합 레시피

상태: 단일 GPU lease에서 FAST 3shape replay와 실제 4epoch를 완료했다. 연산 y/dX/dW gate는 모두 통과했지만 실제 epoch는 8.417579→8.434122s, 0.99804×로 추가 이득이 없어 이 DB 선택은 채택하지 않는다. 운영 DB/설정은 변경하지 않았다. 재현 시에도 GPU 작업은 하나씩 실행한다.

기준은 기존 filtered no-fold actual epoch 8.417579208s다. 이 후보도 fold 없이, 기존12개 d1 safe FindDb 선택에 **실측 통과한 ConvTranspose 3shape의9개 방향 key만** 더한다. 일반 NORMAL DB를 전부 복사하지 않는다.

ConvTranspose의 forward는 MIOpen B, input gradient는 F, weight gradient는 W다. 그래서 F key의 V4R1을 무조건 재필터하지 않는다. 이3개의 F key는 해당 transpose dX 검사에서 통과했다. 새모델/shape에는 이판정을확대하지 않는다.

## 입력과 출력 SHA

- base: `d81ef3d8330bce40526dc885c744b8914de49d679cd33fd2493ef46ce1da1bbf`
- source: `3a623f38cbaee04720c683db5fd58d39ef1b400b481f17400f8efc279129831b`
- output: `b8bd35b3f4d39c44c67fa8b6c7073927cb5bd8b15ab7809947336b0ba2caf33b`

base36key + 검증된9key =45key. 소스두개는SHA불변이다. 자세한 entry는 `hybrid_cache/filtered_transpose_db/merge_manifest.json`.

| 실제 ConvTranspose 입력 / weight | forward B key | dX F key | dW W key |
|---|---|---|---|
| [8, 512, 32] / [512, 256, 16] | `512-1-32-1x16-256-1-320-8-0x3-1x10-1x1-0-NCHW-FP16-B` | `256-1-320-1x16-512-1-32-8-0x3-1x10-1x1-0-NCHW-FP16-F` | `512-1-32-1x16-256-1-320-8-0x3-1x10-1x1-0-NCHW-FP16-W` |
| [8, 256, 320] / [256, 128, 16] | `256-1-320-1x16-128-1-3200-8-0x3-1x10-1x1-0-NCHW-FP16-B` | `128-1-3200-1x16-256-1-320-8-0x3-1x10-1x1-0-NCHW-FP16-F` | `256-1-320-1x16-128-1-3200-8-0x3-1x10-1x1-0-NCHW-FP16-W` |
| [8, 128, 3200] / [128, 64, 4] | `128-1-3200-1x4-64-1-6400-8-0x1-1x2-1x1-0-NCHW-FP16-B` | `64-1-6400-1x4-128-1-3200-8-0x1-1x2-1x1-0-NCHW-FP16-F` | `128-1-3200-1x4-64-1-6400-8-0x1-1x2-1x1-0-NCHW-FP16-W` |

## 순차 실행 명령

작업디렉토리: `~/DevSpace/rdna2-pro-research-20261102/rdna2-ops`.

```bash
~/DevSpace/applio-lab/.venv-rocm10/bin/python bench_conv_catalog_v06.py --set transpose --mode FAST --db results/conv_catalog_v06/hybrid_cache/filtered_transpose_db --out results/conv_catalog_v06/transpose_merged_replay --compare results/conv_catalog_v06/family_fast
```

GPU audit, y/dX/dW reference/control gate, actual kernel 모두 확인한 뒤에만 아래 actual epoch를 실행한다.

```bash
~/DevSpace/applio-lab/.venv-rocm10/bin/python applio_epoch_v06.py --out results/conv_catalog_v06/epoch_filtered_transpose --db results/conv_catalog_v06/hybrid_cache/filtered_transpose_db --epochs 4 --mode FAST --observe-transpose --compare results/conv_catalog_v06/epoch_filtered
```

새 옵션은 F.conv_transpose1d 입력/weight shape·dtype·stride·padding·output_padding·groups·dilation과 call 수만 기록한다. 계산은 original 함수를 그대로 호출하고 종료 시 복원한다. 운영 FoldConv/ConvTranspose/train.py를 수정하지 않는다.

wrapper 수정 전 source는 `source/applio_epoch_v06_pre_transpose.py`, SHA `aa91fe5c027fc0be8906b81eac9db458ec80643fdea27fbdfb504249ee4c312e`로 보존했다. actual epoch는 기존 pretrained/optimizer/scaler/seed/data를 재사용한다. native-repeat floor가 큰 사실을 유지하며 파형 차이를 청각 열화/동등으로 단정하지 않는다. 새 reference tensor SHA는 처음/끝 같아야 하고 initial modelstate·initial eval audio는 대조와 맞아야 한다.

## 보완 실측 결과

원자료: `transpose_merged_replay/result.json`, `epoch_filtered_transpose/result.json`. 대상 세 shape는 각각 64회 적중했다. 초기 modelstate/audio byte exact, reference 5tensor 처음/끝 SHA 일치, 보호 파일/DB SHA 불변, GPU audit valid다. 최종 waveform의 임시 gate 실패는 이미 큰 native-repeat floor와 함께 해석해야 하며 청각 품질의 동등 또는 열화를 뜻하지 않는다. 자세한 표는 `CONV_CATALOG_V06_KO.md` §7.1에 있다.
