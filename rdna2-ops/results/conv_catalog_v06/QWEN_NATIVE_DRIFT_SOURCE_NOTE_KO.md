# Qwen native conditioning 변동: RNG 소스 확인 노트

조사: 2026-11-06 UTC, CPU·읽기 전용. **결론은 원인 미확정이며, global RNG 고정을 해결책으로 추가하지 않았다.** 추가 GPU 실행·운영 소스/설정 변경은 하지 않았다.

## 관측과 판정

기존 2-step 실행 `d878e9d7`의 [runs.json](qwen_model_screen2/runs.json)에서 `warm → baseline` 두 native 요청의 positive/negative conditioning embedding SHA가 모두 달라졌다. 각 요청에서 실제 FP32 patch Conv3D가 두 번 호출됐으며 `X=[2400,3,2,16,16]`, `W=[1152,3,2,16,16]`, bias 있음, `cudnn.enabled=True`, 관련 MIOpen 환경변수 미설정이었다. 후보 Linear는 아직 실행되지 않았다.

native끼리 최종 RGB 최대 절대 오차 `0.1381734311580658`, RMSE `0.0037544189740479606`; byte gate와 엄격 RGB 기준(최대 `2/255`, RMSE `0.25/255`) 모두 실패했다. 따라서 이 실행으로 **후보의 최종 모델 출력/속도를 판정할 수 없다**. gate는 약화하지 않았다. 연산 단위 결과와 모델 판정을 분리한다.

실제 patch Conv의 event 합은 요청당 `3547.0825 / 3553.6327 ms`였다. 별도 NORMAL 연산 실험과의 경로/시간 차이도 아직 원인 규명되지 않았다. 이 값은 wall과 별개인 event 구간이며, 커널 선택을 확인하지 않고 특정 solver의 비용이라고 단정하지 않는다.

## 소스 B 근거: 이번 encode가 RNG를 호출하는가

아래 상대경로의 공통 루트는 `~/DevSpace/qwen-image-21-rdna2-lab01/runtime/ComfyUI/`다. 줄 번호는 아래 해시의 설치 소스 기준이다.

| 소스·줄/함수 | 확인한 사실 | 해석 |
|---|---|---|
| `comfy_extras/nodes_qwen.py:144–182`, `TextEncodeQwenImage21.execute` | VAE `encode`, positive/negative `tokenize → encode_from_tokens_scheduled`를 호출한다. | autoregressive generation/sample 경로가 아니다. |
| `comfy/sd1_clip.py:147` | transformer를 `.eval()`로 설정한다. | 학습 dropout이 원인이라는 근거가 없다. |
| `comfy/text_encoders/qwen3vl.py:62–67`, `Qwen3VL.preprocess_embed` | vision 입력을 FP32로 보내 encoder를 실행한다. | 해당 함수는 RNG를 사용하지 않는다. |
| `comfy/text_encoders/qwen_image21.py:47–77`, `encode_token_weights` | 전체 encoder 실행 후 vision token 위치를 제거한다. | `keep_vision=False`는 vision 실행 생략이나 무작위 sampling이 아니다. |
| `comfy/text_encoders/qwen35.py:668–687`, vision `forward`; `894–920`, `verify_sample` | vision forward와 달리 `rand`·`multinomial`은 별도 MTP generation 내부에 있다. | 파일에 RNG가 존재한다는 사실만으로 이번 encode가 RNG를 쓴다고 볼 수 없다. |
| `comfy/sd.py:834–842`, Qwen21 VAE 선택; `1392–1448`, `VAE.encode` | `wan.vae2_2.WanVAE`를 선택하고 `first_stage_model.encode`를 호출한다. | 아래 mean 반환 경로와 연결된다. |
| `comfy/ldm/wan/vae2_2.py:713–739`, `WanVAE.encode` | 단일 이미지는 `conv1(encoder(x)).chunk(...)[0]`, 다른 경로도 `mu`를 반환한다. | encode는 posterior noise를 샘플링하지 않는다. `randn_like`가 있는 `reparameterize`/`sample`(768–778)은 이 경로에서 호출되지 않는다. |
| `comfy/ops.py:294–320`, `post_cast`; `comfy/float.py:65–84`, `stochastic_rounding` | 일부 재양자화는 `seed_key` 유래 seed를 전달한다. float 함수의 FP32/FP16/BF16 변환은 일반 cast이고 FP8 확률 반올림은 별도 seeded generator를 쓴다. | global Python/NumPy/torch seed 고정만으로 이 경로의 재현성을 제어했다고 볼 수 없다. 실제 해당 branch hit 여부도 별도 확인 사항이다. |

이 조사는 관련 소스 경로에 대한 확인이며, 전체 하위 라이브러리의 RNG 소비나 비결정성을 배제한 증명은 아니다. **현재 근거는 Python/NumPy/torch CPU/CUDA RNG 저장→고정→복원을 해결책으로 넣는 판단을 지지하지 않는다.** 필요하면 RNG state 전후 해시 관찰을 추가할 수 있으나, 원인 확인과 RNG 변경을 구별해야 한다.

## 미확정 원인과 다음 최소 확인

현재 shape/dtype 로그에는 입력·가중치의 값 해시가 없다. 같은 FP32 dtype이어도 첫 load/cast/offload 이후 값이 같다고 입증되지는 않았다. Conv solver·누산 순서의 비결정성, 첫 실행과 이후 실행의 상태 차이, 하류 vision/text encoder의 변동 중 어디서 처음 달라지는지 아직 모른다.

다음 실험이 필요해질 때는 native patch 호출별 **X/W/bias/Y byte digest**만 먼저 추가한다. 동일 요청의 positive/negative 두 호출과 요청 간 해시를 비교한다. X/W/bias가 같은데 Y가 달라지면 Conv 경로를 조사하고, Y까지 같은데 conditioning이 달라지면 하류 encoder를 좁힌다. W/bias부터 달라지면 load/cast/offload 경계를 조사한다. 이 진단에 CPU 복사·해시 계산을 넣은 실행의 시간을 성능 채택 근거로 사용하지 않는다. **이 노트 작성 중에는 이 추가 진단도 실행하지 않았다.**

## 소스·원자료 SHA-256

| 파일 | SHA-256 |
|---|---|
| `comfy_extras/nodes_qwen.py` | `54ce02aa9e167c0b497ce026ac6b558f539427d2d1ed4fc1fcc2ec8e6bc529da` |
| `comfy/sd1_clip.py` | `0ef5a50c43999aee97a65d0b38f0a7e070ec3559745fc8bcb2eaeae74a34decf` |
| `comfy/text_encoders/qwen3vl.py` | `86c9edf603f53fa0c6bab43ce6ec3d908948c50a7060fc706f8481585d7ea890` |
| `comfy/text_encoders/qwen_image21.py` | `c64016aa703f913859c9afb7efe5d1e5120c8fa2820f9f7acc00792987d39136` |
| `comfy/text_encoders/qwen35.py` | `5fe8c3e0d13b821c16108faeebf55622c1a70eeca373fea5697efbb315d07572` |
| `comfy/sd.py` | `ffbd956e375ad62ad00a21418f2974bcb111513929f5b70569a31ab58ad7d470` |
| `comfy/ldm/wan/vae2_2.py` | `a211180534d08b9cdc487697f151333d710d74bff145754c815c30a7816fff41` |
| `comfy/ops.py` | `e13f918c6a64f1dc2b9ce836868734396849db3eeb80313579afbfa90dd6ae89` |
| `comfy/float.py` | `3a6e6c6fb306b5827378f8b2a551984d89b2b9719c24cff39abc423fc1f37c01` |
| `qwen_model_screen2/runs.json` | `aae40f768b411e740c54ca127e217f0ae5168d2b74bc4c508184fa4aadc2fc20` |
