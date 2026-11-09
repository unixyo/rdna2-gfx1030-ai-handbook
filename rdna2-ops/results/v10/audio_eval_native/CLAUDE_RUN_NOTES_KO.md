# B 대 B+native 교체만 — 익명 청취 실행 메모 (Claude, 2026-11-09)

- 목적: 2026-11-09 청취에서 B′(융합+native)가 보류됐다. 프로파일상 속도 이득은 거의 native 교체에서 나왔으므로, 융합을 뺀 묶음을 B와 다시 비교한다.
- 학습: `../final_Bnative1_20261109/`. Pro의 `v10_run_operating.py --arm B_prime --native-selection native_selected/dispatch.json`(융합 선택 없음)을 그대로 썼다. 100 epoch·3,100 step, 초기 G/D SHA는 B와 같다(`732feda6…`). nonfinite 0, scale 감소 12, KFD 큐 최대 7. 프로세스 1,178.24 s, steady 중앙값 11.4387 s. native hit 31/31, 융합 0, v0.8 직접 Conv 21/21.
- 음성: Pro의 `v10_audio_eval.py`를 수정하지 않고 실행했다. 다만 `audio_native_wrapper.py`로 보호 검사의 기준선만 `../protected_before_20261109_claude.json`으로 바꿨다. 새 기준선은 Pro 기준선과 같은 1,022개 파일이다. Claude가 의도적으로 바꾼 3개 파일(lab_hooks.py에 rdna2_v10 opt-in 연결, 카탈로그·가이드 v1.0 편집)만 현재 SHA로 갱신했다.
- 첫 시도(`../audio_eval_native_failed01_no_soundfile/`)는 `/usr/bin/python3`로 실행해서 `soundfile` import에서 멈췄다. GPU 작업 전이었다. 두 번째는 `.venv-rocm10` python으로 실행했다.
- 검증: completed·valid_for_listening·native FP32·같은 generator 입력·운영 보호 모두 true였다. GpuAudit 유효, foreign 0. **B의 WAV 4개는 2026-11-08 Pro 청취 자료의 B WAV와 SHA가 같다.**
- 대응표 `blind_key.json`은 판정 전에 열지 않는다. HTML SHA256은 `be3e4cf4a51d18e7f3b25d9ec65d40f878ed083c9d9e77ed744298a4e997c3e8`(4,318,552 B)다.
