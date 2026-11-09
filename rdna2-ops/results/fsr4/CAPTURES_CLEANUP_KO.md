# captures/ 폴더 정리 기록

- 2026-11-10, 사용자 요청("화질 캡쳐는 필요없음 지우고")으로 `captures/` 전체(약 15 GB)를 지우기로 했다. 이 경로는 휴지통을 지원하지 않아(`gio trash` 거부) 영구 삭제가 되므로, 삭제 명령은 사용자가 직접 실행한다.
- 이유: lossy 익명 A/B 이미지가 합성 테스트 입력이라 눈으로는 구분되지 않았다. 품질 비교는 저장된 수치(§9 MAE)와 코드 분석으로만 판단한다.
- 남긴 것: `analysis/captures_manifest/`에 A/B 묶음의 `manifest.json`, `README_KO.md`, 원래 파일 목록(`FILE_LIST.txt`, 크기·경로)과 개수를 복사했다. 바이트 일치 판정과 SHA는 `analysis/*.json`, `measurements/`에 그대로 있다.
- 영향: 삭제 후에는 보고서 §9와 §11이 가리키는 `captures/ab_review/index.html`과 raw 프레임이 없다. 다시 만들려면 `analysis/build_ab_review.py`와 캡처 실행을 GPU로 다시 돌려야 한다.
- 삭제 명령: `rm -r ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops/results/fsr4/captures` (복구 불가)
