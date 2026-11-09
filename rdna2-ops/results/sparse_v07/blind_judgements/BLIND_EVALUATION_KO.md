# v0.7 Sparse 익명 지각 평가 원응답과 판정

**최종 판정: 지각 gate 실패. 12/12결정이 native를 선호했고, 후보의 상대 열화 severity는 모두 3이었다.** 두 판정자 모두 세 seed 각각에서 좌우 배치를 바꾸어도 같은 인물·얼굴 반복과 신체 연결 문제를 지적했다. 6/6개의 판정자×seed 양방향 쌍에서 동일한 큰 결함이 반복됐다. 후보 동등/우세·severity≤1은 0/12이며 사전 통과 기준 10/12에 미달한다. [기계 판독 집계](UNBLINDED_VERDICT.json)

## 실제 실행한 절차

판정일: 2026-11-07 KST(2026-11-06 UTC). 두 논리 판정자 J1/J2 × 세 seed × 두 좌우 순서의 12결정이다. 판정자와 순서마다 새 문맥으로 총 네 세션을 실행했고, 연구 이력·성능·오차·원본/후보 정체·seed를 넘기지 않았다. 각 세션은 공통 입력 하나, 전체 비교 세 개, 고정 crop 아홉 개를 실제로 열어 보았다고 원응답의 viewed_files에 기록했다. 각각 13개 고유 파일이며, 모든 세션에서 배정된 파일 목록과 일치함을 확인했다.

확인 가능한 모델 표시는 네 세션 모두 **GPT-6 Astra Pro**, 내부 revision은 **unknown**이다. 같은 계열 모델의 독립 문맥 반복이며 **서로 다른 모델 아키텍처의 독립 검증이라고 주장하지 않는다.** 연구자의 육안이나 기존 Claude 육안 평가를 이 12결정에 넣지 않았다. J2/session-B에 한 차례 보낸 중립적인 완료 확인 문구도 집계 JSON에 보존했다. 판정 기준이나 후보 정체·결과는 전달하지 않았다.

프롬프트는 최종 모델 RGB를 보기 전에 고정했다. [동결 기록](../BLIND_JUDGE_PROMPT_V07.json), [실제 네 세션 지시문과 배정](SESSION_PROMPTS.json), [공통 모델 입력·생성 지시](../blind_panels/PUBLIC_TASK.json), [이미지 SHA·crop 좌표](../blind_panels/PUBLIC_CASES.json), [판정 뒤 공개한 대응표](../blind_panels/PRIVATE_MAPPING.json)를 함께 보존했다. full과 crop은 1:1 원본 픽셀을 lossless PNG로 묶고 metadata를 제거했으며, 원본 영역의 pixel byte roundtrip을 확인했다. 위 기준을 결과에 맞춰 바꾸지 않았다.

## 12개 개별 결과

A는 패널 왼쪽, B는 오른쪽이다. 아래 seed·후보 위치·native라는 이름은 **원응답 수집 뒤 coordinator가 mapping을 적용한 정보**다.

| 세션 | seed | 익명 set ID | 후보 위치 | 원응답 선호 → 공개 후 | 후보 상대 severity | 확신 |
|---|---:|---|---|---|---:|---|
| J1/session-A | 20261109 | p05c2321271f549cb | A | B → native | 3 | high |
| J1/session-A | 20261108 | p27c6e8b55e1db2fe | A | B → native | 3 | high |
| J1/session-A | 20261107 | p9f8880d85a773938 | A | B → native | 3 | high |
| J1/session-B | 20261109 | p506af831a3e94ea5 | B | A → native | 3 | high |
| J1/session-B | 20261107 | p77e54ad28ee8a89d | B | A → native | 3 | high |
| J1/session-B | 20261108 | p37d12c010e3506f1 | B | A → native | 3 | high |
| J2/session-A | 20261109 | p05c2321271f549cb | A | B → native | 3 | high |
| J2/session-A | 20261108 | p27c6e8b55e1db2fe | A | B → native | 3 | high |
| J2/session-A | 20261107 | p9f8880d85a773938 | A | B → native | 3 | high |
| J2/session-B | 20261109 | p506af831a3e94ea5 | B | A → native | 3 | high |
| J2/session-B | 20261107 | p77e54ad28ee8a89d | B | A → native | 3 | high |
| J2/session-B | 20261108 | p37d12c010e3506f1 | B | A → native | 3 | high |

순서 반전으로 native/candidate 선호가 뒤집힌 쌍은 0개다. 판정자 간 후보의 채택 결론 충돌도 0개다. seed08의 native 손·스푼·푸딩 접시에 대한 작은 지시 충실도 지적은 판정자·방향에 따라 0/1로 달랐고 원문에 그대로 남겼다. 이를 숨겨 native가 완벽하다고 주장하지 않는다. 후보의 반복 인물과 신체 파손에 대한 severity 3 및 native 선호는 모든 응답에서 일치했다.

## 반복 확인된 결함과 해석

모든 원응답은 후보에서 두 소녀의 얼굴·몸이 위아래 여러 층으로 반복되고, 책·푸딩·팔이 어느 몸에 속하는지 불분명해진다고 설명했다. seed09에서는 일부 얼굴 중복과 인물별 의상 섞임도 지적했다. 각 얼굴의 세부 묘사나 따뜻한 색감이 남아 있다는 점과, 두 사람이 함께 있는 하나의 장면이 무너진다는 점을 구분했다.

이 관찰은 이번 image-only r4에서 전역 장면 일관성이 크게 손상됐다는 **출력 증거**다. 국소 mask만으로 글로벌 인물 배치를 유지하지 못했을 가능성과 일관되지만, 그 원인을 별도 ablation으로 완전히 분리한 결과는 아니다. all-true streaming 대조도 strict RGB에는 실패했으므로 전체 최종 오차를 pruning 하나의 손실이라고 단정하거나 두 RGB 오차를 선형 차감하지 않는다.

엄격 RGB 기준(최대 2/255·RMSE 0.25/255)의 3/3 실패와 late same-mask native 계산 관문 실패는 그대로 유지한다. 지각 평가를 실제 수행해도 이번 후보는 채택 불가다. 이는 세 seed의 한 기존 workflow 표본에 대한 실패이며, 모든 sparse attention이나 다른 모델·영상의 실패로 일반화하지 않는다.

## 판정 원문

아래 파일은 판정자가 반환한 JSON 원문을 그대로 저장한 것이다. 결론 요약과 원응답은 별도 파일이며 결과에 맞춰 응답을 수정하지 않았다.

- [J1_session_A_raw.json](J1_session_A_raw.json) — SHA `3e8366c43be3669f9633f7c82ea6e6627ecd76e68e5e519a30f94b2413c515e5`, 13개 고유 입력/full/crop 열람 기록.
- [J1_session_B_raw.json](J1_session_B_raw.json) — SHA `1770420d98127cf83084bb556e64053418e9916546d92521848c035bb329cc62`, 13개 고유 입력/full/crop 열람 기록.
- [J2_session_A_raw.json](J2_session_A_raw.json) — SHA `d71ad05b310d39a25fa01ef941450b9b637f52f113ba2802fe1cbf3f069d762a`, 13개 고유 입력/full/crop 열람 기록.
- [J2_session_B_raw.json](J2_session_B_raw.json) — SHA `cf470e1c97541a699520888448201db5624f13dd566049d481cbcd784114acd5`, 13개 고유 입력/full/crop 열람 기록.
