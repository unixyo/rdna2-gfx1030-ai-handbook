# 익명 평가 public packet 목록

모든 판정자는 먼저 [PUBLIC_TASK.json](PUBLIC_TASK.json)의 실제 공통 생성 지시와 공통 입력, [JUDGE_PROMPT_KO.txt](JUDGE_PROMPT_KO.txt)의 판정 형식을 읽는다. 아래에는 원본/후보·seed·성능·수치 오차 mapping이 없다. 각 판정자·좌우 순서 세션은 새 문맥에서 세 seed를 각각 판정하며 다른 판정자의 응답이나 반대 순서 세션의 결과를 보지 않는다. 모든 PNG는 1:1 원본 픽셀, resize 없음, 원 PNG metadata 제거, PNG roundtrip에서 source pixels byte 동일 검증 완료다.

공통 입력(생성 결과 아님): a0024.png (`a0024.png`, 연구 폴더에만 있음)

파일당 실제 도구 한도는 2,000,000 bytes다. 여기서 최대 파일은 1,830,270 bytes다. full은 모두 A/B 나란히 같은 canvas로 들어갔다. C1/C2/C3은 사전 고정한 동일 좌표 crop이다.

| case ID | full | C1 | C2 | C3 |
|---|---|---|---|---|
| p05c2321271f549cb | a0000.png (`a0000.png`, 연구 폴더에만 있음) | a0001.png (`a0001.png`, 연구 폴더에만 있음) | a0002.png (`a0002.png`, 연구 폴더에만 있음) | a0003.png (`a0003.png`, 연구 폴더에만 있음) |
| p27c6e8b55e1db2fe | a0004.png (`a0004.png`, 연구 폴더에만 있음) | a0005.png (`a0005.png`, 연구 폴더에만 있음) | a0006.png (`a0006.png`, 연구 폴더에만 있음) | a0007.png (`a0007.png`, 연구 폴더에만 있음) |
| p9f8880d85a773938 | a0008.png (`a0008.png`, 연구 폴더에만 있음) | a0009.png (`a0009.png`, 연구 폴더에만 있음) | a0010.png (`a0010.png`, 연구 폴더에만 있음) | a0011.png (`a0011.png`, 연구 폴더에만 있음) |
| p506af831a3e94ea5 | a0012.png (`a0012.png`, 연구 폴더에만 있음) | a0013.png (`a0013.png`, 연구 폴더에만 있음) | a0014.png (`a0014.png`, 연구 폴더에만 있음) | a0015.png (`a0015.png`, 연구 폴더에만 있음) |
| p77e54ad28ee8a89d | a0016.png (`a0016.png`, 연구 폴더에만 있음) | a0017.png (`a0017.png`, 연구 폴더에만 있음) | a0018.png (`a0018.png`, 연구 폴더에만 있음) | a0019.png (`a0019.png`, 연구 폴더에만 있음) |
| p37d12c010e3506f1 | a0020.png (`a0020.png`, 연구 폴더에만 있음) | a0021.png (`a0021.png`, 연구 폴더에만 있음) | a0022.png (`a0022.png`, 연구 폴더에만 있음) | a0023.png (`a0023.png`, 연구 폴더에만 있음) |

전체 절대 경로·SHA·파일 크기·source_region은 [PUBLIC_CASES.json](PUBLIC_CASES.json)에 있다. CodexPro2 workspace는 `ws_6cac9f528eb957faf52fdab8`, 파일 path의 workspace-relative prefix는 `rdna2-pro-research-20261102/rdna2-ops/results/sparse_v07/blind_panels/`다. `view_image(max_bytes=2000000)`를 사용하고 ImageContent만 표시한다. base64 전체를 text 출력하지 않는다.

익명 평가 원응답과 판정자 메타데이터는 coordinator가 별도 보존한다.
