# 익명 A/B 지각 검토 자료

A/B는 모든 해상도와 장면에서 동일한 두 후보를 가리킨다. 어느 후보인지는 이 폴더 밖의 독립 키 JSON에만 기록했다. 이 묶음에는 매핑을 넣지 않았다.

## 보는 순서

1. `still/1920x1080`와 `still/3840x2160`의 `A.png`, `B.png`를 같은 배율로 번갈아 본다. 정지 입력의 최종 frame239다.
2. 같은 폴더의 `A_crop.png`, `B_crop.png`는 중앙 512×512 원본 픽셀 crop이다. 확대·보간·선명화·텍스트 덧붙이기를 하지 않았다. crop 좌표는 manifest에 있다.
3. `motion/<해상도>/A.mp4`, `B.mp4`와 `brightness/<해상도>/A.mp4`, `B.mp4`를 비교한다. 각 clip은 frame40..55를 순서대로 한 번만 재생한다.
4. 잔상·떨림·얇은 선의 변화가 의심되는 프레임은 각 장면의 `frames/A`, `frames/B`에 있는 같은 번호의 PNG로 확인한다. `crops/A`, `crops/B`는 그 프레임의 동일 좌표 crop이다.

영상은 **30 fps**, 16프레임 약 0.533초다. 새 프레임을 보간하거나 loop를 이어붙이지 않았다. 하네스의 frameTimeDelta는 10ms이므로 표시 재생 속도와 다를 수 있다. 이 짧은 clip의 첫·끝을 플레이어에서 반복하면, 끝과 처음을 잇는 변화는 별도 반복 경계다.

## 표시 변환

입력은 row padding 없는 little-endian RGBA16F다. RGB만 사용하고 alpha는 표시에서 무시한다. 모든 후보·프레임에 똑같이 RGB를 [0,1]로 clip한 다음 선형RGB→sRGB 변환을 적용했다.

```text
L <= 0.0031308 : S = 12.92 * L
그 외         : S = 1.055 * L^(1/2.4) - 0.055
PNG RGB8      : round_to_nearest_even(255 * clamp(S, 0, 1))
```

자동 노출, 프레임별 정규화, 색보정, tone mapping, dithering은 적용하지 않았다. 1보다 큰 값의 HDR 밝기 차이는 clipping으로 표시에서 사라질 수 있다. 따라서 밝기 변화 자료의 PNG/영상만으로 HDR/tone-map 동작을 완전히 판정할 수 없다. 원본 FP16은 연구의 captures 폴더에 그대로 남아 있다.

RGB NaN/Inf가 있는 픽셀은 마젠타(255,0,255)로 표시한다. 현재 전체 검토 프레임의 해당 픽셀 발생 수 합계는 **0**다. 이는 프레임별 합계이며 같은 위치가 여러 번 셀 수 있다. alpha NaN/Inf, RGB 범위 밖 component 수와 finite min/max도 `manifest.json`에 기록했다.

PNG는 원본 해상도 그대로다. MP4는 CPU **libx264rgb, CRF0, RGB 4:4:4 lossless**로 만들었고, 다시 RGB24로 디코드한 전체 바이트의 SHA256과 길이가 변환 PNG 픽셀 순서와 일치하는지 확인했다. 일부 브라우저는 이 codec profile을 재생하지 못할 수 있다. 그때는 PNG frame sequence 또는 해당 profile을 지원하는 로컬 플레이어로 본다.

## 이 자료의 범위

이 장면은 timing kit의 **합성 입력**이다. 움직임은 8개 phase를 반복하며 카메라 이동 입력이 phase7→0에서 되감긴다. 이 경계에서 보이는 변화에는 합성 입력의 wrap이 포함된다. 밝기 장면은 phase 전반에 0.35, 후반에 1.35 배율을 주는 반복이다. 실제 게임 카메라·오브젝트·depth/motion field·밝기 이력을 재현한 캡처가 아니다.

동영상용 수집 조건은 warmup40 이후 frame40..55, 정지컷은 기존 정지 입력 frame239다. 두 후보의 같은 장면·해상도·frame 번호를 비교한다. 전체 프레임과 crop 모두 좌표가 같다. 이 파일 생성·해시 확인은 **사용자의 지각 판정이나 게임 적용 검증 완료를 뜻하지 않는다**. 채택 여부는 별도 판정으로 남아 있다.

`manifest.json`은 A/B별 변환 통계·파일 SHA256·영상 decode 검증을 담는다. 원본 파일 경로·원본 SHA256·이름 매핑은 독립 키 JSON에만 있다.
