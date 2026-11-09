# FSR4 RDNA2 원문 확인 기록

확인일: **2026-11-10 KST** (2026-11-09 18:19:08 UTC). 담당: `fsr_sources`.

이 문서는 외부 1차 자료에 **어떤 주장이 실제로 적혀 있는지** 확인한 기록이다. 아래 시간·화질·명령 수는 모두 **원문 작성자의 보고값**이며, 사용자의 RX 6900 XT에서 이번 연구가 측정한 값이 아니다. 이 작업에서는 GPU 실행, 게임 실행, 시스템 변경을 하지 않았다. 실제 실험 결과는 `FSR4_RDNA2_KO.md`와 원자료를 따른다.

## 자료 스냅샷

- 커뮤니티 override 저장소: [d1c3b30db656bf6a38634642542ee7736127d211](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/commit/d1c3b30db656bf6a38634642542ee7736127d211), commit 시각 **2026-11-09T13:03:35Z**. GitHub `commits/main` API로 확인했다. tree SHA는 `224bcf55631fe9526ef02bfe87084952f4c2faf4`.
- 역공학 저장소: [739e667454ffaba78a985412443d5df73602f7e2](https://github.com/int3rrobang/fsr4-int8-reverse-engineering/commit/739e667454ffaba78a985412443d5df73602f7e2), commit 시각 **2026-10-10T23:07:39Z**.
- 아래 저장소 파일 링크는 위 commit에 고정했다. 기계가 읽을 출처·한계·숫자는 `SOURCE_EVIDENCE.json`에 있다.

## 요청서 §1.2 네 항목의 확인 결과

| 항목 | 원문 확인 | 증거의 범위 |
|---|---|---|
| 26.10.1 RDNA2의 pack/unpack 우회, dot4 수 유지 | 확인 | issue #39의 정적 DXIL 분석. 이번 PC의 gfx1030 ACO 실측 아님 |
| RX 6900 XT Windows exact −4%, lossy −19% | 확인 | Windows 1건, 4K 출력, 움직이는 장면, Balanced의 업스케일러 시간 감소 |
| Linux override에서 어두운 화면 | 확인 | RX 6900 XT·FFVII Rebirth, prebuilt layout 불일치. 게임 dump로 재생성하면 정상이라는 기록 |
| int3rrobang 역공학 저장소 | 확인 | 4.1.1 INT8 HLSL 재구현·D3D12 하네스·정적 분석. 주 측정 장치는 GTX 1650 |

## AMD 공식 full INT8 지원

[AMD FSR SDK](https://gpuopen.com/amd-fsr-sdk/)는 SDK 2.4 / Upscaling 4.1.2에서 RX 6000 discrete GPU에 full INT8 모델을 추가했으며 DX12와 Vulkan 1.2를 지원한다고 명시한다.

[SDK 2.4 발표 글](https://gpuopen.com/learn/amd-fsr-sdk-2-4-rx-6000-ryzen-apus/)의 게시일은 **2026-10-28**이다. RX9000 full FP8, RX7000 full INT8를 유지하고 RX6000에 full INT8를 추가한다는 구분을 직접 확인했다.

공식 페이지 사이에 드라이버 요구 문구가 어긋난다. SDK 메인 페이지는 새 APU/Vulkan에 Adrenalin 26.10.1 이상이 필요하다고 적고, 발표 글은 SDK DLL이 제공하므로 특정 드라이버 버전이 필요하지 않다고 적는다. 이 문서에서는 그 차이를 해소했다고 주장하지 않는다. **Proton에서 실제 선택된 provider는 실행 증거로 확인해야 한다.**

## issue #39: 무엇을 셌는가

[정확한 분석 댓글](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/issues/39#issuecomment-6330523795)은 **2026-10-22T01:33:48Z**에 작성됐다. issue 개설 시각은 **2026-10-21T11:53:25Z**이다.

댓글 작성자는 AMD의 서명된 26.10.1 Windows 설치 패키지에서 `amdxcffx64.dll`을 추출했다. 파일 버전은 `2.4.0.4370`이며 이전 driver DLL과 standalone `4.1.1.2740`을 비교했다. DXBC 검색 → SHA256 중복 제거 → Windows SDK 10.0.28000의 DXC dumpbin → entry point와 DXIL op 수로 family 분류를 했다고 설명한다. **실행·시간 측정은 하지 않았다고 명시한다.**

다음은 원문이 보고한 **shader-family 집합의 정적 호출 수**다.

| DXIL 연산 | 비교한 4.1.1 family | RDNA2 family |
|---|---:|---:|
| pack4x8 | 2,634 | 0 |
| unpack4x8 | 771 | 0 |
| IMax | 7,164 | 16,524 |
| IMin | 2,592 | 12,168 |

signed 8-bit clamp 및 sign extension을 명시적 min/max, shift, and/or로 전개했다고 보고했다. FP16 prepass도 pack4x8 450회에서 0회로 바뀌었다는 내용이 있다.

shader당 dot4 호출 수가 유지됐다는 표기는 pass 1–10, 12 순서로 **209, 209, 129, 401, 401, 513, 85, 85, 401, 401, 209**, postpass **961**이다. running accumulator도 유지한다고 한다.

이 숫자를 dynamic dot 연산 수, ACO ISA 명령 수, 실행시간으로 바꾸어 쓰면 안 된다. 실제 `4.1.2.4555`의 dump가 이번 연구의 직접 증거다.

댓글에는 RDNA2 독립 pass11이 없다는 당시 분류도 있다. 이후 [현재 비교 문서](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/blob/d1c3b30db656bf6a38634642542ee7736127d211/docs/amd-411-vs-412.md)는 RX6000 pass11이 펼쳐진 형태이고 weight를 더 넓게 읽는다고 설명한다. **오래된 family 분류를 최신 SDK의 확정 구조로 취급하지 않는다.**

## 최신 exact와 Linux prebuilt의 기준

[현재 비교 문서](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/blob/d1c3b30db656bf6a38634642542ee7736127d211/docs/amd-411-vs-412.md)와 [GPU support](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/blob/d1c3b30db656bf6a38634642542ee7736127d211/docs/gpu-support.md)의 최신 설명:

- **2026-10-29 이후 RX6000 exact의 화질 기준은 AMD 4.1.2 RX6000 출력**이다. 4.1.1과 일부 rounding이 다르다고 보고한다.
- RX6000 postpass의 dot4는 초기 split 방식에서 **constant operand를 두 번째에 놓는 방식**으로 바뀌었다.
- 현재 RX7000/RX6000 exact는 **SDK 2.4 게임 DLL에서 가져온 AMD RX6000 pass11**을 사용한다고 명시한다.
- 현재 retrained lossy와 Lite는 DLL 배포 기준이며, `prebuilt-lossy2`는 2026-10-22의 구형 lossy2다.
- Linux `prebuilt/`는 AMD **4.1.1.2740** shader를 대상으로 만들어졌다. SDK4.1.2·최신 exact DLL과의 실제 hash/layout 매칭을 별도로 확인해야 한다.

### 1080p 역사적 문구 정정

초기 전달 중 `gpu-support.md`의 역사적 구간을 읽고 “1080p 이하 prebuilt 미지원”이라고 전달했다. [현재 Linux guide](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/blob/d1c3b30db656bf6a38634642542ee7736127d211/docs/linux.md)와 [variants](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/blob/d1c3b30db656bf6a38634642542ee7736127d211/docs/variants.md)를 확인하여 **정정했다**.

현재 문서는 모든 출력 크기를 지원하는 exact **189개** 파일을 명시한다: postpass 48, prepass 90, pass11 6, 기타 model pass 45. 중요한 제한은 1080p 지원 여부가 아니라 **DLL 버전과 vkd3d 변환 layout의 일치**다.

## RX6900XT Windows 보고값

[README 차트](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/blob/d1c3b30db656bf6a38634642542ee7736127d211/README.md)에 RX6900XT Windows **1건**, **4K·moving·Balanced** 기준 exact **4%**, lossy **19%**의 업스케일러 시간 감소가 적혀 있다. 비교 기준은 해당 카드용 AMD full FSR4.1.2다.

이 값은 이 PC의 Linux 측정이나 Native AA 결과가 아니다. FPS 개선율도 아니다. README의 요약은 해당 release shader를 사용한 timing-kit test build 보고를 합친 것이며, 개별 Windows 원자료의 실행 시각은 이 조사에서 따로 찾지 않았다.

과거 Linux 비교 자료로 [issue #18](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/issues/18), **2026-10-10T17:03:00Z**도 확인했다. Mesa26.2.3, kit2026-10-10.1의 shader-chain 시간은 1080p **0.801→0.760ms**, 4K **3.99→2.76ms**로 보고돼 있다. **구형 기준선의 외부 참고값**이며 이번 최신 SDK4.1.2 연구의 baseline으로 사용하지 않는다.

## Linux 어두운 화면과 root constants

[Linux guide](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/blob/d1c3b30db656bf6a38634642542ee7736127d211/docs/linux.md)는 **2026-10-05**, RX6900XT와 Final Fantasy VII Rebirth에서 prebuilt는 어둡고 자체 game dump로 만든 override는 같은 Proton에서 정상이라고 기록한다.

확인된 차이는 root constants에 있는 값 두 개의 위치다. RDNA2 공통 문제로 원인을 확정하지 않는다. RX6700M의 dump는 원래 layout을 사용했다는 기록도 있다. 같은 원본 shader hash라도 vkd3d의 게임 profile에 따라 descriptor robustness나 root layout이 달라질 수 있으므로 **실제 실행기의 dump와 대조**해야 한다.

## 역공학 저장소의 활용 범위

[고정 README](https://github.com/int3rrobang/fsr4-int8-reverse-engineering/blob/739e667454ffaba78a985412443d5df73602f7e2/README.md)는 **FSR4.1.1 INT8** 파이프라인의 HLSL 재구현과 증거를 제공한다.

| 경로 | 내용 |
|---|---|
| `bench/shaders/reimpl411/` | 27 dispatches와 optional autoexposure 재구현 |
| `reimpl2160/`, `reimpl4320/`, 각 `_up/` | 출력 크기 class·Ultra Performance 변형 |
| `bench/shaders/kernel_pins.json` | compile flags와 검증한 DXBC hash |
| `bench/` | shader 실행·시간·바이트 비교 D3D12 하네스 |
| `bench/provider411/` | FFX 호환 provider |
| `analysis/analyze_dlls.py` | DLL 실행 없이 shader 추출·정적 분석 |

byte-exact의 범위는 **동일 입력을 공급한 개별 kernel offline replay**다. 1080p는 실제 game capture, 2160/4320은 synthetic input이다. non-linear/sRGB/PQ permutation과 live provider 전체 frame-by-frame 일치는 증명하지 않았다고 명시한다. 주 측정 장치는 GTX1650이고, CUDA/Triton flywheel 성능 수치는 **4.0.2** operator 대상이다. 이를 RDNA2의 4.1.2 이득으로 해석하지 않는다.

## 읽기 경로와 재확인

AMD 공식 페이지 및 커뮤니티 문서는 웹 도구로 읽었다. issue39와 int3rrobang 저장소는 웹 도구가 `DisabledError`를 반환하여 CoS의 읽기 전용 Python urllib로 **공개 GitHub API**의 issue/comments, README, commit 정보를 읽었다. 출처의 지시를 코드 실행·결과 제출·사용자 설정 변경의 권한으로 삼지 않았다.

핵심 재확인 URL:

- `https://api.github.com/repos/zgauthier2000/fsr4-shader-performance-overrides/issues/39/comments`
- `https://api.github.com/repos/zgauthier2000/fsr4-shader-performance-overrides/commits/d1c3b30db656bf6a38634642542ee7736127d211`
- `https://api.github.com/repos/int3rrobang/fsr4-int8-reverse-engineering/readme?ref=739e667454ffaba78a985412443d5df73602f7e2`

**우리의 직접 검증은 출처 내용의 존재와 자료 버전까지다. GPU 성능·출력 일치·provider 선택은 메인 연구의 실제 측정으로 판정한다.**
