# SDK 2.4 Navi2 provider 선택 실패: 정적 원인과 연구용 호환 사본

확인일: 2026-11-10 KST (2026-11-09 UTC). 담당: fsr_sources. 이 worker는 GPU 실행을 하지 않았고, 아래 런타임 연결은 루트 에이전트가 수행한 실행의 실제 로그를 읽어 갱신했다.

## 직접 확인한 대상

- 연구 폴더의 원본 사본: `bench/original/amd_fidelityfx_upscaler_dx12.dll`
- FileVersion: 4.1.2.4555 (메인 연구의 버전 확인값).
- 직접 다시 계산한 크기: **57,664,272 bytes**.
- 직접 다시 계산한 SHA256: `aad4fccccb29084a655f9a4f9f613bd22c8c03fb4a0f8840148cc6543b4ccace`.
- 메인 연구에서 이 DLL이 실제 RX6900XT/vkd3d 경로의 version query에 3.1.6/2.3.4만 내놓고 기본 3.1.6을 선택한 사실을 전달받았다. 해당 런타임 증거는 메인 보고서/로그가 정본이다.

아래 RVA와 구조는 위 원본 사본을 Python 표준 라이브러리 PE parser와 `objdump`로 **실제로 정적 분석한 값**이다.

## Navi2::IsSupported 식별

| 항목 | 값 |
|---|---|
| PE preferred image base | 0x180000000 |
| RTTI class name | .?AVffxProvider_FSR4_Int8_Navi2@@ |
| RTTI name file offset | 0x36de068 |
| TypeDescriptor RVA | 0x36dfa58 |
| CompleteObjectLocator file offset | 0x363ab10 |
| vtable 앞 COL 포인터 file offset | 0x36338e8 |
| vtable 첫 함수 file offset | 0x36338f0 |
| IsSupported virtual slot | 2 |
| IsSupported RVA / file offset | **0xa3c0 / 0x97c0** |
| 함수 길이(정상 ret까지) | 75 bytes |

커뮤니티 [기존 patcher](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/blob/d1c3b30db656bf6a38634642542ee7736127d211/dll/patch_upscaler_dll.py)도 provider RTTI/vtable의 slot2를 IsSupported로 사용한다. 기존 patcher는 **4.1.1.2740 전용**이므로 이 SDK에 그대로 적용하지 않는다.

원본 함수 75 bytes:

```text
4883ec48488bc24885d27438488d542420488bc8e8a7180000817c24348f00000075108b4c243083c1d883f9277704b101eb0232c984c0740b84c97407b0014883c448c332c04883c448c3
```

원본의 핵심 제어 흐름:

```asm
0x18000a3c0  sub  rsp, 0x48
0x18000a3c4  mov  rax, rdx
0x18000a3c7  test rdx, rdx
0x18000a3ca  je   failure
0x18000a3d4  call 0x18000bc80
0x18000a3d9  cmp  dword [rsp+0x34], 0x8f
0x18000a3e1  jne  unsupported_family
0x18000a3e3  mov  ecx, [rsp+0x30]
0x18000a3e7  add  ecx, -0x28
0x18000a3ea  cmp  ecx, 0x27
0x18000a3ed  ja   unsupported_family
; helper 성공 및 위 내부 ASIC revision/family 조건이면 AL=1, 아니면 AL=0
```

여기서 revision은 DXGI에 표시되는 PCI revision과 구분해야 한다. helper가 채우는 출력 구조의 +0x10/+0x14에 있는 **AMD private chip revision/family**다. 검사 범위는 family **0x8f**, unsigned revision **0x28..0x4f**다.

## 실제 의존하는 Windows 질의

`0xbc80` helper는 DXGI adapter 정보를 가져온 뒤 `0xb9f0` helper를 부른다. 후자는 다음 문자열을 직접 참조한다.

| 문자열 RVA | 내용 |
|---|---|
| 0x3635430 | gdi32.dll |
| 0x3635440 | D3DKMTEnumAdapters |
| 0x3635458 | D3DKMTQueryAdapterInfo |

`D3DKMTQueryAdapterInfo`에 **Type=0**, 최초 buffer size **0x260**을 전달한다. Windows/Wine의 [KMTQUERYADAPTERINFOTYPE](https://github.com/ValveSoftware/wine/blob/7b3ceaf8644bb5d57140903e12897efac69be16a/include/ddk/d3dkmthk.h)에서 Type0은 **KMTQAITYPE_UMDRIVERPRIVATE**다. 반환된 vendor private payload에서 revision/family를 얻으며, 이 질의가 실패하면 helper가 false를 반환한다. 큰 private payload를 위한 buffer-size 재시도도 원본 함수에 있다.

공식 Valve Wine `experimental_11.0`의 commit **7b3ceaf8644bb5d57140903e12897efac69be16a** (2026-10-21T19:32:13Z), [dlls/win32u/d3dkmt.c](https://github.com/ValveSoftware/wine/blob/7b3ceaf8644bb5d57140903e12897efac69be16a/dlls/win32u/d3dkmt.c)의 `NtGdiDdDDIQueryAdapterInfo`는 다음 세 경우만 처리한다.

- CHECKDRIVERUPDATESTATUS
- DRIVERVERSION
- WDDM_2_7_CAPS

나머지는 `type %d not handled.`를 기록하고 **STATUS_NOT_IMPLEMENTED**를 반환한다. UMDRIVERPRIVATE 구현이 없다는 소스 증거다. 이 구조는 원본 DLL이 RX6900XT에서 ML provider를 제공하지 않은 현상을 설명한다.

**판정 상태: 런타임 연결까지 확인.** 원본 DLL query를 `WINEDEBUG=+d3dkmt`로 실행한 `runs/sdk_original_d3dkmt/stderr.log`의 46·48행에 `NtGdiDdDDIQueryAdapterInfo type 0 not handled.`가 두 번 기록됐다. `runs/sdk_original_d3dkmt/result.json`은 exit_code=0, faults=[]이며 완료 시각은 2026-11-09T18:30:49.718754+00:00이다. 정적 분석에서 확인한 Type0 의존성이 현재 설치 Wine의 실제 실행에서도 미구현 경로에 도달했다. 이어 Navi2 predicate만 바꾼 호환 사본은 아래와 같이 full4.1.2 provider 선택과 dispatch에 성공했다. 따라서 원본의 provider 미노출은 **Windows vendor-private query 호환성 gate**로 분류한다. GPU dot4 능력 부족으로 분류할 근거는 없다.

참고로 [amdxc64/main.c](https://github.com/ValveSoftware/wine/blob/7b3ceaf8644bb5d57140903e12897efac69be16a/dlls/amdxc64/main.c)의 자동 driver-upgrade allowlist도 1002:73f0 또는 7440..749f(RDNA3)만 받는다. 이것은 **driver provider 주입 조건**이며 위 embedded SDK IsSupported 실패와 따로 구분한다.

고정 Wine source 파일 SHA256:

세 파일은 각각 `analysis/source_evidence/wine_d3dkmt.c`, `wine_d3dkmthk.h`, `wine_amdxc64_main.c`에도 같은 SHA로 보존했다. `d3dkmt.c`의 기본 로그 채널은 `WINE_DEFAULT_DEBUG_CHANNEL(d3dkmt)`다.

| 파일 | bytes | SHA256 |
|---|---:|---|
| dlls/win32u/d3dkmt.c | 74582 | da3a87c7a4da8efde479218c22983a8e44ca9da1ce4208aae974ed957a0521cc |
| include/ddk/d3dkmthk.h | 40801 | 639ba21e4fe0f8260e973b9ef2f0abb6a0d47ad226613ca394b35489c936c6c1 |
| dlls/amdxc64/main.c | 8472 | fb02aec3a71f61627438d9a45b3c9fe8bf2a039fa215837b93f79fd2ac2ba4a6 |

## 허용한 연구용 호환 사본

루트 에이전트가 community exact의 실제 full4.1.2 실행 성공과 위 원인을 확인한 뒤, **연구 폴더 사본의 Navi2 predicate만 강제로 true로 만드는 비교**를 지시했다.

`tools/force_navi2_compat.py`는 원본 SHA256·크기, PE header, RTTI 이름, type descriptor, complete object locator, vtable slot2, RVA/file offset 및 **원본 함수75bytes 전체**를 모두 확인한다. 파일 offset0x97c0의 첫6bytes만 다음처럼 바꾼다.

```asm
b8 01 00 00 00  ; mov eax, 1
c3              ; ret
```

원본은 true/false를 AL로 반환한다. EAX=1은 AL=1도 설정하고, 함수 진입 직후 return하므로 stack 조정이 필요 없다. 다른 provider predicate, shader, weights, exports와 headers는 바꾸지 않는다.

사본 생성 전후 모든 structurally valid DXBC shader container의 offset/size/전체 SHA256/chunk 목록을 비교하고, **수정 구간 밖 모든 bytes 동일**도 검사한다. 출력 파일과 manifest가 이미 있으면 덮어쓰지 않는다. 입력·출력은 이 연구 폴더 안으로 제한한다.

재현 명령(이 명령은 DLL을 실행하지 않는다):

```bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops/results/fsr4
python3 tools/force_navi2_compat.py --source bench/original/amd_fidelityfx_upscaler_dx12.dll
```

기본 입력은 `sdk/original/amd_fidelityfx_upscaler_dx12.dll`, 출력은 새 `sdk/navi2_compat/amd_fidelityfx_upscaler_dx12.dll`와 `PROVIDER_COMPAT_MANIFEST.json`다.

이 사본의 명칭은 **SDK4.1.2 Navi2 호환 사본**이다. 변경 없는 공식 runtime 측정으로 표현하지 않는다. 원본 Authenticode bytes와 PE checksum은 그대로 남지만 코드 수정으로 원본 서명은 유효하지 않게 된다. 강제 predicate 자체는 argument를 검사하지 않으므로 이 연구에서 확인한 정상 D3D12 device를 제공하는 실행기에서만 사용한다. 실제 provider 선택, dispatch 성공, 출력 일치와 시간은 메인 연구가 별도로 확인한다.

## 추가 비교 기준: 배포된 d-2 reference

[timing-kit README](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/blob/d1c3b30db656bf6a38634642542ee7736127d211/timing-kit/README.md)는 `d-2-driver-rx6000`를 AMD4.1.2 RX6000 reference shaders로 명시하며, shader를 kit 안에서 실행한다고 구분한다. 해당 wrapper는 native SDK runtime 동일성을 뜻하지 않는다.

2026-11-02.14 release의 Windows ZIP은 **108,131,138 bytes**, SHA256 **b9ea1f25c259356531c0491c7339e60b01f410fad179b120c7ddde34de579134**이며, release 게시 시각은 **2026-11-03T01:34:40Z**다. 이 worker는 중복 다운로드하지 않았다. 루트 에이전트가 확보한 d2 wrapper와의 추가 shader/output 비교는 메인 연구 기록을 따른다.

## 정적 분석 재현

```bash
objdump -d --start-address=0x18000a3c0 --stop-address=0x18000a40b bench/original/amd_fidelityfx_upscaler_dx12.dll
objdump -d --start-address=0x18000bc80 --stop-address=0x18000bfe6 bench/original/amd_fidelityfx_upscaler_dx12.dll
objdump -d --start-address=0x18000b9f0 --stop-address=0x18000bc7d bench/original/amd_fidelityfx_upscaler_dx12.dll
```

## 연구용 사본 생성 결과

위 명령을 실제로 실행했고 exit 0을 확인했다. 새 사본은 **57,664,272 bytes**, SHA256 **c3fc8ce90d6294b68fd5845e8867be797048c459b4fecb307c85c15fcfac7d69**다. 바뀐 byte는 정확히 **6개**이며, **DXBC 1,778개 occurrence / 1,389개 unique SHA256**이 모두 동일하다. 원본 SHA와 수정 구간 밖 모든 byte도 동일하다. 생성 후 `objdump`로 새 진입점이 `mov eax,1; ret`인 것을 확인했다.

전체 개별 container 해시와 byte별 변경 manifest는 `sdk/navi2_compat/PROVIDER_COMPAT_MANIFEST.json`에 있다. 출력이 이미 있으므로 같은 명령은 덮어쓰기 대신 거부한다. 다시 생성할 때는 `--output sdk/navi2_compat_repro/amd_fidelityfx_upscaler_dx12.dll`처럼 새로운 연구 하위 경로를 지정한다.

## 루트 에이전트의 호환 사본 런타임 확인

`measurements/sdk412_profile.txt`를 직접 읽어 다음을 확인했다.

```text
V|sdk412|0|f5a5ca1e01001002|4.1.2
CASE|sdk412|1920x1080 Native|1920x1080|phases=8|moving
P|sdk412|automatic|0000000000000000|automatic default|f5a5ca1e01001002|4.1.2|0
CASE|sdk412|3840x2160 Native|3840x2160|phases=8|moving
P|sdk412|automatic|0000000000000000|automatic default|f5a5ca1e01001002|4.1.2|0
```

즉, 두 해상도에서 **provider4.1.2 / id f5a5ca1e01001002**가 실제 선택됐다. `runs/sdk412_profile/result.json`은 status=completed, exit_code=0, faults=[]이고 완료 시각은 **2026-11-09T18:31:01.817848+00:00**이다. 루트가 full dispatch를 수행한 계측 실행의 결과이며, 이 worker가 GPU를 별도로 실행한 결과는 아니다. 출력 일치 및 통상 실행의 교대 시간 분포는 메인 보고서가 정본이다.

이 결과는 **shader·weights를 바꾸지 않은 SDK4.1.2 Navi2 호환 사본의 실행 가능성**을 입증한다. 변경 없는 원본 SDK가 동일 provider를 자동 선택했다거나, 게임의 provider 선택을 확인했다는 뜻은 아니다.
