# FSR4 연구 다운로드 통합 표

작성 UTC: 2026-11-09T18:54:38.841670+00:00

실제 보존 다운로드 **15개, 420,802,912 bytes (0.420802912 GB)**를 다시 읽어 크기와 SHA256을 검증했다. 그 밖에 압축본과 중복되는 로컬 소스 2개도 다시 해시했다. 불일치는 없었다. SDK 헤더는 `downloads/sdk_headers.json`에 기록된 `sdk/` 아래 5개 파일이며, 별도 `sdk_headers/` 폴더는 없다.

미보존 Mesa archive의 기록값 **68,613,948 B**, 추가 원문 fetch 5개의 기록값 **272,261 B**, 추가 확인된 동일 원문 재조회 2회의 **267,164 B**를 합친 명시 항목 payload는 **489,956,285 B**다. 개별 전송 여부가 기록되지 않은 원문 11개 **1,028,339 B**를 모두 별도 fetch로 보수적으로 합해도 **490,984,624 B (0.490984624 GB)**다. 요청한 **100 GB**보다 작은 범위다. 이 계산은 지정된 기록의 파일 payload 합계이고, 전송 재시도·중복 열람·HTTP/TLS·목록 밖 웹/API 응답을 포함한 회선 총량 계측은 아니다.

이번 정리는 새 다운로드·설치 없이 수행했다. deb 2개는 `apt-get --print-uris download`로 현재 저장소 URL만 조회했고, 저장소의 byte 수와 SHA512가 실물과 일치했다. 원래 다운로드 당시 mirror URL은 기록되어 있지 않아, URL 출처를 JSON에 별도로 표시했다.

정본 JSON: `analysis/DOWNLOAD_MANIFEST_ALL.json`. 기존 다운로드 기록과 보고서는 수정하지 않았다.

## 1. 보존된 원본 다운로드: 실물 크기·해시 검증

| 파일 / 저장 위치 | 출처 URL | Bytes | SHA256 | 상태 |
|---|---|---:|---|---|
| `downloads/fsr4-timing-kit-linux-2026-11-06.tar.gz` | [원문](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/releases/download/timing-kit-2026-11-06/fsr4-timing-kit-linux-2026-11-06.tar.gz) | 54,611,189 | `fe99cd28db375447d971732c1114fe29546311ee95513312df23fe90f49725f3` | 실물 재해시 일치 |
| `downloads/overrides-d1c3b30.tar.gz` | [원문](https://api.github.com/repos/zgauthier2000/fsr4-shader-performance-overrides/tarball/d1c3b30db656bf6a38634642542ee7736127d211) | 63,530,999 | `5624395ada3f5a60911b22628ff5d5766945a56350abc23bc5e4739e9fb1c1f5` | 실물 재해시 일치 |
| `downloads/fsr4.1.2-cyboman-rx6000-2026-11-05.zip` | [원문](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/releases/download/dll-2026-11-05/fsr4.1.2-cyboman-rx6000-2026-11-05.zip) | 104,375,648 | `771dce4019f2afa832ef87f0d8f3747696efef74d58f665bd4e584b5e2590aac` | 실물 재해시 일치 |
| `downloads/fsr4-timing-kit-windows-2026-11-02.14.zip` | [원문](https://github.com/zgauthier2000/fsr4-shader-performance-overrides/releases/download/timing-kit-2026-11-02.14/fsr4-timing-kit-windows-2026-11-02.14.zip) | 108,131,138 | `b9ea1f25c259356531c0491c7339e60b01f410fad179b120c7ddde34de579134` | 실물 재해시 일치 |
| `downloads/llvm-mingw-20261104-ucrt-ubuntu-22.04-x86_64.tar.xz` | [원문](https://github.com/mstorsjo/llvm-mingw/releases/download/20261104/llvm-mingw-20261104-ucrt-ubuntu-22.04-x86_64.tar.xz) | 84,104,244 | `919380403de87dca3c581c2053230cd298af5e891286628ba0613b7fed370926` | 실물 재해시 일치 |
| `sdk/api/include/ffx_api.h` | [원문](https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/v2.4.0/Kits/FidelityFX/api/include/ffx_api.h) | 10,848 | `4d9dcf243ef5efbf03340a02f16b51a22053243a22e8a9a0b3ce91c26982b4bf` | 실물 재해시 일치 |
| `sdk/api/include/ffx_api_loader.h` | [원문](https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/v2.4.0/Kits/FidelityFX/api/include/ffx_api_loader.h) | 2,977 | `18d1901dbbc881e8131b91c65ad44bb3b4a1de6c31435d5b75236f16727ba80e` | 실물 재해시 일치 |
| `sdk/api/include/ffx_api_types.h` | [원문](https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/v2.4.0/Kits/FidelityFX/api/include/ffx_api_types.h) | 13,637 | `53f07c113a9ba493f297656e4481b6539aed0afc9f3cc8da682aa477d270c46c` | 실물 재해시 일치 |
| `sdk/api/include/dx12/ffx_api_dx12.h` | [원문](https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/v2.4.0/Kits/FidelityFX/api/include/dx12/ffx_api_dx12.h) | 11,323 | `c9afbc3c4c673723e9e5369a666f0ff54560543b5a93537f8ff453bf1c3bf530` | 실물 재해시 일치 |
| `sdk/upscalers/include/ffx_upscale.h` | [원문](https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/v2.4.0/Kits/FidelityFX/upscalers/include/ffx_upscale.h) | 19,576 | `13091e7fe5fd9c82b1fb7c044b9f08f664d9653f599d2ea500738709236fca4d` | 실물 재해시 일치 |
| `analysis/source_evidence/wine_d3dkmt.c` | [원문](https://raw.githubusercontent.com/ValveSoftware/wine/7b3ceaf8644bb5d57140903e12897efac69be16a/dlls/win32u/d3dkmt.c) | 74,582 | `da3a87c7a4da8efde479218c22983a8e44ca9da1ce4208aae974ed957a0521cc` | 실물 재해시 일치 |
| `analysis/source_evidence/wine_d3dkmthk.h` | [원문](https://raw.githubusercontent.com/ValveSoftware/wine/7b3ceaf8644bb5d57140903e12897efac69be16a/include/ddk/d3dkmthk.h) | 40,801 | `639ba21e4fe0f8260e973b9ef2f0abb6a0d47ad226613ca394b35489c936c6c1` | 실물 재해시 일치 |
| `analysis/source_evidence/wine_amdxc64_main.c` | [원문](https://raw.githubusercontent.com/ValveSoftware/wine/7b3ceaf8644bb5d57140903e12897efac69be16a/dlls/amdxc64/main.c) | 8,472 | `fb02aec3a71f61627438d9a45b3c9fe8bf2a039fa215837b93f79fd2ac2ba4a6` | 실물 재해시 일치 |
| `downloads/debs/glslang-tools_16.4.0-1_amd64.deb` | [원문](http://kr.archive.ubuntu.com/ubuntu/pool/universe/g/glslang/glslang-tools_16.4.0-1_amd64.deb) | 2,750,708 | `b4bef960f3deae8825462da1c4d7eb284affec6a764a4e16e609d8caede6204c` | 실물 재해시 일치 · apt SHA512도 일치 |
| `downloads/debs/spirv-tools_2026.3~rc1-1ubuntu1_amd64.deb` | [원문](http://kr.archive.ubuntu.com/ubuntu/pool/universe/s/spirv-tools/spirv-tools_2026.3%7erc1-1ubuntu1_amd64.deb) | 3,116,770 | `4d99ec9f6f4ee7e7af461d1dfe07a8b5bbba608a251b58314914f5b41c338ca1` | 실물 재해시 일치 · apt SHA512도 일치 |

## 2. 압축본과 중복되는 보존 소스: 총량에 다시 더하지 않음

| 파일 / 저장 위치 | 출처 URL | Bytes | SHA256 | 상태 |
|---|---|---:|---|---|
| `source/zgauthier2000-fsr4-shader-performance-overrides-d1c3b30/tools/model_clamp.py` | [원문](https://raw.githubusercontent.com/zgauthier2000/fsr4-shader-performance-overrides/d1c3b30db656bf6a38634642542ee7736127d211/tools/model_clamp.py) | 5,988 | `5baffb9b88088a4c03f63919ba6703a371e1945c72edde71aaed65f09a547c69` | 로컬 코드 일치 · archive와 중복 |
| `source/zgauthier2000-fsr4-shader-performance-overrides-d1c3b30/tools/model_tail.py` | [원문](https://raw.githubusercontent.com/zgauthier2000/fsr4-shader-performance-overrides/d1c3b30db656bf6a38634642542ee7736127d211/tools/model_tail.py) | 4,429 | `817ef9a929a81d47cdf2984bc904a1f37cd712046ecc2a420d3da5c52afaca61` | 로컬 코드 일치 · archive와 중복 |

## 3. 메모리에서 읽은 원문: 보존된 식별 기록

| 파일 / 저장 위치 | 출처 URL | Bytes | SHA256 | 상태 |
|---|---|---:|---|---|
| `mesa-26.2.4.tar.xz` | [원문](https://archive.mesa3d.org/mesa-26.2.4.tar.xz) | 68,613,948 | `bce5f7fbebb934373b86c999a064d52fb5065878dc57f287f95346648ec832e9` | 기록만 보존 · 실물 미보존 |
| `aco_select_nir_alu.cpp` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/compiler/instruction_selection/aco_select_nir_alu.cpp) | 177,011 | `1c3e092dd6d82c27e08fca55b11f15d89724a0509c7044b2c441351afd4b4a5b` | 식별값 보존 · 개별 전송 미기록 |
| `aco_optimizer.cpp` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/compiler/aco_optimizer.cpp) | 223,082 | `88da817fea2011af66f44c439ca5ced65c33524f2d8c313a08b81ec1eab97242` | 식별값 보존 · 개별 전송 미기록 |
| `radv_instance.c` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/vulkan/radv_instance.c) | 15,848 | `9d6db5588cdffd670fbbb58b75ccd448249c01052100a895b4d9a9fbd63c6ea8` | 식별값 보존 · 개별 전송 미기록 |
| `radv_shader.c` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/vulkan/radv_shader.c) | 156,593 | `48da66a47a6cc66da6ecb998cd203bc940988adf1db465bde060addffb99d415` | 식별값 보존 · 개별 전송 미기록 |
| `ac_shader_util.h` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/common/ac_shader_util.h) | 13,198 | `c911fc892f9a995570ba14287f98eb3c9ef3d5348130221054cd5b71382191ba` | 식별값 보존 · 개별 전송 미기록 |
| `ac_shader_util.c` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/common/ac_shader_util.c) | 64,333 | `91b28d0dc39b95b8fa737ab2a3bd3732518c0641481cb8168af908289e8dcc64` | 식별값 보존 · 개별 전송 미기록 |
| `aco_statistics.cpp` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/compiler/aco_statistics.cpp) | 25,709 | `e67cfaea74c07d595ebe8892a364e707030737b7ed07e263e08b4a9c9bb18868` | 식별값 보존 · 개별 전송 미기록 |
| `nir_opcodes.py` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/compiler/nir/nir_opcodes.py) | 87,677 | `c8ed942f9c130627f79109228980f232ffc0259ae6d1f6a26a17443fe1f7bfd8` | 식별값 보존 · 개별 전송 미기록 |
| `nir_opt_algebraic.py` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/compiler/nir/nir_opt_algebraic.py) | 254,471 | `f63c37ce8072425e8f490546b9800699363be01fda0431ec950c1686b605adbf` | 식별값 보존 · 개별 전송 미기록 |
| `ac_gpu_info.c` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/common/ac_gpu_info.c) | 110,571 | `0cede344881cdac51dcdc2f3ca5ba1ffc7bdf6ee1a89d5eed4c90b0b37fc15e1` | 별도 원문 fetch 기록 · 미보존 |
| `radv_pipeline.c` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/vulkan/radv_pipeline.c) | 50,118 | `b28475667a6b751f875d69a014237442d365b8d38eca672a8ee34e2ea027a06c` | 별도 원문 fetch 기록 · 미보존 |
| `radv_shader_object.c` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/vulkan/radv_shader_object.c) | 29,851 | `e38b43993b4582254a03b1378edc609e26d9f91938db7c31c77c5d0229d59497` | 별도 원문 fetch 기록 · 미보존 |
| `radv_shader_info.c` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/vulkan/radv_shader_info.c) | 68,236 | `d614c9ff88a7160ea944f65a58e15ad7e5186fa5333756293f76e6dd1ec41398` | 별도 원문 fetch 기록 · 미보존 |
| `radv_pipeline_compute.c` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/vulkan/radv_pipeline_compute.c) | 13,485 | `0c36fb5dfcf1c20099e2137c7368b5199816006a87352763c79de6b4eaf1096f` | 별도 원문 fetch 기록 · 미보존 |

## 총량과 확인 범위

| 구분 | Bytes | 판정 |
|---|---:|---|
| 실물 재검증한 원본 다운로드 15개 | 420,802,912 | 보존 파일 payload 확정 |
| 압축본과 중복되는 로컬 source 2개 | 10,417 | 재해시했지만 다운로드 합계에 중복 가산하지 않음 |
| 미보존 Mesa archive 기록 | 68,613,948 | 기록값; 현재 재해시 불가 |
| 별도 추가 원문 fetch 5개 | 272,261 | 기록값; 현재 재해시 불가 |
| 보존 원본 + 명시된 archive/추가 fetch | 489,689,121 | 지정된 기록의 payload 합계 |
| 추가 확인된 같은 원문 재조회 2회 | 267,164 | 새 고유 파일은 아니지만 추가 요청 byte로 기록 |
| 명시된 payload + 알려진 재조회 | 489,956,285 | 재조회 2회를 포함한 기록 범위 합계 |
| 불명확한 개별 fetch 11개도 전부 추가한 값 | 490,984,624 | 알려진 재조회와 중복 가능성을 모두 가산한 보수적 합계 |
| 전체 회선 전송량 | 미계측 | 반복 fetch·재시도·웹/API 응답을 포함하는 정확한 총량은 알 수 없음 |

ISA_METHOD의 초기 Mesa source 9개와 model_clamp/model_tail 2개는 archive 추출인지 별도 raw 요청인지 구분이 남아 있지 않다. 따라서 독립 다운로드로 확정 합산하지 않는다. 추가 source fetch 5개는 별도 요청이었다는 기록이 있다. 모든 미보존 항목의 SHA는 기존 원문 식별 기록이며 이번에 원문을 다시 내려받아 검증했다는 뜻이 아니다.

기존 vendor SDK DLL·Proton/Wine·드라이버 사본, 압축 해제 결과 전체, 생성한 harness/shader·GPU 캡처·A/B 산출물은 새 다운로드 합계에서 제외했다. 이 표는 DOWNLOADS.json, downloads의 지정 JSON/deb, SDK 헤더 5개, Wine 원문 3개, ISA_METHOD/ISA_ADDITIONAL_SOURCE_FETCHES 식별 기록을 대상으로 한다.

## 추가 확인된 동일 원문 재조회

ISA 담당이 subgroup 각주를 확인하며 아래 두 파일을 같은 raw URL에서 다시 읽었다고 보고했다. SHA는 기존 기록과 같고 메모리에서만 읽었으므로 새 고유 파일로 추가하지 않았다. 알려진 추가 요청 payload267,164 B를 위 총량에 별도 반영했다. 다른 반복 요청은 전송 이력이 없어 계량하지 않았다.

| 파일 | 출처 URL | Bytes | SHA256 | 상태 |
|---|---|---:|---|---|
| `radv_shader.c` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/vulkan/radv_shader.c) | 156,593 | `48da66a47a6cc66da6ecb998cd203bc940988adf1db465bde060addffb99d415` | 동일 원문 재조회 · 메모리에서만 읽음 |
| `ac_gpu_info.c` | [원문](https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.2.4/src/amd/common/ac_gpu_info.c) | 110,571 | `0cede344881cdac51dcdc2f3ca5ba1ffc7bdf6ee1a89d5eed4c90b0b37fc15e1` | 동일 원문 재조회 · 메모리에서만 읽음 |
