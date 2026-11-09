# TunableOp 비결정 원인 조사 v1.0

작성·갱신: 2026-11-08. **상태: 개별 GEMM 13개와 Qwen·Anima 모델의 원인 경로·연구 필터 검증 완료.**

Anima는 FP32 GSU4 5행만 Default로 바꾸자 두 새 프로세스의 conditioning·같은 mode의 latent·RGB가 모두 같아졌다. cand 이미지 총 6개가 하나의 SHA였고, filtered Tunable+manual_vt의 warm denoise **45.106621초, off 대비 1.78945배**로 기존 속도 이득을 유지했다.

Qwen은 FP32 GSU4 2행만 바꾼 뒤 동일 프로세스의 25-step tbase 세 결과가 하나의 latent·RGB SHA로 고정됐다. 원본은 세 결과가 모두 달랐다. warm denoise는 원본 **55.715586초**, 필터 **56.448651초**였고, 같은 조건을 확인한 off 기준은 **57.286772초**다. Qwen 필터의 새 프로세스 재현성은 이번에 측정하지 않았다.

운영 파일에 필터를 적용하지 않았다. GPU는 주 세션만 직렬 실행했고 이 분석 담당은 torch를 import하지 않았다. 생성 PNG30개의 파일·RGB SHA와 크기를 모두 직접 재확인했고 RGB RMSE도 계산했다(§15). 최종 통합 보호 검사에서도 **1022개 파일의 변경·누락0개**, errors 없음이며 GPU 종료 audit도 통과했다. 개별 실측은 §10, Anima는 §11, Qwen은 §13, 최종 핵심 숫자5개는 §16을 본다.

## 1. 이번에 확인한 사실

| 항목 | Qwen | Anima |
|---|---:|---:|
| 원본 CSV 전체 행 | 34 | 62 |
| validator 제외 GEMM 행 | 29 | 57 |
| 설치 Tensile DB와 정확히 매핑 | 25 | 57 |
| Default | 3 | 0 |
| reserved/기타 ID로 추가 trace 필요 | 1 (ID −9) | 0 |
| 매핑된 GSU1 | 22 | 45 |
| 매핑된 GSU4 | 3 | 12 |
| 그중 FP32 GSU4 | 2 | 5 |

Qwen 모델 이름의 INT8은 이 CSV의 GEMM 입력 dtype를 뜻하지 않는다. 실제 행은 float와 Half이다. CSV의 행 수와 GEMM 수를 구분했고, 현재 파일의 Anima 수는 과거 문서의 “61 shape”와 다르다. 원자료는 tunable_cpu/inventory.json에 모든 행·원본 SHA·설치 DB SHA와 함께 저장했다.

설치 runtime은 MOSS .venv-moss의 PyTorch 2.12.0+rocm10.2.0a20261011, torch git 9ccf2fd0aab44087b58af23ad20bc961c29bb852, HIP 7.17.26405, rocBLAS 5.8.0.3d8f334b다. Applio의 .venv-rocm10과 혼동하지 않는다. §1의 inventory·설치 소스 조사는 CPU에서 수행했다. §10의 GPU 실측도 이 runtime을 result.json에 기록해 별도로 확인했다.

### 1.1 음수 솔루션 번호의 정확한 의미

설치 버전과 같은 rocBLAS commit 3d8f334b의 소스에서 다음 변환을 확인했다.

~~~text
c_rocblas_solutions_reserved = 10
Tensile_index = -rocBLAS_solution_index - 11
~~~

따라서 Gemm_Rocblas_-1086325192는 설치 Tensile index 1086325181이다. 단순히 부호를 바꾸거나 CSV 번호를 배열의 순번으로 사용하면 다른 커널을 가리킨다. 음수는 Tensile과 hipBLASLt를 구분하는 인코딩이며 그 자체가 오류의 증거는 아니다. −9는 이 버전 소스의 reserved 영역에 있으므로 일반 Tensile 변환을 적용하지 않았다. 해당 행은 Qwen float TN, m=4096,n=1,k=4096이다. 실제 rocBLAS 호출/커널 trace로 어떤 경로가 실행되는지 확인해야 한다.

근거:

- 정확한 소스: https://github.com/ROCm/rocm-libraries/blob/3d8f334b/projects/rocblas/library/src/tensile_host.cpp (map_index_rocblas_to_tensile)
- 상수: https://github.com/ROCm/rocm-libraries/blob/3d8f334b/projects/rocblas/library/src/include/definitions.hpp
- 공식 저장소 maintainer의 설명: https://github.com/ROCm/rocm-libraries/issues/8119 (2026-09-02 설명 참조)
- 로컬 정확 사본: tunable_cpu/sources/tensile_host.cpp, definitions.hpp, tensile_host.hpp.

### 1.2 GSU4와 atomic은 같은 말이 아니다

설치 gfx1030 Tensile .dat 71개 내의 고유 solution 5,337개를 의존 패키지 없이 CPU 파싱했다. .co는 CCOB 압축 offload bundle이므로 clang-offload-bundler로 gfx1030 ELF를 추출했다. llvm-objdump의 --disassemble-symbols만 쓰면 local label 앞에서 본문이 잘리는 문제가 있어, ELF의 해당 global function 주소부터 다음 global function 주소까지 전체를 읽었다.

| 실제 선택된 행 | 설치 커널 요약 | static assembly 결과 |
|---|---|---|
| Qwen entry 16: float TN 64×2400×4096 | MT64×64×16, GSU4, globalAccumulation=0 | buffer_atomic_cmpswap 명령 256개 |
| Anima entry 2: float TN 1024×32×1024 | MT32×32×8, GSU4, globalAccumulation=0 | buffer_atomic_cmpswap 명령 64개 |
| Qwen entry 19: Half NT 640×640×1152 | MT128×128×32, GSU4, globalAccumulation=2, workspaceSizePerElemC=16 | atomic 명령 0개 |

숫자는 disassembly에 나타난 **정적 명령 수**이고, 한 GEMM에서 실제 실행된 atomic 횟수가 아니다. 앞의 두 커널에 atomic compare-and-swap가 있다는 것은 확인한 사실이며, §10에서 정확히 그 커널의 고정 입력 비결정도 재현했다. Half GSU4는 32회 바이트가 같고 별도 PostGSU4 커널이 trace에 나타났다. FP32의 원자적 누산과 workspace 합산을 한 범주로 뭉뚱그리지 않는다. 모델의 이미지 차이에 대한 인과 판정은 모델 단계 결과와 구분한다.

원자료:

- tunable_cpu/assembly_summary.json: 커널 이름, 원 .co SHA, ELF 주소 범위, 명령 수.
- tunable_cpu/qwen_16.s, anima_2.s, qwen_19.s.
- tunable_cpu/TensileLibrary_Type_*.elf: 운영 .co를 읽어 연구 폴더에 푼 사본.

## 2. 이전에 확인됐던 사실과 이번 상태

이 표는 이전 실험의 정본 기록을 옮긴 것이며 v1.0에서 새로 재현한 결과가 아니다.

| 모델 | 기존 관측 | 이번 v1.0 상태 |
|---|---|---|
| Qwen INT8 | 같은 프로세스·CSV·seed의 Tunable on끼리 PNG max 152–245, RMSE 3–7/255. off는 바이트 일치 | entry0·16 비결정·실제 sampler hit 확인. 원본25-step tbase3회 모두 다르고, 2행 Default 필터 뒤3회 동일. 속도55.715586→56.448651초 |
| Anima FP16 | 같은 프로세스 안은 바이트 일치. 같은 CSV를 새 프로세스에서 TUNING=0으로 읽으면 RGB RMSE 4–7/255. off는 결정적 | FP32 5행 비결정·실제 CLIP hit·conditioning 재사용 확인. 5행 Default 필터 p0/p1의 조건·같은 mode 이미지가 바이트 일치. off 대비1.78945배 유지 |
| MOSS INT8 | 프로세스 안에서도 다름. 첫 tuning 중 GSU1 커널에서 메모리 오류, rotating buffer 0으로 한 번 끝까지 실행 | 이번에는 미착수. Qwen을 먼저 사용 |

기존 근거: PROJECT/DEEP_RESEARCH_MERGED_20261107_KO.md §5·§6.1·§6.2·§6.4, RDNA2_OPTIMIZATION_GUIDE_KO.md §9.20. 기존 Anima 경로 1.79배는 약 81.01초 native 대비 약 45.20초 Tunable+manual_vt denoise의 비율이다. 이번에는 같은 세션에서 off native 80.715881초와 filtered Tunable+manual_vt 45.106621초를 새로 재서 1.78945배를 확인했다. 과거 baseline을 새 후보 시간의 분모로 재사용하지 않았다.

## 3. CSV만으로 복원 가능한 것과 불가능한 것

설치 torch/include/ATen/cuda/tunable/GemmCommon.h의 Signature()가 저장하는 것은 dtype, transa/transb, m/n/k, lda/ldb/ldc, batch count다. alpha/beta, batch stride, 입력 값, storage offset, pointer alignment는 포함하지 않는다. 특히 Anima의 batched Half 행에는 lda=1024인데 head dimension=64인 것도 있다. 이를 무조건 packed batch라고 가정하면 실제 모델 레이아웃을 재현했다고 할 수 없다.

이번 첫 프로브는 Qwen entry 16의 **nonbatched** GEMM을 고른다. 물리 column-major A=(4096,64), B=(4096,2400), C=(64,2400), leading dimension=(4096,4096,64), A transpose로 요청 geometry를 만든다. seed 20261108의 CPU 생성 입력을 GPU로 한 번 옮겨 재사용하고 alpha=1,beta=0으로 고정한다. tensor storage는 40,984,576 bytes다. 이것은 같은 dtype·shape·transpose·leading dimension·solution을 시험하는 합성 입력이며 모델에서 캡처한 값은 아니다.

v10_tunable_probe.py는 GPU replay 전에 비어 있는 Tunable DB로 한 번 호출하여 **실제 dispatcher가 기록한 untuned op/signature**를 읽는다. 요청 행과 완전히 같아야 다음 단계로 진행한다. 이어 사본 CSV를 read_file()로 읽고 get_results()의 행과 요청 솔루션도 확인한다. 그러므로 transpose/출력 stride가 달라 다른 shape를 측정했는데 성공으로 보고하는 경로를 차단했다.

batched 프로브는 canonical packed stride를 명시적으로 가정하며 --batch-strides로 정확한 A/B/C stride를 줄 수 있다. interleaved/겹치는 layout은 임의 추정하지 않고 실패시켜 실제 tensor capture가 필요한 상태로 남긴다.

## 4. 원인에 대한 현재 판정

1. **확인:** Qwen16·Anima2의 같은 입력·같은 솔루션·같은 signature가 프로세스 안과 사이에서 다른 출력을 냈다. 실제 선택 커널은 설치 DB에 매핑한 FP32 GSU4 atomic 커널과 일치한다. off와 Default 치환은 관측한 모든 반복에서 일정했다. **원리 설명에 대한 추론:** 병렬 atomic 누산 순서에 따라 FP32 반올림이 달라지는 경로와 증거가 일치한다. atomic의 실행 순서 자체를 추적한 것은 아니다.
2. Anima에서는 FP32 GSU4 GEMM이 텍스트 조건을 달라지게 하고, 프로세스 안의 조건 재사용이 그 차이를 반복 때 가리는 경로를 확인했다. 근거는 5행의 고정 입력 비결정, 실제 CLIP hit, token 동일·conditioning 차이, 반복 시 조건 재사용, 5행만 Default로 바꾼 뒤 두 새 프로세스의 조건·같은 mode RGB가 같아진 결과다. 원본 두 모델 프로세스 중 하나는 profiler/로그 진단 실행이라는 제한은 그대로 남긴다. 5행 각각의 이미지 차이 기여와 모든 prompt에 대한 보장은 하지 않는다.
3. 캡처한 sampler는 두 모델 모두 CSV miss가0이었다. Qwen CLIP에는 missing shape가 있었으나 같은 프로세스의 비교는 같은 conditioning을 재사용했다. 다른 prompt·dtype/layout·입력 값·allocator alignment의 경로는 이 결과의 보장 범위 밖이다. model wrapper는 missing CSV를 따로 남긴다.
4. Qwen의 reserved −9 행은 자동 변환/필터링하지 않았다. 해당 내부 경로는 미분류이며 이번 두 행 필터의 원인 판정에 필요한 추가 작업으로 남기지 않았다.
5. Qwen에서는 비결정이 확인된 FP32 두 shape가 실제 sampler 구간에서 반복 실행됐다. 1-step 진단과 원본25-step 실행 모두 같은 conditioning·입력 latent를 재사용했는데 출력 latent·RGB가 달랐다. 그 두 행만 Default로 바꾼25-step 필터에서는 tbase3회가 같아졌다. off base도 같았다. Anima의 문제5행이 CLIP에만 있었던 것과 실행 위치가 다르며, 이 차이가 이번 모델들의 프로세스 안/사이 관측을 설명한다.

## 5. 준비한 코드와 안전 범위

- v10_tunable_inventory.py: CPU 전용. CSV와 설치 Tensile DB 매핑, SHA와 metadata 기록.
- v10_tunable_probe.py: --run이 없으면 CPU plan. 실행 시 자식 프로세스 안에서 GpuAudit 확보 후 torch import. 실제 signature gate, 선택 행 gate, on/off 고정 입력 32회, 바이트 SHA·finite·RMSE·NRMSE, event timing, optional profiler, CPU output 저장. torch.cuda.tunable.write_file()를 호출하지 않는다.
- v10_tunable_filter.py: 원본 SHA 확인 후 **새 연구 CSV**만 만든다. --entry로 실제 원인 행을 고르거나, 탐색 후보로 fp32_split_default/all_split_default 정책을 쓸 수 있다.
- v10_tunable_model.py: 기존 Qwen/Anima runner를 읽어 동일 프로세스에서 실행하는 연구 wrapper. GpuAudit, private caches/FindDb/CSV, 안전 변수, TUNING=0, missing shape 기록, 생성 PNG의 raw RGB SHA, runner/workflow/CSV SHA를 기록한다. 운영 runner는 고치지 않는다.
- v10_tunable_capture.py: 기존 runner의 instrument(nodes)가 끝난 직후 CLIP.encode_from_tokens_scheduled와 KSampler.sample을 연구 프로세스 안에서 감싼다. tokens·positive/negative conditioning·입력/출력 latent의 모든 tensor leaf에 raw-byte SHA를 기록하고 seed, mode, 그 시점까지 실제 CLIP 호출 수를 남긴다. 기존 sampler 타이머 바깥에서 hash를 계산한다. unknown object는 주소를 해시하지 않고 타입과 미해시 사실을 기록한다.
- v10_tunable_rgb_compare.py: CPU 전용 Pillow 분석. 저장된 PNG를 RGB로 읽어 protocol의 파일·RGB SHA와 크기를 재확인하고, 지정한 이미지 쌍의 8-bit RGB 최대차·RMSE를 계산한다. 모델·GPU·torch는 불러오지 않고 이미지 파일도 바꾸지 않는다.

Qwen 기존 runner의 main 호출에는 private user directory가 없고 temp가 공유 HERE/temp였다. 원본을 바꾸지 않고 wrapper가 **정확히 runtime main.py에 진입할 때만** sys.argv의 --user-directory를 out/user, --temp-directory를 out/temp로 지정한다. 기존 reference input 경로·workflow·seed는 유지한다. 실제 runtime argv와 runner/workflow/CSV의 전후 SHA를 protocol.json에 저장한다.

--trace-csv-hits는 설치 C++ TuningResultsManager::Lookup의 level-3 “ResultEntry found for op,signature” 로그를 private 파일에 받는다. CLIP와 sampler 구간의 파일 offset으로 정확한 CSV hit와 missing signature를 구분한다. **get_results()에 로드된 행을 실제 hit라고 세지 않는다.** --profile-conditioning은 이 기록과 함께 text-encoder 구간의 GPU trace를 저장하므로 선택된 SGEMM 이름이 실제 text-encoder에서 실행됐는지 확인할 수 있다. 이 모드는 instrumented_diagnostic이고 valid_for_speed_comparison=false다. 속도·1.79배 비교는 두 플래그를 빼고 재는 별도 실행으로 판정한다.

연구 필터의 현재 상태:

| 파일 | 바뀐 행 | 상태 |
|---|---|---|
| tunable_cpu/qwen_fp32_split_default0.csv | entry 0,16을 Default로 | 고정25-step workflow의 동일 프로세스 재현성·속도 검증 완료. 새 프로세스 비교는 미실행 |
| tunable_cpu/anima_fp32_split_default0.csv | entry 2,5,7,13,15를 Default로 | 고정 workflow의 새 프로세스 conditioning·RGB 재현성과 off 대비1.78945배 검증 완료 |

두 파일은 확인한 workflow·seed·버전 안에서 재현성과 속도를 검증한 연구 필터로 승격한다. Anima는 새 프로세스 두 개, Qwen은 동일 프로세스의 반복이라는 검증 범위가 다르다. 운영 파일에 적용하지 않았고 다른 prompt·새 shape의 결정성을 보장하는 범용 필터는 아니다. GSU4 Half를 일괄 삭제하지 않았다.

## 6. 재개 명령

모든 GPU 명령은 Applio GPU 작업 완료 후 **주 세션이 하나씩** 실행한다. 처음 명령은 GPU를 쓰지 않는 inventory 재생성이다.

~~~bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
python3 -B v10_tunable_inventory.py
~~~

Qwen 최소 원인 프로브와 Default 대조의 재현 예다. 아래 out은 완료한 원자료를 덮어쓰지 않는 새 이름이며, 현재 완료 결과를 다시 실행하라는 지시는 아니다.

~~~bash
MOSS_PY=~/DevSpace/moss-image-rdna2-lab01/.venv-moss/bin/python
"$MOSS_PY" -B v10_tunable_probe.py --entry 16 --out results/v10/tunable_qwen16_original_reproduction_v1 --repeats 32 --profile --save-outputs --run
"$MOSS_PY" -B v10_tunable_probe.py --entry 16 --solution Default --out results/v10/tunable_qwen16_default_reproduction_v1 --repeats 32 --profile --save-outputs --run
~~~

새 프로세스에서 같은 입력은 첫 명령의 out만 p1로 바꿔 실행하고 input_storage_sha256와 output on/off 첫 SHA를 대조한다. Qwen entry 0·19도 필요할 때 같은 방식으로 확인한다. Anima는 --csv ~/DevSpace/anima-rdna2-lab01/results/tunableop1/tunableop_results0.csv --entry 2를 쓰며 out은 별도 이름으로 정한다. seed는 공통 20261108이다.

Anima 기존 모델 runner와 실제 workflow:

- runner: ~/DevSpace/anima-rdna2-lab01/scripts/anima_lab.py
- workflow: ~/DevSpace/anima-rdna2-lab01/workflows/anima_t2i_1024_api.json
- runtime: ~/DevSpace/moss-image-rdna2-lab01/runtime/ComfyUI
- 기본 조건: 1024×1024, 30 steps, seed 20261107, manual_vt, private port 8477.

text-encoder의 실제 CSV hit·kernel과 조건 캐시를 먼저 확인하는 진단 명령:

~~~bash
"$MOSS_PY" -B v10_tunable_model.py --model anima --modes base,cand,cand --trace-csv-hits --profile-conditioning --out results/v10/tunable_anima_conditioning_reproduction_v1 --run
~~~

conditioning_capture.json에서 CLIP 호출이 첫 positive/negative 두 번만 있었는지, 반복 sampler의 clip_calls_seen이 더 늘지 않았는지, conditioning SHA가 같은지 확인한다. 새 프로세스와 필터 전후를 비교해 token SHA는 같은데 conditioning SHA가 달라졌는지 본다. 선택 SGEMM의 exact CSV hit와 text-encoder trace까지 연결된 뒤에만 조건 캐시 가설의 근거로 사용한다.

완료한 Anima 비교의 재현 명령 예다. 기존 결과는 유지하고 새 out 이름을 사용한다.

~~~bash
"$MOSS_PY" -B v10_tunable_model.py --model anima --modes base,cand,cand,cand --out results/v10/tunable_anima_original_reproduction_v1 --run
"$MOSS_PY" -B v10_tunable_model.py --model anima --modes base,cand,cand,cand --csv results/v10/tunable_cpu/anima_fp32_split_default0.csv --out results/v10/tunable_anima_filtered_reproduction_v1_p0 --run
"$MOSS_PY" -B v10_tunable_model.py --model anima --modes base,cand,cand,cand --csv results/v10/tunable_cpu/anima_fp32_split_default0.csv --out results/v10/tunable_anima_filtered_reproduction_v1_p1 --run
"$MOSS_PY" -B v10_tunable_model.py --model anima --tunable off --modes base,base,base --out results/v10/tunable_anima_off_reproduction_v1 --run
~~~

원본 모델의 p1까지 같은 비진단 설정으로 비교하려면 original 명령의 out만 새 p1로 바꾼다. 이번 실제 원본 모델 쌍은 diagnostic와 clean original_p0이며 그 제한은 §11.1에 적었다. 각 실행의 첫 요청/첫 mode는 warm 비용과 텍스트 캐시 상태를 구분한다. 1.79배 비교의 분모는 filtered manual_vt cand의 warm 중앙값, 분자는 off native base의 warm 중앙값이다. on baseline base(약 61초)를 off의 약 81초 baseline과 섞지 않는다.

Qwen full-model은 v10_tunable_model.py --model qwen을 사용한다. 기존 runner는 PROJECT/rehearsal_qwen_20261106/scripts/rehearsal_tunableop.py이며, workflow는 ~/DevSpace/qwen-image-21-rdna2-lab01/workflows/request-gemini-edit-640x960-api.json이다. 기본 25 steps, seed 20261106, modes=base,tbase,tbase,base다. wrapper는 per-mode API toggle이 작동하도록 ENABLED 환경변수 강제값을 제거한다. TUNING=0은 유지한다.

Qwen의 첫 mode가 base이면 Tunable off 상태에서 conditioning을 만들고 뒤의 tbase가 그것을 재사용할 수 있다. 따라서 on 상태의 실제 text-encoder hit를 확인하는 진단은 --modes tbase,tbase로 시작한다. base부터 시작한 속도 측정의 conditioning 결과를 on으로 새로 계산한 조건처럼 해석하지 않는다.

## 7. 검사와 남은 관문

CPU 실행으로 inventory 전체 82행 매핑, AST 문법, Qwen entry 16 layout plan, Qwen/Anima model plan과 workflow 경로를 확인했다. CPU plan 후 sys.modules에 torch가 없음을 확인했다. 새 CSV는 원본 SHA를 확인하고 별도 파일에 생성했다.

**완료:** §10의 13개 개별 GEMM 실행은 signature·loaded solution·입력 전후 SHA·finite·GpuAudit 관문을 통과했다. original 선택 커널은 모두 trace와 일치하며, Qwen16·Anima2의 두 새 프로세스도 고정 입력 비교를 통과했다. Default의 정확한 GSU1 커널은 trace에서 직접 관측했다. §11의 전체 모델 진단은 source/CSV/workflow 보호 SHA·finite·capture·GpuAudit를 통과했으며 diagnostic이므로 속도 비교에서 제외했다.

**완료한 C 범위:** Qwen의 개별 FP32 GEMM·실제 sampler hit·같은 프로세스 재현성 회복·속도, Anima의 조건 캐시 경로·새 프로세스 재현성 회복·1.79배 이득 유지다. MOSS는 이번 범위에서 미착수다. Qwen 필터의 새 프로세스 검증과 모든 prompt/shape로의 일반화는 미실행이다.

**통합 마무리 완료:** PNG 무결성 재확인과 원본/필터 RGB RMSE 계산은 §15에서 완료했다. 주 세션이 만든 protected_after.json(2026-11-08 11:51:03.989085 UTC)을 읽어1022개 파일·3,241,275,343bytes의 changed=[]·missing=[]·errors={}·unchanged=true를 확인했다. gpu_idle_after.json(11:51:04 UTC)은 GpuAudit valid, foreign/error0, KFD client0, gpu_idle=true다. 검사를 다시 실행하지 않고 완료 정본을 사용했다.

다운로드한 것은 정확 버전 소스 4개, 합계 130,717 bytes뿐이며 모델이나 패키지는 설치하지 않았다. URL·파일명·bytes·SHA는 tunable_cpu/sources/DOWNLOADS.json에 둔다. 설치된 .co에서 푼 ELF는 로컬 사본이며 다운로드가 아니다.

## 8. 초기 CPU inventory의 숫자 5개

1. Qwen CSV의 실제 GEMM 수: **29**
2. Anima CSV의 실제 GEMM 수: **57**
3. Qwen의 매핑된 GSU4 수: **3** (그중 FP32 2)
4. Anima의 매핑된 GSU4 수: **12** (그중 FP32 5)
5. 확인한 Qwen entry16 커널의 static atomic opcode 수: **256** (실제 실행 횟수/비결정 재현 수가 아님)

~~~bash
python3 -B - <<'PY'
import json
from pathlib import Path
p=Path('results/v10/tunable_cpu')
r=json.loads((p/'inventory.json').read_text())
a=json.loads((p/'assembly_summary.json').read_text())
print('Qwen GEMM',len(r['csvs'][0]['entries']))
print('Anima GEMM',len(r['csvs'][1]['entries']))
print('Qwen GSU4',r['csvs'][0]['globalSplitU_counts'].get('4',0))
print('Anima GSU4',r['csvs'][1]['globalSplitU_counts'].get('4',0))
print('Qwen entry16 static atomics',next(x['atomic_instruction_count'] for x in a if x['tag']=='qwen_16'))
PY
~~~

## 9. 저장된 결과를 읽는 CPU 분석 helper

v10_tunable_analyze.py는 torch·ROCm·이미지 decoder를 import하지 않고 JSON과 Chrome trace만 읽는다. --probe는 LABEL=DIR, --model은 ROLE:LABEL=DIR 형식이며 ROLE은 original/filter/off 중 하나다. 산출물은 지정한 새 분석 폴더의 analysis.json과 SUMMARY_KO.md다. 입력 결과는 수정하지 않는다.

~~~bash
python3 -B v10_tunable_analyze.py \
  --probe qwen16_original_p0=results/v10/tunable_qwen16_original_p0 \
  --probe qwen16_original_p1=results/v10/tunable_qwen16_original_p1 \
  --probe qwen16_default_p0=results/v10/tunable_qwen16_default_p0 \
  --model original:anima_original_p0=results/v10/tunable_anima_original_p0 \
  --model original:anima_original_p1=results/v10/tunable_anima_original_p1 \
  --model filter:anima_filtered_p0=results/v10/tunable_anima_filtered_p0 \
  --model filter:anima_filtered_p1=results/v10/tunable_anima_filtered_p1 \
  --model off:anima_off_p0=results/v10/tunable_anima_off_p0 \
  --warmup first-per-mode \
  --out results/v10/tunable_analysis
~~~

아직 없는 결과를 함께 지정해도 실행하며 그 항목은 pending으로 남긴다. 예전 runs.json만 존재하는 경우에는 시간을 읽어 보여 주지만, protocol·audit·조건 캡처·보호 SHA가 없으므로 v1.0 검증 완료로 바꾸지 않는다. 내부 NaN/Inf, audit 실패, signature 불일치, 입력 변경 등도 정상 표본으로 합치지 않는다.

- 프로브: 실제 op/signature, CSV에 로드된 solution, 원래·실제 stride, seed/alpha/beta, A/B/C 초기 storage SHA, A/B 전후 SHA, finite, on/off 출력 종류 수, event 시간, audit를 비교한다. 프로세스 간 비교는 고정 입력·geometry·runtime이 모두 같은지 먼저 확인한다. 선택 커널은 inventory의 Tensile 이름과 on.trace.json의 실제 이름을 따로 비교한다. 실제 이름에서 끝의 AMD kernel descriptor 접미사 .kd만 제거하고 원래 이름도 보존한다. trace 누락·kernel event 0개·Default/reserved 정적 매핑 미확정은 pending이다.
- 모델: mode별 warm denoise, 조건·입력/출력 latent·RGB 반복 SHA, CLIP 호출 수, token SHA, seed를 요약한다. 이미지 파일 SHA가 protocol에 기록된 값과 같은지 확인한 뒤 저장된 RGB SHA를 사용한다. .pt를 불러오지 않는다.
- conditioning 진단: exact CSV hit의 선택 solution을 설치 inventory로 풀고 같은 CLIP trace 구간에 예상 커널이 있는지 확인한다. 이것은 **같은 CLIP 구간의 연결**이며 개별 GEMM launch의 일대일 correlation으로 과장하지 않는다.
- 속도: 기본 warm-up 규칙은 **mode별 첫 성공 실행 제외**다. 각 arm에 warm 표본 2개 이상, 완료·배타 GPU·보호 SHA 통과, 진단 로깅/프로파일 없는 실행이 있어야 off native base 대비 original/filter 후보 속도 비율을 계산한다. Anima 후보 mode는 cand, Qwen은 tbase다. 규칙 변경은 --warmup으로 명시하고 JSON에 남긴다.

Anima off 기준은 §6의 base,base,base 명령으로 warm base 표본 2개를 확보했다. Qwen의 기본 base,tbase,tbase,base는 mode별 warm-up 제외 후 tbase 한 표본만 남는다. 속도까지 평가할 실행은 --modes base,tbase,tbase,tbase,base처럼 tbase를 최소 세 번 지정하고, off 별도 실행은 base 세 번 이상을 둔다. 결과를 보고 유리한 warm-up 규칙으로 바꾸지 않는다.

이번 Qwen의 25-step 두 실행은 base,tbase,tbase,tbase,base다. **필터 결과가 나오기 전에 확정한 비교 기준:** 주 비교는 original과 filtered의 tbase warm 표본 2개씩이다. 보조 off 비율은 각 실행의 마지막 base 하나씩, 총 2개의 중앙값을 기준으로 계산한다(배속=off 시간/on 시간). 두 마지막 base가 모두 tunable=false이고 conditioning·입력 SHA·workflow·seed·runtime이 같을 때만 합친다. 첫 base는 계속 warm-up으로 제외하고 별도 full baseline은 추가하지 않는다. 표본 범위를 명시하고 1-step 진단은 모든 속도 비교에서 제외한다.

CPU 확인은 실제 이전 Qwen/Anima runs.json의 구조와 숫자를 읽어 수행했다. 두 기록 모두 v1.0 protocol 부재를 pending으로 유지했고, 존재하지 않는 새 결과에서 속도를 만들어내지 않았다. 기존 native smoke Chrome trace의 17 launch/1 Tensile 이름도 파서로 읽었다. 이는 trace 파서 확인이며 C의 GPU 재현 근거가 아니다. 원자료는 tunable_cpu/helper_check/validation.json에 있다. 이 확인 프로세스에 torch import와 GPU 실행은 없었다.

## 10. 고정 입력 GEMM 실측 — 2026-11-08

원자료는 results/v10/tunable_{아래 label}/result.json, on.trace.json, off.trace.json, rocblas_profile.log다. 통합 CPU 분석은 tunable_analysis/current/analysis.json이며 13개 실행 모두 completed=true, valid_for_timing=true다. 원본 CSV에 있는 exact op/signature를 먼저 dispatcher로 확인한 뒤 같은 CSV 솔루션을 읽었다. A/B storage SHA는 실행 전후 같고 모든 결과가 finite이며 foreign GPU client와 audit error는 0이다. seed 20261108, alpha=1, beta=0, 반복 32회, 시간은 12회 묶음 5개의 GPU event 중앙값이다.

| label | on / off 출력 종류 수 (각 32회) | on 반복 최대 절대차 | on / off 중앙값 ms | original 솔루션의 trace |
|---|---:|---:|---:|---|
| qwen16_original_p0 | 32 / 1 | 0.000030517578125 | 0.109094 / 0.231899 | MT64×64×16 FP32 GSU4 일치 |
| qwen16_original_p1 | 32 / 1 | 0.000030517578125 | 0.109538 / 0.232946 | 같은 커널 일치 |
| qwen16_default_p0 | 1 / 1 | 0 | 0.235177 / 0.236056 | on/off 모두 MT64×32×8 GSU1 관측 |
| qwen0_original_p0 | 32 / 1 | 0.000030517578125 | 0.386093 / 0.524569 | MT128×64×16 FP32 GSU4 일치 |
| qwen0_default_p0 | 1 / 1 | 0 | 0.520846 / 0.519703 | on/off 모두 MT128×128×16 GSU1 관측 |
| qwen19_original_p0 | 1 / 1 | 0 | 0.045876 / 0.062003 | Half GSU4와 별도 PostGSU4 합산 관측 |
| anima2_original_p0 | 32 / 1 | 0.0000152587890625 | 0.023978 / 0.030876 | MT32×32×8 FP32 GSU4 일치 |
| anima2_original_p1 | 32 / 1 | 0.0000152587890625 | 0.023152 / 0.031080 | 같은 커널 일치 |
| anima2_default_p0 | 1 / 1 | 0 | 0.030859 / 0.031272 | on/off 모두 MT16×16×8 GSU1 관측 |
| anima5_original_p0 | 32 / 1 | 0.000030517578125 | 0.044811 / 0.057250 | MT64×32×8 FP32 GSU4 일치 |
| anima7_original_p0 | 32 / 1 | 0.000030517578125 | 0.052852 / 0.141710 | 같은 커널 일치 |
| anima13_original_p0 | 32 / 1 | 0.0000152587890625 | 0.079083 / 0.153517 | MT128×64×16 FP32 GSU4 일치 |
| anima15_original_p0 | 32 / 1 | 0.000030517578125 | 0.105170 / 0.226952 | 같은 커널 일치 |

Qwen16은 float TN 64×2400×4096, Qwen0은 float TN 4096×164×4096, Anima2는 float TN 1024×32×1024다. Anima5/7은 1024×32×2048/3072, Anima13/15는 1024×156×2048/3072다. Qwen19는 Half NT 640×640×1152다. 정확한 leading dimension과 solution ID, 원본 CSV SHA는 각 result.json과 통합 분석에 있다. 모델의 INT8/FP16 명칭을 개별 GEMM dtype로 쓰지 않는다.

**확인된 사실:** Qwen16과 Anima2의 p0/p1 쌍은 op·signature·seed·alpha/beta·stride·실제 tensor geometry·초기 A/B/C storage SHA·runtime·선택 솔루션이 모두 같다. 각 쌍의 on 출력은 두 프로세스 합계 **64개 모두 다르고**, off는 합계 64회에서 **한 SHA로 같았다**. 파일에 없는 shape가 새로 선택된 경우가 아니라, 이미 지정한 같은 솔루션 안에서도 생기는 비결정이다.

세 Default 대조는 on/off 바이트가 같고 오차가 모두 0이다. 해당 shape에서 원본 솔루션을 Default로 치환하면 관측한 비결정은 사라진다. 그 대가로 Qwen16은 약 0.109→0.235 ms, Qwen0은 0.386→0.521 ms, Anima2는 0.024→0.031 ms로 느려진다. 이것은 **개별 합성 GEMM의 비용**이며 모델 속도 손실이나 1.79배 유지 판정이 아니다.

**오차의 구분:** 표의 최대 절대차는 같은 on 솔루션의 반복끼리 비교한 값이다. on과 off의 서로 다른 알고리즘을 비교하면 Qwen16 p0 최대차는 0.000885009765625, NRMSE는 0.00000125856921096이다. 이를 on 반복 비결정 크기로 쓰면 안 된다. Qwen19는 on 반복끼리와 off 반복끼리는 바이트가 같지만 on/off 사이는 최대차 0.0625, NRMSE 0.0000159153935764이다. **알고리즘 사이의 반올림 차이와 반복 비결정은 별개**다.

GSU4라는 이름만으로 필터링해서는 안 된다는 대조도 얻었다. Qwen19의 workspace+PostGSU4 경로는 32회 모두 같은 출력이었다. §1.2의 assembly와 결합하면 FP32 atomic 경로와 Half workspace 경로를 구분할 근거가 있다. 모든 입력·모든 GSU4에 대한 결정성 보장은 아니다.

trace 제한: qwen16_original_p0의 off trace에는 kernel event가 없어 그 trace의 커널 선택은 pending이다. 같은 결과의 off SHA·시간·audit는 유효하고, 별도 p1 및 Default 실행에서는 GSU1 이름을 실제 관측했다. 빈 trace를 실패한 측정이나 커널 불일치로 바꾸지 않았다. 모든 original on trace는 예상 이름과 일치한다. Default는 정적 CSV→특정 이름 매핑이 없어 자동 selected_trace_status가 pending이며, 표의 GSU1 이름은 trace 직접 관측이다.

통합 분석 재생성은 GPU를 쓰지 않는다.

~~~bash
python3 -B - <<'PY'
import subprocess,sys
labels=['qwen16_original_p0','qwen16_original_p1','qwen16_default_p0',
        'qwen0_original_p0','qwen0_default_p0','qwen19_original_p0',
        'anima2_original_p0','anima2_original_p1','anima2_default_p0',
        'anima5_original_p0','anima7_original_p0','anima13_original_p0','anima15_original_p0']
cmd=[sys.executable,'-B','v10_tunable_analyze.py']
for label in labels:
    cmd += ['--probe',f'{label}=results/v10/tunable_{label}']
cmd += ['--model','original:anima_diagnostic=results/v10/tunable_anima_conditioning_diagnostic',
        '--out','results/v10/tunable_analysis/current']
subprocess.run(cmd,check=True)
PY
~~~

## 11. 실제 Anima conditioning 진단 — 완료

2026-11-08 tunable_anima_conditioning_diagnostic의 base,cand,cand 전체 실행이 완료됐다. protocol.json의 completed/valid_for_timing=true, source CSV·runner·workflow 보호 SHA 불변, finite·capture·배타 GPU 관문을 확인했다. profiler·verbose가 켜진 진단이므로 valid_for_speed_comparison=false다. 분석은 tunable_analysis/current/analysis.json에 함께 저장했다.

| 구간 | 전체 kernel event | exact CSV hit | FP32 후보의 실제 hit | 매핑 이름·호출 수 |
|---|---:|---:|---|---|
| CLIP 0, token 길이 32 | 1,573 | 8행, 합계 253회, miss 0 | entry2 56회, entry5 28회, entry7 28회 | 모든 선택 이름과 이름별 호출 합계 일치 |
| CLIP 1, token 길이 156 | 1,516 | 8행, 합계 253회, miss 0 | entry13 28회, entry15 28회 | 모든 선택 이름과 이름별 호출 합계 일치 |

두 구간에서 Tunable enabled=true이며 출력 tensor는 finite다. entry2에서 합성 입력 비결정을 보인 정확한 GSU4 커널이 실제 text-encoder에서 실행됐다. 5/7은 같은 −1086325257 솔루션, 13/15는 Qwen0과 같은 −1086325182 솔루션을 쓴다. 이제 다섯 shape 모두 §10에서 고정 입력 비결정을 재현했다. 각 행의 개별 이미지 차이 기여는 아직 분리하지 않았다.

실행 로그의 profiler thread/duplicate flow 경고 때문에 runtime flow의 일대일 연결과 trace 시간을 인과 근거로 사용하지 않는다. 저장된 JSON의 kernel 이름·개수는 독립된 C++ Lookup 로그의 구간별 합계와 전부 맞았다. 이 보고서가 사용하는 범위는 **같은 완료된 CLIP 구간에서 exact hit와 선택 이름·호출 수가 일치했다**는 사실이다. CPU 교차 원자료와 SHA는 tunable_analysis/anima_conditioning_stage_observation.json에 있다.

**조건 재사용 확인:** CLIP 호출은 총 2회이고 sampler는 3회였다. 세 sampler 모두 clip_calls_seen=2, seed=20261107, Tunable enabled=true였다. positive conditioning·negative conditioning·latent input SHA는 세 요청에서 각각 하나로 같다. 두 cand 요청의 출력 latent SHA와 raw RGB SHA도 각각 하나로 같았다. base↔cand의 출력 차이는 manual_vt 알고리즘 변경까지 포함하므로 프로세스 재현성 비교에 섞지 않는다.

**CSV 누락 가설의 범위:** 각 sampler 구간에서 exact CSV hit는 35행(4 float, 31 Half), 합계 17,154회이며 miss는 모두 0이다. FP32 필터 5행의 sampler hit는 0이고 모두 CLIP 구간에 있었다. 따라서 이 진단의 CLIP·sampler 호출은 기존 CSV로 덮였고, 해당 구간에서 파일에 없는 shape가 Default로 선택됐다는 설명은 필요하지 않다. 모든 prompt·다른 shape·미캡처 VAE 구간까지 일반화하지 않는다. 구간별 원자료는 conditioning_capture.json, CPU 요약은 tunable_analysis/anima_sampler_observation.json이다.

### 11.1 새 original 모델 프로세스 완료

tunable_anima_original_p0는 base,cand,cand,cand 4요청을 완료했고 completed/valid_for_timing/valid_for_speed_comparison=true다. 입력 CSV·runner·workflow SHA와 GPU audit도 통과했다. cand의 첫 실행을 제외한 두 warm denoise는 45.328187초와 45.134030초, 중앙값은 **45.231108초**다. off 대비 속도는 같은 세션의 기준이 완료된 §11.3에서 계산한다.

diagnostic와 새 original_p0의 model·runner·workflow·runtime·CSV·seed·steps, 두 token SHA는 전부 같다. 두 CLIP output SHA는 각각 달랐고 positive/negative conditioning도 달랐다. 같은 mode끼리 비교하면 입력 latent SHA는 같지만 출력 latent·RGB SHA는 다르다. 각 프로세스 안의 cand 반복은 각각 2회/3회에서 동일했고 CLIP 호출 수는 각각 2회였다. 이는 조건을 한 번 만들어 반복 재사용하는 실제 경로와 일치한다.

이 교차 비교는 한쪽이 진단 실행이고 다른 쪽은 비진단 실행이라는 설정 차이를 포함한다. 프로세스 이외의 모든 조건을 통제한 모델 A/A라고 부르지 않는다. 개별 GEMM의 동일 설정 p0/p1 비결정은 §10에서 별도로 확인했다. 원자료와 비교 항목은 tunable_analysis/anima_original_process_comparison.json에 있다. clean GPU 측정 중에는 작은 JSON의 SHA만 비교했고, 측정 종료 뒤 §15에서 실제 PNG와 RGB RMSE까지 확인했다.

### 11.2 filtered p0/p1 — 새 프로세스 재현성 완료

tunable_anima_filtered_p0와 p1의 base,cand,cand,cand 각 4요청 모두 completed/valid_for_timing/valid_for_speed_comparison=true다. 각각 CSV·runner·workflow SHA 불변, GpuAudit, 캡처의 finite 20개 snapshot을 확인했다. FP32 후보 5행만 Default로 바꾼 연구 CSV이며 원본 CSV는 그대로다.

| 비진단 실행 | cand 시간 3개 s | 첫 cand 제외 warm 중앙값 s | 같은 프로세스 cand 결과 |
|---|---|---:|---|
| original p0 | 45.039394, 45.328187, 45.134030 | 45.231108 | conditioning·latent·RGB 각각 SHA 1개 |
| filtered p0 | 45.394185, 45.076212, 45.099231 | 45.087722 | conditioning·latent·RGB 각각 SHA 1개 |
| filtered p1 | 45.108219, 45.116619, 45.114011 | 45.115315 | conditioning·latent·RGB 각각 SHA 1개 |

filtered 두 프로세스에서 첫 cand를 각각 제외한 warm 표본 4개의 중앙값은 **45.106621초**다. original/filtered는 1.00276으로 약 0.28% 차이를 성능 향상이라고 주장하지 않는다. warm denoise는 같은 수준이다. 같은 seed·token과 모델/runner/workflow/runtime 조건에서 필터가 만든 conditioning은 original과 다르다. 필터를 바꾸기 전 출력을 바이트 그대로 보존한다는 의미가 아니다.

완료한 filtered p0/p1은 model·runner·workflow·runtime·CSV SHA·seed·steps·modes가 모두 같았다. token SHA와 positive/negative conditioning SHA도 같다. base 두 결과는 같은 mode끼리, cand 총 여섯 결과도 같은 mode끼리 입력 latent·출력 latent·RGB SHA가 각각 하나로 같다. 두 프로세스의 CLIP은 각 2회이며 sampler 네 번 모두 그 조건을 재사용했다. 따라서 이 workflow에서 필터가 새 프로세스 사이의 조건·이미지 재현성을 회복했다는 판정이다.

실제 소스 CSV SHA는 original `4e6526d45dca0fbf591ff3a802350fe08a90e46123a584f3581cd09088d9edcf`, filter `8a0fdd106358b7eeebbfaa025774af235daa91d92965b2d9f206e5041daa2573`으로 기존 기록과 같다. CPU 원자료는 tunable_analysis/anima_filter_p0_observation.json과 anima_filter_replay_comparison.json이다. GPU 측정 중 JSON으로 비교했고, 이후 §15에서 PNG의 실제 내용과 RGB 오차도 확인했다.

### 11.3 off 기준 완료 — 1.79배 유지 확인

tunable_anima_off_p0의 base 3회는 81.131251, 80.666068, 80.765695초였다. 첫 요청을 제외한 warm 중앙값은 **80.715881초**다. 세 sampler 모두 tunable_enabled=false, 결과 RGB SHA는 1개였으며 완료·배타 GPU·source SHA·finite 관문을 통과했다. model·runner·workflow·runtime·seed·steps는 on 실행과 같다.

| 경로 | warm 표본 수 | warm denoise 중앙값 s | 같은 세션 off native 대비 |
|---|---:|---:|---:|
| Tunable off, native attention base | 2 | 80.715881 | 1.00000배 |
| original Tunable, manual_vt cand | 2 | 45.231108 | 1.78452배 |
| filtered Tunable, manual_vt cand (p0+p1) | 4 | 45.106621 | **1.78945배** |

이 배속은 Tunable과 manual_vt를 함께 쓴 기존 경로의 배속이다. 5행 필터 자체가1.79배를 만든 것으로 해석하지 않는다. 필터를 넣어 새 프로세스 재현성을 회복하면서 기존 경로의 속도 이득을 유지했다는 결과다. 바꾼 5행이 이번 workflow의 CLIP에서만 실행되고 denoise hit가0이었다는 관측과도 일치한다.

최종 Anima 판정은 **확인한 조건에서 원인 경로를 식별했고, 연구 필터의 재현성과 속도를 검증했다**다. 다른 prompt/shape·dtype나 모델까지 자동 일반화하지 않으며, 이미지의 지각 품질이 더 좋아졌다는 주장은 하지 않는다. 원본 두 모델 프로세스 비교의 진단 설정 차이는 §11.1에 그대로 남긴다. source/protocol 원자료는 각 실행 폴더, CPU 숫자는 tunable_analysis/anima_speed_observation.json에 있다.

## 12. Claude가 GPU 없이 읽을 현재 핵심 숫자 5개

1. Qwen16 original p0 같은 프로세스의 출력 종류 수: **32 / 32회**.
2. Qwen16 original p0+p1의 on 출력 종류 수: **64 / 64회** (입력·솔루션 동일), off는 1개.
3. Anima2 original p0+p1의 on 출력 종류 수: **64 / 64회** (입력·솔루션 동일), off는 1개.
4. Qwen16 Default on/off 최대 절대차: **0**, 32회 반복도 각각 1개 SHA.
5. Qwen19 Half GSU4 on 출력 종류 수: **1 / 32회**.

~~~bash
python3 -B - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('results/v10/tunable_analysis/current/analysis.json').read_text())
p={x['label']:x for x in r['probes']}
def pair(a,b):
    return next(x for x in r['probe_comparisons'] if x['a']==a and x['b']==b)
print('Qwen16 within',p['qwen16_original_p0']['modes']['on']['distinct'])
print('Qwen16 cross',pair('qwen16_original_p0','qwen16_original_p1')['outputs'])
print('Anima2 cross',pair('anima2_original_p0','anima2_original_p1')['outputs'])
print('Qwen16 Default max_abs',p['qwen16_default_p0']['on_vs_off']['max_abs'])
print('Qwen19 Half GSU4 distinct',p['qwen19_original_p0']['modes']['on']['distinct'])
PY
~~~

## 13. Qwen 실제 모델 진단과25-step 필터 비교 — 완료

tunable_qwen_diagnostic_p0는 처음부터 tbase,tbase, steps=1, seed=20261106으로 실행했다. 두 CLIP과 두 sampler 모두 Tunable enabled=true다. completed/valid_for_timing=true, GpuAudit·source CSV/runner/workflow SHA·finite·private user/temp 관문을 통과했으며 verbose/rocBLAS 로깅 진단이므로 valid_for_speed_comparison=false다. 이 결과를25-step 품질·속도 결과로 사용하지 않는다.

| 구간 | exact CSV hit | missing signature | entry0 / entry16 실제 hit |
|---|---|---|---|
| CLIP 0 | 0 | 16행, 496호출 | 0 / 0 |
| CLIP 1 | 0 | 16행, 496호출 | 0 / 0 |
| sampler 0, 1-step | 17행, 457호출 | 0 | 2 / 1 |
| sampler 1, 1-step | 17행, 457호출 | 0 | 2 / 1 |

CLIP에는 CSV에 없는 shape가 있었다. 이 진단은 TUNING=0이므로 그 부분은 Default 경로이며, Qwen 전체 모델이 CSV로 완전히 덮였다고 보고하지 않는다. 반면 캡처한 두 sampler는 miss0이고, 두 요청의 positive/negative conditioning·입력 latent SHA는 각각 동일했다. clip_calls_seen도2에서늘지않았다. 출력 latent SHA와 raw RGB SHA는 각각2개로달랐다. 따라서 이번 **같은 프로세스의 반복 차이**를 CLIP을 다시 실행했기 때문이라고 설명할 수 없다.

rocblas_profile.log의 실제 호출도 두 목표 행과 일치한다.

| 행 | 실제 f32 geometry / ld | alpha / beta | 실제 solution_index | 전체 호출 수 |
|---|---|---|---:|---:|
| Qwen entry0 | TN, M4096 N164 K4096 / 4096,4096,4096 | 1 / 0 | −1086325182 | 4 |
| Qwen entry16 | TN, M64 N2400 K4096 / 4096,4096,64 | 1 / 0 | −1086325192 | 2 |

이 geometry·transpose·leading dimension·alpha/beta·선택 번호는 §10에서 고정 입력 비결정을 재현한 프로브와 같다. atomics_mode=atomics_allowed, algo=1로 기록됐다. 이번 모델 진단은 선택 GPU kernel trace를 만들지 않았으므로, 전체 모델에서 커널 이름을 직접 관측했다고 바꾸지 않는다. **모델의 exact CSV hit와 실제 rocBLAS 선택 번호**, 그리고 별도 고정 입력 프로브의 **그 번호에 대응하는 실제 kernel trace·출력 비결정**을 연결한 증거다.

원자료: tunable_qwen_diagnostic_p0/protocol.json, conditioning_capture.json, rocblas_profile.log. CPU 요약과 세 원자료 SHA는 tunable_analysis/qwen_diagnostic_observation.json에 있다. 좁게 다시 읽는 명령은 GPU를 쓰지 않는다.

~~~bash
rg -n -- 'solution_index: -10863251(82|92)' results/v10/tunable_qwen_diagnostic_p0/rocblas_profile.log
~~~

### 13.1 원본25-step 실행 — 비결정 재현 완료

tunable_qwen_original_p0의 base,tbase,tbase,tbase,base 실행을 완료했다. completed/valid_for_speed_comparison=true, GpuAudit·source SHA 불변·finite를 확인했다. 캡처의 미해시 객체는0개다.

첫 base에서 Tunable off로 CLIP 두 번을 실행한 뒤 모든5요청이 같은 positive conditioning·negative conditioning·입력 latent SHA를 재사용했다. sampler의 enabled 순서는 false,true,true,true,false다. **tbase 세 요청의 출력 latent·RGB는 각각3개로 전부 달랐고, 앞/뒤 base 두 요청은 각각 하나로 같았다.** 따라서 full25-step에서도 같은 입력의 Tunable on 반복 비결정을 재현했다.

| mode | 요청별 denoise s | 첫 mode 제외 warm 중앙값 s | 출력 latent/RGB 종류 수 |
|---|---|---:|---:|
| original base (off) | 75.262523, 57.003924 | 57.003924 (표본1개) | 1 / 1 |
| original tbase (on) | 55.812901, 55.553719, 55.877454 | **55.715586** (표본2개) | 3 / 3 |

이 원본 실행만으로는 마지막 off 표본이 하나이므로 최종 off 배속을 내지 않았다. §13.2에서 필터가 완료된 뒤 미리 정한 동일 조건 검사를 통과한 두 마지막 base를 합친다. 원자료는 tunable_qwen_original_p0/protocol.json·conditioning_capture.json, CPU 요약은 tunable_analysis/qwen_original_observation.json이다.

### 13.2 두 행 Default 필터 — 동일 프로세스 재현성 회복

tunable_qwen_filtered_p0의 같은25-step·seed·modes 실행도 완료했다. 원본과 필터 모두 model·runner·workflow·runtime·seed·steps·modes, token SHA, positive/negative conditioning SHA가 같았고 source SHA·GpuAudit·finite를 통과했다. 미해시 객체는 둘 다0개다. 바꾼 연구 CSV의 SHA는 `587cffd10e920969e1a23d3b97a4918c50ddce6508781b75e7293ebb25e256ba`이며 실행 뒤에도 같았다.

필터의 tbase3회는 같은 conditioning·입력 latent 아래 **출력 latent SHA1개, RGB SHA1개**로 같아졌다. 원본 tbase는 각각3개였다. 두 실행에 있는 off base 총4결과도 하나의 RGB SHA였다. 필터 on과 off의 이미지는 서로 다르다. 따라서 판정은 **동일 입력 반복의 재현성을 회복했다**는 것이며, 서로 다른 알고리즘의 on/off 출력을 바이트 그대로 같게 만들었다는 뜻이 아니다.

| 경로 | 요청별 denoise s | warm 표본 수 | warm 중앙값 s |
|---|---|---:|---:|
| original tbase | 55.812901, 55.553719, 55.877454 | 2 | 55.715586 |
| filtered tbase | 56.444425, 56.417654, 56.479648 | 2 | **56.448651** |
| 두 실행의 마지막 off base | 57.003924, 57.569621 | 2 | **57.286772** |

주 비교에서 필터의 관측 warm 시간은 원본보다 **0.733065초, 1.32%** 길다. 각 arm warm 표본2개이므로 이를 정밀한 고정 비용이나 통계적으로 확정한 손실이라고 과장하지 않는다. 보조 off 기준은 두 마지막 base가 모두 tunable=false이고, conditioning·입력 latent SHA·workflow·seed·runtime이 같다는 검사를 통과한 뒤 합쳤다. 이 기준으로 original은1.02820배, filtered는1.01485배다. 작은 배속 차이를 큰 실사용 이득으로 포장하지 않는다.

**확인된 원인 범위:** 고정 입력 최소 GEMM에서 같은 GSU4 솔루션의 비결정, 실제 sampler의 같은 signature·rocBLAS 번호, full25-step에서 같은 조건의 출력 차이, 두 행만 Default로 바꾼 뒤 반복 일치라는 연결을 확보했다. CLIP은 CSV miss가 있었지만 이25-step 비교에서는 첫 off 요청에서 만든 같은 조건을 재사용했고, sampler의 진단 구간은 miss0이었다. Qwen의 모든 입력이나 필터의 새 프로세스 결정성까지 증명한 것은 아니다.

CPU 원자료는 tunable_analysis/qwen_final_observation.json이며 source CSV 전후 SHA, 각 mode의 실제 enabled 값·입력/출력 SHA·시간을 담는다. PNG 내용 재해시와 RGB RMSE는 §15에서 완료했다.

## 14. 현재 세션 이어받기

- CoS background **session27965는exit0으로 완료**됐다. batch 상태는 results/v10/tunable_qwen_model_batch/protocol.json, 개별 로그는 같은 폴더의 label.log를 읽는다.
- diagnostic_p0·original_p0·filtered_p0 모두 완료했다. 두25-step 실행은 modes=base,tbase,tbase,tbase,base, seed=20261106이다. 필터는 FP32 entry0·16만 Default로 바꾼 연구 CSV다.
- **완료한13개 GEMM 프로브, Anima의 모든 모델 실행, Qwen의 모든 모델 실행을 다시 실행하지 않는다.** 기존 result/protocol과 저장된 SHA를 사용한다.
- clean GPU 속도 측정이 끝난 뒤 주 세션이 GPU가 비었다고 확인했고, CPU에서 PNG30개 재검증·RGB 비교11쌍·최종 helper 분석을 완료했다. final/analysis.json의13개 probe와8개 모델 실행은 모두 verified다. Anima 진단 실행의 speed comparison이 pending인 것은 진단을 속도 판정에서 제외한 정상 상태이며, 미완료 모델 실행을 뜻하지 않는다.
- **통합 종료 확인 완료:** [protected_after.json](protected_after.json)은1022개 파일·3,241,275,343bytes, changed=[]·missing=[]·errors={}·unchanged=true다. [gpu_idle_after.json](gpu_idle_after.json)은 KFD client0·foreign/error0·GpuAudit valid·gpu_idle=true다. 두 정본을 읽어 확인했으며 after 검사와 완료 GPU 실험을 다시 실행하지 않는다. C의 실행·분석·보고 마무리는 완료됐다. 명시한 범위 밖의 Qwen 필터 새 프로세스·MOSS·다른 prompt 검증은 미실행 상태를 유지한다.

## 15. PNG 내용 확인과 RGB RMSE — 완료

GPU 모델·속도 측정 종료 확인 뒤 CPU에서 생성 PNG30개를 직접 읽었다. protocol에 기록된 파일 SHA·raw RGB SHA·크기가 모두 맞았고 비교한11쌍의 입력 파일도 모두 통과했다. v10_tunable_rgb_compare.py는 Pillow만 사용했고 torch import·GPU 실행·이미지 변경은 없었다. 최종 종합 JSON은 tunable_analysis/final/analysis.json, PNG 검증과 모든 pixel 수치는 tunable_analysis/rgb_final.json이다.

| 비교 | RGB RMSE, 0–255 단위 | 최대 절대차 | 해석 범위 |
|---|---:|---:|---|
| Anima original, diagnostic↔clean cand | **6.162048** | 216 | 같은 token·seed의 새 프로세스, 진단 설정 차이 포함 |
| Anima filtered p0↔p1 cand | **0** | 0 | 같은 비진단 설정의 새 프로세스 |
| Anima filtered 같은 프로세스 반복 | **0** | 0 | 같은 cand mode |
| Anima off 반복 | **0** | 0 | 같은 off base mode |
| Qwen original25-step, tbase1↔2 | **6.527086** | 241 | 같은 프로세스·조건·입력 |
| Qwen original25-step, tbase1↔3 | **5.541087** | 241 | 같은 프로세스·조건·입력 |
| Qwen original25-step, tbase2↔3 | **4.717203** | 198 | 같은 프로세스·조건·입력 |
| Qwen filtered25-step, tbase1↔2와1↔3 | **0** | 0 | 같은 프로세스; 새 프로세스 검증은 미실행 |
| Qwen 두 실행의 마지막 off base | **0** | 0 | 서로 다른 프로세스, 두 요청 모두off·조건 동일 |
| Qwen1-step 진단 반복 | 1.795528 | 57 | 1-step 진단이며25-step 품질 수치가 아님 |

Anima의 관측6.162/255는 이전4–7/255 기록 범위에, Qwen25-step의4.717–6.527/255도 이전3–7/255 범위에 들어간다. 이전 범위를 모든 실행에 적용하는 보장으로 바꾸지 않는다. Anima 원본 쌍의 진단 설정 차이는 이 pixel 비교에도 그대로 적용된다. 필터 뒤의0은 실제 RGB 내용이 같은 것을 뜻하며 지각 품질이 좋아졌다는 판단이 아니다. 실제 PNG 크기는 Anima1024×1024, Qwen960×640이었다.

CPU 재현 명령은 기존 이미지를 읽기만 한다. 새 출력 이름을 쓰므로 완료 원자료를 덮어쓰지 않는다.

~~~bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
~/DevSpace/moss-image-rdna2-lab01/.venv-moss/bin/python -B v10_tunable_rgb_compare.py \
  --recipe results/v10/tunable_analysis/rgb_recipe.json \
  --out results/v10/tunable_analysis/rgb_reproduction_v1.json
~~~

## 16. Claude가 GPU 없이 읽을 최종 핵심 숫자5개

1. Qwen 원본25-step 같은 프로세스의 세 쌍 중 최대 RGB RMSE: **6.527085785 / 255**.
2. Qwen 필터25-step 같은 프로세스의 반복 RGB RMSE: **0**. 새 프로세스 검증은 미실행이다.
3. Anima 필터의 두 새 프로세스 사이 RGB RMSE: **0**.
4. Anima filtered Tunable+manual_vt의 off native 대비 배속: **1.789446419배**. off warm2개, filter warm4개다.
5. Qwen filtered tbase의 off 대비 보조 배속: **1.014847496배**. on warm2개, 조건을 확인해 합친 두 마지막 off2개 기준이다. 주 비교 original→filter의 warm 시간은55.715586→56.448651초다.

~~~bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
python3 -B - <<'PY'
import json
from pathlib import Path
p=Path('results/v10/tunable_analysis')
rgb=json.loads((p/'rgb_final.json').read_text())
anima=json.loads((p/'anima_speed_observation.json').read_text())
qwen=json.loads((p/'qwen_final_observation.json').read_text())
c={r['label']:r for r in rgb['comparisons']}
print('Qwen original max repeat RMSE /255',max(v['rmse_8bit'] for k,v in c.items() if k.startswith('qwen_original_')))
print('Qwen filtered repeat RMSE',max(v['rmse_8bit'] for k,v in c.items() if k.startswith('qwen_filtered_')))
print('Anima filtered cross-process RMSE',c['anima_filtered_cross_process']['rmse_8bit'])
print('Anima speedup (off n2 / filter n4)',anima['filtered_speedup'])
print('Qwen speedup (pooled off n2 / filter n2)',qwen['off_div_filtered'])
PY
~~~


