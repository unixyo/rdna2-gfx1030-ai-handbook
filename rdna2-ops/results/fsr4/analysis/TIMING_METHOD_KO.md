# FSR4 timing kit의 시간·셰이더 대응·출력 비교 해석

작성: 2026-11-09. 대상은 `exact_profile` 진단 실행이다. 이 문서는 GPU를 다시 실행하지 않고 원자료와 동결된 측정 소스를 분석했다. 아래 경로는 `rdna2-ops/results/fsr4/` 기준이다.

갱신: 실제 pass 비용 표에는 `WARM_PROFILE_TIMING_KO.md`와 `warm_profiles_summary.json`의 warm40/timed200 자료를 사용한다. 이 문서의 초기 warm4/timed8 숫자는 timestamp 정의와 초기 진단의 한계를 기록하기 위해 보존했다. 일반 전체 시간의 7회 교대 baseline은 `BASELINE_ANALYSIS_KO.md` 및 `baseline_analysis.json`에 있다. 첫 candidate 정식 교대는 관측된 warm40 ramp 때문에 실행 전에 warm400/timed200으로 고정했고, `clamp_all_TRIAL_DECLARATION.json`에 기록했다. candidate는 이 새 reference와만 비교하며 이전 baseline40을 섞지 않는다.

## 이 프로파일에서 확인된 범위

`measurements/exact_profile.txt`는 warmup 4프레임, timed 8프레임, Native AA, moving scene, pass 계측을 기록한다. 실제 provider query 결과는 `4.1.2-exact-r2-cyboman`, version ID `f5a5ca1e01001001`이고 자동 선택이었다. 각 해상도에서 28개 dispatch, 각 dispatch에서 8개 표본, 총 448개 pass 표본이 있다. 두 해상도는 1920×1080 및 3840×2160이며 입력과 출력 크기가 같다. 원자료는 다음 파일이다.

- `measurements/exact_profile.txt`: provider, CASE, PSO, PIPE, jitter 부호별 T 집계.
- `measurements/exact_profile_frames.csv`: 시간 순서가 보존된 전체 dispatch 구간 GPU 시간, 16행.
- `measurements/exact_profile_passes.csv`: 개별 dispatch 시간·다음 구간 시간, 448행.
- `pipelines/exact_profile/*.dxbc`: 실제 CreateComputePipelineState에 전달된 shader container.
- `dumps/exact_profile/*.dxil`: vkd3d가 덤프한 shader container.
- `runs/exact_profile/stderr.log`: RADV/ACO 컴파일 출력.

이 실행은 셰이더 식별과 비용 분해를 위한 짧은 진단이다. 성능 채택에 사용할 warmup과 반복 조건을 충족한 교대 측정 결과가 아니다. 특히 4K의 처음과 마지막 전체 시간은 3.52720 → 3.21236 ms로 약 8.93% 감소했다. 이 표만으로 정상 상태의 1080p/4K 비례 관계나 작은 후보 이득을 판정하면 안 된다.

## GPU 시간의 정의

동결 소스 `harness/fsr4time_local.c`의 818–828행에서 전체 timestamp는 `ffx.Dispatch`가 기록하는 GPU 명령들의 바깥에 놓인다. 따라서 CSV의 `gpu_ms`는 이 GPU 구간의 시간이다. CPU의 FFX 호출 시간, 업로드, 제출 후 CPU fence 대기, 창 표시와 프레젠트는 포함하지 않는다. 매 프레임은 `Submit()`에서 fence를 기다리고 allocator/list를 reset한다(171–179행). 실제 게임처럼 여러 프레임을 겹쳐 제출하는 경로를 재현한 값은 아니다.

pass 계측은 Dispatch 함수 직전과 직후에 timestamp 두 개를 추가한다(623–629행). 각 pass의 `gpu_us`는 그 두 timestamp 사이 시간이다. `gap_us`는 해당 pass 종료에서 다음 pass 시작까지이며, 마지막 pass는 전체 종료 timestamp까지다(830–835행).

프레임마다 다음 항등식이 성립한다.

```text
전체 GPU 구간
 = 첫 dispatch 이전 구간
 + 모든 dispatch 구간의 합
 + 각 dispatch 다음 gap의 합
```

gap에는 resource barrier, 명령 처리·스케줄링·캐시/파이프라인 진행과 계측 영향 등이 섞일 수 있다. gap 전체를 barrier 한 종류의 비용이라고 확정하거나, shader 구간만 더한 값을 업스케일러 전체 시간으로 부르면 안 된다. timestamp가 추가되는 pass 모드 자체도 수행 흐름에 영향을 줄 수 있으므로 후보의 채택 여부는 `run_kind=totals`인 일반 실행으로 판단한다. 캡처 실행은 `run_kind=capture`이며 이 역시 별도 진단이다.

아래는 동일한 8프레임에서 계산한 산술평균이다. 단위는 µs다.

| 항목 | 1080p Native | 4K Native |
|---|---:|---:|
| 전체 GPU 구간 | 1162.240 | 3360.495 |
| dispatch 구간 합 | 1064.725 | 3272.210 |
| 뒤따르는 gap 합 | 95.965 | 87.100 |
| 첫 dispatch 이전 잔여 구간 | 1.550 | 1.185 |
| 전체 GPU 구간 중앙값 | 1161.940 | 3356.820 |

잔여 구간은 프레임별로 전체에서 두 합을 빼서 구했다. 1080p에서는 1.52–1.56 µs, 4K에서는 1.08–1.28 µs다. 부동소수점 표시 오차를 제외하면 전체 시간과 분해 합이 일치한다.

업스트림 `Median()`은 짝수 표본 수에서 정렬된 `v[n/2]`를 반환하는 upper median이다(428행). 이 문서의 중앙값은 가운데 두 값의 평균이다. 이후 분석에서도 `frames.csv`에서 `statistics.median`으로 일관되게 재계산해야 R행의 표시값과 정의가 섞이지 않는다. `T`행의 even/odd 또한 일반적인 프레임 번호의 짝수/홀수가 아니라 `jitterOffset.x >= 0` / `< 0` 구분이다(830, 842, 862행). 이번 Native 8상에서는 결과적으로 부호가 번갈지만 정의 자체는 부호다.

## dispatch와 shader의 대응

`analysis/map_profile_pso.py`가 CASE별 PIPE→PSO 관계를 읽고, 각 PSO 파일 전체의 SHA256을 vkd3d `.dxil` 파일 전체의 SHA256과 비교한다. 56개 실제 바인딩(28×2해상도) 모두 유일한 vkd3d dump에 바이트 단위로 대응한다. 결과는 `analysis/exact_profile_pso_map.json`이다. 이 매핑에는 DXBC SHA256, DXIL bitcode SHA256, entry point, 원래 PSO 번호, vkd3d hash, groups, pass 시간과 원시 entry metadata가 들어 있다.

CASE 밖이나 종료 정리에 등장한 NULL 파이프라인은 실행 shader가 아니다. PSO 번호는 생성 순서이고 PIPE 번호는 최초 바인딩 순서라서 서로 다르다. 이번 28개 dispatch의 PIPE 번호는 dispatch 번호와 같다. dispatch 0은 **PSO 27**, dispatch 1은 **PSO 0**이다.

첫 dispatch의 vkd3d hash는 두 해상도 모두 `996136f00380aba8`이다. entry point가 `main`인 이 shader에는 `spdCounter`, `spdIntermediateR/G/B/A`, luminance 계수 0.2126/0.7152/0.0722와 log2/exp2 축약 경로가 있다. 컨텍스트 생성도 AUTO_EXPOSURE를 켠다(788행). 이 증거로 첫 dispatch를 SPD 기반 노출 계산 단계로 식별한다. 네트워크 prepass는 다음 dispatch 1이다. 이름에 남아 있는 `fp8_no_scale`는 entry point의 역사적 이름이며, 실제 네트워크의 INT8 여부는 ISA의 dot4 명령과 함께 판단해야 한다.

다음은 식별된 주요 pass의 진단 평균이다. 모든 값은 µs이며, 각 네트워크 pass 뒤의 padding pass는 따로 합쳤다.

| dispatch | 실제 역할 / entry point 끝부분 | 1080p | 4K |
|---:|---|---:|---:|
| 0 | SPD/노출, `main` | 44.300 | 234.620 |
| 1 | `prepass` | 162.690 | 708.795 |
| 3 | `pass1` | 83.165 | 241.590 |
| 5 | `pass2` | 82.945 | 236.620 |
| 7 | `pass3` | 16.305 | 40.535 |
| 9 | `pass4` | 44.400 | 113.980 |
| 11 | `pass5` | 44.395 | 113.810 |
| 13 | `pass6` | 22.625 | 46.250 |
| 15 | `pass7` | 46.935 | 101.140 |
| 17 | `pass8` | 47.010 | 100.910 |
| 19 | `pass9` | 72.835 | 171.680 |
| 21 | `pass10` | 43.790 | 112.930 |
| 23 | `pass11` | 68.135 | 172.530 |
| 25 | `pass12` | 82.980 | 234.270 |
| 27 | `postpass` | 200.920 | 641.635 |
| 2,4,…,26 합 | `pass0_post`…`pass12_post` | 1.295 | 0.915 |

`pass0_post`의 DXIL에는 경계 좌표 계산 및 zero raw-buffer store가 실제로 있다. padding dispatch의 짧은 시간만 보고 무작업으로 분류할 수 없다. 이 13개 dispatch 뒤 gap 합은 1080p 82.375 µs, 4K 74.345 µs다. 이 값은 해당 gap을 다른 설계로 모두 없앨 수 있다는 뜻이 아니다. 제거·융합 후보라면 경계 값과 전체 출력의 동일성, 일반 전체 시간 감소를 따로 확인해야 한다.

DXIL entry properties에는 이 자료에서 shader flags와 NumThreads만 관측되었으며 명시적인 WaveSize 항목이 없다. NumThreads 32/64/256은 thread-group 크기다. 이를 실제 wave 크기로 대체하지 않는다. RADV 텍스트의 컴파일 블록 순서와 PSO 생성 순서 연결은 별도 ISA 분석의 순서 기반 추론이며, 위에서 검증한 dispatch→DXIL 해시 대응과 증거 수준을 구분해야 한다.

## 동일 출력 비교의 전제

이번 pass 프로파일은 프레임 출력 캡처를 하지 않았다(859–864행에서 집계 후 context를 종료). 따라서 이 프로파일 자체는 byte equality를 입증하지 않는다.

동일성 검증에서는 동일한 harness 바이너리, 해상도·render size, scene, warmup 및 timed 프레임 수, reset 위치, 입력과 jitter 순서, exposure/HDR/sharpening 설정을 사용해야 한다. 이번 설정은 HDR와 auto exposure 활성, sharpening 비활성, `preExposure=1`, `frameTimeDelta=10ms`, depth=0.5, 첫 프레임에서만 reset이다(762, 788, 805–816행). 최종 한 프레임만 비교하면 두 실행의 `WARM+TIMED-1`과 그 이전 history가 같아야 한다.

`HashOutput()`은 GPU 출력을 RGBA16F로 readback하고 행 padding을 제거한 `width×height×8` 바이트를 저장한다(393–418행). 1080p는 16,588,800바이트, 4K는 66,355,200바이트다. C행의 64비트 fingerprint는 빠른 식별에 사용할 수 있지만, exact 판정 근거는 같은 frame의 raw 파일에 대한 `cmp` 또는 전체 바이트 비교와 SHA256 기록이다. 8비트 이미지 미리보기는 양자화 때문에 exact 증거로 사용할 수 없다. 여러 시점의 일치는 시험한 입력·history 범위의 결과이며 모든 가능한 입력의 수학적 동일성 증명은 아니다.

시간에 따른 검증은 `FSR4TIME_CAPTURE`로 연속 8–16프레임을 저장하고, timing 모드와 분리한다. stationary, moving, brightness를 동일 조건으로 각각 비교한다. 캡처 사이에는 readback과 fence 대기가 추가되므로 해당 시간 분포를 성능 판정 자료에 섞지 않는다.

입력은 업스트림의 합성 fixture를 유지한다. World는 192픽셀 주기의 패턴이고(231–240행), Native AA의 8개 phase 텍스처를 미리 만든 뒤 `frame % phases`로 재사용한다(771–782, 805행). moving은 phase마다 출력 픽셀 기준 x=12, y=6만큼 이동하지만 phase 7→0에서 입력은 처음 이미지로 돌아간다. 따라서 연속적인 실제 카메라 이동을 재현한 것이 아니며, wrap 경계에서 motion vector와 이미지 변화가 맞지 않는 구간이 있다. 이 불연속도 fixture의 일부로 양쪽에 동일하게 적용한다. brightness는 정지 패턴에 phase 전반 0.35, 후반 1.35 배율을 적용하는 별도 진단이다. 합성 캡처만으로 게임의 잔상·떨림·얇은 선 품질 전체를 채택 판정하지 않는다.

## CPU만 사용하는 재현

아래는 기존 덤프를 다시 분석할 뿐 GPU를 실행하지 않는다.

```sh
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops/results/fsr4
python3 analysis/map_profile_pso.py exact_profile
```

스크립트는 `analysis/exact_profile_pso_map.json`을 다시 쓴다. 현재 측정의 한 provider·한 scene 프로파일을 전제로 크기별로 결합한다. 다른 profile 이름을 인자로 줄 수 있지만 여러 provider 또는 여러 scene을 한 파일에 혼합했으면 CASE 키를 확장한 뒤 사용해야 한다. LLVM은 기존 DXIL bitcode를 CPU에서 텍스트 IR로 읽는 용도로만 쓰이며 생성된 GPU shader는 실행하지 않는다.

일반 전체 시간, 짧은 pass 프로파일, 출력 검증을 별도 자료로 남기는 것이 이 하네스의 해석 경계다. 게임 프레임 시간이나 공식 SDK DLL의 Navi2 provider 경로는 이 진단으로 검증되지 않았다.

