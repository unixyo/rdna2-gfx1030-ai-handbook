# rdna2-ops v0.7 — gfx1030 operator catalog and reusable experiments

Start with the **[Korean operator catalog](../RDNA2_OP_CATALOG_KO.md)**, then the **[detailed handbook](RDNA2_OPTIMIZATION_GUIDE_KO.md)**.
It includes applicability, exact commands, accuracy gates, actual results,
rejected candidates, rollback, and the checklist for a new model. Existing
MOSS/ComfyUI deployment is one measured example, not the project boundary.

## v0.7 — INT4, Sparse Attention and IQ prefill, 2026-11-07

- [INT4 weight GEMM](results/int4_v07/INT4_GEMM_V07_KO.md): actual MOSS/Qwen/Anima
  weights and nine Linear shapes, group 32/64/128, symmetric/affine packing,
  signed dot8, unpack-to-dot4 and FP16 paths. All 324 same-storage compute
  gates pass. Splitting G128 into two BK64 integer pieces reduces Triton
  spill metadata and improves the old W4 kernels 1.80–2.03x, but still loses
  against the models' actual native baselines. MOSS W4 reference/custom
  latent and RGB bytes match; the original-to-W4 path fails strict RGB.
- [Sparse Attention](results/sparse_v07/SPARSE_ATTENTION_V07_KO.md): nine
  actual Qwen step/layer maps, five operator inputs and three full 25-step
  model seeds. An image-only r4 mask improves the early target operator
  22.666→18.600 ms and extra allocator peak 3102.594→38.000 MiB.
  A same-process warm model request improves 50.8745→46.3941 s, but the late
  compute gate and every strict RGB gate fail. Twelve blinded decisions,
  full images, fixed crops, reversed sides, raw prompts and responses are
  retained separately. All 12 prefer native and assign candidate severity 3;
  the perceptual gate also fails. These are fresh contexts of the same base
  vision model, not distinct model architectures.
- [Strata prefill](results/strata_v07/STRATA_PREFILL_V07_KO.md): the actual
  pack contains several IQ types, with IQ3_XXS GU in 6/48 layers.
  Three routing distributions and 54 operator rows are byte exact; the
  skew-tail average-J path improves GU/down 1.090/1.080x. First-request
  stage evidence is I/O-contaminated, and two priming runs hit the unchanged
  swap protection gate. Warm 9.7K/19.8K/119K model A/B and the warm
  compute-versus-PCIe bottleneck remain **unverified**.
- [Protection and repairs](RDNA2_OPTIMIZATION_GUIDE_KO.md): all 31 existing
  application files and 48 Strata operating files have unchanged before/after
  hashes. The operating service remains inactive/dead. Missing kernels.hip
  and two FP8 config byte/hash discrepancies are repaired without semantic
  changes. Triton remains 3.8, with no downloads or environment rebuild.
- [Five core reproduction commands](results/V07_REPRO_CORE_KO.md) distinguish
  old-candidate speedups from native replacement, same-storage arithmetic
  from representation loss, and first-use from warm timing.

The current package version is 0.7.0. The v0.7 manifest and SHA256SUMS
describe current selected bytes; the v0.6 manifest remains a historical
snapshot. Large captured tensors and private operating-derived Strata
configuration copies stay in their existing research-PC result directories.
No v0.7 candidate is enabled automatically in an operating model.

## v0.6 — operator cells, 2026-11-07

The entry point is organized by operation, dtype, shape, layout and backend.
MOSS, Qwen, Anima and Applio supply existing workloads for validation; they do
not define the scope of the catalog. Each cell uses a native kernel reference,
three or more operator shapes, a relevant final-model A/B when the operator
candidate passes, and linked raw results. Failed replacements remain recorded.

- [rocBLAS layout selection](results/LAYOUT_BLAS_V06_KO.md): 48 FP16/FP32
  mm/bmm combinations, prepared and conversion-inclusive timing, exact kernel
  names, memory and numerical evidence. NT helps one measured FP16 PV family
  and regresses the tested Qwen FP32 case. Large A-transpose costs reject TN
  as a per-call transformation.
- [FP16 Attention](results/FP16_ATTENTION_V06_KO.md): 33 operator comparisons
  and a 10-request Anima 30-step A/B. Prescale+VT does not improve the existing
  direct VT workflow. Preserving the original 4D matmul path passes final
  latent/RGB byte equality. Two additional warm requests take 67.775/67.727 s
  versus same-server native 80.607/80.720 s, a median 1.1906x. The original
  first-candidate 83.019 s remains recorded; warm gains do not promise a gain
  on a new server's first Triton request.
- The [Comfy research harness](research_comfy_v05.py) now borrows
  `AttentionTensorContainer.peek()` without consuming the input. Actual
  capture preserves Anima latent/RGB bytes. The configured capture target
  and candidate target can differ and are both recorded.
- [BF16 GEMM](BF16_GEMM_V06_KO.md): three shapes and six paths, actual native
  BBS kernels, conversion-inclusive timings and separately measured guards.
  FP32 GEMM at the real BF16 VAE SDPA boundary passes strict RGB but does not
  materially accelerate decode. Anima's request-local BF16 Linear replacement
  is faster in a two-step diagnostic and fails final RGB; it is not adopted.
- [FP8-storage GEMM](FP8_WEIGHT_GEMM_V06_KO.md): both E4M3fn/E5M2, three
  shapes each, exhaustive byte-pattern decoding, software decode, prepared
  FP16, fused dot2 and INT8. The E4M3fn model cache is byte exact but slower
  at 30 steps with 3.83 GiB extra explicit cache. A cache-hit staging fix also
  fails its predeclared speed screen. Projection-only INT8 fails final RGB.
  Raw BF16-to-FP8 representation loss is measured separately. E5M2 has no
  final-model A/B claim.
- [Convolution](results/conv_catalog_v06/CONV_CATALOG_V06_KO.md): BF16 Conv2D
  through FP32, with the original BF16 bias boundary, accelerates actual MOSS
  VAE decode 2.30–2.63x in regular/tiled and normal/low-VRAM paths, with strict
  final RGB acceptance. This is a decode result, not a whole-generation speedup.
  Filtering a reproduced wrong forward solver from a research FindDb cuts
  actual Applio steady epoch time 9.310→8.418 s. Training audio equivalence
  remains unresolved because native repetition already exceeds the provisional
  waveform gate. Five convolution families have three-shape evidence each.
  Qwen exact-tile Conv3D→Linear passes three operator shapes; its actual model
  comparison stops before the candidate because native conditioning is unstable.
  Adding nine validated ConvTranspose1D DB keys improves the three operators
  but not the actual training epoch (8.4176→8.4341 s), so it is not adopted.
- [LayerNorm/RMSNorm/adaLN](results/NORM_FUSIONS_V06_KO.md): 12 operator cases,
  27 native/candidate paths, with PyTorch and existing Comfy Kitchen HIP baselines.
  Replacing statistics fails strict final Anima RGB despite small operator error.
  Keeping native LayerNorm and fusing only modulation/cast preserves model bytes
  in both 30-step requests. Its 1.40–2.65x operator gain becomes only 1.0039x
  denoising gain in this workload (n=2), not a large model speedup.

The shared [GPU audit](gpu_guard_v06.py) combines a research-job lock with KFD
client polling. Observed external clients or audit errors invalidate timings.
It does not detect all graphics load or clients shorter than the poll interval.
Research adapters are context managers, restored on exit; operating configs,
installed runtimes and existing model files are not patched.
The final [operating-file audit](results/v06_operating_files_audit.json) verifies
15 tracked MOSS/Qwen/Anima source/config/workflow hashes unchanged. Actual
Applio epoch results separately verify 16 protected files. These are bounded
file checks, not a claim that all compiler/runtime cache files are unchanged.

## v0.5 — actual MOSS/Qwen follow-up, 2026-11-06

The six Qwen rehearsal documentation defects are incorporated into the handbook:
server-side profiling harness, separate operator/device denominators, actual
Attention callable discovery, backend-specific strided-layout regressions,
post-GEMM scale folding, and distinct INT8 implementations.

- [INT8 GEMM + dequant](rdna2_ops/int8_gemm.py): MOSS's four measured shapes
  are byte exact and 2.887–3.333× faster at the operator level. The actual
  denoising+VAE median is **4.897742 → 2.148737 s, 2.279×**, relative to the
  existing view+auto baseline, with identical latent/RGB bytes. The canonical
  summary excludes warmup and the first candidate run; baseline n=3,
  candidate n=2. [Raw paired requests](results/v05_moss_int8_tuned_ab/runs.json).
- The same new INT8 candidate is **slower than Qwen patientx on all six
  measured shapes**. That backend already fuses dequant; it remains the
  reference. No Qwen whole-model INT8 replacement was applied.
- [FP32 tiled Attention](rdna2_ops/fp32_attention.py): actual D128 inputs,
  spill-free final tile, no global quadratic score allocation. Qwen K4964
  uses 75.5 MiB additional allocated peak versus alpha's 3102.6 MiB, but
  is slower (54.597 versus 22.787 ms). MOSS also regresses. Kept as an explicit
  memory experiment, with no final-RGB acceptance claim.
- [Mixed precision at FP32 boundaries](rdna2_ops/mixed_attention.py): five
  real MOSS candidates fail the unchanged final RGB gate. The exact control
  passes; INT8 code flips persist even after three-product correction.
  FP16-input/FP32-output BMM also selects a slow HSS fallback in this build.
- [Sage-inspired rectangular Attention](rdna2_ops/sage_attention_v05.py):
  actual dot4/dot2, at best 1.129× faster than Qwen alpha, but full operator
  error and actual 8-step RGB both fail. RGB max=0.5908203, RMSE=0.0135610;
  thresholds remain 2/255 and 0.25/255. This is an independent prototype,
  not official SageAttention package support.
- [Settings-driven ComfyUI harness](research_comfy_v05.py) uses existing
  interpreters/runtime/options and process-local hooks. GPU activity tables
  use Chrome trace kernel/memcpy/memset categories and exclude GPU range
  annotations. [Actual warm/profile/baseline verification](results/v05_moss_profile_verified/profile_verification_summary.json)
  is byte exact. Earlier incorrect tables remain as `_pre_fix.json`, with
  canonical corrections and provenance.

No MOSS/Qwen operating default, installed package, model or environment was
changed for v0.5. The ten tracked operating source/config/workflow files have
identical before/after hashes; this is a bounded check, not a directory-wide
audit. [Operating-file audit](results/v05_operating_files_audit.json).
All v0.5 research servers exited after their owned jobs completed. New kernels
are explicit research APIs; the existing production adapter is unchanged.

The package keeps `rdna2-ops/` and the sibling `rehearsal_qwen_20261106/`
directory together so handbook links and rehearsal baseline imports resolve.
Large captured QKV `.pt` tensors and new raw Chrome traces remain in the
research PC's corresponding result folders; the package includes source,
configs, compact evidence and trace hashes. Recreate a capture using handbook
§2.3 before replaying a captured-input benchmark on another host.

## Historical v0.4 measurements

The preceding v0.4 measurements on the same existing environment include:

- Explicit score-free [tiled Attention](rdna2_ops/tiled_attention.py):
  FP16 S4097/8192/16384, D64/128, FP32 controls, latency/memory/error.
- Experimental [INT8 QK + FP16/FP32 PV](rdna2_ops/quantized_attention.py):
  generated gfx1030 dot4c/dot2c, quantization-inclusive measurements.
- [FP16 GEMM](rdna2_ops/gemm.py): numerically valid, slower than rocBLAS;
  retained as a rejected replacement candidate and reproducible comparison.
- [Actual precision diagnosis](results/precision_diagnosis_v04_fixed/analysis.json):
  FP32 arithmetic with FP16 QKV/output rounding reproduces the FP16 run's
  entire trajectory; the first activation INT8 boundary amplifies differences.
- New kernels are explicit forward-only experiments. Production auto dispatch
  and environment/package installation remain unchanged.

The historical v0.3 operational evidence follows below; it is not re-labeled
as an end-to-end result for the new approximate Attention kernels.

Research and production follow-up: **2026-11-05**, actual **Radeon RX 6900 XT / gfx1030**, existing Ubuntu environment.

The existing `custom_nodes/MOSS_RDNA2_INT8` now owns the source-level INT8 transpose-copy removal and selective exact FP32 attention. The existing `scripts/start_moss_comfy.sh` defaults to `MOSS_RDNA2_INT8_LAYOUT=view` and `MOSS_RDNA2_ATTENTION=auto`. No dependency reinstall, environment rebuild, model replacement, bytecode swap, or installed Kitchen/Comfy source edit is required.

On the actual launcher server, the original 512×512 four-step workflow measured **7.446393 → 4.762543 seconds** for denoising + VAE, **1.56353× / 36.04% less time**. Three warm alternating rounds used identical cached text conditioning and seed; sampler and decoder really executed. Every latent and decoded RGB tensor matched in **dtype, shape and all bytes**. INT8 alone measured 4.994781 seconds. Timing excludes startup, text encoding and PNG saving.

A second actual image geometry, **768×512 / attention length 1573**, measured **8.355858 → 7.849930 seconds** when adding attention auto to INT8 view, with full latent/RGB byte equality. The automatic attention policy now covers the measured layout family `[1,24,S,128]`, **128 ≤ S ≤ 4096**, rather than one sequence length. Twenty length/layout microbenchmarks include boundaries and odd lengths.

**[Production integration, precise scope, rollback, precision experiments and reproduction](production/README.md)** contains the operating instructions. The INT8 optimization is restricted to FP32 input/output; FP16/BF16 paths, including the external text encoder, call the original eager function. Reduced-precision attention failed the predeclared image-error thresholds on all three tested seeds and is not enabled. The accepted production path retains FP32 arithmetic. Existing GroupNorm+SiLU and explicit VAE contexts remain available below; they are not enabled by this launcher change.

The final FP32-only guard was checked again on the actual server: one additional warm pair measured **7.431528 → 4.749214 seconds**, all bytes equal, with **672 optimized INT8 calls and 168 optimized / 8 native attention calls**. This final smoke pair is supplementary to the three-round table. See [final scope A/B](results/production_final_scope.json).

Evidence: [actual server A/B](results/production_server_shared.json), [768×512 A/B](results/production_range_768x512.json), [precision experiment](results/production_precision.json), [INT8 scope and rollback](results/production_backend_scope.json). Independent native server restarts already changed text conditioning; exact-output claims here are for paired runs with the same conditioning, not cross-process determinism.

In the historical v0.3 run, normal production startup was verified with instrumentation and counters disabled; the original workflow completed and saved its PNG, and that run left its server on port 8299. This is a dated execution record, not the current server state. See [normal-server execution record](results/normal_production_smoke.json).

## Verified environment

- Existing Ubuntu/Python 3.12 installation and MOSS project environment.
- PyTorch `2.12.0+rocm10.2.0a20261011`; runtime HIP `7.17.26405`.
- Existing Triton `3.8.0+git669b31ac.rocm10.2.0a20261011`.
- Existing HIP compiler `7.15.26333`, AMD clang 23; the built normalization binary targets `gfx1030`.
- The reproduction commands use the existing `.venv-moss` interpreter and `scripts/with_local_storage.sh` cache/storage wrapper. The HIP library is already built on the tested host.

## Actual ComfyUI profile: where the time went

The original API graph ran through the installed, unmodified `main.py`, including its real nodes and memory management. The first instrumented run was **cold**: its stage wall times include loading and profiler overhead and are diagnostic, not the warm speed benchmark.

The table attributes each GPU kernel/copy/memset activity once to its operator, leaving unmatched activities explicit. Percentages use the sum of activity durations inside the denoising and VAE markers: **8,906.422 ms**. Overlapping GPU activity can overlap in elapsed wall time. Inclusive parent SDPA durations are not added to these operator totals.

| Rank | Operator / attribution | GPU ms | Share | Activities |
|---:|---|---:|---:|---:|
| 1 | `aten::copy_` | 4,975.509 | 55.86% | 4,134 |
| 2 | `comfy_kitchen::int8_linear` | 1,202.680 | 13.50% | 504 |
| 3 | Unattributed `_dot_i8_i32_kernel.kd` | 343.641 | 3.86% | 168 |
| 4 | `aten::mm` | 325.202 | 3.65% | 1,096 |
| 5 | `aten::add_` | 274.926 | 3.09% | 472 |
| 6 | Unattributed host-to-device copy | 230.652 | 2.59% | 713 |
| 7 | `aten::bmm` | 214.159 | 2.40% | 358 |
| 8 | `aten::mul` | 200.466 | 2.25% | 2,414 |
| 9 | `comfy_kitchen::dequantize_int8_convrot_weight_dtype` | 126.612 | 1.42% | 168 |
| 10 | `aten::_softmax` | 124.172 | 1.39% | 177 |

The INT8 direct-copy kernel alone accounted for 4,744.781 ms / 53.27%, motivating removal of a transpose materialization that the installed stride-aware INT8 backend did not need. Actual SDPA attribution was 476.353 ms in denoising and 1.099 ms in VAE. All **177 calls used the math backend**, despite fused-backend enable flags being set:

| Stage | dtype | Q/K/V shape | Calls |
|---|---|---|---:|
| Denoising | FP32 | `[1,24,1061,128]` | 168 |
| Denoising | FP32 | `[1,24,37,128]` | 8 |
| VAE | FP16 | `[1,1,4096,384]` | 1 |

The main FP32 input stride was `[3259392,128,3072,1]`; these calls had no mask, dropout or causal attention. Source: [actual_comfy_profile.json](results/actual_comfy_profile.json), including the raw kernel ranking and backend observations.

## Historical v0.2 warm full-workflow A/B results

These are synchronized **CPU wall times for the complete sampler plus VAE decode calls**, measured without the profiler. Model/text caches remain usable, but sampler and decoder cache hits are disabled so each same-seed request really executes. Warmup rows are excluded. Each row below has three native and three optimized measurements with alternating pair order; each experiment has its own baseline.

| Enabled change | Native median s | Optimized median s | Speedup | Latent / image max error |
|---|---:|---:|---:|---:|
| Exact FP32 attention | 7.438691 | 7.324319 | 1.01562× | 0 / 0 |
| INT8 transpose-copy removal | 7.437611 | 5.001646 | 1.48703× | 0 / 0 |
| Both together | **7.445725** | **4.762299** | **1.56347×** | **0 / 0** |

The combined optimized samples were **4.762286, 5.302309 and 4.762299 s**; the full observed range is retained rather than hiding the slower sample. All recorded latent/image outputs were finite and passed the exact-output gate with zero maximum error and RMSE. Three warm pairs establish the reported measurement, not a universal speedup across models or system load. The combined result does not enable VAE GroupNorm fusion, and the timing excludes text encoding, image saving and API polling outside the two marked stages.

Evidence: [attention A/B](results/comfy_attention_ab_exact.json), [INT8 A/B](results/comfy_int8_ab.json), [combined A/B](results/comfy_combined_ab.json).

### Rejected approximation

The first attention candidate applied scaling after GEMM and used a Triton softmax. Its random-input microbenchmark was about **1.75× faster**, with maximum operator error around `7e-7`, but the real same-seed image diverged: maximum difference **0.8528**, RMSE **0.09786**. Repeated native generation was identical. That candidate was rejected; its microbenchmark gain is not a result of the shipped path. Its [historical raw result](results/rejected_attention_ab.json) retains ComfyUI execution-success fields, which do not mean accuracy passed. The current harness makes numerical failure an explicit failed result.

## Operators and integration

### Exact FP32 math attention

The current path retains native Q/K square-root pre-scaling, rocBLAS batch matrix multiplies and `torch.softmax`. A small Triton kernel checks the first probability of each row and scans original scores only when that value is NaN; only all-negative-infinity rows are replaced with zero. Ordinary rows are not written by the repair kernel. This avoids the native safety path's full-matrix `isneginf`/`where` traffic while preserving its semantics.

```python
import torch
from rdna2_ops.attention import scaled_dot_product_attention

with torch.inference_mode():
    output = scaled_dot_product_attention(q, k, v)  # backend="auto"
```

`auto` selects FP32 `[1,24,S,128]` on gfx1030 for `128 <= S <= 4096`, with interleaved Q/K and interleaved or packed-QKV V strides. The two layout families passed twenty length cases, including odd lengths and interval boundaries. Other heads/batches/layouts, FP16/BF16, masks, causal attention, dropout, GQA, gradients, autocast, compilation and tracing use native SDPA. `backend="torch"` forces native; `backend="rdna2"` explicitly requests the broader eligible batch-one FP32 path for validation and raises on unsupported contracts. This is math attention with a materialized score matrix, not a memory-efficient attention implementation.

The measured shape took **2.54447 → 1.84830 ms** per ordinary call, with zero output error. The short S=37 candidate was slower and remains on native in automatic dispatch. Three random-shape comparisons had zero maximum error. Special-value checks covered softmax lengths 1/37/1061/4096, all `-inf`, NaNs, `+inf`, negative overflow, negative scale and zero scale. See [attention microbenchmark](results/attention_math_native_softmax.json) and [runtime contracts](results/runtime_contracts.json).

### INT8 transpose-copy adapter

This adapter passes `weight.T` directly to the **already installed, verified stride-aware INT8 backend**, instead of materializing `weight.T.contiguous()`. It changes neither quantization nor the integer matrix multiplication. Three layout tests produced identical INT32 results; the actual image A/B also produced zero latent/image error. See [int8_layout.json](results/int8_layout.json).

Use the context only after the application's existing backend initialization, with an exclusively owned workload:

```python
import inspect
from pathlib import Path
import torch
import comfy_kitchen.backends.eager.quantization as quant
from rdna2_ops.comfy_int8 import kitchen_int8_without_transpose_copy

backend = quant._int8_matmul_accumulate
expected = Path("~/DevSpace/moss-image-rdna2-lab01/scripts/rdna2_int8_matmul.py")
if Path(inspect.getsourcefile(backend)).resolve() != expected:
    raise RuntimeError("This example requires the tested stride-aware backend")

with torch.inference_mode(), kitchen_int8_without_transpose_copy(backend):
    result = run_denoising()  # The caller's existing workload.
```

The supplied callable must be the identical backend currently installed in Kitchen; a different project must validate its backend's stride support before using this adapter. The context replaces only the existing eager module's `int8_linear` export with the source implementation. Backend priority, constraints and capabilities stay unchanged; the original `quantization.int8_linear` and its direct aliases remain untouched. It applies only to FP32 inputs and FP32 outputs and restores the original export on exit, including exceptions. **The change is process-wide**: other threads would see it while active, and nested/concurrent contexts are rejected. An already-active production patch must be disabled before using this separate context. The existing stride-aware GEMM remains installed.

### GroupNorm + SiLU: eager inference

```python
import torch
from rdna2_ops import group_norm_silu, GroupNormSiLU

with torch.inference_mode():
    y = group_norm_silu(x, 32, weight, bias, eps=1e-5)

layer = GroupNormSiLU(32, 320).to(device="cuda", dtype=torch.float16)
with torch.inference_mode():
    y = layer(x)
```

The HIP path accepts contiguous NC... FP32/FP16/BF16 tensors on gfx1030, optional same-dtype/device length-C affine vectors, finite nonnegative epsilon and signed-INT32-sized input. Small groups use one kernel; groups above 8192 values use partial statistics plus a fused output kernel. Wave32 reductions, centered FP32 statistics and an explicit cast before SiLU preserve the two-operator dtype boundary. No custom backward is provided.

`backend="auto"` falls back to native for unsupported inputs, autograd/AD, autocast, tracing, `torch.compile` and active GPU graph capture. `backend="hip"` forces the supported eager path and raises during capture. **Use `backend="torch"` throughout graph warmup, capture and replay infrastructure**: warmup is not detectable as capture, and replay does not call the Python dispatcher.

The current runtime checks passed native-equivalent capture and tracing with a new input, plus compiled accuracy. FP16 `[1,320,64,64]` graph timing was **42.6605 µs native versus 14.4510 µs compiled auto**; that compiled result uses PyTorch's compiler fallback. HIP graph timing is now `policy_disabled`. `validate_bench.py` retains ordinary-call timing and reports that policy; `bench_compiled.py` captures native/compiled callables only. See [runtime_contracts.json](results/runtime_contracts.json).

## Actual VAE: FP16/BF16, regular/tiled and real low-VRAM loading

The VAE tests use the **captured denoiser latent** `[1,128,32,32]`, the installed `VAELoader`/`VAEDecode`/`VAEDecodeTiled`, and the existing `flux2-vae_small.safetensors` checkpoint. Outputs are 512×512 RGB. Each arm uses its own corresponding native decode path. Five alternating rounds measure three complete regular decodes or one tiled decode per round; tables report median wall time.

| dtype | Loading | Decode | Native ms | Fused ms | Speedup |
|---|---|---|---:|---:|---:|
| FP16 | Normal | Regular | **45.04** | **43.20** | **1.0427×** |
| FP16 | Normal | Tiled | 245.03 | 242.17 | 1.0118× |
| FP16 | Low-VRAM | Regular | 38.66 | 37.22 | 1.0386× |
| FP16 | Low-VRAM | Tiled | 296.20 | 293.54 | 1.0091× |
| BF16 | Normal | Regular | 179.30 | 178.41 | 1.0050× |
| BF16 | Normal | Tiled | 908.59 | 900.06 | 1.0095× |
| BF16 | Low-VRAM | Regular | 172.09 | 170.44 | 1.0096× |
| BF16 | Low-VRAM | Tiled | 936.33 | 931.09 | 1.0056× |

Tiled decode uses tile size 256 and overlap 64. The normal and low-VRAM runs produced the same error statistics for each dtype/path:

| dtype / decode | Maximum RGB difference | RGB RMSE | Maximum difference in 8-bit channel steps |
|---|---:|---:|---:|
| FP16 regular | 0.000976563 | 0.000121774 | 0.2490 |
| FP16 tiled | 0.000493288 | 0.000063062 | 0.1258 |
| BF16 regular | **0.010437012** | 0.000839100 | **2.6614** |
| BF16 tiled | 0.003234267 | 0.000428700 | 0.8247 |

All eight cases passed the preset finite-output limits: FP16 max/RMSE ≤ 0.01/0.001; BF16 ≤ 0.04/0.006. GN fusion is tolerance-based, not bitwise. BF16's measured gain is only about **0.5–1%**, with up to **2.66 eight-bit channel steps** of difference; it remains an explicit choice with those accuracy/performance tradeoffs.

The revised helper uses ComfyUI's own **`CastBiasWeightContext`**, including weight functions, dtype casting, prefetch/offload lifetime and interrupt polling. All 14 compatible blocks were reached: 28 cast/HIP calls for regular decode and 644 for tiled decode. Module/parameter identities and `state_dict` names stay intact while active, and original instance forwards are restored afterward. The low-VRAM cases actually exercised `model_lowvram=true`, `dynamic=false`, **124 CPU-weight modules and `loaded_bytes=0`**, rather than merely setting a label.

```python
import torch
from rdna2_ops.comfy import fuse_comfy_vae_decoder

stats = {}
with torch.inference_mode(), fuse_comfy_vae_decoder(
        vae.first_stage_model.decoder, stats=stats):
    image = vae.decode(latent)
```

Use a decoder owned exclusively by this context; nesting/concurrent use is rejected. Unknown forward overrides and hooked blocks are skipped, and incompatible global hooks are rejected. Compilation/tracing/capture call the original forwards. For a graph lifecycle, select `backend="torch"` for the whole context.

Evidence: [FP16 normal](results/vae_fp16_normal.json), [FP16 low-VRAM](results/vae_fp16_lowvram.json), [BF16 normal](results/vae_bf16_normal.json), [BF16 low-VRAM](results/vae_bf16_lowvram.json).

## Reproduce with the existing tested environment

Run GPU jobs sequentially. These commands reuse the existing launcher wrapper, interpreter, models and workflow; they write new result directories.

```bash
cd ~/DevSpace/rdna2-pro-research-20261102/rdna2-ops
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MOSS_RDNA2_INT8_PATCH=1

rdna2_python() {
  bash ~/DevSpace/moss-image-rdna2-lab01/scripts/with_local_storage.sh \
    ~/DevSpace/moss-image-rdna2-lab01/.venv-moss/bin/python -u "$@"
}

# One cold, instrumented run of the actual API graph.
rdna2_python profile_comfy_workflow.py \
  --project ~/DevSpace/moss-image-rdna2-lab01 \
  --out results/reproduce_profile --seed 20261105

# Warm native/optimized comparison: attention + INT8 copy removal.
rdna2_python profile_comfy_workflow.py \
  --project ~/DevSpace/moss-image-rdna2-lab01 \
  --out results/reproduce_combined --seed 20261105 \
  --compare-attention --compare-int8 --compare-rounds 3
```

The harness owns a loopback ComfyUI server on port 8299 and exits after its requests finish; `--port` selects another free port. It refuses an occupied port before starting its client. It loads `workflows/sft_fast4_int8_512.json` from the project. To isolate an optimization, supply only `--compare-attention` or only `--compare-int8` and a distinct `--out` directory. The current A/B harness warms both arms and requires finite, zero-error latent and image outputs.

Validate the actual VAE paths against the preserved latent:

```bash
for rdna2_dtype in fp16 bf16; do
  for rdna2_load_mode in normal lowvram; do
    rdna2_python validate_comfy_vae.py \
      --project ~/DevSpace/moss-image-rdna2-lab01 \
      --latent results/actual_comfy_profile/actual_latent.pt \
      --dtype "$rdna2_dtype" --load-mode "$rdna2_load_mode" \
      --output "results/reproduce_vae_${rdna2_dtype}_${rdna2_load_mode}.json"
  done
done
```

Each invocation validates regular and tiled decode. Low-VRAM mode uses the existing Comfy CLI controls `--lowvram --disable-dynamic-vram --reserve-vram 15.4` within that process and verifies the resulting memory-management branch. The release evidence archive `results/comfy_evidence.zip` contains the trace, captured latent, generated image and completion data; the latent is also provided under `results/actual_comfy_profile/`. A fresh profile's latent can be used from `results/reproduce_profile/actual_latent.pt`.

For the operator checks in the same environment:

```bash
rdna2_python validate_bench.py --output results/reproduce_group_norm.json
rdna2_python validate_runtime_contracts.py --output results/reproduce_runtime_contracts.json
```

These checks write new JSON files and preserve the archived evidence. No installation or environment-reconstruction step is required on the tested host.

## Historical v0.1 evidence

These JSON files are retained unchanged as historical measurements, including old HIP graph experiments. **Their HIP graph fields are not supported v0.2 execution paths** and must not be presented as current graph speedups.

| File | Historical scope |
|---|---|
| [validation_benchmark_final.json](results/validation_benchmark_final.json) | 35 GroupNorm correctness cases and 12 ordinary-call/graph shape-dtype measurements |
| [compiled_comparison.json](results/compiled_comparison.json) | Earlier eager/compiled/HIP comparison; compiled Triton already won graph replay |
| [comfy_blocks_v1.json](results/comfy_blocks_v1.json) | Original installed ResnetBlock implementation with synthetic inputs/weights |
| [pretrained_vae.json](results/pretrained_vae.json) | Pretrained decoder with seeded random latents and the earlier integration helper |

The independent GN reference computes centered grouped statistics and affine arithmetic in FP64, rounds to the input dtype before SiLU, and uses fixed FP32 rtol/atol 3e-4/3e-5, FP16 3e-3/3e-3 and BF16 3e-2/3e-2. The original coverage includes nullable affine parameters, constants, 8191/8192/8193 group boundaries, storage offsets, a non-default stream, large mean/small variance and native gradient/AD/autocast fallback. Current graph/tracing/compile and nonfinite-attention behavior is recorded separately in `runtime_contracts.json`.

See [SOURCES.md](SOURCES.md) for the broader RDNA2 kernel survey. The measured attention, layout and normalization work is reusable; the specific full-image result is limited to the recorded workflow and hardware.

## Licenses

Independently written kernels and integration code use the root MIT license. The Kitchen-derived INT8 eager arithmetic preserves Comfy Org attribution under Apache-2.0; full terms are in `rdna2_ops/LICENSE.kitchen` and `production/MOSS_RDNA2_INT8/LICENSE.apache-2.0`.
