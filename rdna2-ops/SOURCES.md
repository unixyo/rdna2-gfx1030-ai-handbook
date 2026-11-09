# Sources and validation scope — v0.6, 2026-11-07 KST

All measurements come from the actual RX 6900 XT / gfx1030. Installed-source inspection explains the implementation and runtime restrictions; GPU and whole-image tests establish the measured results. The existing environment was retained. Operating instructions and reproduction details belong in the accompanying README documentation.

## Historical v0.3 production evidence

The existing MOSS `scripts/start_moss_comfy.sh`, storage wrapper and original API graph were used. Integration lives in the existing `MOSS_RDNA2_INT8` custom node. The adapter replaces only Kitchen's exported eager `int8_linear` callable, preserving backend identity, priority and capabilities. It passes `weight.T` to the existing stride-aware gfx1030 INT32 GEMM; it does not replace function bytecode or edit installed ComfyUI/Kitchen package files.

| Actual launcher comparison | Native median | Optimized median | Result |
|---|---:|---:|---|
| v0.3 denoising + VAE | 7.446393 s | 4.762543 s | 1.56353×; latent/RGB byte equality |

There are three warm samples per mode, with modes alternated within one server instance. Separate native-server restarts changed positive/negative conditioning hashes **before sampling**; the paired measurement holds that conditioning fixed. The upstream conditioning variation was not diagnosed or changed. Altered FP16/BF16 attention precision was **not adopted as a production default**.

## Code provenance and licenses

- `rdna2_ops/csrc/group_norm_silu.hip` is independently written and MIT licensed. The earlier “No upstream kernel was copied” statement describes that HIP kernel, not all shipped Python code.
- `production/MOSS_RDNA2_INT8/backend.py` adapts **Comfy Kitchen 0.2.40** `backends/eager/quantization.py::int8_linear`. It retains the Apache-2.0 copyright/SPDX notice; the license is included as `production/MOSS_RDNA2_INT8/LICENSE.apache-2.0`. The inspected upstream file SHA256 is `4412eb5bbd12cb2c9a09c5c9e076e744e16bf39632fe624e4df739b84681ef12`.
- The adaptation preserves quantization, scale order, chunking, dtype conversion, bias and residual behavior while removing transpose materialization for the measured FP32 path. MOSS's existing `scripts/rdna2_int8_matmul.py` supplies the GEMM; this is reuse of an installed kernel.
- Upstream: [Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen). The installed version/file hash is the provenance pin; upstream `main` is not assumed identical.

## Installed revisions

| Component | Inspected version / revision |
|---|---|
| ComfyUI | `086099f5d8886163ef77dc354f8382cb7431582a` |
| PyTorch | `2.12.0+rocm10.2.0a20261011`, git `9ccf2fd0aab44087b58af23ad20bc961c29bb852` |
| Runtime HIP | `7.17.26405` |
| Bundled AOTriton | `0.14.50`, git `7a6963a7265d2cf0fc48c6a63464386134d32483` |
| Triton | Existing `3.8.0+git669b31ac.rocm10.2.0a20261011` |

Comfy source paths below are relative to `~/DevSpace/moss-image-rdna2-lab01/runtime/ComfyUI/`. Kitchen/Torch sources are under that project's `.venv-moss/lib/python3.12/site-packages/`.

## Why denoising QKV is FP32

| Source checked in the installation | Consequence |
|---|---|
| `comfy/supported_models.py:2053`; `model_management.py:2005–2010` | MOSS permits BF16/FP32, excluding FP16; gfx1030 declines BF16 compute when `manual_cast=False`. |
| `comfy/sd.py:2429–2433`; `model_management.py:1189–1211` | Quantized loading calls `unet_manual_cast(None, ..., supported_dtypes)`, selecting FP32 independently of BF16 weight storage. |
| `comfy/model_base.py:227–233,269–274`; `comfy/ops.py:1536,1579–1583` | Latent/context are cast to the inference dtype; quantized Linear preserves the requested input/compute dtype. |
| `comfy/ldm/moss_image/model.py:62–82,142–148`; Kitchen `backends/hip/__init__.py:2411–2412` | QKV is already FP32; QK RMSNorm/RoPE preserves that dtype. |
| `comfy/ldm/modules/attention.py:594–635` | The PyTorch attention wrapper reshapes/transposes without an additional upcast. |

Pinned primary sources: [model policy](https://github.com/Comfy-Org/ComfyUI/blob/086099f5d8886163ef77dc354f8382cb7431582a/comfy/supported_models.py), [dtype selection](https://github.com/Comfy-Org/ComfyUI/blob/086099f5d8886163ef77dc354f8382cb7431582a/comfy/model_management.py), [quantized loading](https://github.com/Comfy-Org/ComfyUI/blob/086099f5d8886163ef77dc354f8382cb7431582a/comfy/sd.py), [input conversion](https://github.com/Comfy-Org/ComfyUI/blob/086099f5d8886163ef77dc354f8382cb7431582a/comfy/model_base.py).

**Application precision setting:** bare Torch initializes reduced-precision math SDPA to false, but Comfy `model_management.py:567–569` enables it for Torch ≥ 2.5. The [pinned math implementation](https://github.com/pytorch/pytorch/blob/9ccf2fd0aab44087b58af23ad20bc961c29bb852/aten/src/ATen/native/transformers/attention.cpp) promotes FP16/BF16 QKV to FP32 only when this flag is false. Bare-Torch defaults therefore do not describe the initialized Comfy process. The accepted FP32 path retains native Q/K pre-scaling, BMM and softmax, changing only the all-negative-infinity-row safety repair.

## Fused-backend restrictions in this installation

- [Pinned PyTorch SDPA selection](https://github.com/pytorch/pytorch/blob/9ccf2fd0aab44087b58af23ad20bc961c29bb852/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp): Flash excludes FP32; AOTriton Efficient accepts FP32/FP16/BF16. Both ROCm paths require the AOTriton GPU check. FP32 alone does not explain every math fallback.
- Bundled AOTriton's [GPU classifier](https://github.com/ROCm/aotriton/blob/7a6963a7265d2cf0fc48c6a63464386134d32483/v3src/hip-dependent/util.cc) has no gfx1030 entry. Its [availability check](https://github.com/ROCm/aotriton/blob/7a6963a7265d2cf0fc48c6a63464386134d32483/modules/flash/csrc/attn_check.cc) requires matching kernels. The experimental-enable flag does not supply missing architecture support.
- [Pinned PyTorch CK preference checks](https://github.com/pytorch/pytorch/blob/9ccf2fd0aab44087b58af23ad20bc961c29bb852/aten/src/ATen/Context.cpp): `ckSupported()` permits gfx942/gfx950, and getter/setter enforce architecture plus build support.
- Installed `torch/include/ATen/hip/HIPConfig.h:10` has `AT_CUDNN_ENABLED() = 0`. Kitchen `backends/hip/__init__.py:3351–3357` gates INT8 attention on `has_wmma()`. Comfy's `--use-ck-attention` means **Comfy Kitchen**, distinct from PyTorch's Composable Kernel backend.

Forced-backend probes complement these static checks; a rejected backend is not an executed speedup.

## Pretranspose feasibility

Already-prepared contiguous RHS storage measured **2.16–2.55× faster than stride-view GEMM** on four actual matrix shapes. This is a microbenchmark result, not a deployed whole-workflow gain.

A column-major logical `[N,K]` qdata tensor can survive ordinary QuantizedTensor `.to()`/clone. Dynamic gathered storage does not preserve it: Comfy `memory_management.py:183` rebuilds inner tensors with `.view(template.shape)` and no stride, and `model_management.py:1556–1578` copies into those row-major destinations. Kitchen `tensor/int8.py:362` also forces `.contiguous()` in ordinary linear dispatch; `tensor/base.py:335–342` requantizes without a physical-layout preservation contract.

Dynamic `ON_LOAD` runs after staging, before operation-time materialization/LoRA application (`model_patcher.py:2125–2127`). Partial eviction (`:2131–2156`) has no equivalent public callback. Model-level callbacks therefore do not establish a safe pointer-cache lifecycle. Keeping all four matrix types across 42 layers would additionally consume about **4.43 GiB**. No storage conversion or persistent transpose cache was deployed; production retains the validated view path.

## Historical evidence and broader sources

The **v0.2** combined result was **7.445725 → 4.762299 s**, three warm pairs with zero latent/image error. Its raw JSON remains historical evidence, separate from the v0.3 launcher result. The first approximate-softmax candidate improved a microbenchmark but failed whole-image accuracy and was rejected. Cold-profile activity shares are diagnostic, not warm stage wall times.

Initial selection and the broader candidate review used:

- [PyTorch GroupNorm](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/group_norm_kernel.cu), [RMSNorm](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/layer_norm.cpp), and [CUDA/HIP softmax](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/SoftMax.cu). Native/fused/compiled baselines were considered. HIP GroupNorm is eager-only; custom HIP graph timing is policy-disabled.
- [AMD RDNA2 ISA](https://gpuopen.com/download/rdna2-shader-instruction-set-architecture.pdf), [Triton RDNA1/RDNA2 support removal](https://github.com/triton-lang/triton/pull/12042), and [Triton math semantics](https://github.com/triton-lang/triton/blob/main/python/triton/language/semantic.py). Existing compiler/runtime versions were preserved and measured.
- [PyTorch TunableOp](https://docs.pytorch.org/docs/main/cuda.tunable.html), [rocBLAS solution enumeration](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/reference/beta-features.html), [AITER](https://github.com/ROCm/aiter), and [rocSPARSE SpMM](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/generic.html): follow-on inventory, not completed implementations or claimed workload gains.

## v0.4 handbook and new experiments — 2026-11-06

The Korean [optimization handbook](RDNA2_OPTIMIZATION_GUIDE_KO.md) distinguishes
**A: measurements on this GPU**, **B: source inspection**, and **C: official
documentation/algorithm descriptions**. No environment rebuild, dependency
upgrade, model download or production-default change was made for these runs.

### A: measured artifacts

| Artifact | Exact scope |
|---|---|
| `results/tiled_attention_fp16_v04.json` | B1/H2, interleaved FP16, S4097/8192/16384, D64/128, 16x32 and 32x64 tiles; full native MATH comparison and sampled FP64 reference |
| `results/tiled_attention_d128_32x32_v04.json` | Separate paired D128 32x32 trial, S8192/16384; baseline is from the same trial |
| `results/tiled_attention_fp32_v04.json` | B1/H2/S8192, D64/128, FP32 operands/output without downcast; D128 slower than native |
| `results/quantized_attention_v04.json` | B1/H4, FP16, per-token INT8 QK, K smoothing on/off, FP16/FP32 PV; quantization-inclusive alternating timing and full-output error |
| `results/quantized_attention_isa_v04.json` | Independent repeat with dot2c, dot4c, assembly SHA256, register/spill/shared metadata; not pooled with the first timing run |
| `results/quantized_vs_tiled_v04.json` | Same B1/H4 contiguous Q/K/V for native, FP16 tiled and all four INT8 variants; direct extra speed/accuracy/memory tradeoff, including v_fmac_f32 |
| `results/gemm_v04.json` | FP16 GEMM with FP32 accumulation, rocBLAS/Tensile versus explicit Triton; independent FP64 checks and hipBLASLt preference fallback probe |
| `results/precision_diagnosis_v04_fixed/precision_diagnosis.json` | Same-process native replay and six precision/control variants; attention, denoiser, quantizer and final-output evidence |
| `results/precision_diagnosis_v04_fixed/analysis.json` | Compact analysis with full-trajectory hash equivalence and bounded real-QKV overflow/rounding probes |
| `results/precision_diagnosis_scopes_v04/precision_diagnosis.json` | Separate-process native/short-only/long-only controlled experiment |
| `results/precision_diagnosis_scopes_v04/analysis.json` | Scope-specific image errors and first activation quantization boundary; never aligned to a different process's native image |

Precision traces are diagnostic, not performance measurements. Tiled/INT8
synthetic operator results are not end-to-end 1024px/video quality claims.
Historical raw results and source copies are byte-identical to the on-host
research files. The prior package's extra scope-test argument annotation is
preserved separately in `results/production_backend_scope_notes.json`, rather
than presented as a field emitted by the raw measurement.
The first diagnostic attempt included cold LoRA weight re-quantization in the
activation call index and failed alignment. That failure remains remotely in
`results/precision_diagnosis_v04/`; the fixed diagnostic separately counts the
168 weight calls and compares the 672 activation calls. It does not relax
shape/call alignment checks to hide a mismatch.

### B: compiler/library source boundaries

The installed Triton version is
`3.8.0+git669b31ac.rocm10.2.0a20261011`. The matching ROCm source prefix was
inspected rather than assuming current upstream main has the same behavior:

- [FMA.cpp at 669b31ac](https://github.com/ROCm/triton/blob/669b31ac/third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/FMA.cpp):
  FP16-to-FP32 chooses vectorSize=2/`llvm.amdgcn.fdot2`; INT8-to-INT32 chooses
  vectorSize=4/`llvm.amdgcn.sdot4`. Inspected file blob
  `0c9a991d0e003e1f4b672946f392b9f955787ebc`.
- [DotOpToLLVM.cpp](https://github.com/ROCm/triton/blob/669b31ac/third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM.cpp):
  blocked encoding goes to `convertAMDFMADot`; file blob
  `d08e6703e0c3b114185f0e4585ca80bbd3525c3c`.
- [AccelerateAMDMatmul.cpp](https://github.com/ROCm/triton/blob/669b31ac/third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp):
  blocked SIMD lowering and the even-K FP16/FP32 dot legality; file blob
  `722f7146e842f770a1d79ab01f1a0626de64df5e`.
- [TheRock target configuration](https://github.com/ROCm/TheRock/blob/main/cmake/therock_amdgpu_targets.cmake):
  gfx1030 explicitly excludes hipBLASLt; inspected file blob
  `ee147646c8d5699f8a3222fc6667f94f09e42fe2`. This is an upstream build policy,
  distinct from the actual installed PyTorch fallback measured above.
- [Upstream Triton PR #12042](https://github.com/triton-lang/triton/pull/12042),
  merged 2026-09-30, removes RDNA1/RDNA2 target support from main. Existing
  3.8-based successful execution must not be generalized to an untested upgrade.

### C: algorithms and APIs

- [FlashAttention](https://arxiv.org/abs/2205.14135) and the
  [official Triton fused Attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html):
  online softmax avoids a global quadratic score/probability allocation.
- [SageAttention](https://arxiv.org/abs/2410.02367) and the
  [authors' INT8 QK/FP16 PV implementation](https://github.com/thu-ml/SageAttention/blob/main/sageattention/triton/attn_qk_int8_per_block.py):
  motivation for quantized QK and K centering. Our per-token prototype and
  FP32-accumulating SIMD PV are not the paper's NVIDIA tensor-core implementation.
- [rocBLAS design notes](https://rocmdocs.amd.com/projects/rocBLAS/en/latest/conceptual/rocblas-design-notes.html):
  `ROCBLAS_USE_HIPBLASLT=0` forces Tensile; preference value 1 may fall back.
- [Triton dot API](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)
  and [AMD gfx1030 instruction reference](https://rocm.docs.amd.com/projects/llvm-project/en/docs-7.14.1/LLVM/llvm/html/AMDGPU/AMDGPUAsmGFX1030.html):
  operand/accumulator semantics and SIMD instruction definitions. The actual
  emitted dot2c/dot4c and spills are backed by the A artifacts, not inferred
  solely from these documents.

## v0.5 actual MOSS/Qwen evidence and source boundaries

All new GPU jobs use the existing gfx1030 environments and Triton 3.8.0.
Research wrappers are process-local. The installed MOSS/Qwen operating source
and settings are not edited. The Korean handbook §9.6–9.9 contains the complete
tables, contracts, rejected candidates and reproduction commands.

| Evidence | What it establishes | Limit |
|---|---|---|
| `results/v05_moss_int8_tuned_ab/runs.json` | Same-server actual 4-step latent/RGB equality and 2.279× denoise+VAE median | Existing view+auto baseline; canonical baseline n=3/candidate n=2 after first candidate exclusion |
| `results/int8_gemm_moss_shapes_v05.json`, `results/int8_gemm_qwen_shapes_v05.json` | Actual observed shapes, synthetic INT8 tensor operator comparisons, exact output and generated dot4 | Qwen candidate loses all six shapes; no Qwen INT8 whole-model replacement result |
| `results/fp32_attention_qwen_pv_chunks_v05.json`, `results/fp32_attention_moss_no_spill_v05.json` | Actual captured FP32 values and strides, spill-zero tile, latency and additional allocator peak | No final RGB measurement because all tested speed replacements lose |
| `results/v05_moss_mixed_boundaries/runs.json` | Exact FP32 control, five rejected partial-FP16 variants, quantizer boundary code changes and final RGB | Diagnostic copies affect wall; those full-run walls are not ordinary-inference speed results |
| `results/mixed_gemm_backends_v05.json` | Actual warmed FP32/FP16 input/output combinations and selected BMM kernels | Casts excluded; operation profile timings are not whole Attention timings |
| `results/fp16_fp32_dispatch_source_v05.json` | Installed HS/HH/SS Tensile artifacts, headers, file hashes and symbol metadata | Installed wheel lacks implementation `.cpp`; official main is explanatory, not a matching-commit proof |
| `results/v05_sage_qwen_4964.json` | Independent INT8 QK/FP16 PV rectangular implementation, actual input speed/memory/error and generated dot4/dot2 | Every full-output operator gate fails; sampled FP64 rows alone are insufficient |
| `results/v05_qwen_sage_rgb/runs.json` | Same-server 8-step actual RGB failure and exact native restoration | Only one candidate request; baseline is original split, not alpha |
| `results/v05_moss_profile_verified/profile_verification_summary.json` | Correct raw-trace category totals, separately attributed CPU view, byte-exact actual profile run | Summed activities are not a stream union or wall duration |
| `results/v05_operating_files_audit.json` | Ten explicitly tracked operating source/config/workflow hashes unchanged | Not an audit of every file in the two project trees |

The new `int8_gemm.py`, `fp32_attention.py`, `mixed_attention.py` and
`sage_attention_v05.py` are research implementations in this project. The Sage
prototype builds on this project's existing per-token quantizer and online
softmax, not an assertion that official SageAttention supports this GPU. The
native Qwen and alpha baselines are imported from the existing runtime and
the preserved rehearsal, rather than reconstructed as convenient substitutes.

The BMM dtype interpretation was cross-checked against primary sources:

- [PyTorch CUDABlas.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDABlas.cpp):
  the ROCm half strided-batched helper selects the C/D storage dtype from the
  requested output type; both HHS and HSS use FP32 compute.
- [rocBLAS programming guide](https://rocm.docs.amd.com/projects/rocBLAS/en/develop/how-to/Programmers_Guide.html):
  HHS and HSS datatype combinations. The measured HSS fallback choice and
  latency are A evidence from this installation, not a prediction from docs.

The initial v0.5 `device_type == CUDA` aggregation incorrectly included
`gpu_user_annotation`. Corrected canonical GPU tables use complete raw trace
events with category `kernel`, `gpu_memcpy`, or `gpu_memset`. Original incorrect
reports remain as `_kernels_pre_fix.json`; correction metadata references the
original report, retained trace SHA and new aggregator SHA. No model-output
or elapsed-stage values were changed by this reporting correction.

Large captured tensors and v0.5 raw Chrome traces remain on the research PC.
The portable package includes compact JSON, source, ISA/TTGIR snapshots and
the original Qwen rehearsal. Its `SHA256SUMS` describes packaged bytes; it does
not claim that omitted large files were bundled.

## v0.6 operator catalog evidence (2026-11-07 KST)

- Entry: [operator catalog](../RDNA2_OP_CATALOG_KO.md), using its section 7
  native-kernel / candidate / three-shape / final-output / record contract.
- Layout A: [48-combination raw evidence](results/layout_blas_v06.json) and
  [kernel-name dictionary](results/LAYOUT_BLAS_V06_KO.md). The tensor storage
  labels NN/NT/TN must not be treated as identical rocBLAS API transpose flags.
- FP16 Attention A: [33-combination experiment](results/fp16_attention_v06_expanded.json)
  and [actual 30-step Anima A/B](results/v06_anima_attention_ab/runs.json).
  The operator field named byte_equal uses torch.equal and therefore states
  exact value equality, not signed-zero byte equality. Model digest equality
  is an actual byte-level check. Native and candidate timing samples remain
  separate from diagnostic profiler activity totals.
- Capture A/B: the installed AttentionTensorContainer class was read, the
  original shape-access error reproduced, and non-consuming peek validated.
  [CPU result](results/capture_container_cpu_v06.json),
  [actual capture and output check](results/v06_anima_capture/runs.json).
- Hardware C: AMD's [floating-point dot product builtins table](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/docs-1.0.0/compiler-builtins/cross-arch/arithmetic-ref/dot-float-builtins.html)
  lists RDNA2 fdot2 support and no BF16 fdot2 or FP8 dot4. The
  [HIP 6.3.2 hardware table](https://rocm.docs.amd.com/projects/HIP/en/docs-6.3.2/reference/hardware_features.html)
  separately lists BF16 and FP8 arithmetic as unsupported on RDNA2. These
  tables are not evidence of performance or of inability to store/decode the
  formats in software; actual kernel/profile/ISA measurements decide that.
- Existing Anima report supplied by the user:
  `~/DevSpace/anima-rdna2-lab01/REPORT_ANIMA_FP16_KO.md`, SHA256
  `e309b3b83772213803e91b5ed2b0639f0a90737b3cf81388e593564084cac200`.
  A byte-identical [snapshot](results/sources/REPORT_ANIMA_FP16_KO_20261107.md)
  is included for the original direct-path and tiled-path comparison.
  Its earlier visual-equivalence observation is not an independent perceptual
  acceptance for new candidates. Its overlapped tiled measurement is excluded
  as described in that report.
- Existing Applio ROCm comparison supplied by the user:
  `~/DevSpace/rdna2-pro-research-20261102/conv_miopen_20261107/`.
  NORMAL dilated observations include workspace-related skipped candidates;
  they are not a clean proof about every optimal NORMAL solver.
- BF16 A/B: [three-shape raw comparison](results/v06_bf16_gemm/results.json),
  [actual MOSS BF16 VAE SDPA boundary](results/v06_bf16_vae_sdpa_fp32/results.json),
  [request-local Anima BF16 diagnostic](results/v06_anima_bf16_compute_2step/runs.json),
  and [complete report](BF16_GEMM_V06_KO.md). Conversion timing excludes the
  separately measured half-range guards. The Anima candidate also changes
  flatten/bias ordering and fails final RGB; it is not an isolated precision
  experiment. Original VAE control bit_equal uses torch.equal and represents
  value equality, with signed-zero bytes not distinguished.
- FP8 A/B: [both-format three-shape evidence](results/v06_fp8_gemm/results.json),
  [E4M3fn thirty-step cache A/B](results/v06_anima_fp8_e4m3fn_30step/runs.json),
  [owned-cache follow-up](results/v06_anima_fp8_owned_2step/runs.json), and
  [complete report](FP8_WEIGHT_GEMM_V06_KO.md). The experiments use identical
  raw, unscaled FP8 values for the compute-path comparison. Original BF16
  checkpoint to raw FP8 representation loss is a separate CPU comparison
  of stored final outputs. E5M2 final-model and scaled/FNUZ checkpoint paths
  are not covered. Decoder bit-pattern comparisons and model byte digests
  are stronger than the operator harness's torch.equal field.
- Convolution A/B/C: [complete operator/epoch/model report](results/conv_catalog_v06/CONV_CATALOG_V06_KO.md),
  [BF16 Conv2D three-shape comparison](results/conv_catalog_v06/bf16_conv2d_micro/results.json),
  [actual normal VAE decode](results/conv_catalog_v06/bf16_conv2d_vae/results.json),
  [actual low-VRAM VAE decode](results/conv_catalog_v06/bf16_conv2d_vae_lowvram/results.json),
  and [filtered-DB Applio training](results/conv_catalog_v06/epoch_filtered/result.json).
  The microbenchmark has per-method allocated peaks. VAE JSON peak fields
  cover an A/B interval and must not be read as candidate-only memory savings.
  FindDb and PerfDb serve different purposes; the research recipe preserves
  failed solver selections and filters only validated keys. MIOpen's pinned
  source uses first-read static FindMode state, so per-call environment changes
  are not a reliable dispatch mechanism. Training speed is measured, while
  final audio equivalence is unresolved by the native-repeat floor.
- Conv family A/B: [FAST](results/conv_catalog_v06/family_fast/result.json),
  [NORMAL](results/conv_catalog_v06/family_normal/result.json), and pinned
  grouped-solver source in `results/conv_catalog_v06/source/`. Missing CK
  warnings do not imply that all convolution calls fail. The measured fallback
  cost does not estimate a hypothetical CK-present speedup.
- Qwen exact-tile Conv3D A/B: [normal baseline](results/conv_catalog_v06/qwen_patch_fp32_native_normal/result.json),
  [Linear candidate](results/conv_catalog_v06/qwen_patch_fp32_linear_vs_normal/result.json),
  [actual model attempt](results/conv_catalog_v06/qwen_model_screen2/runs.json),
  and [native-drift source note](results/conv_catalog_v06/QWEN_NATIVE_DRIFT_SOURCE_NOTE_KO.md).
  Native conditioning changes before the candidate runs, so neither candidate
  RGB error nor model speedup is established. The read-only RNG investigation
  does not establish the underlying cause.
- Normalization A/B: [12 operator cases and 27 native/candidate paths](results/norm_fusions_v06.json),
  [actual Anima model requests](results/v06_anima_norm_ab/runs.json), and
  [complete report](results/NORM_FUSIONS_V06_KO.md). Actual baselines include
  PyTorch vectorized norm and Comfy Kitchen HIP adaLN; the latter is a different
  package from AMD Composable Kernel. Modulation-only byte equality is checked
  with uint8 buffers and model digests, while three statistics replacements
  fail final RGB. Its roughly 0.39% model-time reduction is a small n=2
  observation, not the 1.40–2.65x operator speedup.
- The same Anima requests provide the two additional guarded FP16 Attention
  warm measurements: 67.775/67.727 s versus native 80.607/80.720 s, 1.1906x
  with identical conditioning/latent/RGB digests across both experiments.
  The earlier 83.019 s first candidate remains recorded. A first-Triton
  initialization hypothesis is not a measured decomposition of that delay.
- Normalization source scope: [measured adapter](results/sources/norm_switches_v06_measured_model.py)
  and [pre-filter adapter](results/sources/norm_switches_v06_before_measured_filter.py)
  are preserved. The current adapter adds a conservative original-Block
  fallback for any custom mlp_patch because the earlier orchestration did
  not preserve arbitrary extra-dict chaining. Its arithmetic AST is unchanged.
  Existing CPU contract evidence covers single x-only patches and value
  equality, not arbitrary patch chains or an unrecorded adapter source hash.
- ConvTranspose1D A: [validated-key FAST replay](results/conv_catalog_v06/transpose_merged_replay/result.json)
  and [actual epoch](results/conv_catalog_v06/epoch_filtered_transpose/result.json).
  Operator gains reproduce, but the actual epoch is 0.99804x the filtered
  baseline. Training output remains inconclusive against native-repeat drift.
- Final bounded protection check: [15 operating files](results/v06_operating_files_audit.json)
  are unchanged, with Applio's 16 protected items checked in actual epoch JSON.
  This does not claim a full filesystem or compiler-cache audit. The current
  generic harness also has an opt-in instrument_module extension for an
  actual conditioning-stage hook, with cleanup inside the GPU audit lifetime.
  [The previous measured harness](results/sources/research_comfy_v06_before_instrument.py)
  is preserved; its older measurements are not assigned the new source hash.
