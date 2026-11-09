// SPDX-License-Identifier: MIT
// Original, standalone HIP GroupNorm + SiLU kernels for RDNA2 (wave32).
// Compile for the real device, e.g. hipcc --offload-arch=gfx1030 -O3 -shared -fPIC.
// No PyTorch headers, device allocation, device synchronization, or matrix ISA.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {
constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr int kWaves = kThreads / kWave;
constexpr int kChunk = 8192;
constexpr int kOutputItems = 4;
constexpr int kOutputTile = kThreads * kOutputItems;
constexpr uint64_t kMaxGrid = 2147483647ULL;

// Keep BF16 storage independent of ROCm's optional native BF16 instructions.
struct BFloat16Storage {
  uint16_t bits;
};

template <typename T> struct Scalar;
template <> struct Scalar<float> {
  __device__ __forceinline__ static float load(float value) { return value; }
  __device__ __forceinline__ static float store(float value) { return value; }
};
template <> struct Scalar<__half> {
  __device__ __forceinline__ static float load(__half value) {
    return __half2float(value);
  }
  __device__ __forceinline__ static __half store(float value) {
    return __float2half_rn(value);
  }
};
template <> struct Scalar<BFloat16Storage> {
  __device__ __forceinline__ static float load(BFloat16Storage value) {
    return __uint_as_float(static_cast<uint32_t>(value.bits) << 16);
  }
  __device__ __forceinline__ static BFloat16Storage store(float value) {
    uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) == 0x7f800000u &&
        (bits & 0x007fffffu) != 0) {
      // Preserve NaNs, including NaNs whose payload would truncate to zero.
      return {static_cast<uint16_t>((bits >> 16) | 0x0040u)};
    }
    bits += 0x7fffu + ((bits >> 16) & 1u);  // round-to-nearest, ties-to-even
    return {static_cast<uint16_t>(bits >> 16)};
  }
};

// The returned total is valid in thread 0. The caller must synchronize before
// reusing scratch; every caller below does so while publishing its statistic.
__device__ __forceinline__ float block_sum(float value, float* scratch) {
  const int lane = threadIdx.x & (kWave - 1);
  const int wave = threadIdx.x / kWave;
#pragma unroll
  for (int offset = kWave / 2; offset > 0; offset /= 2) {
    value += __shfl_down(value, offset, kWave);
  }
  if (lane == 0) scratch[wave] = value;
  __syncthreads();
  value = threadIdx.x < kWaves ? scratch[threadIdx.x] : 0.0f;
  if (wave == 0) {
#pragma unroll
    for (int offset = kWave / 2; offset > 0; offset /= 2) {
      value += __shfl_down(value, offset, kWave);
    }
  }
  return value;
}

template <typename T>
__device__ __forceinline__ T affine_silu(float value, float anchor,
                                        float mean_delta, float inv_std, const T* weight,
                                        const T* bias, int channel) {
  // Keep the common offset separate: reconstructing mean = anchor + delta
  // would round away a small fractional mean on a high-DC input.
  float normalized = ((value - anchor) - mean_delta) * inv_std;
  if (weight) normalized *= Scalar<T>::load(weight[channel]);
  if (bias) normalized += Scalar<T>::load(bias[channel]);
  // F.silu(F.group_norm(x)) materializes GroupNorm in x's dtype. Preserve
  // that rounding boundary even though the intermediate never enters VRAM.
  const float rounded = Scalar<T>::load(Scalar<T>::store(normalized));
  return Scalar<T>::store(rounded / (1.0f + expf(-rounded)));
}

// For small groups, values remain in registers through two-pass statistics
// and output. Large groups reuse this kernel to produce independent partial
// (mean_delta, sum-of-squared-deviations) pairs for each 8192-element chunk.
// Every chunk uses the same group anchor, so deltas can be merged directly.
template <typename T, int Items, bool StatsOnly>
__global__ __launch_bounds__(kThreads) void row_kernel(
    const T* __restrict__ x, const T* __restrict__ weight,
    const T* __restrict__ bias, T* __restrict__ y,
    float* __restrict__ partials, int64_t group_size, int groups,
    int channels_per_group, int spatial, int parts, float eps) {
  const int row = StatsOnly ? static_cast<int>(blockIdx.x) / parts
                            : static_cast<int>(blockIdx.x);
  const int part = StatsOnly ? static_cast<int>(blockIdx.x) % parts : 0;
  const int64_t offset = static_cast<int64_t>(part) * kChunk;
  const int64_t remaining = group_size - offset;
  const int count = static_cast<int>(remaining < kChunk ? remaining : kChunk);
  const int64_t group_base = static_cast<int64_t>(row) * group_size;
  const int64_t base = group_base + offset;
  const float anchor = Scalar<T>::load(x[group_base]);

  __shared__ float scratch[kWaves];
  __shared__ float mean_delta;
  __shared__ float inv_std;
  float values[Items];
  float delta_sum = 0.0f;
#pragma unroll
  for (int item = 0; item < Items; ++item) {
    const int j = threadIdx.x + item * kThreads;
    const bool valid = j < count;
    values[item] = valid ? Scalar<T>::load(x[base + j]) : anchor;
    delta_sum += valid ? values[item] - anchor : 0.0f;
  }
  const float total_delta = block_sum(delta_sum, scratch);
  if (threadIdx.x == 0) mean_delta = total_delta / count;
  __syncthreads();

  float centered_sum = 0.0f;
#pragma unroll
  for (int item = 0; item < Items; ++item) {
    const int j = threadIdx.x + item * kThreads;
    const float centered = (values[item] - anchor) - mean_delta;
    if (j < count) centered_sum += centered * centered;
  }
  const float m2 = block_sum(centered_sum, scratch);
  if constexpr (StatsOnly) {
    if (threadIdx.x == 0) {
      const int64_t stat = 2 * static_cast<int64_t>(blockIdx.x);
      partials[stat] = mean_delta;
      partials[stat + 1] = m2;
    }
  } else {
    if (threadIdx.x == 0) inv_std = rsqrtf(m2 / count + eps);
    __syncthreads();
    const int first_channel = (row % groups) * channels_per_group;
#pragma unroll
    for (int item = 0; item < Items; ++item) {
      const int j = threadIdx.x + item * kThreads;
      if (j < count) {
        const int channel = first_channel + j / spatial;
        y[base + j] = affine_silu(values[item], anchor, mean_delta, inv_std,
                                  weight, bias, channel);
      }
    }
  }
}

template <typename T>
__global__ __launch_bounds__(kThreads) void large_epilogue(
    const T* __restrict__ x, const T* __restrict__ weight,
    const T* __restrict__ bias, T* __restrict__ y,
    const float* __restrict__ partials, int64_t group_size, int groups,
    int channels_per_group, int spatial, int parts, int tiles, float eps) {
  const int row = static_cast<int>(blockIdx.x) / tiles;
  const int tile = static_cast<int>(blockIdx.x) % tiles;
  const int64_t stat_base = 2 * static_cast<int64_t>(row) * parts;
  const float delta_anchor = partials[stat_base];
  const int64_t base = static_cast<int64_t>(row) * group_size;
  const float anchor = Scalar<T>::load(x[base]);
  __shared__ float scratch[kWaves];
  __shared__ float mean_delta;
  __shared__ float inv_std;

  // Partial means are deltas from the common input anchor. Recenter them
  // for the merge, retaining the global mean as a delta throughout.
  // The final chunk may be shorter than kChunk.
  float delta_sum = 0.0f;
  for (int part = threadIdx.x; part < parts; part += kThreads) {
    const int64_t remaining = group_size - static_cast<int64_t>(part) * kChunk;
    const int count = static_cast<int>(remaining < kChunk ? remaining : kChunk);
    delta_sum += (partials[stat_base + 2 * part] - delta_anchor) * count;
  }
  const float total_delta = block_sum(delta_sum, scratch);
  if (threadIdx.x == 0) mean_delta = delta_anchor + total_delta / group_size;
  __syncthreads();

  float m2_sum = 0.0f;
  for (int part = threadIdx.x; part < parts; part += kThreads) {
    const int64_t remaining = group_size - static_cast<int64_t>(part) * kChunk;
    const int count = static_cast<int>(remaining < kChunk ? remaining : kChunk);
    const float delta = partials[stat_base + 2 * part] - mean_delta;
    m2_sum += partials[stat_base + 2 * part + 1] + delta * delta * count;
  }
  const float total_m2 = block_sum(m2_sum, scratch);
  if (threadIdx.x == 0) inv_std = rsqrtf(total_m2 / group_size + eps);
  __syncthreads();

  const int first_channel = (row % groups) * channels_per_group;
#pragma unroll
  for (int item = 0; item < kOutputItems; ++item) {
    const int64_t j = static_cast<int64_t>(tile) * kOutputTile + threadIdx.x +
                      item * kThreads;
    if (j < group_size) {
      const int channel = first_channel + static_cast<int>(j / spatial);
      y[base + j] = affine_silu(Scalar<T>::load(x[base + j]), anchor, mean_delta,
                                inv_std, weight, bias, channel);
    }
  }
}

struct Shape {
  int64_t group_size;
  int rows;
  int parts;
  int tiles;
};

bool shape_info(int n, int c, int spatial, int groups, Shape* shape) {
  if (n < 0 || c <= 0 || spatial <= 0 || groups <= 0 || c % groups != 0)
    return false;
  const uint64_t group_size = static_cast<uint64_t>(c / groups) * spatial;
  const uint64_t rows = static_cast<uint64_t>(n) * groups;
  const uint64_t parts = (group_size + kChunk - 1) / kChunk;
  const uint64_t tiles = (group_size + kOutputTile - 1) / kOutputTile;
  if (rows > kMaxGrid || parts > kMaxGrid || tiles > kMaxGrid) return false;
  if (rows != 0 && (parts > kMaxGrid / rows || tiles > kMaxGrid / rows ||
                   group_size > static_cast<uint64_t>(INT64_MAX) / rows))
    return false;
  *shape = {static_cast<int64_t>(group_size), static_cast<int>(rows),
            static_cast<int>(parts), static_cast<int>(tiles)};
  return true;
}

template <typename T, int Items>
hipError_t launch_small(const void* x, const void* weight, const void* bias,
                        void* y, const Shape& shape, int c, int spatial,
                        int groups, float eps, hipStream_t stream) {
  hipLaunchKernelGGL((row_kernel<T, Items, false>), dim3(shape.rows),
                    dim3(kThreads), 0, stream, static_cast<const T*>(x),
                    static_cast<const T*>(weight), static_cast<const T*>(bias),
                    static_cast<T*>(y), nullptr, shape.group_size, groups,
                    c / groups, spatial, 1, eps);
  return hipGetLastError();
}

template <typename T>
hipError_t dispatch(const void* x, const void* weight, const void* bias, void* y,
                    const Shape& shape, int c, int spatial, int groups,
                    float eps, hipStream_t stream, void* workspace) {
  if (shape.group_size <= kChunk) {
#define LAUNCH_ITEMS(count)                                                    \
    return launch_small<T, count>(x, weight, bias, y, shape, c, spatial, groups, \
                                   eps, stream)
    if (shape.group_size <= 256) { LAUNCH_ITEMS(1); }
    if (shape.group_size <= 512) { LAUNCH_ITEMS(2); }
    if (shape.group_size <= 1024) { LAUNCH_ITEMS(4); }
    if (shape.group_size <= 2048) { LAUNCH_ITEMS(8); }
    if (shape.group_size <= 4096) { LAUNCH_ITEMS(16); }
    LAUNCH_ITEMS(32);
#undef LAUNCH_ITEMS
  }
  hipLaunchKernelGGL((row_kernel<T, 32, true>),
                    dim3(static_cast<uint32_t>(shape.rows) * shape.parts),
                    dim3(kThreads), 0, stream, static_cast<const T*>(x),
                    static_cast<const T*>(weight), static_cast<const T*>(bias),
                    static_cast<T*>(y), static_cast<float*>(workspace),
                    shape.group_size, groups, c / groups, spatial,
                    shape.parts, eps);
  hipError_t status = hipGetLastError();
  if (status != hipSuccess) return status;
  hipLaunchKernelGGL((large_epilogue<T>),
                    dim3(static_cast<uint32_t>(shape.rows) * shape.tiles),
                    dim3(kThreads), 0, stream, static_cast<const T*>(x),
                    static_cast<const T*>(weight), static_cast<const T*>(bias),
                    static_cast<T*>(y), static_cast<const float*>(workspace),
                    shape.group_size, groups, c / groups, spatial,
                    shape.parts, shape.tiles, eps);
  return hipGetLastError();
}
}  // namespace

extern "C" size_t rdna2_group_norm_silu_workspace_bytes(int n, int c, int spatial,
                                                       int groups) {
  Shape shape;
  if (!shape_info(n, c, spatial, groups, &shape) || shape.group_size <= kChunk)
    return 0;
  return static_cast<size_t>(shape.rows) * shape.parts * 2 * sizeof(float);
}

// dtype: 0 = float32, 1 = float16, 2 = bfloat16 storage with FP32 arithmetic.
// Inputs/output are contiguous NCHW (S = H*W or the product of spatial dims).
// Weight/bias are nullable length-C arrays with the same dtype as x.
// workspace must hold workspace_bytes() bytes and stay alive until stream
// work completes. Use distinct workspace for simultaneously executing calls.
// Returns a hipError_t value. Launches asynchronously on the supplied stream.
// This is an inference forward primitive; the Python layer owns autograd policy.
extern "C" int rdna2_group_norm_silu(
    const void* x, const void* weight, const void* bias, void* y, int n, int c,
    int spatial, int groups, float eps, int dtype, void* stream, void* workspace) {
  Shape shape;
  if (!shape_info(n, c, spatial, groups, &shape) || dtype < 0 || dtype > 2 ||
      !std::isfinite(eps) || eps < 0.0f)
    return static_cast<int>(hipErrorInvalidValue);
  if (shape.rows == 0) return static_cast<int>(hipSuccess);
  if (!x || !y || (shape.group_size > kChunk && !workspace))
    return static_cast<int>(hipErrorInvalidValue);
  const hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);
  if (dtype == 0)
    return static_cast<int>(dispatch<float>(x, weight, bias, y, shape, c, spatial,
                                            groups, eps, hip_stream, workspace));
  if (dtype == 1)
    return static_cast<int>(dispatch<__half>(x, weight, bias, y, shape, c, spatial,
                                             groups, eps, hip_stream, workspace));
  return static_cast<int>(dispatch<BFloat16Storage>(
      x, weight, bias, y, shape, c, spatial, groups, eps, hip_stream, workspace));
}

extern "C" const char* rdna2_hip_error_string(int status) {
  return hipGetErrorString(static_cast<hipError_t>(status));
}
