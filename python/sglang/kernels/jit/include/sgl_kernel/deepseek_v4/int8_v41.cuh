#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <cstdint>

// Small helpers shared by the DeepSeek-V4.1 INT8 KV-cache writers
// (`main_norm_rope_int8_v41_maca.cuh` and `store_int8_v41_maca.cuh`). Both quantize a
// 512-wide latent into the V4.1 INT8 paged layout with one fp32 scale per
// 32-dim group, then scatter the packed row; only the prologue differs (one
// does rmsnorm + RoPE, the other takes an already-normed/rotated input). The
// constants, the group abs-max reduction, the round-to-nearest int8 pack and
// the quantize + store tail are identical, so they live here instead of being
// repeated in each kernel.
//
// All functions are `SGL_DEVICE` (= `__forceinline__ __device__`) so including
// this header in multiple translation units is ODR-safe. The math is kept
// byte-for-byte the same as the two kernels it replaced: `scale = absmax / 127`
// and `inv_scale = 127 / absmax` are recomputed (not `1 / scale`) so the packed
// bytes stay bitwise-identical.

namespace sglang {

namespace deepseek_v4::int8_v41 {

// ============================================================================
// V4.1 INT8 KV cache layout (per token, 576 bytes):
//   [0, 512)   INT8 data: 448 nope + 64 rope, 16 groups of 32 dims (rope is
//              quantized into the data region, unlike the V4 bf16-rope layout).
//   [512, 576) FP32 scales: 16 scales, one per 32-dim group.
// ============================================================================
constexpr int kNopeDim = 448;
constexpr int kRopeDim = 64;
constexpr int kHeadDim = kNopeDim + kRopeDim;               // 512
constexpr int kQuantBlock = 32;
constexpr int kNumGroups = kHeadDim / kQuantBlock;          // 16
constexpr int kDataBytesPerToken = kHeadDim;                // 512
constexpr int kScaleBytesPerToken = kNumGroups * 4;         // 64
constexpr int kBytesPerToken = kDataBytesPerToken + kScaleBytesPerToken;  // 576

// Shared block/warp configuration: 256 threads (8 warps), one block per token,
// and the last warp additionally carries the RoPE pair.
constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;  // 8

// Reduce the absolute max across one 32-dim group. A group is covered by 16
// consecutive threads (each owns a 2-element pack), so this is a 4-step
// butterfly over masks 1, 2, 4, 8; on return every lane of the group holds the
// same max.
SGL_DEVICE float group_max_abs(float val) {
  val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, 1));
  val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, 2));
  val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, 4));
  val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, 8));
  return val;
}

// Pack two floats to INT8 with round-to-nearest and clamp to [-128, 127].
SGL_DEVICE char2 pack_round(float x, float y) {
  auto clip = [](float v) -> int8_t {
    return static_cast<int8_t>(
        device::math::max(device::math::min(roundf(v), 127.0f), -128.0f));
  };
  return char2{clip(x), clip(y)};
}

// The quantize + paged-store tail shared by both writers. `tx` is the thread's
// index within the token row (0..255, owning the `tx`-th 2-element pack),
// `v0` / `v1` are its two normed (and roped) fp32 values, and `data_ptr` /
// `scale_ptr` are the token row's 512 B int8 data base and 64 B fp32 scale
// base inside the page. The caller has already skipped negative `out_loc` /
// index rows, so every lane of the warp reaches the group reduction here.
SGL_DEVICE void quantize_store_row(
    uint32_t tx, float v0, float v1, uint8_t* data_ptr, uint8_t* scale_ptr) {
  const auto local_max = fmaxf(fabs(v0), fabs(v1));
  const auto absmax = fmaxf(group_max_abs(local_max), 1e-4f);
  const auto scale = absmax / 127.0f;
  const auto inv_scale = 127.0f / absmax;

  reinterpret_cast<char2*>(data_ptr)[tx] = pack_round(v0 * inv_scale, v1 * inv_scale);

  // One fp32 scale per 32-dim group: lane 0 of each 16-thread group writes it.
  if ((tx & 15) == 0) {
    reinterpret_cast<float*>(scale_ptr)[tx >> 4] = scale;
  }
}

}  // namespace deepseek_v4::int8_v41

}  // namespace sglang
