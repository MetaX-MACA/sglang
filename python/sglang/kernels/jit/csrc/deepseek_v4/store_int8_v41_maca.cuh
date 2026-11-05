#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/int8_v41.cuh>

#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>
#include <type_traits>

namespace sglang {

// ============================================================================
// V4.1 INT8 KV cache layout (per token, 576 bytes):
//   [0, 512)   INT8 data  ??16 groups ? 32 dims (448 nope + 64 rope)
//   [512, 576) FP32 scales ??16 ? 4 bytes
//
// Kernel: BF16 input (N, 512) ??INT8 quantize all 512 dims ??paged store.
// 8 warps, 256 threads per block, 1 block per token.
// Each warp handles 64 elements (2 ? 32-dim groups).
// ============================================================================

// Layout constants, the group abs-max reduction, the round-to-nearest int8 pack
// and the quantize + store tail are shared with main_norm_rope_int8_v41_maca.cuh; see
// <sgl_kernel/deepseek_v4/int8_v41.cuh>.

struct FusedStoreCacheInt8V41Param {
  const void* __restrict__ input;   // (N, 512) BF16
  void* __restrict__ cache;         // INT8 paged cache
  const void* __restrict__ indices; // (N,) token slot ids
};

// ============================================================================
// Kernel: BF16 input ??INT8 quantize (per-32 dim) + paged store.
// 256 threads (8 warps), 1 block per token.
// Each thread processes 2 elements (BF16 load ??INT8 pack).
// ============================================================================
template <typename Float, typename IndicesT, uint32_t kPageBits, bool kUsePDL>
__global__ void fused_store_flashmla_cache_int8_v41(
    const FusedStoreCacheInt8V41Param param) {
  using namespace device;

  constexpr int64_t kPageBytes =
      static_cast<int64_t>(deepseek_v4::int8_v41::kBytesPerToken) << kPageBits;

  const auto input = param.input;
  auto cache = param.cache;
  const auto indices = param.indices;
  const auto bid = static_cast<int>(blockIdx.x);
  const auto tid = static_cast<int>(threadIdx.x);

  PDLWaitPrimary<kUsePDL>();

  // Prefetch the index
  const auto index = static_cast<const IndicesT*>(indices)[bid];

  // Guard: DP padding / out-of-window sentinel can produce negative indices.
  if (index < 0) {
    PDLTriggerSecondary<kUsePDL>();
    return;
  }

  // Load 2 BF16 values per thread (256 threads ? 2 = 512 dims)
  using Float2 = packed_t<Float>;
  const auto elems = static_cast<const Float2*>(input)[tid + bid * 256];
  const auto [x, y] = cast<fp32x2_t>(elems);

  // Each warp handles 64 dims = 2 ? 32-dim groups
  // Thread tid in warp wid: global dim indices = wid*64 + (tid%32)*2, wid*64 + (tid%32)*2+1
  // Group assignment: (wid*2 + (tid%32)/16) ??which 32-dim group
  const int32_t page = index >> kPageBits;
  const int32_t offset = index & ((1 << kPageBits) - 1);
  const auto page_ptr = static_cast<uint8_t*>(cache) + page * kPageBytes;
  const auto data_ptr = page_ptr + offset * deepseek_v4::int8_v41::kDataBytesPerToken;
  const auto scale_ptr =
      page_ptr + (static_cast<int64_t>(deepseek_v4::int8_v41::kDataBytesPerToken) << kPageBits)
      + offset * deepseek_v4::int8_v41::kScaleBytesPerToken;

  // Shared per-32 quantize + int8/paged-store tail (see int8_v41.cuh): two
  // values per thread, one fp32 scale per 32-dim group.
  deepseek_v4::int8_v41::quantize_store_row(tid, x, y, data_ptr, scale_ptr);

  PDLTriggerSecondary<kUsePDL>();
}

// ============================================================================
// Kernel wrapper struct for JIT compilation
// ============================================================================
template <typename Float, typename IndicesT, typename StoreT, uint32_t kPageSize, bool kUsePDL>
struct FusedStoreCacheFlashMLAInt8V41Kernel {
  static constexpr int32_t kLogSize = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes =
      static_cast<int64_t>(deepseek_v4::int8_v41::kBytesPerToken) * kPageSize;
  static constexpr auto kernel =
      fused_store_flashmla_cache_int8_v41<Float, IndicesT, kLogSize, kUsePDL>;

  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");
  static_assert(1 << kLogSize == kPageSize);
  static_assert(std::is_same_v<StoreT, int8_t>, "StoreT must be int8_t for V4.1 INT8");

  static void run(
      tvm::ffi::TensorView input,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView indices) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({N, deepseek_v4::int8_v41::kHeadDim})  // input: BF16 (N, 512)
        .with_strides({deepseek_v4::int8_v41::kHeadDim, 1})  // must be contiguous along dim 1
        .with_dtype<Float>()
        .with_device(device_)
        .verify(input);
    TensorMatcher({-1, -1})  // cache: INT8 paged
        .with_strides({kPageBytes, 1})
        .with_dtype<int8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({N})  // indices
        .with_dtype<IndicesT>()
        .with_device(device_)
        .verify(indices);

    const auto num_tokens = static_cast<int>(N.unwrap());
    const auto params = FusedStoreCacheInt8V41Param{
        .input = input.data_ptr(),
        .cache = cache.data_ptr(),
        .indices = indices.data_ptr(),
    };
    const auto num_blocks = num_tokens;
    LaunchKernel(num_blocks, deepseek_v4::int8_v41::kBlockSize, device_.unwrap())
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang

