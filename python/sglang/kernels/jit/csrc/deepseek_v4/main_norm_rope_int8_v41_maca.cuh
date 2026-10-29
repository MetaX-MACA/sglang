#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
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

// Layout constants, the group abs-max reduction, the round-to-nearest int8 pack
// and the quantize + store tail are shared with store_int8_v41_maca.cuh and live in
// <sgl_kernel/deepseek_v4/int8_v41.cuh>. Reference:
// fused_deepseek_v41_qnorm_rope_kv_rope_int8_insert in mcoplib.

// 8 warps per block: block-per-token work-item dispatch (K kernel).
#define INT8_K_KERNEL __global__ __launch_bounds__(deepseek_v4::int8_v41::kBlockSize, 8)

struct FusedKNormRopeInt8Params {
  const void* __restrict__ kv;          // (B, kHeadDim) DType
  const void* __restrict__ kv_weight;   // (kHeadDim,) DType
  const float* __restrict__ freqs_cis;  // (max_pos, kRopeDim) fp32
  const void* __restrict__ positions;   // (B,) PosT
  const int32_t* __restrict__ out_loc;  // (B,) int32 -> cache slot id
  void* __restrict__ kvcache;           // int8 cache
  int64_t kv_stride_batch;
  uint32_t batch_size;
  float eps;
};

// ============================================================================
// K kernel: rmsnorm + RoPE + INT8 quantize (per-32 dim) + paged store.
// ============================================================================
template <typename DType, int64_t kHeadDim, int64_t kRopeDim, typename PosT, int32_t kPageBits, bool kUsePDL>
INT8_K_KERNEL void fused_k_norm_rope_int8_flashmla(const FusedKNormRopeInt8Params params) {
  using namespace device;

  constexpr int64_t kVecSize = 2;
  constexpr uint32_t kRopeWarp = deepseek_v4::int8_v41::kNumWarps - 1;
  constexpr int64_t kInt8PageBytes = static_cast<int64_t>(deepseek_v4::int8_v41::kBytesPerToken) << kPageBits;

  static_assert(kHeadDim == deepseek_v4::int8_v41::kBlockSize * kVecSize);
  static_assert(kRopeDim == kWarpThreads * kVecSize);
  static_assert(kHeadDim - kRopeDim == kRopeWarp * kWarpThreads * kVecSize);

  using Storage = AlignedVector<DType, kVecSize>;
  using Float2 = AlignedVector<float, kVecSize>;

  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpThreads;
  const auto lane_id = tx % kWarpThreads;
  const auto work_id = blockIdx.x;
  if (work_id >= params.batch_size) return;

  const auto input_ptr = static_cast<const DType*>(params.kv) + work_id * params.kv_stride_batch;
  const auto position = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[work_id]);
  const auto out_loc = params.out_loc[work_id];
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  PDLWaitPrimary<kUsePDL>();
  Float2 data, freq;

  // ======== Part 1: RMSNorm ========
  {
    __shared__ float partial_sums[deepseek_v4::int8_v41::kNumWarps];

    Storage input_vec, weight_vec;
    input_vec.load(input_ptr, tx);
    weight_vec.load(params.kv_weight, tx);
    if (warp_id == kRopeWarp) freq.load(freqs_cis, lane_id);

    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto x = cast<float>(input_vec[i]);
      sum_of_squares += x * x;
    }
    const auto warp_sum = warp::reduce_sum(sum_of_squares);
    if (lane_id == 0) partial_sums[warp_id] = warp_sum;
    __syncthreads();
    sum_of_squares = warp::reduce_sum<deepseek_v4::int8_v41::kNumWarps>(partial_sums[lane_id % deepseek_v4::int8_v41::kNumWarps]);
    const auto norm_factor = math::rsqrt(sum_of_squares / kHeadDim + params.eps);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto x = cast<float>(input_vec[i]);
      const auto w = cast<float>(weight_vec[i]);
      data[i] = x * norm_factor * w;
    }
  }

  if (out_loc < 0) return;

  const int32_t page = out_loc >> kPageBits;
  const int32_t offset = out_loc & ((1 << kPageBits) - 1);

  PDLTriggerSecondary<kUsePDL>();

  // ======== Part 2: RoPE on warp 7 ========
  if (warp_id == kRopeWarp) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto freq_real = freq[0];
    const auto freq_imag = freq[1];
    data[0] = x_real * freq_real - x_imag * freq_imag;
    data[1] = x_real * freq_imag + x_imag * freq_real;
  }

  // ======== Part 3: INT8 quantize + store (shared tail) ========
  const auto page_ptr = static_cast<uint8_t*>(params.kvcache) + page * kInt8PageBytes;
  const auto data_ptr = page_ptr + offset * deepseek_v4::int8_v41::kDataBytesPerToken;
  const auto scale_ptr = page_ptr + (static_cast<int64_t>(deepseek_v4::int8_v41::kDataBytesPerToken) << kPageBits)
                       + offset * deepseek_v4::int8_v41::kScaleBytesPerToken;

  deepseek_v4::int8_v41::quantize_store_row(tx, data[0], data[1], data_ptr, scale_ptr);
}

// ============================================================================
// Kernel wrapper struct for JIT compilation
// ============================================================================
template <typename DType, int64_t kHeadDim, int64_t kRopeDim, uint32_t kPageSize, bool kUsePDL>
struct FusedKNormRopeInt8FlashMLAKernel {
  static constexpr int32_t kLogPageSize = std::countr_zero(kPageSize);
  static constexpr int64_t kInt8PageBytes = static_cast<int64_t>(deepseek_v4::int8_v41::kBytesPerToken) * kPageSize;
  static constexpr int64_t kPageStride = kInt8PageBytes;

  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");
  static_assert(1 << kLogPageSize == kPageSize);
  static_assert(kHeadDim == 512 && kRopeDim == 64, "INT8 FlashMLA layout requires (512, 64)");

  // PosT dispatch template
  template <typename PosT>
  static constexpr auto kernel = fused_k_norm_rope_int8_flashmla<
      DType, kHeadDim, kRopeDim, PosT, kLogPageSize, kUsePDL>;

  static void forward(
      const tvm::ffi::TensorView kv,
      const tvm::ffi::TensorView kv_weight,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView out_loc,
      const tvm::ffi::TensorView kvcache,
      float eps) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, kHeadDim})
        .with_strides({-1, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(kv);
    TensorMatcher({kHeadDim})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(kv_weight);
    TensorMatcher({-1, kRopeDim})
        .with_dtype<float>()
        .with_device(device_)
        .verify(freqs_cis);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({B})
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device_)
        .verify(positions);
    TensorMatcher({B})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(out_loc);
    TensorMatcher({-1, -1})
        .with_strides({kPageStride, 1})
        .with_dtype<int8_t>()  // kernel expects int8 layout
        .with_device(device_)
        .verify(kvcache);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    if (batch_size == 0) return;

    const auto params = FusedKNormRopeInt8Params{
        .kv = kv.data_ptr(),
        .kv_weight = kv_weight.data_ptr(),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .positions = positions.data_ptr(),
        .out_loc = static_cast<const int32_t*>(out_loc.data_ptr()),
        .kvcache = kvcache.data_ptr(),
        .kv_stride_batch = kv.stride(0),
        .batch_size = batch_size,
        .eps = eps,
    };

    // PosT dispatch
    const auto k_int32 = kernel<int32_t>;
    const auto k_int64 = kernel<int64_t>;
    const auto k = pos_dtype.is_type<int32_t>() ? k_int32 : k_int64;
    LaunchKernel(batch_size, deepseek_v4::int8_v41::kBlockSize, device_.unwrap())
        .enable_pdl(kUsePDL)(k, params);
  }
};

}  // namespace sglang

