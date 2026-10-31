#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {

namespace device::hisparse {

/// NOTE: We call nope+rope as a "value" here.
/// Paged C4 cache layout:
/// VALUE 0, VALUE 1, ..., VALUE 63,
/// SCALE 0, SCALE 1, ..., SCALE 63,
/// [Padding to align to 576 bytes]
inline constexpr int64_t kPageSize = 64;
inline constexpr int64_t kPageBits = 6;  // log2(kPageSize)
inline constexpr int64_t kValueBytes = 576;
inline constexpr int64_t kScaleBytes = 8;
/// NOTE: FlashMLA requires each page to be aligned to 576 bytes
inline constexpr int64_t kItemBytes = kValueBytes + kScaleBytes;
inline constexpr int64_t kPageBytes = host::div_ceil(kItemBytes * kPageSize, 576) * 576;
inline constexpr int64_t kScaleOffset = kValueBytes * kPageSize;

struct PointerInfo {
  int64_t* value_ptr;
  int64_t* scale_ptr;
};

SGL_DEVICE PointerInfo get_pointer_paged(void* cache, int32_t index) {
  using namespace device;
  static_assert(1 << kPageBits == kPageSize);
  const int32_t page_num = index >> kPageBits;
  const int32_t page_offset = index & (kPageSize - 1);
  const auto page_ptr = pointer::offset(cache, page_num * kPageBytes);
  const auto value_ptr = pointer::offset(page_ptr, page_offset * kValueBytes);
  const auto scale_ptr = pointer::offset(page_ptr, kScaleOffset + page_offset * kScaleBytes);
  return {static_cast<int64_t*>(value_ptr), static_cast<int64_t*>(scale_ptr)};
}

// BF16 pages are a flat run of 512-element rows; FP8 V4 pages keep
// values and scales in separate runs and include page-end alignment padding.
SGL_DEVICE void transfer_item(
    void* dst_cache, void* src_cache, const int32_t dst_index, const int32_t src_index,
    const int64_t item_bytes = kItemBytes) {
#if defined(USE_ROCM) || defined(__MACACC__)
  constexpr int kWaveSize = 64;
#else
  constexpr int kWaveSize = 32;
#endif
  const int lane = threadIdx.x % kWaveSize;
  if (item_bytes == 1024) {
    const auto src = static_cast<const int64_t*>(src_cache) + static_cast<int64_t>(src_index) * 128;
    auto dst = static_cast<int64_t*>(dst_cache) + static_cast<int64_t>(dst_index) * 128;
    for (int i = lane; i < 128; i += kWaveSize) dst[i] = src[i];
    return;
  }
  const auto [dst_value, dst_scale] = get_pointer_paged(dst_cache, dst_index);
  const auto [src_value, src_scale] = get_pointer_paged(src_cache, src_index);
  // Treat the split runs as one 73-word copy, with one owner per word.
  for (int i = lane; i < 73; i += kWaveSize) {
    if (i < 72) dst_value[i] = src_value[i];
    else *dst_scale = *src_scale;
  }
}

}  // namespace device::hisparse

}  // namespace sglang
