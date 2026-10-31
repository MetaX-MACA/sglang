#pragma once

#include <sgl_kernel/runtime.cuh>

namespace sglang {

inline int64_t get_device_accessible_ptr(tvm::ffi::TensorView tensor, int64_t device_index) {
  using namespace host;
  RuntimeCheck(device_index >= 0, "Target device index must be non-negative");
  const auto device = tensor.device();
  RuntimeCheck(
      device.device_type == kDLCPU || device.device_type == kDLGPUHost ||
          (device.device_type == kDLCUDA && device.device_id == device_index),
      "Expected registered host memory or a tensor on the target device");
  const tvm::ffi::CUDADeviceGuard guard(static_cast<int>(device_index));
  return reinterpret_cast<int64_t>(runtime::get_device_accessible_ptr(tensor));
}

}  // namespace sglang
