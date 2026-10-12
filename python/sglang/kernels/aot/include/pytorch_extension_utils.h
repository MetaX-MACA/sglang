/*
 * Copyright (c) 2026 by SGLang team.
 * Licensed under the Apache License, Version 2.0.
 */
#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// AOT operators still take at::Tensor arguments. Keep their validation helpers
// local instead of depending on FlashInfer's private PyTorch extension header,
// which is absent from its TVM FFI distributions, including MetaX builds.
#include "pytorch_extension_utils_rocm.h"

#define CHECK_SHAPE(a, b) TORCH_CHECK((a).sizes() == (b).sizes(), #a " and " #b " must have the same shape")
