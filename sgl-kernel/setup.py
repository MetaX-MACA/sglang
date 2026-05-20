# Copyright 2025 SGLang Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import shutil
import sys
from pathlib import Path

import torch
import flashinfer
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()
maca_path = Path(os.getenv("MACA_PATH")).resolve()
flashinfer_path = Path(flashinfer.__file__).parent.resolve()

if "bdist_wheel" in sys.argv and "--plat-name" not in sys.argv:
    sys.argv.extend(["--plat-name", "manylinux2014_x86_64"])


def _get_cuda_version():
    if torch.version.cuda:
        return tuple(map(int, torch.version.cuda.split(".")))
    return (0, 0)


def _get_device_sm():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    return 0


def _get_version():
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


operator_namespace = "sgl_kernel"

deepgemm = root / "3rdparty" / "deepgemm"
include_dirs = [
    root / "include",
    root / "csrc",
    maca_path / "include",
    flashinfer_path / "data" / "include",
    flashinfer_path / "data" / "include" / "gemm",
    flashinfer_path / "data" / "csrc",
    "cublas",
]


class CustomBuildPy(build_py):
    def run(self):
        self.copy_deepgemm_to_build_lib()
        self.make_jit_include_symlinks()
        build_py.run(self)

    def make_jit_include_symlinks(self):
        # Make symbolic links of third-party include directories
        build_include_dir = os.path.join(self.build_lib, "deep_gemm/include")
        os.makedirs(build_include_dir, exist_ok=True)

        third_party_include_dirs = [
            cutlass.resolve() / "include" / "cute",
            cutlass.resolve() / "include" / "cutlass",
        ]

        for d in third_party_include_dirs:
            dirname = str(d).split("/")[-1]
            src_dir = d
            dst_dir = f"{build_include_dir}/{dirname}"
            assert os.path.exists(src_dir)
            if os.path.exists(dst_dir):
                assert os.path.islink(dst_dir)
                os.unlink(dst_dir)
            os.symlink(src_dir, dst_dir, target_is_directory=True)

    def copy_deepgemm_to_build_lib(self):
        """
        This function copies DeepGemm to python's site-packages
        """
        dst_dir = os.path.join(self.build_lib, "deep_gemm")
        os.makedirs(dst_dir, exist_ok=True)

        # Copy deepgemm/deep_gemm to the build directory
        src_dir = os.path.join(str(deepgemm.resolve()), "deep_gemm")

        # Remove existing directory if it exists
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)

        # Copy the directory
        shutil.copytree(src_dir, dst_dir)


nvcc_flags = [
    # "-DNDEBUG",
    # f"-DOPERATOR_NAMESPACE={operator_namespace}",
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-gencode=arch=compute_75,code=sm_75",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
    "-std=c++17",
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
    "-DCUTLASS_VERSIONS_GENERATED",
    "-DCUTE_USE_PACKED_TUPLE=1",
    "-DCUTLASS_TEST_LEVEL=0",
    "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "--ptxas-options=-v",
    "--expt-relaxed-constexpr",
    # "--expt-extended-lambda"
    # "--threads=32"
    # "-Xcompiler=-Wconversion",
    # "-Xcompiler=-fno-strict-aliasing",
    "-DFLASHINFER_ENABLE_BF16",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]
nvcc_flags_fp8 = [
    "-DFLASHINFER_ENABLE_FP8",
    "-DFLASHINFER_ENABLE_FP8_E4M3",
    "-DFLASHINFER_ENABLE_FP8_E5M2",
]

sources = [
    "csrc/allreduce/custom_all_reduce.cu",
    # "csrc/attention/cascade.cu",
    "csrc/attention/merge_attn_states.cu",
    # "csrc/attention/cutlass_mla_kernel.cu",
    "csrc/attention/vertical_slash_index.cu",
    "csrc/elementwise/activation.cu",
    "csrc/elementwise/fused_add_rms_norm_kernel.cu",
    # "csrc/elementwise/rope.cu",
    "csrc/elementwise/pos_enc.cu",
    # "csrc/elementwise/cast.cu",
    "csrc/gemm/awq_kernel.cu",
    "csrc/moe/moe_align_kernel.cu",
    "csrc/moe/moe_fused_gate.cu",
    "csrc/moe/moe_topk_softmax_kernels.cu",
    "csrc/moe/prepare_moe_input.cu",
    "csrc/speculative/eagle_utils.cu",
    "csrc/speculative/speculative_sampling.cu",
    "csrc/speculative/packbit.cu",
    "csrc/grammar/apply_token_bitmask_inplace_cuda.cu",
    "csrc/kvcacheio/transfer.cu",
    # "csrc/memory/store.cu",
    "csrc/common_extension.cc",
    # "csrc/moe/masked_group_gemm_kernel.cu",
    # "csrc/moe/masked_group_gemm_kernel_opt.cu",
    "csrc/moe/moe_fused_gate_opt.cu",
    # "csrc/elementwise/fused_rotary_emb.cu",
    # "csrc/quantization/int8_quant_kernels.cu",
    # "csrc/quantization/quantize_kernel.cu",
    # "csrc/gemm/marlin/gptq_marlin.cu",
    # "csrc/gemm/marlin/gptq_marlin_repack.cu",
    # "csrc/gemm/marlin/awq_marlin_repack.cu",
    # "csrc/gemm/gptq/gptq_kernel.cu",
    # "csrc/moe/marlin_moe_wna16/ops.cu",
    # "csrc/gemm/dsv3_fused_a_gemm.cu",
    # "csrc/gemm/dsv3_router_gemm_bf16_out.cu",
    # "csrc/gemm/dsv3_router_gemm_entry.cu",
    # "csrc/gemm/dsv3_router_gemm_float_out.cu",
    "csrc/quantization/int8_quant_kernels.cu",
    # "csrc/quantization/quantize_kernel.cu",
    # "csrc/elementwise/fused_layernorm_dynamic_per_token_quant_custom.cu",
    # "csrc/elementwise/fused_rotary_emb.cu",
    "csrc/moe/cutlass_moe/scaled_mm_c2x.cu",
    "csrc/mamba/causal_conv1d.cu",
    "csrc/cutlass_w8a8/scaled_mm_entry.cu",
    "csrc/elementwise/topk.cu",
    "csrc/elementwise/concat_mla.cu",
    "csrc/elementwise/copy.cu",
    "csrc/moe/moe_sum.cu",
    "csrc/moe/moe_sum_reduce.cu",
    "csrc/speculative/ngram_utils.cu",
    "csrc/memory/weak_ref_tensor.cpp",
    "csrc/moe/kimi_k2_moe_fused_gate.cu",
    "csrc/moe/fused_qknorm_rope_kernel.cu",
    "csrc/moe/moe_topk_sigmoid_kernels.cu",
    # "csrc/sgl_diffusion/elementwise/timestep_embedding.cu"
    # "csrc/quantization/gguf/gguf_kernel.cu"
]

enable_bf16 = os.getenv("SGL_KERNEL_ENABLE_BF16", "1") == "1"
enable_fp8 = os.getenv("SGL_KERNEL_ENABLE_FP8", "0") == "1"
enable_fp4 = os.getenv("SGL_KERNEL_ENABLE_FP4", "0") == "1"
enable_sm90a = os.getenv("SGL_KERNEL_ENABLE_SM90A", "0") == "1"
enable_sm100a = os.getenv("SGL_KERNEL_ENABLE_SM100A", "0") == "1"
cuda_version = _get_cuda_version()
sm_version = _get_device_sm()

if torch.cuda.is_available():
    if cuda_version >= (12, 0) and sm_version >= 90:
        nvcc_flags.append("-gencode=arch=compute_90a,code=sm_90a")
    if cuda_version >= (12, 8) and sm_version >= 100:
        nvcc_flags.append("-gencode=arch=compute_100,code=sm_100")
        nvcc_flags.append("-gencode=arch=compute_100a,code=sm_100a")
    else:
        nvcc_flags.append("-use_fast_math")
    if sm_version >= 90:
        nvcc_flags.extend(nvcc_flags_fp8)
    if sm_version >= 80:
        nvcc_flags.append("-DFLASHINFER_ENABLE_BF16")
else:
    # compilation environment without GPU
    if enable_sm100a:
        nvcc_flags.append("-gencode=arch=compute_100a,code=sm_100a")
    if enable_sm90a:
        nvcc_flags.append("-gencode=arch=compute_90a,code=sm_90a")
    if enable_fp4:
        nvcc_flags.append("-DENABLE_NVFP4=1")
    if enable_fp8:
        nvcc_flags.extend(nvcc_flags_fp8)
    if enable_bf16:
        nvcc_flags.append("-DFLASHINFER_ENABLE_BF16")

for flag in [
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
]:
    try:
        torch.utils.cpp_extension.COMMON_NVCC_FLAGS.remove(flag)
    except ValueError:
        pass

cxx_flags = ["-O3"]
libraries = ["c10", "torch", "torch_python", "mctlassEx"]
extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", "-L/usr/lib/x86_64-linux-gnu", "-Lmaca_path/lib"]

TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])
version_ge_1_1 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 0):
    version_ge_1_1 = ["-DVERSION_GE_1_1"]
version_ge_1_3 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    version_ge_1_3 = ["-DVERSION_GE_1_3"]
version_ge_1_5 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 4):
    version_ge_1_5 = ["-DVERSION_GE_1_5"]
version_dependent_macros = version_ge_1_1 + version_ge_1_3 + version_ge_1_5

compile_flags_for_mctlass_grouped_gemm_int8 = [
    "-mllvm",
    "-metaxgpu-disable-bsm-offset=0",
    "-mllvm",
    "-structurizecfg-skip-uniform-regions=true",
    "-mllvm",
    "-metaxgpu-igroup=true",
    "-mllvm",
    "-misched-postra=true"
]

ext_modules = [
    CUDAExtension(
        name="sgl_kernel.common_ops",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "nvcc": nvcc_flags,
            "cxx": cxx_flags,
        },
        libraries=libraries,
        extra_link_args=extra_link_args,
        py_limited_api=False,
    ),
    CUDAExtension(
        name="grouped_gemm_cuda",
        sources=[
            "csrc/moe/grouped_gemm.cpp",
            "csrc/moe/grouped_gemm_cuda.cu",
        ],
        include_dirs=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")],
        extra_compile_args={
            "cxx": ["-O3"] + version_dependent_macros,
            "cucc": [
                "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
            ] + version_dependent_macros,
        },
    ),
    # CUDAExtension(
    #     name="moe_fused_w4a16",
    #     sources=[
    #         "csrc/moe/moe_fused_w4a16.cpp",
    #         "csrc/moe/moe_fused_w4a16_cuda.cu",
    #     ],
    #     include_dirs=include_dirs,
    #     extra_compile_args={
    #         "cxx": ["-O3"] + version_dependent_macros,
    #         "cucc": [
    #             "-O3",
    #             "-U__CUDA_NO_HALF_OPERATORS__",
    #             "-U__CUDA_NO_HALF_CONVERSIONS__",
    #             "--expt-relaxed-constexpr",
    #             "--expt-extended-lambda",
    #         ] + version_dependent_macros,
    #     },
    #     libraries=libraries,
    #     extra_link_args=extra_link_args,
    # ),
#     CUDAExtension(
#         name="grouped_gemm_mctlass_int8",
#         sources=[
#             "csrc/moe/grouped_gemm_mctlass_int8.cpp",
#             "csrc/moe/grouped_gemm_mctlass_int8_cuda.cu",
#         ],
#         include_dirs=include_dirs,
#         extra_compile_args={
#             "cxx": ["-O3"] + version_dependent_macros,
#             "cucc": [
#                 "-O3",
#                 "-U__CUDA_NO_HALF_OPERATORS__",
#                 "-U__CUDA_NO_HALF_CONVERSIONS__",
#                 "--expt-relaxed-constexpr",
#                 "--expt-extended-lambda",
#             ] + version_dependent_macros,
#             "nvcc": nvcc_flags + compile_flags_for_mctlass_grouped_gemm_int8,
#         },
#     ),
]

version = "0.3.21"
maca_ai_version = os.environ.get('MACA_AI_VERSION', '0.0.0.0')
version += "+maca"+maca_ai_version

setup(
    name="sgl-kernel",
    version=version,
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=True),
        # "build_py": CustomBuildPy,
    },
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
