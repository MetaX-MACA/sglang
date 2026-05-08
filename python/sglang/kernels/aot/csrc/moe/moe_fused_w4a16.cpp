#include <torch/extension.h>

int64_t mctlass_moe_w4a16_gemm_kernel_mnk(int64_t num_valid_tokens, int64_t N, int64_t K, int64_t group);
void mctlass_fused_moe_kernel_w4a16(
    torch::Tensor const& a, torch::Tensor const& b, torch::Tensor& c, torch::Tensor const& b_scales, torch::Tensor const& zp_b,
    torch::Tensor const& moe_weight, torch::Tensor const& token_ids, torch::Tensor const& expert_ids, 
    torch::Tensor const& num_tokens_post_padded, int64_t N, int64_t K, int64_t EM, int64_t num_valid_tokens, int64_t topk, bool mul_routed_weight);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mctlass_moe_w4a16_gemm_kernel_mnk", &mctlass_moe_w4a16_gemm_kernel_mnk, "gemm_kernel_mnk.");
  m.def("mctlass_fused_moe_kernel_w4a16", &mctlass_fused_moe_kernel_w4a16, "fused_moe_w4a16");
}
