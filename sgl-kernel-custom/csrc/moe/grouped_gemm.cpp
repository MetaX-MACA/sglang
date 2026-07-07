#include <torch/extension.h>

namespace grouped_gemm {

void GroupedGemm(torch::Tensor a,
         torch::Tensor b,
         torch::Tensor c,
         torch::Tensor batch_sizes,
         bool trans_a, bool trans_b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gmm", &GroupedGemm, "Grouped GEMM.");
}

} // namespace grouped_gemm
