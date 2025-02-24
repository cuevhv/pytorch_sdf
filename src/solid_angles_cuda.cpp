#include <torch/extension.h>
#include <vector>

// Function declaration (implemented in CUDA)
torch::Tensor solid_angles_cuda(torch::Tensor points, torch::Tensor triangles, float thresh = 1e-8);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solid_angles_cuda", &solid_angles_cuda, "Compute solid angles (CUDA)");
}