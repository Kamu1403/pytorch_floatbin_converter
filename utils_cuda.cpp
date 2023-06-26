#include <torch/extension.h>

#include <vector>

// CUDA forward declarations
void float2bin_cuda(
    torch::Tensor from,
    torch::Tensor to);

void bin2float_cuda(
    torch::Tensor from,
    torch::Tensor to);
// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void float2bin(
    torch::Tensor from,
    torch::Tensor to) {
  CHECK_INPUT(from);
  CHECK_INPUT(to);

  float2bin_cuda(from,to);
  return;
}

void bin2float(
    torch::Tensor from,
    torch::Tensor to) {
  CHECK_INPUT(from);
  CHECK_INPUT(to);

  bin2float_cuda(from,to);
  return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("float2bin", &float2bin, "Utils float2bin (CUDA)");
  m.def("bin2float", &bin2float, "Utils bin2float (CUDA)");
}