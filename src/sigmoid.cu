#include <sigmoid.cuh>
#include <utils.cuh>

namespace my_tensor {
namespace {
__global__ void CudaForward(const float* bottom_data, float* top_data, int n) {
  CUDA_KERNEL_LOOP(i, n) {
    *(top_data + i) = 1.0f / (1.0f + std::exp(-*(bottom_data + i)));
  }
}

__global__ void CudaBackward(
  const float* top_diff, const float* top_data, float* bottom_diff, int n) {
  CUDA_KERNEL_LOOP(i, n) {
    *(bottom_diff + i) =
      *(top_diff + i) * *(top_data + i) * (1 - *(top_data + i));
  }
}
}  // namespace

void Sigmoid::Forward(
  const std::shared_ptr<Tensor> bottom, std::shared_ptr<Tensor> top) {
  if (bottom->OnCPU() && top->OnGPU() ||
      bottom->OnGPU() && top->OnCPU()) {
    throw std::runtime_error("Device no match.");
  }
  const float* bottom_data = bottom->GetData();
  float* top_data = top->GetMutableData();
  int n = bottom->GetSize();
  if (bottom->OnCPU()) {
    for (int i = 0; i < n; i++) {
      *(top_data + i) = 1.0f / (1.0f + std::exp(-*(bottom_data + i)));
    }
    bottom_data = nullptr;
    top_data = nullptr;
  } else {
    CudaForward<<<CudaGetBlocks(n), kCudaThreadNum>>>(bottom_data, top_data, n);
    cudaDeviceSynchronize();
  }
}

void Sigmoid::Backward(
  const std::shared_ptr<Tensor> top, std::shared_ptr<Tensor> bottom) {
  if (bottom->OnCPU() && top->OnGPU() ||
      bottom->OnGPU() && top->OnCPU()) {
    throw std::runtime_error("Device no match.");
  }
  const float *top_diff = top->GetDiff();
  const float *top_data = top->GetData();
  float *bottom_diff = bottom->GetMutableDiff();
  int n = top->GetSize();
  if (bottom->OnCPU()) {
    for (int i = 0; i < n; i++) {
      *(bottom_diff + i) =
        *(top_diff + i) * *(top_data + i) * (1 - *(top_data + i));
    }
  } else {
    CudaBackward<<<CudaGetBlocks(n), kCudaThreadNum>>>(
      top_diff, top_data, bottom_diff, n);
    cudaDeviceSynchronize();
  }
}
}  // namespace my_tensor
