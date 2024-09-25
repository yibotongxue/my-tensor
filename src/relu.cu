#include <relu.cuh>
#include <utils.cuh>

#include <stdexcept>
#include <numeric>

namespace my_tensor {
__global__ void CudaForward(const float* bottom_data, float* top_data, int n) {
  CUDA_KERNEL_LOOP(i, n) {
    *(top_data + i) = *(bottom_data + i) > 0 ? *(bottom_data + i) : 0;
  }
}

void Relu::Forward(const std::shared_ptr<Tensor> bottom, std::shared_ptr<Tensor> top) {
  if (bottom->OnCPU() && top->OnGPU() ||
      bottom->OnGPU() && top->OnCPU()) {
    throw std::runtime_error("Device not match");
  }
  auto shape = bottom->GetShape();
  int n = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  const float* bottom_data = bottom->GetData();
  float* top_data = top->GetMutableData();
  if (bottom->OnCPU()) {
    for (int i = 0; i < n; i++) {
      *(top_data + i) = *(bottom_data + i) > 0 ? *(bottom_data + i) : 0;
    }
  } else {
    CudaForward<<<CudaGetBlocks(n), kCudaThreadNum>>>(bottom_data, top_data, n);
    cudaDeviceSynchronize();
  }
}

__global__ void CudaBackward(
  const float *top_diff, const float *bottom_data, float *bottom_diff, int n) {
  CUDA_KERNEL_LOOP(i, n) {
    *(bottom_diff + i) = *(bottom_data + i) > 0 ? *(top_diff + i) : 0;
  }
}

void Relu::Backward(const std::shared_ptr<Tensor> top, std::shared_ptr<Tensor> bottom) {
  if (top->OnCPU() && bottom->OnGPU() ||
      top->OnGPU() && bottom->OnCPU()) {
    throw std::runtime_error("Device not match");
  }
  auto shape = top->GetShape();
  int n = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  const float* bottom_data = bottom->GetData();
  const float* top_diff = top->GetDiff();
  float* bottom_diff = bottom->GetMutableDiff();
  if (top->OnCPU()) {
    for (int i = 0; i < n; i++) {
      *(bottom_diff + i) = *(bottom_data + i) > 0 ? *(top_diff + i) : 0;
    }
  } else {
    CudaBackward<<<CudaGetBlocks(n), kCudaThreadNum>>>(top_diff, bottom_data, bottom_diff, n);
    cudaDeviceSynchronize();
  }
}
} // namespace my_tensor
