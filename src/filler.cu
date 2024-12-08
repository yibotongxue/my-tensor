// Copyright 2024 yibotongxue

#include <curand_kernel.h>
#include <thrust/fill.h>

#include "common.hpp"
#include "filler.hpp"

namespace my_tensor {

namespace {
__global__ void AdjustUniformRangeKernel(float *data, size_t size, float a,
                                         float b) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = a + (b - a) * data[idx];
  }
}
}  // namespace

template <typename T>
void ZeroFiller<T>::FillGPU(TensorPtr<T> tensor) {
  T *data = tensor->GetGPUDataPtr();
  CUDA_CHECK(cudaMemset(data, 0, tensor->GetSize() * sizeof(T)));
}

template <typename T>
void ConstantFiller<T>::FillGPU(TensorPtr<T> tensor) {
  thrust::fill(tensor->GetGPUData().begin(), tensor->GetGPUData().end(),
               static_cast<T>(val_));
}

template <>
void XavierFiller<>::FillGPU(TensorPtr<> tensor) {
  int n = tensor->GetSize();
  float limit = std::sqrt(6.0f / (n_in_ + n_out_));
  curandGenerateUniform(MyTensorContext::curand_generator(),
                        tensor->GetGPUDataPtr(), n);
  AdjustUniformRangeKernel<<<CudaGetBlocks(n), kCudaThreadNum>>>(
      tensor->GetGPUDataPtr(), n, -limit, limit);
}

template <>
void HeFiller<>::FillGPU(TensorPtr<> tensor) {
  float *data = tensor->GetGPUDataPtr();
  int n = tensor->GetSize();
  float limit = std::sqrt(2.0f / n_);
  curandGenerateNormal(MyTensorContext::curand_generator(),
                       tensor->GetGPUDataPtr(), n, 0.0f, limit);
}

template class Filler<>;
template class ZeroFiller<>;
template class ConstantFiller<>;
template class XavierFiller<>;
template class HeFiller<>;
}  // namespace my_tensor
