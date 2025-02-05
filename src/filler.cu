// Copyright 2024 yibotongxue

#include <thrust/device_ptr.h>

#include "filler.hpp"
#include "utils.hpp"

namespace my_tensor {

template <Arithmetic T>
void ZeroFiller<T>::FillGPU(TensorPtr<T> tensor) {
  T *data = tensor->GetGPUDataPtr();
  CUDA_CHECK(cudaMemset(data, 0, tensor->GetSize() * sizeof(T)));
}

template <Arithmetic T>
void ConstantFiller<T>::FillGPU(TensorPtr<T> tensor) {
  auto data_ptr = PTR_CAST(tensor->GetGPUDataPtr());
  thrust::fill(data_ptr, data_ptr + tensor->GetSize(), val_);
}

__global__ static void XavierFillerKernel(float *data, float limit, int n) {
  CUDA_KERNEL_LOOP(i, n) {
    curandState state;
    curand_init(1234 + i, i, 0, &state);
    data[i] = curand_uniform(&state) * 2 * limit - limit;
  }
}

template <>
void XavierFiller<>::FillGPU(TensorPtr<float> tensor) {
  int n = tensor->GetSize();
  float limit = std::sqrt(6.0f / (n_in_ + n_out_));
  XavierFillerKernel<<<CudaGetBlocks(n), kCudaThreadNum>>>(
      tensor->GetGPUDataPtr(), limit, n);
}

__global__ static void HeFillKernel(float *data, float limit, int n) {
  CUDA_KERNEL_LOOP(i, n) {
    curandState state;
    curand_init(1234 + i, i, 0, &state);
    data[i] = curand_normal(&state) * limit;
  }
}

template <>
void HeFiller<>::FillGPU(TensorPtr<float> tensor) {
  float *data = tensor->GetGPUDataPtr();
  int n = tensor->GetSize();
  float limit = std::sqrt(2.0f / n_);
  HeFillKernel<<<CudaGetBlocks(n), kCudaThreadNum>>>(data, limit, n);
}

template class Filler<float>;
template class ZeroFiller<float>;
template class ConstantFiller<float>;
template class XavierFiller<float>;
template class HeFiller<float>;
}  // namespace my_tensor
