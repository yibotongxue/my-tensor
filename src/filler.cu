// Copyright 2024 yibotongxue

#include "filler.hpp"

namespace my_tensor {

template <typename T>
void ZeroFiller<T>::Fill(TensorPtr<T> tensor) {
  T *data = tensor->GetGPUDataPtr();
  CUDA_CHECK(cudaMemset(data, 0, tensor->GetSize() * sizeof(T)));
}

template <typename T>
void ConstantFiller<T>::Fill(TensorPtr<T> tensor) {
  auto &tensor_data = tensor->GetGPUData();
  thrust::fill(tensor_data.begin(), tensor_data.end(), val_);
}

__global__ static void XavierFillerKernel(float *data, float limit, int n) {
  CUDA_KERNEL_LOOP(i, n) {
    curandState state;
    curand_init(1234 + i, i, 0, &state);
    data[i] = curand_uniform(&state) * 2 * limit - limit;
  }
}

template <>
void XavierFiller<>::Fill(TensorPtr<> tensor) {
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
void HeFiller<>::Fill(TensorPtr<> tensor) {
  float *data = tensor->GetGPUDataPtr();
  int n = tensor->GetSize();
  float limit = std::sqrt(2.0f / n_);
  HeFillKernel<<<CudaGetBlocks(n), kCudaThreadNum>>>(data, limit, n);
}

template class Filler<>;
template class ZeroFiller<>;
template class ConstantFiller<>;
template class XavierFiller<>;
template class HeFiller<>;
}  // namespace my_tensor
