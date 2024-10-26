// Copyright 2024 yibotongxue

#include <relu.cuh>
#include <utils.cuh>

#include <thrust/transform.h>

namespace my_tensor {

namespace {
template <typename T>
struct ReluOperator {
  __device__ __host__ T operator()(T x) { return x > 0 ? x : 0; }
};

template <typename T>
struct ReluGradOperator {
  __device__ __host__ T operator()(T data, T diff) {
    return data > 0 ? diff : 0;
  }
};
}  // namespace

template <typename T>
void Relu<T>::ForwardCPU(const TensorPtr<T> bottom, TensorPtr<T> top) {
  CHECK_SAME_SHAPE(bottom, top)
  thrust::transform(bottom->GetCPUData().begin(), bottom->GetCPUData().end(),
                    top->GetCPUData().begin(), ReluOperator<T>());
}

template <typename T>
void Relu<T>::BackwardCPU(const TensorPtr<T> top, TensorPtr<T> bottom) {
  CHECK_SAME_SHAPE(bottom, top)
  thrust::transform(bottom->GetCPUData().begin(), bottom->GetCPUData().end(),
                    top->GetCPUDiff().begin(), bottom->GetCPUDiff().begin(),
                    ReluGradOperator<T>());
}

template <typename T>
void Relu<T>::ForwardGPU(const TensorPtr<T> bottom, TensorPtr<T> top) {
  CHECK_SAME_SHAPE(top, bottom)
  thrust::transform(bottom->GetGPUData().begin(), bottom->GetGPUData().end(),
                    top->GetGPUData().begin(), ReluOperator<T>());
}

template <typename T>
void Relu<T>::BackwardGPU(const TensorPtr<T> top, TensorPtr<T> bottom) {
  CHECK_SAME_SHAPE(top, bottom)
  thrust::transform(bottom->GetGPUData().begin(), bottom->GetGPUData().end(),
                    top->GetGPUDiff().begin(), bottom->GetGPUDiff().begin(),
                    ReluGradOperator<T>());
}

template class Relu<>;
}  // namespace my_tensor
