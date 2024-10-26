// Copyright 2024 yibotongxue

#include <sigmoid.cuh>
#include <utils.cuh>
#include <thrust/transform.h>

namespace my_tensor {
namespace {
template <typename T>
struct SigmoidOperator {
  __host__ __device__ T operator()(T x) {
    return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
  }
};

template <typename T>
struct SigmoidGradOperator {
  __host__ __device__ T operator()(T top_diff, T top_data) {
    return top_diff * top_data * (1 - top_data);
  }
};
}  // namespace

template <typename T>
void Sigmoid<T>::ForwardCPU(const TensorPtr<T> bottom, TensorPtr<T> top) {
  CHECK_SAME_SHAPE(bottom, top)
  thrust::transform(bottom->GetCPUData().begin(), bottom->GetCPUData().end(),
    top->GetCPUData().begin(), SigmoidOperator<T>());
}

template <typename T>
void Sigmoid<T>::BackwardCPU(const TensorPtr<T> top, TensorPtr<T> bottom) {
  CHECK_SAME_SHAPE(bottom, top)
  thrust::transform(top->GetCPUDiff().begin(), top->GetCPUDiff().end(),
    top->GetCPUData().begin(), bottom->GetCPUDiff().begin(),
    SigmoidGradOperator<T>());
}

template <typename T>
void Sigmoid<T>::ForwardGPU(const TensorPtr<T> bottom, TensorPtr<T> top) {
  CHECK_SAME_SHAPE(top, bottom)
  thrust::transform(bottom->GetGPUData().begin(), bottom->GetGPUData().end(),
    top->GetGPUData().begin(), SigmoidOperator<T>());
}

template <typename T>
void Sigmoid<T>::BackwardGPU(const TensorPtr<T> top, TensorPtr<T> bottom) {
  CHECK_SAME_SHAPE(top, bottom)
  thrust::transform(top->GetGPUDiff().begin(), top->GetGPUDiff().end(),
    top->GetGPUData().begin(), bottom->GetGPUDiff().begin(),
    SigmoidGradOperator<T>());
}

template class Sigmoid<>;
}  // namespace my_tensor
