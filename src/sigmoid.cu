// Copyright 2024 yibotongxue

#include <thrust/transform.h>

#include <vector>

#include "error.h"
#include "sigmoid.cuh"
#include "utils.cuh"

namespace my_tensor {
template <typename T>
void Sigmoid<T>::CheckTensorCount(const std::vector<TensorPtr<T>>& bottom,
                                  const std::vector<TensorPtr<T>>& top) const {
  if (bottom.size() != 1) {
    throw SigmoidError("The bottom of sigmoid layer should have one tensor.");
  }
  if (top.size() != 1) {
    throw SigmoidError("The top of sigmoid layer should have one tensor.");
  }
}

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
void Sigmoid<T>::ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                            const std::vector<TensorPtr<T>>& top) {
  CHECK_SAME_SHAPE(bottom[0], top[0])
  thrust::transform(bottom[0]->GetCPUData().begin(),
                    bottom[0]->GetCPUData().end(), top[0]->GetCPUData().begin(),
                    SigmoidOperator<T>());
}

template <typename T>
void Sigmoid<T>::BackwardCPU(const std::vector<TensorPtr<T>>& top,
                             const std::vector<TensorPtr<T>>& bottom) {
  CHECK_SAME_SHAPE(bottom[0], top[0])
  thrust::transform(top[0]->GetCPUDiff().begin(), top[0]->GetCPUDiff().end(),
                    top[0]->GetCPUData().begin(),
                    bottom[0]->GetCPUDiff().begin(), SigmoidGradOperator<T>());
}

template <typename T>
void Sigmoid<T>::ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                            const std::vector<TensorPtr<T>>& top) {
  CHECK_SAME_SHAPE(top[0], bottom[0])
  thrust::transform(bottom[0]->GetGPUData().begin(),
                    bottom[0]->GetGPUData().end(), top[0]->GetGPUData().begin(),
                    SigmoidOperator<T>());
}

template <typename T>
void Sigmoid<T>::BackwardGPU(const std::vector<TensorPtr<T>>& top,
                             const std::vector<TensorPtr<T>>& bottom) {
  CHECK_SAME_SHAPE(top[0], bottom[0])
  thrust::transform(top[0]->GetGPUDiff().begin(), top[0]->GetGPUDiff().end(),
                    top[0]->GetGPUData().begin(),
                    bottom[0]->GetGPUDiff().begin(), SigmoidGradOperator<T>());
}

template class Sigmoid<>;
}  // namespace my_tensor
