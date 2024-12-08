// Copyright 2024 yibotongxue

#include <thrust/transform.h>

#include <vector>

#include "error.hpp"
#include "sigmoid.hpp"
#include "utils.hpp"

namespace my_tensor {
namespace {
template <typename T>
struct SigmoidOperator {
  __device__ T operator()(T x) {
    return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
  }
};

template <typename T>
struct SigmoidGradOperator {
  __device__ T operator()(T top_diff, T top_data) {
    return top_diff * top_data * (1 - top_data);
  }
};
}  // namespace

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
