// Copyright 2024 yibotongxue

#include <thrust/device_ptr.h>
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
  auto bottom_ptr = PTR_CAST(bottom[0]->GetGPUDataPtr());
  auto top_ptr = PTR_CAST(top[0]->GetGPUDataPtr());
  thrust::transform(bottom_ptr, bottom_ptr + bottom[0]->GetSize(), top_ptr,
                    SigmoidOperator<T>());
}

template <typename T>
void Sigmoid<T>::BackwardGPU(const std::vector<TensorPtr<T>>& top,
                             const std::vector<TensorPtr<T>>& bottom) {
  CHECK_SAME_SHAPE(top[0], bottom[0])
  auto top_diff_ptr = PTR_CAST(top[0]->GetGPUDiffPtr());
  auto top_data_ptr = PTR_CAST(top[0]->GetGPUDataPtr());
  auto bottom_diff_ptr = PTR_CAST(bottom[0]->GetGPUDiffPtr());
  thrust::transform(top_diff_ptr, top_diff_ptr + top[0]->GetSize(),
                    top_data_ptr, bottom_diff_ptr, SigmoidGradOperator<T>());
}

template class Sigmoid<>;
}  // namespace my_tensor
