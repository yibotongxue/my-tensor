// Copyright 2024 yibotongxue

#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include <vector>

#include "error.hpp"
#include "relu.hpp"
#include "utils.hpp"

namespace my_tensor {

namespace {
template <Arithmetic T>
struct ReluOperator {
  __device__ T operator()(T x) { return x > 0 ? x : 0; }
};

template <Arithmetic T>
struct ReluGradOperator {
  __device__ T operator()(T data, T diff) { return data > 0 ? diff : 0; }
};
}  // namespace

template <Arithmetic T>
void Relu<T>::ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                         const std::vector<TensorPtr<T>>& top) {
  CHECK_SAME_SHAPE(top[0], bottom[0])
  auto bottom_ptr = PTR_CAST(bottom[0]->GetGPUDataPtr());
  auto top_ptr = PTR_CAST(top[0]->GetGPUDataPtr());
  thrust::transform(bottom_ptr, bottom_ptr + bottom[0]->GetSize(), top_ptr,
                    ReluOperator<T>());
}

template <Arithmetic T>
void Relu<T>::BackwardGPU(const std::vector<TensorPtr<T>>& top,
                          const std::vector<TensorPtr<T>>& bottom) {
  CHECK_SAME_SHAPE(top[0], bottom[0])
  auto bottom_data_ptr = PTR_CAST(bottom[0]->GetGPUDataPtr());
  auto top_diff_ptr = PTR_CAST(top[0]->GetGPUDiffPtr());
  auto bottom_diff_ptr = PTR_CAST(bottom[0]->GetGPUDiffPtr());
  thrust::transform(bottom_data_ptr, bottom_data_ptr + bottom[0]->GetSize(),
                    top_diff_ptr, bottom_diff_ptr, ReluGradOperator<T>());
}

template class Relu<>;
}  // namespace my_tensor
