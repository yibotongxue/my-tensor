// Copyright 2024 yibotongxue

#include <thrust/transform.h>

#include <vector>

#include "error.hpp"
#include "relu.hpp"
#include "utils.hpp"

namespace my_tensor {

namespace {
template <typename T>
struct ReluOperator {
  __device__ T operator()(T x) { return x > 0 ? x : 0; }
};

template <typename T>
struct ReluGradOperator {
  __device__ T operator()(T data, T diff) { return data > 0 ? diff : 0; }
};
}  // namespace

template <typename T>
void Relu<T>::ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                         const std::vector<TensorPtr<T>>& top) {
  CHECK_SAME_SHAPE(top[0], bottom[0])
  thrust::transform(bottom[0]->GetGPUData().begin(),
                    bottom[0]->GetGPUData().end(), top[0]->GetGPUData().begin(),
                    ReluOperator<T>());
}

template <typename T>
void Relu<T>::BackwardGPU(const std::vector<TensorPtr<T>>& top,
                          const std::vector<TensorPtr<T>>& bottom) {
  CHECK_SAME_SHAPE(top[0], bottom[0])
  thrust::transform(bottom[0]->GetGPUData().begin(),
                    bottom[0]->GetGPUData().end(), top[0]->GetGPUDiff().begin(),
                    bottom[0]->GetGPUDiff().begin(), ReluGradOperator<T>());
}

template class Relu<>;
}  // namespace my_tensor
