// Copyright 2024 yibotongxue

#include "relu.hpp"

#include <algorithm>
#include <vector>

#include "error.hpp"

namespace my_tensor {

template <typename T>
void Relu<T>::CheckTensorCount(const std::vector<TensorPtr<T>>& bottom,
                               const std::vector<TensorPtr<T>>& top) const {
  if (bottom.size() != 1) {
    throw ReluError("The bottom of relu layer should have one tensor.");
  }
  if (top.size() != 1) {
    throw ReluError("The top of relu layer should have one tensor.");
  }
}

template <typename T>
void Relu<T>::Reshape(const std::vector<TensorPtr<T>>& bottom,
                      const std::vector<TensorPtr<T>>& top) const {
  top[0]->Resize(bottom[0]->GetShape());
}

template <typename T>
void Relu<T>::ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                         const std::vector<TensorPtr<T>>& top) {
  std::ranges::transform(
      bottom[0]->GetCPUData(), top[0]->GetCPUData().begin(),
      [](T val) -> T { return std::max(val, static_cast<T>(0)); });
}

template <typename T>
void Relu<T>::BackwardCPU(const std::vector<TensorPtr<T>>& top,
                          const std::vector<TensorPtr<T>>& bottom) {
  std::ranges::transform(
      bottom[0]->GetCPUData(), top[0]->GetCPUDiff(),
      bottom[0]->GetCPUDiff().begin(),
      [](T val1, T val2) -> T { return val1 > 0 ? val2 : 0; });
}

template class Relu<>;
}  // namespace my_tensor
