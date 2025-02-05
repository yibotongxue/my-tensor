// Copyright 2024 yibotongxue

#include "relu.hpp"

#include <algorithm>
#include <vector>

#include "error.hpp"

namespace my_tensor {

template <Arithmetic T>
void Relu<T>::CheckTensorCount(const std::vector<TensorPtr<T>>& bottom,
                               const std::vector<TensorPtr<T>>& top) const {
  if (bottom.size() != 1) {
    throw ReluError("The bottom of relu layer should have one tensor.");
  }
  if (top.size() != 1) {
    throw ReluError("The top of relu layer should have one tensor.");
  }
}

template <Arithmetic T>
void Relu<T>::Reshape(const std::vector<TensorPtr<T>>& bottom,
                      const std::vector<TensorPtr<T>>& top) const {
  top[0]->Resize(bottom[0]->GetShape());
}

template <Arithmetic T>
void Relu<T>::ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                         const std::vector<TensorPtr<T>>& top) {
  auto&& bottom_data = SPAN_DATA(bottom[0], T);
  std::ranges::transform(bottom_data, top[0]->GetCPUDataPtr(), [](T val) -> T {
    return std::max(val, static_cast<T>(0));
  });
}

template <Arithmetic T>
void Relu<T>::BackwardCPU(const std::vector<TensorPtr<T>>& top,
                          const std::vector<TensorPtr<T>>& bottom) {
  auto&& bottom_data = SPAN_DATA(bottom[0], T);
  auto&& top_diff = SPAN_DIFF(top[0], T);
  std::ranges::transform(
      bottom_data, top_diff, bottom[0]->GetCPUDiffPtr(),
      [](T val1, T val2) -> T { return val1 > 0 ? val2 : 0; });
}

template class Relu<float>;
}  // namespace my_tensor
