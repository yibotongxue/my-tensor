// Copyright 2024 yibotongxue

#include "sigmoid.hpp"

#include <algorithm>
#include <vector>

#include "error.hpp"

namespace my_tensor {
template <Arithmetic T>
void Sigmoid<T>::CheckTensorCount(const std::vector<TensorPtr<T>>& bottom,
                                  const std::vector<TensorPtr<T>>& top) const {
  if (bottom.size() != 1) {
    throw SigmoidError("The bottom of sigmoid layer should have one tensor.");
  }
  if (top.size() != 1) {
    throw SigmoidError("The top of sigmoid layer should have one tensor.");
  }
}

template <Arithmetic T>
void Sigmoid<T>::Reshape(const std::vector<TensorPtr<T>>& bottom,
                         const std::vector<TensorPtr<T>>& top) const {
  top[0]->Resize(bottom[0]->GetShape());
}

template <Arithmetic T>
void Sigmoid<T>::ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                            const std::vector<TensorPtr<T>>& top) {
  auto&& bottom_data = SPAN_DATA(bottom[0], T);
  std::ranges::transform(bottom_data, top[0]->GetCPUDataPtr(), [](T val) -> T {
    return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-val));
  });
}

template <Arithmetic T>
void Sigmoid<T>::BackwardCPU(const std::vector<TensorPtr<T>>& top,
                             const std::vector<TensorPtr<T>>& bottom) {
  auto&& top_diff = SPAN_DIFF(top[0], T);
  auto&& top_data = SPAN_DATA(top[0], T);
  std::ranges::transform(top_diff, top_data, bottom[0]->GetCPUDiffPtr(),
                         [](T diff, T data) -> T {
                           return diff * data * (static_cast<T>(1) - data);
                         });
}

template class Sigmoid<float>;
}  // namespace my_tensor
