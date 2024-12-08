// Copyright 2024 yibotongxue

#include "sigmoid.hpp"

#include <algorithm>
#include <vector>

#include "error.hpp"

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

template <typename T>
void Sigmoid<T>::Reshape(const std::vector<TensorPtr<T>>& bottom,
                         const std::vector<TensorPtr<T>>& top) const {
  top[0]->Resize(bottom[0]->GetShape());
}

template <typename T>
void Sigmoid<T>::ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                            const std::vector<TensorPtr<T>>& top) {
  std::ranges::transform(
      bottom[0]->GetCPUData(), top[0]->GetCPUData().begin(), [](T val) -> T {
        return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-val));
      });
}

template <typename T>
void Sigmoid<T>::BackwardCPU(const std::vector<TensorPtr<T>>& top,
                             const std::vector<TensorPtr<T>>& bottom) {
  std::ranges::transform(top[0]->GetCPUDiff(), top[0]->GetCPUData(),
                         bottom[0]->GetCPUDiff().begin(),
                         [](T diff, T data) -> T {
                           return diff * data * (static_cast<T>(1) - data);
                         });
}

template class Sigmoid<>;
}  // namespace my_tensor
