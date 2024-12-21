// Copyright 2024 yibotongxue

#include "filler.hpp"

#include <algorithm>
#include <ranges>  // NOLINT

#include "common.hpp"

namespace my_tensor {

template <typename T>
void ZeroFiller<T>::FillCPU(TensorPtr<T> tensor) {
  std::ranges::fill(tensor->GetCPUData(), 0);
}

template <typename T>
void ConstantFiller<T>::FillCPU(TensorPtr<T> tensor) {
  std::ranges::fill(tensor->GetCPUData(), val_);
}

template <>
void XavierFiller<>::FillCPU(TensorPtr<> tensor) {
  int n = tensor->GetSize();
  float limit = std::sqrt(6.0f / (n_in_ + n_out_));
  auto gen = MyTensorContext::random_eigine();
  std::uniform_real_distribution<float> dis(-limit, limit);
  auto func = [&dis, &gen]() -> float { return dis(gen); };
  std::ranges::generate(tensor->GetCPUData(), func);
}

template <>
void HeFiller<>::FillCPU(TensorPtr<> tensor) {
  float *data = tensor->GetCPUDataPtr();
  int n = tensor->GetSize();
  float limit = std::sqrt(2.0f / n_);
  auto gen = MyTensorContext::random_eigine();
  std::normal_distribution<float> dis(-limit, limit);
  auto func = [&dis, &gen]() -> float { return dis(gen); };
  std::ranges::generate(tensor->GetCPUData(), func);
}

template class Filler<>;
template class ZeroFiller<>;
template class ConstantFiller<>;
template class XavierFiller<>;
template class HeFiller<>;
}  // namespace my_tensor
