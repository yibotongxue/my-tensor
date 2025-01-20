// Copyright 2024 yibotongxue

#include "filler.hpp"

#include <algorithm>
#include <ranges>  // NOLINT

#include "common.hpp"

namespace my_tensor {

template <Arithmetic T>
void ZeroFiller<T>::FillCPU(TensorPtr<T> tensor) {
  std::ranges::fill(SPAN_DATA(tensor, T), 0);
}

template <Arithmetic T>
void ConstantFiller<T>::FillCPU(TensorPtr<T> tensor) {
  std::ranges::fill(SPAN_DATA(tensor, T), val_);
}

template <>
void XavierFiller<>::FillCPU(TensorPtr<float> tensor) {
  int n = tensor->GetSize();
  float limit = std::sqrt(6.0f / (n_in_ + n_out_));
  auto& gen = MyTensorContext::random_eigine();
  std::uniform_real_distribution<float> dis(-limit, limit);
  auto func = [&dis, &gen]() -> float { return dis(gen); };
  std::ranges::generate(SPAN_DATA(tensor, float), func);
  // PRINT_DATA(tensor, float);
}

template <>
void HeFiller<>::FillCPU(TensorPtr<float> tensor) {
  int n = tensor->GetSize();
  float limit = std::sqrt(2.0f / n_);
  auto& gen = MyTensorContext::random_eigine();
  std::normal_distribution<float> dis(0, limit);
  auto func = [&dis, &gen]() -> float { return dis(gen); };
  std::ranges::generate(SPAN_DATA(tensor, float), func);
}

template class Filler<>;
template class ZeroFiller<>;
template class ConstantFiller<>;
template class XavierFiller<>;
template class HeFiller<>;
}  // namespace my_tensor
