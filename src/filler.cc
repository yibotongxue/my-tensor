// Copyright 2024 yibotongxue

#include "filler.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <ranges>  // NOLINT

#include "common.hpp"

namespace my_tensor {

template <typename T>
void ZeroFiller<T>::FillCPU(TensorPtr<T> tensor) {
  T *data = tensor->GetCPUDataPtr();
  memset(data, 0, tensor->GetSize() * sizeof(T));
}

template <typename T>
void ConstantFiller<T>::FillCPU(TensorPtr<T> tensor) {
  std::ranges::fill(tensor->GetCPUData(), static_cast<T>(val_));
}

template <>
void XavierFiller<>::FillCPU(TensorPtr<> tensor) {
  int n = tensor->GetSize();
  float limit = std::sqrt(6.0f / (n_in_ + n_out_));
  std::uniform_real_distribution<float> dist(-limit, limit);
  auto gen = MyTensorContext::random_eigine();
  auto func = [&gen, &dist]() -> float { return dist(gen); };
  std::ranges::generate(tensor->GetCPUData(), func);
}

template <>
void HeFiller<>::FillCPU(TensorPtr<> tensor) {
  float *data = tensor->GetGPUDataPtr();
  int n = tensor->GetSize();
  float limit = std::sqrt(2.0f / n_);
  std::normal_distribution<float> dist(0.0f, limit);
  auto gen = MyTensorContext::random_eigine();
  auto func = [&gen, &dist]() -> float { return dist(gen); };
  std::ranges::generate(tensor->GetCPUData(), func);
}

template class Filler<>;
template class ZeroFiller<>;
template class ConstantFiller<>;
template class XavierFiller<>;
template class HeFiller<>;
}  // namespace my_tensor
