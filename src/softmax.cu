// Copyright 2024 yibotongxue

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include "error.h"
#include "layer-parameter.hpp"
#include "softmax.cuh"
#include "tensor.cuh"

namespace my_tensor {

template <typename T>
void Softmax<T>::CheckTensorCount(const std::vector<TensorPtr<T>>& bottom,
                                  const std::vector<TensorPtr<T>>& top) const {
  if (bottom.size() != 1) {
    throw SoftmaxError(
        "The bottom of convolution layer should have one tensor.");
  }
  if (top.size() != 1) {
    throw SoftmaxError("The top of convolution layer should have one tensor.");
  }
}

template <typename T>
void Softmax<T>::LayerSetUp(const std::vector<TensorPtr<T>>& bottom,
                            const std::vector<TensorPtr<T>>& top) {
  if (bottom[0]->GetShape().size() != 2) {
    throw SoftmaxError("The bottom of softmax should be two dimension.");
  }
  std::shared_ptr<SoftmaxParameter> param =
      std::dynamic_pointer_cast<SoftmaxParameter>(this->layer_param_);
  channels_ = param->channels_;
  if (bottom[0]->GetShape()[1] != channels_) {
    throw SoftmaxError(
        "The channels of bottom of softmax not match the layer.");
  }
  batch_size_ = bottom[0]->GetShape()[0];
}

template <typename T>
void Softmax<T>::ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                            const std::vector<TensorPtr<T>>& top) {
  CheckShape(bottom[0], top[0]);
  const auto& bottom_data = bottom[0]->GetCPUData();
  auto& top_data = top[0]->GetCPUData();
  for (int i = 0; i < batch_size_; i++) {
    T max_value = *std::max_element(bottom_data.begin() + i * channels_,
                                    bottom_data.begin() + (i + 1) * channels_);
    std::transform(bottom_data.begin() + i * channels_,
                   bottom_data.begin() + (i + 1) * channels_,
                   top_data.begin() + i * channels_, [max_value](T val) -> T {
                     return static_cast<T>(std::exp(val - max_value));
                   });
    T sum_value = std::accumulate(top_data.begin() + i * channels_,
                                  top_data.begin() + (i + 1) * channels_, T(0),
                                  std::plus<T>());
    std::transform(top_data.begin() + i * channels_,
                   top_data.begin() + (i + 1) * channels_,
                   top_data.begin() + i * channels_, [sum_value](T val) -> T {
                     return static_cast<T>(val / sum_value);
                   });
  }
}

template <typename T>
void Softmax<T>::CheckShape(const TensorPtr<T> bottom,
                            const TensorPtr<T> top) const {
  if (bottom->GetShape().size() != 2) {
    throw SoftmaxError(
        "The bottom of softmax layer should be a two dimension tensor.");
  }
  if (top->GetShape().size() != 2) {
    throw SoftmaxError(
        "The top of softmax layer should be a two dimension tensor.");
  }
  CHECK_SAME_SHAPE(bottom, top)
  if (bottom->GetShape()[0] != batch_size_) {
    throw SoftmaxError(
        "The batch size of bottom of softmax layer not match layer.");
  }
  if (bottom->GetShape()[1] != channels_) {
    throw SoftmaxError(
        "The channels size of bottom of softmax layer not match layer.");
  }
}

template class Softmax<>;

}  // namespace my_tensor
