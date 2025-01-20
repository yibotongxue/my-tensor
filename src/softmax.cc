// Copyright 2024 yibotongxue

#include "softmax.hpp"

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include "error.hpp"
#include "layer-parameter.hpp"
#include "tensor.hpp"

namespace my_tensor {

template <Arithmetic T>
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

template <Arithmetic T>
void Softmax<T>::Reshape(const std::vector<TensorPtr<T>>& bottom,
                         const std::vector<TensorPtr<T>>& top) const {
  top[0]->Resize({batch_size_, channels_});
}

template <Arithmetic T>
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

template <Arithmetic T>
void Softmax<T>::ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                            const std::vector<TensorPtr<T>>& top) {
  CheckShape(bottom[0], top[0]);
  auto&& bottom_data = SPAN_DATA(bottom[0], T);
  auto&& top_data = SPAN_DATA(top[0], T);
  auto bottom_view = std::views::all(bottom_data);
  for (int i = 0; i < batch_size_; i++) {  // for each row
    auto sub_view = bottom_view | std::views::drop(i * channels_) |
                    std::views::take(channels_);
    auto max_postion = std::ranges::max_element(sub_view);
    T max_value = *max_postion;
    auto exp_view = sub_view | std::views::transform([max_value](T val) -> T {
                      return static_cast<T>(std::exp(val - max_value));
                    });
    T sum_value =
        std::accumulate(exp_view.begin(), exp_view.end(), T(0), std::plus<T>());
    auto norm_view = exp_view | std::views::transform([sum_value](T val) -> T {
                       return static_cast<T>(val / sum_value);
                     });
    std::ranges::copy(norm_view, top_data.begin() + i * channels_);
  }
}

template <Arithmetic T>
void Softmax<T>::CheckShape(const TensorPtr<T> bottom,
                            const TensorPtr<T> top) const {
#ifdef DEBUG
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
#endif  // DEBUG
}

template class Softmax<float>;

}  // namespace my_tensor
