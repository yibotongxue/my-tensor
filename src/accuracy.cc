// Copyright 2024 yibotongxue

#include "accuracy.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <ranges>  // NOLINT
#include <vector>

#include "error.hpp"
#include "layer-factory.hpp"
#include "softmax.hpp"

namespace my_tensor {

template <typename T>
void Accuracy<T>::CheckTensorCount(const std::vector<TensorPtr<T>>& bottom,
                                   const std::vector<TensorPtr<T>>& top) const {
  if (bottom.size() != 2) {
    throw AccuracyError(
        "The bottom of loss layer should has two tensor, one as input and "
        "other as label.");
  }
  if (top.size() != 1) {
    throw AccuracyError("The top of loss layer should has only one tensor.");
  }
}

template <typename T>
void Accuracy<T>::Reshape(const std::vector<TensorPtr<T>>& bottom,
                          const std::vector<TensorPtr<T>>& top) const {
  top[0]->Resize({1});
}

template <typename T>
void Accuracy<T>::LayerSetUp(const std::vector<TensorPtr<T>>& bottom,
                             const std::vector<TensorPtr<T>>& top) {
  if (bottom[0]->GetShape().size() != 2) {
    throw AccuracyError("The input of accuracy should be two dimension.");
  }
  if (bottom[1]->GetShape().size() != 1) {
    throw AccuracyError("The label of accuracy should be one dimension.");
  }
  std::shared_ptr<AccuracyParameter> param =
      std::dynamic_pointer_cast<AccuracyParameter>(this->layer_param_);
  int features = param->features_;
  if (bottom[0]->GetShape()[1] != features) {
    throw AccuracyError("The input feature of accuracy not match the feature.");
  }
  features_ = features;
  batch_size_ = bottom[0]->GetShape()[0];
}

template <typename T>
void Accuracy<T>::ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                             const std::vector<TensorPtr<T>>& top) {
  CheckShape(bottom[0], bottom[1], top[0]);
  const auto& bottom_data = SPAN_DATA(bottom[0], T);
  const auto& label = SPAN_DATA(bottom[1], T);
  auto bottom_data_view = std::views::all(bottom_data);
  int correct{0};
  for (int i : std::views::iota(0, batch_size_)) {
    auto row_view = bottom_data_view | std::views::drop(i * features_) |
                    std::views::take(features_);
    int predict = std::ranges::max_element(row_view) - row_view.begin();
    if (predict == label[i]) {
      correct++;
    }
  }
  top[0]->GetCPUDataPtr()[0] =
      static_cast<T>(correct) / static_cast<T>(batch_size_);
}

template <typename T>
void Accuracy<T>::BackwardCPU(const std::vector<TensorPtr<T>>& top,
                              const std::vector<TensorPtr<T>>& bottom) {
  throw AccuracyError("Unimplemention error.");
}

template <typename T>
void Accuracy<T>::CheckShape(const TensorPtr<T> input, const TensorPtr<T> label,
                             const TensorPtr<T> output) const {
#ifdef DEBUG
  if (input->GetShape().size() != 2) {
    throw AccuracyError(
        "The input of loss with softmax layer should be a two dimension "
        "tensor.");
  }
  if (label->GetShape().size() != 1) {
    throw AccuracyError(
        "The label of accuracy layer should be a one dimension "
        "tensor.");
  }
  if (input->GetShape()[0] != batch_size_) {
    throw AccuracyError(
        "The input batch size not match that of accuracy layer.");
  }
  if (input->GetShape()[1] != channels_) {
    throw AccuracyError("The input channels not match that of accuracy layer.");
  }
  if (label->GetShape()[0] != batch_size_) {
    throw AccuracyError(
        "The label batch size not match that of accuracy layer.");
  }
  if (output->GetShape().size() != 1 || output->GetShape()[0] != 1) {
    throw AccuracyError(
        "The output of accuracy layer should be a one dimension "
        "contains only one element.");
  }
#endif  // DEBUG
}

template class Accuracy<>;

}  // namespace my_tensor
