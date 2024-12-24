// Copyright 2024 yibotongxue

#include "loss-with-softmax.hpp"

#include <algorithm>
#include <memory>
#include <ranges>  // NOLINT
#include <vector>

#include "error.hpp"
#include "layer-factory.hpp"
#include "softmax.hpp"

namespace my_tensor {

template <typename T>
void LossWithSoftmax<T>::CheckTensorCount(
    const std::vector<TensorPtr<T>>& bottom,
    const std::vector<TensorPtr<T>>& top) const {
  if (bottom.size() != 2) {
    throw LossWithSoftmaxError(
        "The bottom of loss layer should has two tensor, one as input and "
        "other as label.");
  }
  if (top.size() != 1) {
    throw LossWithSoftmaxError(
        "The top of loss layer should has only one tensor.");
  }
}

template <typename T>
void LossWithSoftmax<T>::Reshape(const std::vector<TensorPtr<T>>& bottom,
                                 const std::vector<TensorPtr<T>>& top) const {
  top[0]->Resize({1});
}

template <typename T>
void LossWithSoftmax<T>::LayerSetUp(const std::vector<TensorPtr<T>>& bottom,
                                    const std::vector<TensorPtr<T>>& top) {
  if (bottom[0]->GetShape().size() != 2) {
    throw LossWithSoftmaxError(
        "The input of loss with softmax should be two dimension.");
  }
  if (bottom[1]->GetShape().size() != 1) {
    throw LossWithSoftmaxError(
        "The label of loss with softmax should be one dimension.");
  }
  std::shared_ptr<LossWithSoftmaxParameter> param =
      std::dynamic_pointer_cast<LossWithSoftmaxParameter>(this->layer_param_);
  channels_ = param->channels_;
  if (bottom[0]->GetShape()[1] != channels_) {
    throw SoftmaxError(
        "The channels of bottom of softmax not match the layer.");
  }
  batch_size_ = bottom[0]->GetShape()[0];
  softmax_bottom_ = {bottom[0]};
  softmax_top_ = {std::make_shared<Tensor<T>>(bottom[0]->GetShape())};
  auto softmax_param = CreateLayerParameter("Softmax");
  softmax_param->name_ = param->name_;
  auto temp_param = std::dynamic_pointer_cast<SoftmaxParameter>(softmax_param);
  temp_param->channels_ = param->channels_;
  softmax_.reset();
  softmax_ = CreateLayer<T>(temp_param);
  softmax_->SetUp(softmax_bottom_, softmax_top_);
}

template <typename T>
void LossWithSoftmax<T>::ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                                    const std::vector<TensorPtr<T>>& top) {
  CheckShape(bottom[0], bottom[1], top[0]);
  softmax_->ForwardCPU(softmax_bottom_, softmax_top_);
  auto&& softmax_top_data = softmax_top_[0]->GetCPUDataSpan();
  auto&& label_data = bottom[1]->GetCPUDataSpan();
  auto&& top_data = top[0]->GetCPUDataSpan();
  T result{0};
  for (int i = 0; i < batch_size_; i++) {
    result -= std::log(softmax_top_data[i * channels_ + label_data[i]]);
  }
  top_data[0] = result / batch_size_;
}

template <typename T>
void LossWithSoftmax<T>::BackwardCPU(const std::vector<TensorPtr<T>>& top,
                                     const std::vector<TensorPtr<T>>& bottom) {
  CheckShape(bottom[0], bottom[1], top[0]);
  auto&& softmax_top_data = softmax_top_[0]->GetCPUDataSpan();
  auto&& label_data = bottom[1]->GetCPUDataSpan();
  auto&& bottom_diff = bottom[0]->GetCPUDiffSpan();
  float batch_size = static_cast<float>(batch_size_);
  std::ranges::transform(softmax_top_data, bottom_diff.begin(),
                         [batch_size](T val) -> T { return val / batch_size; });
  for (int i = 0; i < batch_size_; i++) {
    int idx = i * channels_ + label_data[i];
    bottom_diff[idx] -= 1.0f / batch_size;
  }
}

template <typename T>
void LossWithSoftmax<T>::CheckShape(const TensorPtr<T> input,
                                    const TensorPtr<T> label,
                                    const TensorPtr<T> output) const {
#ifdef DEBUG
  if (input->GetShape().size() != 2) {
    throw LossWithSoftmaxError(
        "The input of loss with softmax layer should be a two dimension "
        "tensor.");
  }
  if (label->GetShape().size() != 1) {
    throw LossWithSoftmaxError(
        "The label of loss with softmax layer should be a one dimension "
        "tensor.");
  }
  if (input->GetShape()[0] != batch_size_) {
    throw LossWithSoftmaxError(
        "The input batch size not match that of loss with softmax layer.");
  }
  if (input->GetShape()[1] != channels_) {
    throw LossWithSoftmaxError(
        "The input channels not match that of loss with softmax layer.");
  }
  if (label->GetShape()[0] != batch_size_) {
    throw LossWithSoftmaxError(
        "The label batch size not match that of loss with softmax layer.");
  }
  if (output->GetShape().size() != 1 || output->GetShape()[0] != 1) {
    throw LossWithSoftmaxError(
        "The output of loss with softmax layer should be a one dimension "
        "contains only one element.");
  }
#endif  // DEBUG
}

template class LossWithSoftmax<>;

}  // namespace my_tensor
