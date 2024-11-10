// Copyright 2024 yibotongxue

#include <thrust/count.h>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <memory>
#include <ranges>
#include <vector>

#include "error.h"
#include "layer-factory.cuh"
#include "loss-with-softmax.cuh"
#include "softmax.cuh"

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
  int expect_size = 1;
  if (top[0]->GetSize() != expect_size) {
    throw LossWithSoftmaxError(
        "The top size not match loss with softmax layer.");
  }
  top[0]->Reshape({1});
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
float LossWithSoftmax<T>::GetAccuracy(const TensorPtr<T> label) const {
  const TensorPtr<T> predict =
      std::dynamic_pointer_cast<Softmax<T>>(softmax_)->GetPredict();
  thrust::device_vector<int> temp(batch_size_, 0);
  thrust::transform(
      predict->GetGPUData().begin(), predict->GetGPUData().end(),
      label->GetGPUData().begin(), temp.begin(),
      [] __device__(T val1, T val2) -> int { return val1 == val2 ? 1 : 0; });
  int same_count = thrust::reduce(temp.begin(), temp.end());
  return static_cast<float>(same_count) / static_cast<float>(batch_size_);
}

template <typename T>
void LossWithSoftmax<T>::ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                                    const std::vector<TensorPtr<T>>& top) {
  CheckShape(bottom[0], bottom[1], top[0]);
  softmax_->ForwardCPU(softmax_bottom_, softmax_top_);
  const auto& softmax_top_data = softmax_top_[0]->GetCPUData();
  const auto& label_data = bottom[1]->GetCPUData();
  auto& top_data = top[0]->GetCPUData();
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
  const auto& softmax_top_data = softmax_top_[0]->GetCPUData();
  const auto& label_data = bottom[1]->GetCPUData();
  auto& bottom_diff = bottom[0]->GetCPUDiff();
  thrust::copy(softmax_top_data.begin(), softmax_top_data.end(),
               bottom_diff.begin());
  for (int i = 0; i < batch_size_; i++) {
    int idx = i * channels_ + label_data[i];
    bottom_diff[idx] -= 1;
  }
}

template <typename T>
void LossWithSoftmax<T>::ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                                    const std::vector<TensorPtr<T>>& top) {
  CheckShape(bottom[0], bottom[1], top[0]);
  softmax_->ForwardGPU(softmax_bottom_, softmax_top_);
  const T* softmax_top_data = softmax_top_[0]->GetGPUDataPtr();
  const T* label_data = bottom[1]->GetGPUDataPtr();
  auto& top_data = top[0]->GetGPUData();
  thrust::device_vector<T> temp_data(batch_size_);
  int channels = channels_;
  thrust::transform(
      thrust::counting_iterator(0), thrust::counting_iterator(batch_size_),
      temp_data.begin(),
      [softmax_top_data, label_data, channels] __device__(int i) -> T {
        return -std::log(
            softmax_top_data[i * channels + static_cast<int>(label_data[i])]);
      });
  top_data[0] =
      thrust::reduce(temp_data.begin(), temp_data.end()) / batch_size_;
}

template <typename T>
void LossWithSoftmax<T>::BackwardGPU(const std::vector<TensorPtr<T>>& top,
                                     const std::vector<TensorPtr<T>>& bottom) {
  CheckShape(bottom[0], bottom[1], top[0]);
  const auto& softmax_top_data = softmax_top_[0]->GetGPUData();
  const T* label_ptr = bottom[1]->GetGPUDataPtr();
  auto& bottom_diff = bottom[0]->GetGPUDiff();
  thrust::copy(softmax_top_data.begin(), softmax_top_data.end(),
               bottom_diff.begin());
  T* bottom_ptr = RAW_PTR(bottom_diff);
  int channels = channels_;
  thrust::for_each(
      thrust::counting_iterator(0), thrust::counting_iterator(batch_size_),
      [label_ptr, bottom_ptr, channels] __device__(int i) -> void {
        bottom_ptr[i * channels + static_cast<int>(label_ptr[i])] -= 1;
      });
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
