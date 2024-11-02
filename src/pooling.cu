// Copyright 2024 yibotongxue

#include <iostream>
#include <memory>
#include <vector>

#include "error.h"
#include "layer-parameter.hpp"
#include "pooling.cuh"
#include "tensor.cuh"

namespace my_tensor {

template <typename T>
void Pooling<T>::SetUp(const TensorPtr<T> bottom) {
  std::shared_ptr<PoolingParameter> param =
      std::dynamic_pointer_cast<PoolingParameter>(this->layer_param_);
  assert(param.get() != nullptr);
  input_channels_ = param->input_channels_;
  kernel_h_ = param->kernel_h_;
  kernel_w_ = param->kernel_w_;
  stride_h_ = param->stride_h_;
  stride_w_ = param->stride_w_;
  if (bottom->GetShape().size() != 4) {
    throw PoolingError(
        "The input of pooling layer should be 4 dimension tensor.");
  }
  batch_size_ = bottom->GetShape()[0];
  if (bottom->GetShape()[1] != input_channels_) {
    throw PoolingError("The input channels not match.");
  }
  input_height_ = bottom->GetShape()[2];
  input_width_ = bottom->GetShape()[3];
  output_height_ = (input_height_ - kernel_h_) / stride_h_ + 1;
  output_width_ = (input_width_ - kernel_w_) / stride_w_ + 1;
  const std::vector<int> mask_shape = {batch_size_, input_channels_,
                                       output_height_, output_width_};
  mask_.reset();
  mask_ = std::make_shared<Tensor<int>>(mask_shape);
}

template <typename T>
void Pooling<T>::ForwardCPU(const TensorPtr<T> bottom, TensorPtr<T> top) {
  CheckShape(bottom, top);
  const auto& bottom_data = bottom->GetCPUData();
  auto& top_data = top->GetCPUData();
  auto& mask_data = mask_->GetCPUData();
  int input_im_size = input_height_ * input_width_;
  int output_im_size = output_height_ * output_width_;
  for (int t = 0; t < batch_size_ * input_channels_; t++) {
    for (int i = 0; i < output_height_; i++) {
      for (int j = 0; j < output_width_; j++) {
        int h_start = i * stride_h_;
        int w_start = j * stride_w_;
        T val = static_cast<T>(__FLT_MIN__);
        int mask_idx = -1;
        for (int x = 0; x < kernel_h_; x++) {
          for (int y = 0; y < kernel_w_; y++) {
            int temp_idx =
                t * input_im_size + (h_start + x) * input_width_ + w_start + y;
            T temp = bottom_data[temp_idx];
            if (temp > val) {
              val = temp;
              mask_idx = temp_idx;
            }
          }
        }
        top_data[t * output_im_size + i * output_width_ + j] = val;
        mask_data[t * output_im_size + i * output_width_ + j] = mask_idx;
      }
    }
  }
}

template <typename T>
void Pooling<T>::CheckShape(const TensorPtr<T> bottom,
                            const TensorPtr<T> top) const {
  const auto& bottom_shape = bottom->GetShape();
  const auto& top_shape = top->GetShape();
  if (bottom_shape.size() != 4) {
    throw PoolingError(
        "The input of pooling layer should be 4 dimension tensor.");
  }
  if (top_shape.size() != 4) {
    throw PoolingError(
        "The output of pooling layer should be 4 dimension tensor.");
  }
  if (bottom_shape[0] != top_shape[0]) {
    throw PoolingError(
        "The input and output of pooling layer should have the same batch "
        "size.");
  }
  if (bottom_shape[1] != top_shape[1]) {
    throw PoolingError(
        "The input and output of pooling layer should have the same channels.");
  }
  if (bottom_shape[2] != input_height_ || bottom_shape[3] != input_width_) {
    throw PoolingError("The input shape not match the pooling layer.");
  }
  if (top_shape[2] != output_height_ || top_shape[3] != output_width_) {
    throw PoolingError("The output shape not match the pooling layer.");
  }
}

template class Pooling<>;

}  // namespace my_tensor
