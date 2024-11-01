// Copyright 2024 yibotongxue

#include <memory>
#include <vector>

#include "blas.cuh"
#include "conv.cuh"
#include "error.h"
#include "filler-parameter.hpp"
#include "filler.cuh"
#include "im2col.cuh"

namespace my_tensor {

template <typename T>
void Convolution<T>::SetUp(const TensorPtr<T> bottom) {
  if (this->layer_param_->type_ != ParamType::kConvolution) {
    throw LayerError("Layer not match type.");
  }
  if (bottom->GetShape().size() != 4) {
    throw ConvError("The dimension of the inputshould be 4.");
  }
  std::shared_ptr<ConvolutionParameter> param =
      std::dynamic_pointer_cast<ConvolutionParameter>(this->layer_param_);
  assert(param.get() != nullptr);
  kernel_height_ = param->kernel_size_;
  kernel_width_ = param->kernel_size_;
  input_channels_ = param->input_channels_;
  output_channels_ = param->output_channels_;
  const std::vector<int> kernel_shape{output_channels_, input_channels_,
                                      kernel_height_, kernel_width_};
  kernel_.reset();
  kernel_ = std::make_shared<Tensor<T>>(kernel_shape);
  FillerPtr<T> kernel_filler = CreateFiller<T>(param->kernel_filler_parameter_);
  kernel_filler->Fill(kernel_);
  const std::vector<int> bias_shape{output_channels_};
  bias_.reset();
  bias_ = std::make_shared<Tensor<T>>(bias_shape);
  FillerPtr<T> bias_filler = CreateFiller<T>(param->bias_filler_parameter_);
  bias_filler->Fill(bias_);
  batch_size_ = bottom->GetShape()[0];
  if (bottom->GetShape()[1] != input_channels_) {
    throw ConvError("The input channels not match.");
  }
  height_ = bottom->GetShape()[2];
  width_ = bottom->GetShape()[3];
  const std::vector<int> col_shape{
      batch_size_, input_channels_ * kernel_height_ * kernel_width_,
      height_ * width_};
  col_cache_.reset();
  col_cache_ = std::make_shared<Tensor<T>>(col_shape);
}

template <typename T>
void Convolution<T>::ForwardCPU(const TensorPtr<T> bottom, TensorPtr<T> top) {
  CheckShape(bottom, top);
  const auto& kernel_data = kernel_->GetCPUData();
  const auto& bias_data = bias_->GetCPUData();
  const auto& bottom_data = bottom->GetCPUData();
  auto& top_data = top->GetCPUData();
  int kernel_size = kernel_height_ * kernel_width_;
  int im_size = height_ * width_;
  Im2col_CPU(batch_size_, bottom->GetCPUDataPtr(), input_channels_, height_,
             width_, kernel_height_, kernel_width_,
             col_cache_->GetCPUDataPtr());
  const auto& col_data = col_cache_->GetCPUData();
  for (int t = 0; t < batch_size_; t++) {
    for (int i = 0; i < output_channels_; i++) {
      for (int j = 0; j < im_size; j++) {
        T val = bias_data[i];
        for (int k = 0; k < input_channels_ * kernel_size; k++) {
          val += kernel_data[i * input_channels_ * kernel_size + k] *
                 col_data[t * input_channels_ * im_size * kernel_size +
                          k * im_size + j];
        }
        top_data[t * output_channels_ * im_size + i * im_size + j] = val;
      }
    }
  }
}

template <typename T>
void Convolution<T>::BackwardCPU(const TensorPtr<T> top, TensorPtr<T> bottom) {
  CheckShape(bottom, top);
  const auto& kernel_data = kernel_->GetCPUData();
  const auto& top_diff = top->GetCPUDiff();
  const auto& bottom_data = bottom->GetCPUData();
  auto& bottom_diff = bottom->GetCPUDiff();
  auto& kernel_diff = kernel_->GetCPUDiff();
  int kernel_size = kernel_height_ * kernel_width_;
  int im_size = height_ * width_;
  const auto& col_data = col_cache_->GetCPUData();
  auto& col_diff = col_cache_->GetCPUDiff();
  auto& bias_diff = bias_->GetCPUDiff();
  // top = kernel * temp
  // [n * output_channels_ * im_size] = [(n *) output_channels_ *
  // (input_channels_ * kernel_size)]
  // * [n * (input_channels_ * kernel_size) * im_size]
  // partial temp
  for (int t = 0; t < batch_size_; t++) {
    for (int i = 0; i < input_channels_ * kernel_size; i++) {
      for (int j = 0; j < im_size; j++) {
        T val = 0;
        for (int k = 0; k < output_channels_; k++) {
          val += kernel_data[k * input_channels_ * kernel_size + i] *
                 top_diff[t * output_channels_ * im_size + k * im_size + j];
        }
        col_diff[t * input_channels_ * kernel_size * im_size + i * im_size +
                 j] = val;
      }
    }
  }
  Col2im_CPU(batch_size_, col_cache_->GetCPUDiffPtr(), input_channels_, height_,
             width_, kernel_height_, kernel_width_, bottom->GetCPUDiffPtr());
  // partial kernel
  for (int i = 0; i < output_channels_; i++) {
    for (int j = 0; j < input_channels_ * kernel_size; j++) {
      T val = 0;
      for (int t = 0; t < batch_size_; t++) {
        for (int k = 0; k < im_size; k++) {
          val += top_diff[t * output_channels_ * im_size + i * im_size + k] *
                 col_data[t * input_channels_ * kernel_size * im_size +
                          j * im_size + k];
        }
      }
      kernel_diff[i * input_channels_ * kernel_size + j] = val;
    }
  }
  // partial bias
  for (int i = 0; i < output_channels_; i++) {
    T val = 0;
    for (int t = 0; t < batch_size_; t++) {
      for (int j = 0; j < im_size; j++) {
        val += top_diff[t * output_channels_ * im_size + i * im_size + j];
      }
    }
    bias_diff[i] = val;
  }
}

// TODO(yibotongxue) Add bias to top
template <typename T>
void Convolution<T>::ForwardGPU(const TensorPtr<T> bottom, TensorPtr<T> top) {
  CheckShape(bottom, top);
  int kernel_size = kernel_height_ * kernel_width_;
  int im_size = height_ * width_;
  Im2col_GPU(batch_size_, bottom->GetGPUDataPtr(), input_channels_, height_,
             width_, kernel_height_, kernel_width_,
             col_cache_->GetGPUDataPtr());
  matmul(kernel_->GetGPUDataPtr(), col_cache_->GetGPUDataPtr(),
         top->GetGPUDataPtr(), output_channels_, input_channels_ * kernel_size,
         im_size, batch_size_, 1);
}

// TODO(yibotongxue) Add bias backward
template <typename T>
void Convolution<T>::BackwardGPU(const TensorPtr<T> top, TensorPtr<T> bottom) {
  CheckShape(bottom, top);
  int kernel_size = kernel_height_ * kernel_width_;
  int im_size = height_ * width_;
  // partial temp
  transpose_matmul(kernel_->GetGPUDataPtr(), top->GetGPUDiffPtr(),
                   col_cache_->GetGPUDiffPtr(), input_channels_ * kernel_size,
                   output_channels_, im_size, batch_size_, 1);
  Col2im_GPU(batch_size_, col_cache_->GetGPUDiffPtr(), input_channels_, height_,
             width_, kernel_height_, kernel_width_, bottom->GetGPUDiffPtr());
  // partial kernel
  std::vector<int> batch_kernel_shape{batch_size_, output_channels_,
                                      input_channels_, kernel_height_,
                                      kernel_width_};
  auto batch_kernel = std::make_shared<Tensor<T>>(batch_kernel_shape);
  matmul_transpose(top->GetGPUDiffPtr(), col_cache_->GetGPUDataPtr(),
                   batch_kernel->GetGPUDiffPtr(), output_channels_, im_size,
                   input_channels_ * kernel_size, batch_size_);
  col_sum(batch_kernel->GetGPUDiffPtr(), kernel_->GetGPUDiffPtr(), batch_size_,
          output_channels_ * input_channels_ * kernel_height_ * kernel_width_);
}

template <typename T>
void Convolution<T>::CheckShape(const TensorPtr<T> bottom,
                                const TensorPtr<T> top) const {
  const std::vector<int>& kernel_shape = kernel_->GetShape();
  const std::vector<int>& bottom_shape = bottom->GetShape();
  const std::vector<int>& top_shape = top->GetShape();
  if (bottom_shape.size() != 4) {
    throw ConvError("The dimension of the inputshould be 4.");
  }
  if (top_shape.size() != 4) {
    throw ConvError("The dimension of the output should be 4.");
  }
  if (bottom_shape[0] != top_shape[0]) {
    throw ConvError("The batch size of input and output not match.");
  }
  if (bottom_shape[2] != top_shape[2]) {
    throw ConvError("The height_ of input and output not match.");
  }
  if (bottom_shape[3] != top_shape[3]) {
    throw ConvError("The width_ size of input and output not match.");
  }
  if (bottom_shape[1] != kernel_shape[1]) {
    throw ConvError("The input channels_ not match kernel.");
  }
  if (top_shape[1] != kernel_shape[0]) {
    throw ConvError("The output channels_ not match kernel.");
  }
}

template class Convolution<>;

}  // namespace my_tensor
