// Copyright 2024 yibotongxue

#include <memory>
#include <vector>

#include "blas.cuh"
#include "conv.cuh"
#include "error.h"
#include "im2col.cuh"

namespace my_tensor {

#define DEFINE_SHAPE_DATA                                              \
  const std::vector<int>& bottom_shape = bottom->GetShape();           \
  const std::vector<int>& top_shape = top->GetShape();                 \
  const std::vector<int>& kernel_shape = this->params_[0]->GetShape(); \
  int n = bottom_shape[0], c_in = bottom_shape[1];                     \
  int height = bottom_shape[2], width = bottom_shape[3];               \
  int c_out = top_shape[1];                                            \
  int kernel_height = kernel_shape[2];                                 \
  int kernel_width = kernel_shape[3];

template <typename T>
Convolution<T>::Convolution(const std::vector<TensorPtr<T>>& params)
    : Layer<T>(params) {
  if (this->params_.size() != 1) {
    throw ConvError("Convolution parameters number should be 1.");
  }
  const std::vector<int>& kernel_shape = this->params_[0]->GetShape();
  if (kernel_shape.size() != 4) {
    throw ConvError("Convolution kernel should be 4 dimension tensor.");
  }
}

template <typename T>
void Convolution<T>::ForwardCPU(const TensorPtr<T> bottom, TensorPtr<T> top) {
  CheckShape(bottom, top);
  DEFINE_SHAPE_DATA
  const auto& kernel_data = this->params_[0]->GetCPUData();
  const auto& bottom_data = bottom->GetCPUData();
  auto& top_data = top->GetCPUData();
  int kernel_size = kernel_height * kernel_width;
  int im_size = height * width;
  std::vector<int> temp_shape{n, c_in * kernel_size, im_size};
  auto temp_tensor = std::make_shared<Tensor<T>>(temp_shape);
  Im2col_CPU(n, bottom->GetCPUDataPtr(), c_in, height, width, kernel_height,
             kernel_width, temp_tensor->GetCPUDataPtr());
  const auto& temp_data = temp_tensor->GetCPUData();
  for (int t = 0; t < n; t++) {
    for (int i = 0; i < c_out; i++) {
      for (int j = 0; j < im_size; j++) {
        T val = 0;
        for (int k = 0; k < c_in * kernel_size; k++) {
          val += kernel_data[i * c_in * kernel_size + k] *
                 temp_data[t * c_in * im_size * kernel_size + k * im_size + j];
        }
        top_data[t * c_out * im_size + i * im_size + j] = val;
      }
    }
  }
}

template <typename T>
void Convolution<T>::BackwardCPU(const TensorPtr<T> top, TensorPtr<T> bottom) {
  CheckShape(bottom, top);
  DEFINE_SHAPE_DATA
  const auto& kernel_data = this->params_[0]->GetCPUData();
  const auto& top_diff = top->GetCPUDiff();
  const auto& bottom_data = bottom->GetCPUData();
  auto& bottom_diff = bottom->GetCPUDiff();
  auto& kernel_diff = this->params_[0]->GetCPUDiff();
  int kernel_size = kernel_height * kernel_width;
  int im_size = height * width;
  std::vector<int> temp_shape{n, c_in * kernel_size, im_size};
  auto temp_tensor = std::make_shared<Tensor<T>>(temp_shape);
  Im2col_CPU(n, bottom->GetCPUDataPtr(), c_in, height, width, kernel_height,
             kernel_width, temp_tensor->GetCPUDataPtr());
  const auto& temp_data = temp_tensor->GetCPUData();
  auto& temp_diff = temp_tensor->GetCPUDiff();
  // top = kernel * temp
  // [n * c_out * im_size] = [(n *) c_out * (c_in * kernel_size)]
  // * [n * (c_in * kernel_size) * im_size]
  // partial temp
  for (int t = 0; t < n; t++) {
    for (int i = 0; i < c_in * kernel_size; i++) {
      for (int j = 0; j < im_size; j++) {
        T val = 0;
        for (int k = 0; k < c_out; k++) {
          val += kernel_data[k * c_in * kernel_size + i] *
                 top_diff[t * c_out * im_size + k * im_size + j];
        }
        temp_diff[t * c_in * kernel_size * im_size + i * im_size + j] = val;
      }
    }
  }
  Col2im_CPU(n, temp_tensor->GetCPUDiffPtr(), c_in, height, width,
             kernel_height, kernel_width, bottom->GetCPUDiffPtr());
  // partial kernel
  for (int i = 0; i < c_out; i++) {
    for (int j = 0; j < c_in * kernel_size; j++) {
      T val = 0;
      for (int t = 0; t < n; t++) {
        for (int k = 0; k < im_size; k++) {
          val += top_diff[t * c_out * im_size + i * im_size + k] *
                 temp_data[t * c_in * kernel_size * im_size + j * im_size + k];
        }
      }
      kernel_diff[i * c_in * kernel_size + j] = val;
    }
  }
}

template <typename T>
void Convolution<T>::ForwardGPU(const TensorPtr<T> bottom, TensorPtr<T> top) {
  CheckShape(bottom, top);
  DEFINE_SHAPE_DATA
  int kernel_size = kernel_height * kernel_width;
  int im_size = height * width;
  std::vector<int> temp_shape{n, c_in * kernel_size, im_size};
  auto temp_tensor = std::make_shared<Tensor<T>>(temp_shape);
  Im2col_GPU(n, bottom->GetGPUDataPtr(), c_in, height, width, kernel_height,
             kernel_width, temp_tensor->GetGPUDataPtr());
  matmul(this->params_[0]->GetGPUDataPtr(), temp_tensor->GetGPUDataPtr(),
         top->GetGPUDataPtr(), c_out, c_in * kernel_size, im_size, n, 1);
}

template <typename T>
void Convolution<T>::BackwardGPU(const TensorPtr<T> top, TensorPtr<T> bottom) {
  CheckShape(bottom, top);
  DEFINE_SHAPE_DATA
  int kernel_size = kernel_height * kernel_width;
  int im_size = height * width;
  std::vector<int> temp_shape{n, c_in * kernel_size, im_size};
  auto temp_tensor = std::make_shared<Tensor<T>>(temp_shape);
  Im2col_GPU(n, bottom->GetGPUDataPtr(), c_in, height, width, kernel_height,
             kernel_width, temp_tensor->GetGPUDataPtr());
  // partial temp
  transpose_matmul(this->params_[0]->GetGPUDataPtr(), top->GetGPUDiffPtr(),
                   temp_tensor->GetGPUDiffPtr(), c_in * kernel_size, c_out,
                   im_size, n, 1);
  Col2im_GPU(n, temp_tensor->GetGPUDiffPtr(), c_in, height, width,
             kernel_height, kernel_width, bottom->GetGPUDiffPtr());
  // partial kernel
  std::vector<int> batch_kernel_shape{n, c_out, c_in, kernel_height,
                                      kernel_width};
  auto batch_kernel = std::make_shared<Tensor<T>>(batch_kernel_shape);
  matmul_transpose(top->GetGPUDiffPtr(), temp_tensor->GetGPUDataPtr(),
                   batch_kernel->GetGPUDiffPtr(), c_out, im_size,
                   c_in * kernel_size, n);
  col_sum(batch_kernel->GetGPUDiffPtr(), this->params_[0]->GetGPUDiffPtr(), n,
          c_out * c_in * kernel_height * kernel_width);
}

template <typename T>
void Convolution<T>::CheckShape(const TensorPtr<T> bottom,
                                const TensorPtr<T> top) const {
  const std::vector<int>& kernel_shape = this->params_[0]->GetShape();
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
    throw ConvError("The height of input and output not match.");
  }
  if (bottom_shape[3] != top_shape[3]) {
    throw ConvError("The width size of input and output not match.");
  }
  if (bottom_shape[1] != kernel_shape[1]) {
    throw ConvError("The input channels not match kernel.");
  }
  if (top_shape[1] != kernel_shape[0]) {
    throw ConvError("The output channels not match kernel.");
  }
}

template class Convolution<>;

#undef DEFINE_SHAPE_DATA

}  // namespace my_tensor
