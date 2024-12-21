// Copyright 2024 yibotongxue

#include <memory>
#include <vector>

#include "blas.hpp"
#include "conv.hpp"
#include "error.hpp"
#include "filler-parameter.hpp"
#include "filler.hpp"
#include "im2col.hpp"

namespace my_tensor {

template <typename T>
void Convolution<T>::ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                                const std::vector<TensorPtr<T>>& top) {
  CheckShape(bottom[0], top[0]);
  int kernel_size = kernel_height_ * kernel_width_;
  int im_size = height_ * width_;
  Im2col_GPU(batch_size_, bottom[0]->GetGPUDataPtr(), input_channels_, height_,
             width_, kernel_height_, kernel_width_,
             col_cache_->GetGPUDataPtr());
  matmul_gpu(kernel_->GetGPUDataPtr(), col_cache_->GetGPUDataPtr(),
             top[0]->GetGPUDataPtr(), output_channels_,
             input_channels_ * kernel_size, im_size, batch_size_, 1);
  add_row_vector_gpu(top[0]->GetGPUDataPtr(), bias_->GetGPUDataPtr(),
                     output_channels_, im_size, batch_size_);
}

template <typename T>
void Convolution<T>::BackwardGPU(const std::vector<TensorPtr<T>>& top,
                                 const std::vector<TensorPtr<T>>& bottom) {
  CheckShape(bottom[0], top[0]);
  int kernel_size = kernel_height_ * kernel_width_;
  int im_size = height_ * width_;
  // partial temp
  transpose_matmul_gpu(kernel_->GetGPUDataPtr(), top[0]->GetGPUDiffPtr(),
                       col_cache_->GetGPUDiffPtr(),
                       input_channels_ * kernel_size, output_channels_, im_size,
                       batch_size_, 1);
  Col2im_GPU(batch_size_, col_cache_->GetGPUDiffPtr(), input_channels_, height_,
             width_, kernel_height_, kernel_width_, bottom[0]->GetGPUDiffPtr());
  // partial kernel
  std::vector<int> batch_kernel_shape{batch_size_, output_channels_,
                                      input_channels_, kernel_height_,
                                      kernel_width_};
  auto batch_kernel = std::make_shared<Tensor<T>>(batch_kernel_shape);
  matmul_transpose_gpu(top[0]->GetGPUDiffPtr(), col_cache_->GetGPUDataPtr(),
                       batch_kernel->GetGPUDiffPtr(), output_channels_, im_size,
                       input_channels_ * kernel_size, batch_size_);
  col_sum_gpu(
      batch_kernel->GetGPUDiffPtr(), kernel_->GetGPUDiffPtr(), batch_size_,
      output_channels_ * input_channels_ * kernel_height_ * kernel_width_);
  // partial bias
  float* temp_diff = nullptr;
  cudaMalloc(&temp_diff, batch_size_ * output_channels_ * sizeof(float));
  row_sum_gpu(top[0]->GetGPUDiffPtr(), temp_diff, output_channels_, im_size,
              batch_size_);
  col_sum_gpu(temp_diff, bias_->GetGPUDiffPtr(), batch_size_, output_channels_);
  cudaFree(temp_diff);
}

template class Convolution<>;

}  // namespace my_tensor
