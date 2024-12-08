// Copyright 2024 yibotongxue

#include <thrust/fill.h>
#include <thrust/scatter.h>

#include <iostream>
#include <memory>
#include <vector>

#include "error.hpp"
#include "layer-parameter.hpp"
#include "pooling.hpp"
#include "tensor.hpp"
#include "utils.hpp"

namespace my_tensor {

namespace {
template <typename T>
__global__ void PoolingKernel(const int nthreads, const T* const bottom_data,
                              const int n, const int input_w,
                              const int input_size, const int output_w,
                              const int output_size, const int kernel_h,
                              const int kernel_w, const int stride_h,
                              const int stride_w, T* top_data, int* mask_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int t = index / output_size;
    int h_start = (index % output_size) / output_w * stride_h;
    int w_start = (index % output_w) * stride_w;
    T val = static_cast<T>(-__FLT_MAX__);
    int idx = -1;
    int row_idx = t * input_size + h_start * input_w + w_start;
    for (int i = 0; i < kernel_h; i++) {
      int col_idx = row_idx;
      for (int j = 0; j < kernel_w; j++) {
        if (val < bottom_data[col_idx]) {
          val = bottom_data[col_idx];
          idx = col_idx;
        }
        col_idx += 1;
      }
      row_idx += input_w;
    }
    top_data[index] = val;
    mask_data[index] = idx;
  }
}
}  // namespace

template <typename T>
void Pooling<T>::ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                            const std::vector<TensorPtr<T>>& top) {
  CheckShape(bottom[0], top[0]);
  int input_size = input_height_ * input_width_;
  int output_size = output_height_ * output_width_;
  int n = batch_size_ * input_channels_;
  int nthreads = n * output_size;
  PoolingKernel<<<CudaGetBlocks(nthreads), kCudaThreadNum>>>(
      nthreads, bottom[0]->GetGPUDataPtr(), n, input_width_, input_size,
      output_width_, output_size, kernel_h_, kernel_w_, stride_h_, stride_w_,
      top[0]->GetGPUDataPtr(), mask_->GetGPUDataPtr());
}

template <typename T>
void Pooling<T>::BackwardGPU(const std::vector<TensorPtr<T>>& top,
                             const std::vector<TensorPtr<T>>& bottom) {
  CheckShape(bottom[0], top[0]);
  const auto& top_diff = top[0]->GetGPUDiff();
  const auto& mask_data = mask_->GetGPUData();
  auto& bottom_diff = bottom[0]->GetGPUDiff();
  thrust::fill(bottom_diff.begin(), bottom_diff.end(), 0);
  thrust::scatter(top_diff.begin(), top_diff.end(), mask_data.begin(),
                  bottom_diff.begin());
}

template class Pooling<>;

}  // namespace my_tensor
