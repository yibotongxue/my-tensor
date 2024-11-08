// Copyright 2024 yibotongxue

#include <thrust/fill.h>
#include <thrust/scatter.h>

#include <iostream>
#include <memory>
#include <vector>

#include "error.h"
#include "layer-parameter.h"
#include "pooling.cuh"
#include "tensor.cuh"
#include "utils.cuh"

namespace my_tensor {

template <typename T>
void Pooling<T>::CheckTensorCount(const std::vector<TensorPtr<T>>& bottom,
                                  const std::vector<TensorPtr<T>>& top) const {
  if (bottom.size() != 1) {
    throw PoolingError(
        "The bottom of convolution layer should have one tensor.");
  }
  if (top.size() != 1) {
    throw PoolingError("The top of convolution layer should have one tensor.");
  }
}

template <typename T>
void Pooling<T>::Reshape(const std::vector<TensorPtr<T>>& bottom,
                         const std::vector<TensorPtr<T>>& top) const {
  int expect_size =
      batch_size_ * input_channels_ * output_height_ * output_width_;
  if (top[0]->GetSize() != expect_size) {
    throw PoolingError("The top size not match pooling layer.");
  }
  top[0]->Reshape(
      {batch_size_, input_channels_, output_height_, output_width_});
}

template <typename T>
void Pooling<T>::LayerSetUp(const std::vector<TensorPtr<T>>& bottom,
                            const std::vector<TensorPtr<T>>& top) {
  std::shared_ptr<PoolingParameter> param =
      std::dynamic_pointer_cast<PoolingParameter>(this->layer_param_);
  assert(param.get() != nullptr);
  input_channels_ = param->input_channels_;
  kernel_h_ = param->kernel_h_;
  kernel_w_ = param->kernel_w_;
  stride_h_ = param->stride_h_;
  stride_w_ = param->stride_w_;
  if (bottom[0]->GetShape().size() != 4) {
    throw PoolingError(
        "The input of pooling layer should be 4 dimension tensor.");
  }
  batch_size_ = bottom[0]->GetShape()[0];
  if (bottom[0]->GetShape()[1] != input_channels_) {
    throw PoolingError("The input channels not match.");
  }
  input_height_ = bottom[0]->GetShape()[2];
  input_width_ = bottom[0]->GetShape()[3];
  output_height_ = (input_height_ - kernel_h_) / stride_h_ + 1;
  output_width_ = (input_width_ - kernel_w_) / stride_w_ + 1;
  const std::vector<int> mask_shape = {batch_size_, input_channels_,
                                       output_height_, output_width_};
  mask_.reset();
  mask_ = std::make_shared<Tensor<int>>(mask_shape);
}

template <typename T>
void Pooling<T>::ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                            const std::vector<TensorPtr<T>>& top) {
  CheckShape(bottom[0], top[0]);
  const auto& bottom_data = bottom[0]->GetCPUData();
  auto& top_data = top[0]->GetCPUData();
  auto& mask_data = mask_->GetCPUData();
  int input_im_size = input_height_ * input_width_;
  int output_im_size = output_height_ * output_width_;
  for (int t = 0; t < batch_size_ * input_channels_; t++) {
    for (int i = 0; i < output_height_; i++) {
      for (int j = 0; j < output_width_; j++) {
        int h_start = i * stride_h_;
        int w_start = j * stride_w_;
        T val = static_cast<T>(-__FLT_MAX__);
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
void Pooling<T>::BackwardCPU(const std::vector<TensorPtr<T>>& top,
                             const std::vector<TensorPtr<T>>& bottom) {
  CheckShape(bottom[0], top[0]);
  const auto& top_diff = top[0]->GetCPUDiff();
  const auto& mask_data = mask_->GetCPUData();
  auto& bottom_diff = bottom[0]->GetCPUDiff();
  thrust::fill(bottom_diff.begin(), bottom_diff.end(), 0);
  thrust::scatter(top_diff.begin(), top_diff.end(), mask_data.begin(),
                  bottom_diff.begin());
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

template <typename T>
void Pooling<T>::CheckShape(const TensorPtr<T> bottom,
                            const TensorPtr<T> top) const {
#ifdef DEBUG
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
#endif  // DEBUG
}

template class Pooling<>;

}  // namespace my_tensor
