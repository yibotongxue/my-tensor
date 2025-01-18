// Copyright 2025 yibotongxue

#include <memory>
#include <vector>

#include "batchnorm.hpp"
#include "blas.hpp"
#include "error.hpp"
#include "memory-util.hpp"

namespace my_tensor {

template <typename T>
void BatchNorm<T>::ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                              const std::vector<TensorPtr<T>>& top) {
  const T* bottom_data = bottom[0]->GetGPUDataPtr();
  T* top_data = top[0]->GetGPUDataPtr();
  const T* gama_data = gama_->GetGPUDataPtr();
  const T* beta_data = beta_->GetGPUDataPtr();
  T* mean_data = mean_->GetGPUDataPtr();
  T* variance_data = sqrt_variance_->GetGPUDataPtr();
  if (this->is_train_) {
    Fill_GPU<T>(mean_data, channels_, 0);
    Fill_GPU<T>(variance_data, channels_, 0);
    // compute mean
    row_sum_gpu(bottom_data, temp_cache1_->GetGPUDataPtr(), channels_,
                spatial_size_, batch_size_);
    col_sum_gpu(temp_cache1_->GetGPUDataPtr(), mean_data, batch_size_,
                channels_, 1);
    scale_gpu(
        mean_data, channels_,
        static_cast<T>(1.0) / static_cast<T>(batch_size_ * spatial_size_));
    // compute variance
    MyMemcpyGPU2GPU(standarded_cache_->GetGPUDataPtr(), bottom_data,
                    bottom[0]->GetSize() * sizeof(T));
    add_row_vector_gpu<T>(standarded_cache_->GetGPUDataPtr(), mean_data,
                          channels_, spatial_size_, batch_size_, -1);
    square_gpu(standarded_cache_->GetGPUDataPtr(),
               temp_cache2_->GetGPUDataPtr(), standarded_cache_->GetSize());
    row_sum_gpu(temp_cache2_->GetGPUDataPtr(), temp_cache1_->GetGPUDataPtr(),
                channels_, spatial_size_, batch_size_);
    col_sum_gpu(temp_cache1_->GetGPUDataPtr(), variance_data, batch_size_,
                channels_, 1);
    scale_gpu(
        variance_data, channels_,
        static_cast<T>(1.0) / static_cast<T>(batch_size_ * spatial_size_));
    sqrt_gpu(variance_data, variance_data, channels_);
    // compute mean cache and variance cache
    scale_gpu(mean_cache_->GetGPUDataPtr(), channels_, move_scale_factor_);
    add_two_vec_gpu(mean_cache_->GetGPUDataPtr(), mean_data,
                    static_cast<T>(1.0 - move_scale_factor_), channels_);
    scale_gpu(sqrt_variance_cache_->GetGPUDataPtr(), channels_,
              move_scale_factor_);
    add_two_vec_gpu(sqrt_variance_cache_->GetGPUDataPtr(), variance_data,
                    static_cast<T>(1.0 - move_scale_factor_), channels_);
    // compute standarded data
    divide_row_vector_gpu<T>(standarded_cache_->GetGPUDataPtr(), variance_data,
                             channels_, spatial_size_, batch_size_,
                             static_cast<T>(1e-5));
  } else {
    // FIXME not sure if this is correct
    mean_data = mean_cache_->GetGPUDataPtr();
    variance_data = sqrt_variance_cache_->GetGPUDataPtr();
    MyMemcpyGPU2GPU(standarded_cache_->GetGPUDataPtr(), bottom_data,
                    bottom[0]->GetSize() * sizeof(T));
    add_row_vector_gpu<T>(standarded_cache_->GetGPUDataPtr(), mean_data,
                          channels_, spatial_size_, batch_size_, -1);
    divide_row_vector_gpu<T>(standarded_cache_->GetGPUDataPtr(), variance_data,
                             channels_, spatial_size_, batch_size_,
                             static_cast<T>(1e-5));
  }
  // compute top data
  MyMemcpyGPU2GPU(top_data, standarded_cache_->GetGPUDataPtr(),
                  standarded_cache_->GetSize() * sizeof(T));
  multiply_row_vector_gpu<T>(top_data, gama_data, channels_, spatial_size_,
                             batch_size_);
  add_row_vector_gpu<T>(top_data, beta_data, channels_, spatial_size_,
                        batch_size_);
}

template <typename T>
void BatchNorm<T>::BackwardGPU(const std::vector<TensorPtr<T>>& top,
                               const std::vector<TensorPtr<T>>& bottom) {
  const T* top_diff = top[0]->GetGPUDiffPtr();
  const T* top_data = top[0]->GetGPUDataPtr();
  const T* bottom_data = bottom[0]->GetGPUDataPtr();
  T* bottom_diff = bottom[0]->GetGPUDiffPtr();
  const T* gama_data = gama_->GetGPUDataPtr();
  T* gama_diff = gama_->GetGPUDiffPtr();
  T* beta_diff = beta_->GetGPUDiffPtr();
  const T* mean_data = mean_->GetGPUDataPtr();
  const T* variance_data = sqrt_variance_->GetGPUDataPtr();
  // compute beta diff
  row_sum_gpu(top_diff, temp_cache1_->GetGPUDataPtr(), channels_, spatial_size_,
              batch_size_);
  col_sum_gpu(temp_cache1_->GetGPUDataPtr(), beta_diff, batch_size_, channels_,
              1);
  // compute gama diff
  multiply_two_vec_gpu<T>(top_diff, standarded_cache_->GetGPUDataPtr(),
                          temp_cache2_->GetGPUDataPtr(),
                          batch_size_ * channels_ * spatial_size_);
  row_sum_gpu(temp_cache2_->GetGPUDataPtr(), temp_cache1_->GetGPUDataPtr(),
              channels_, spatial_size_, batch_size_);
  col_sum_gpu(temp_cache1_->GetGPUDataPtr(), gama_diff, batch_size_, channels_,
              1);
  // MyMemFreeGPU(batch_size_times_channel_times_spatial_size_length_cache_->GetGPUDataPtr());
  // MyMemFreeGPU(temp_data);
  // compute bottom diff
  // batch_size_times_channel_times_spatial_size_length_cache_->GetGPUDataPtr()
  // is the partial diff of top_data to standarded_cache
  int n = batch_size_ * spatial_size_;
  MyMemcpyGPU2GPU(temp_cache2_->GetGPUDataPtr(), top_diff,
                  sizeof(T) * n * channels_);
  multiply_row_vector_gpu<T>(temp_cache2_->GetGPUDataPtr(), gama_data,
                             channels_, spatial_size_, batch_size_);
  row_sum_gpu(temp_cache2_->GetGPUDataPtr(), temp_cache1_->GetGPUDataPtr(),
              channels_, spatial_size_, batch_size_);
  MyMemcpyGPU2GPU(bottom_diff, temp_cache2_->GetGPUDataPtr(),
                  sizeof(T) * n * channels_);
  scale_gpu<T>(bottom_diff, n * channels_, n);
  // channel_length_cache_->GetGPUDataPtr() is the sum of
  // batch_size_times_channel_times_spatial_size_length_cache_->GetGPUDataPtr()
  col_sum_gpu(temp_cache1_->GetGPUDataPtr(), temp_cache_->GetGPUDataPtr(),
              batch_size_, channels_, 1);
  add_row_vector_gpu<T>(bottom_diff, temp_cache_->GetGPUDataPtr(), channels_,
                        spatial_size_, batch_size_, -1);
  multiply_two_vec_gpu<T>(temp_cache2_->GetGPUDataPtr(),
                          standarded_cache_->GetGPUDataPtr(),
                          temp_cache2_->GetGPUDataPtr(), n * channels_);
  row_sum_gpu(temp_cache2_->GetGPUDataPtr(), temp_cache1_->GetGPUDataPtr(),
              channels_, spatial_size_, batch_size_);
  col_sum_gpu(temp_cache1_->GetGPUDataPtr(), temp_cache_->GetGPUDataPtr(),
              batch_size_, channels_, 1);
  multiply_row_vector_gpu<T>(standarded_cache_->GetGPUDataPtr(),
                             temp_cache_->GetGPUDataPtr(), channels_,
                             spatial_size_, batch_size_);
  add_two_vec_gpu<T>(bottom_diff, standarded_cache_->GetGPUDataPtr(), -1,
                     n * channels_);
  divide_row_vector_gpu<T>(bottom_diff, variance_data, channels_, spatial_size_,
                           batch_size_, 1e-5);
  scale_gpu<T>(bottom_diff, n * channels_, 1.0f / n);
}

template class BatchNorm<float>;

}  // namespace my_tensor
