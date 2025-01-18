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

namespace {
template <typename T>
__global__ void MutiplyRowVectorAndAssign(
    T* const target, const T* const source, const T* const vec,
    const int channels, const int spatial_size, const int batch_size) {
  extern __shared__ T shared_vec[];
  int thread_id = threadIdx.x;
  if (thread_id < channels) {
    shared_vec[thread_id] = vec[thread_id];
  }
  __syncthreads();
  CUDA_KERNEL_LOOP(idx, batch_size * channels * spatial_size) {
    target[idx] = source[idx] * shared_vec[(idx / spatial_size) % channels];
  }
}

template <typename T>
__global__ void ScaleBottomAndUpdateTempCache2(
    T* const bottom_diff, T* const temp_cache2, const T* const temp_cache,
    const T* const standarded_cache, const int batch_size, const int channels,
    const int spatial_size) {
  extern __shared__ T shared_vec[];
  int thread_id = threadIdx.x;
  if (thread_id < channels) {
    shared_vec[thread_id] = temp_cache[thread_id];
  }
  __syncthreads();
  int n = batch_size * spatial_size;
  CUDA_KERNEL_LOOP(idx, batch_size * channels * spatial_size) {
    bottom_diff[idx] =
        temp_cache2[idx] * n - shared_vec[(idx / spatial_size) % channels];
    temp_cache2[idx] *= standarded_cache[idx];
  }
}

template <typename T>
__global__ void ComputeBottomDiff(const T* const standarded_cache,
                                  const T* const temp_cache,
                                  T* const bottom_diff,
                                  const T* const variance_data,
                                  const int batch_size, const int channels,
                                  const int spatial_size) {
  extern __shared__ T shared_vec[];
  int thread_id = threadIdx.x;
  if (thread_id < channels) {
    shared_vec[thread_id] = temp_cache[thread_id];
  } else if (thread_id < 2 * channels) {
    shared_vec[thread_id] = variance_data[thread_id - channels];
  }
  __syncthreads();
  float n = batch_size * spatial_size;
  CUDA_KERNEL_LOOP(idx, batch_size * channels * spatial_size) {
    bottom_diff[idx] -=
        standarded_cache[idx] * shared_vec[(idx / spatial_size) % channels];
    bottom_diff[idx] /=
        (shared_vec[(idx / spatial_size) % channels + channels] + 1e-5) * n;
  }
}
}  // namespace

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
  MutiplyRowVectorAndAssign<<<CudaGetBlocks(channels_ * spatial_size_ *
                                            batch_size_),
                              kCudaThreadNum, channels_ * sizeof(T)>>>(
      temp_cache2_->GetGPUDataPtr(), top_diff, gama_data, channels_,
      spatial_size_, batch_size_);
  row_sum_gpu(temp_cache2_->GetGPUDataPtr(), temp_cache1_->GetGPUDataPtr(),
              channels_, spatial_size_, batch_size_);
  // channel_length_cache_->GetGPUDataPtr() is the sum of
  // batch_size_times_channel_times_spatial_size_length_cache_->GetGPUDataPtr()
  col_sum_gpu(temp_cache1_->GetGPUDataPtr(), temp_cache_->GetGPUDataPtr(),
              batch_size_, channels_, 1);
  ScaleBottomAndUpdateTempCache2<<<CudaGetBlocks(batch_size_ * channels_ *
                                                 spatial_size_),
                                   kCudaThreadNum, channels_ * sizeof(T)>>>(
      bottom_diff, temp_cache2_->GetGPUDataPtr(), temp_cache_->GetGPUDataPtr(),
      standarded_cache_->GetGPUDataPtr(), batch_size_, channels_,
      spatial_size_);
  row_sum_gpu(temp_cache2_->GetGPUDataPtr(), temp_cache1_->GetGPUDataPtr(),
              channels_, spatial_size_, batch_size_);
  col_sum_gpu(temp_cache1_->GetGPUDataPtr(), temp_cache_->GetGPUDataPtr(),
              batch_size_, channels_, 1);
  ComputeBottomDiff<<<CudaGetBlocks(batch_size_ * channels_ * spatial_size_),
                      kCudaThreadNum, 2 * channels_ * sizeof(T)>>>(
      standarded_cache_->GetGPUDataPtr(), temp_cache_->GetGPUDataPtr(),
      bottom_diff, variance_data, batch_size_, channels_, spatial_size_);
}

template class BatchNorm<float>;

}  // namespace my_tensor
