// Copyright 2025 yibotongxue

#include "batchnorm.hpp"

#include <memory>
#include <vector>

#include "blas.hpp"
#include "error.hpp"
#include "memory-util.hpp"

namespace my_tensor {

template <typename T>
void BatchNorm<T>::CheckTensorCount(
    const std::vector<TensorPtr<T>>& bottom,
    const std::vector<TensorPtr<T>>& top) const {
  if (bottom.size() != 1) {
    throw BatchNormError(
        "The bottom of batchnorm layer should have one tensor.");
  }
  if (top.size() != 1) {
    throw BatchNormError("The top of batchnorm layer should have one tensor.");
  }
}

template <typename T>
void BatchNorm<T>::Reshape(const std::vector<TensorPtr<T>>& bottom,
                           const std::vector<TensorPtr<T>>& top) const {
  top[0]->Resize(bottom[0]->GetShape());
}

template <typename T>
void BatchNorm<T>::LayerSetUp(const std::vector<TensorPtr<T>>& bottom,
                              const std::vector<TensorPtr<T>>& top) {
  if (bottom[0]->GetShape().size() != 4) {
    throw BatchNormError(
        "Input of batchnorm layer should be four dimesion tensor.");
  }
  std::shared_ptr<BatchNormParameter> param =
      std::dynamic_pointer_cast<BatchNormParameter>(this->layer_param_);
  assert(param.get() != nullptr);
  channels_ = param->channels_;
  if (bottom[0]->GetShape()[1] != channels_) {
    throw BatchNormError(
        "The channels of input tensor is not equal to the "
        "channels of batchnorm layer.");
  }
  std::vector<int> shape = {1, channels_, 1, 1};
  gama_ = std::make_shared<Tensor<T>>(shape);
  beta_ = std::make_shared<Tensor<T>>(shape);
  if (MyTensorContext::on_cpu()) {
    Fill_CPU<T>(gama_->GetCPUDataPtr(), channels_, 1);
    Fill_CPU<T>(beta_->GetCPUDataPtr(), channels_, 0);
  } else {
    Fill_GPU<T>(gama_->GetGPUDataPtr(), channels_, 1);
    Fill_GPU<T>(beta_->GetGPUDataPtr(), channels_, 0);
  }
  mean_cache_ = std::make_shared<Tensor<T>>(shape);
  sqrt_variance_cache_ = std::make_shared<Tensor<T>>(shape);
  mean_ = std::make_shared<Tensor<T>>(shape);
  sqrt_variance_ = std::make_shared<Tensor<T>>(shape);
  standarded_cache_ = std::make_shared<Tensor<T>>(bottom[0]->GetShape());
  scale_factor_ = 1.0;
  move_scale_factor_ = param->move_scale_factor_;
  batch_size_ = bottom[0]->GetShape()[0];
  spatial_size_ = bottom[0]->GetShape()[2] * bottom[0]->GetShape()[3];
}

template <typename T>
void BatchNorm<T>::ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                              const std::vector<TensorPtr<T>>& top) {
  const T* bottom_data = bottom[0]->GetCPUDataPtr();
  T* top_data = top[0]->GetCPUDataPtr();
  const T* gama_data = gama_->GetCPUDataPtr();
  const T* beta_data = beta_->GetCPUDataPtr();
  T* mean_data = mean_->GetCPUDataPtr();
  T* variance_data = sqrt_variance_->GetCPUDataPtr();
  if (this->is_train_) {
    Fill_CPU<T>(mean_data, channels_, 0);
    Fill_CPU<T>(variance_data, channels_, 0);
    // compute mean
    T* temp_data;
    MyMallocCPU(reinterpret_cast<void**>(&temp_data),
                sizeof(T) * batch_size_ * channels_);
    row_sum_cpu(bottom_data, temp_data, channels_, spatial_size_, batch_size_);
    col_sum_cpu(temp_data, mean_data, batch_size_, channels_, 1);
    scale_cpu(
        mean_data, channels_,
        static_cast<T>(1.0) / static_cast<T>(batch_size_ * spatial_size_));
    // compute variance
    MyMemcpyCPU2CPU(standarded_cache_->GetCPUDataPtr(), bottom_data,
                    bottom[0]->GetSize() * sizeof(T));
    add_row_vector_cpu<T>(standarded_cache_->GetCPUDataPtr(), mean_data,
                          channels_, spatial_size_, batch_size_, -1);
    T* temp_temp_data;
    MyMallocCPU(reinterpret_cast<void**>(&temp_temp_data),
                sizeof(T) * batch_size_ * channels_ * spatial_size_);
    square_cpu(standarded_cache_->GetCPUDataPtr(), temp_temp_data,
               standarded_cache_->GetSize());
    row_sum_cpu(temp_temp_data, temp_data, channels_, spatial_size_,
                batch_size_);
    MyMemFreeCPU(temp_temp_data);
    col_sum_cpu(temp_data, variance_data, batch_size_, channels_, 1);
    scale_cpu(
        variance_data, channels_,
        static_cast<T>(1.0) / static_cast<T>(batch_size_ * spatial_size_));
    sqrt_cpu(variance_data, variance_data, channels_);
    MyMemFreeCPU(temp_data);
    // compute mean cache and variance cache
    scale_cpu(mean_cache_->GetCPUDataPtr(), channels_, move_scale_factor_);
    add_two_vec_cpu(mean_cache_->GetCPUDataPtr(), mean_data,
                    static_cast<T>(1.0 - move_scale_factor_), channels_);
    scale_cpu(sqrt_variance_cache_->GetCPUDataPtr(), channels_,
              move_scale_factor_);
    add_two_vec_cpu(sqrt_variance_cache_->GetCPUDataPtr(), variance_data,
                    static_cast<T>(1.0 - move_scale_factor_), channels_);
    // update scale factor
    scale_factor_ = 1.0 + move_scale_factor_ * scale_factor_;
    sqrt_variance_->GetCPUDataPtr();
    // compute standarded data
    divide_row_vector_cpu<T>(standarded_cache_->GetCPUDataPtr(), variance_data,
                             channels_, spatial_size_, batch_size_,
                             static_cast<T>(1e-5));
  } else {
    // FIXME not sure if this is correct
    MyMemcpyCPU2CPU(mean_data, mean_cache_->GetCPUDataPtr(),
                    mean_cache_->GetSize());
    scale_cpu(mean_data, channels_, scale_factor_);
    MyMemcpyCPU2CPU(variance_data, sqrt_variance_cache_->GetCPUDataPtr(),
                    sqrt_variance_cache_->GetSize());
    scale_cpu(variance_data, channels_, scale_factor_);
    add_row_vector_cpu<T>(standarded_cache_->GetCPUDataPtr(), mean_data,
                          channels_, spatial_size_, batch_size_, -1);
    divide_row_vector_cpu<T>(standarded_cache_->GetCPUDataPtr(), variance_data,
                             channels_, spatial_size_, batch_size_,
                             static_cast<T>(1e-5));
  }
  // compute top data
  MyMemcpyCPU2CPU(top_data, standarded_cache_->GetCPUDataPtr(),
                  standarded_cache_->GetSize() * sizeof(T));
  multiply_row_vector_cpu<T>(top_data, gama_data, channels_, spatial_size_,
                             batch_size_);
  temp_cache_ = tensor_sum_cpu<T>(top_data, channels_);
  add_row_vector_cpu<T>(top_data, beta_data, channels_, spatial_size_,
                        batch_size_);
}

template <typename T>
void BatchNorm<T>::BackwardCPU(const std::vector<TensorPtr<T>>& top,
                               const std::vector<TensorPtr<T>>& bottom) {
  const T* top_diff = top[0]->GetCPUDiffPtr();
  const T* top_data = top[0]->GetCPUDataPtr();
  const T* bottom_data = bottom[0]->GetCPUDataPtr();
  T* bottom_diff = bottom[0]->GetCPUDiffPtr();
  const T* gama_data = gama_->GetCPUDataPtr();
  T* gama_diff = gama_->GetCPUDiffPtr();
  T* beta_diff = beta_->GetCPUDiffPtr();
  const T* mean_data = mean_->GetCPUDataPtr();
  const T* variance_data = sqrt_variance_->GetCPUDataPtr();
  // compute beta diff
  T* temp_data;
  MyMallocCPU(reinterpret_cast<void**>(&temp_data),
              sizeof(T) * batch_size_ * channels_);
  row_sum_cpu(top_diff, temp_data, channels_, spatial_size_, batch_size_);
  col_sum_cpu(temp_data, beta_diff, batch_size_, channels_, 1);
  // compute gama diff
  T* temp_temp_data;
  MyMallocCPU(reinterpret_cast<void**>(&temp_temp_data),
              sizeof(T) * batch_size_ * channels_ * spatial_size_);
  multiply_two_vec_cpu<T>(top_diff, standarded_cache_->GetCPUDataPtr(),
                          temp_temp_data,
                          batch_size_ * channels_ * spatial_size_);
  row_sum_cpu(temp_temp_data, temp_data, channels_, spatial_size_, batch_size_);
  col_sum_cpu(temp_data, gama_diff, batch_size_, channels_, 1);
  // MyMemFreeCPU(temp_temp_data);
  // MyMemFreeCPU(temp_data);
  // compute bottom diff
  // temp_temp_data is the partial diff of top_data to standarded_cache
  int n = batch_size_ * spatial_size_;
  MyMemcpyCPU2CPU(temp_temp_data, top_diff, sizeof(T) * n * channels_);
  multiply_row_vector_cpu<T>(temp_temp_data, gama_data, channels_,
                             spatial_size_, batch_size_);
  row_sum_cpu(temp_temp_data, temp_data, channels_, spatial_size_, batch_size_);
  MyMemcpyCPU2CPU(bottom_diff, temp_temp_data, sizeof(T) * n * channels_);
  scale_cpu<T>(bottom_diff, n * channels_, n);
  // temp_temp_temp_data is the sum of temp_temp_data
  T* temp_temp_temp_data;
  MyMallocCPU(reinterpret_cast<void**>(&temp_temp_temp_data),
              sizeof(T) * channels_);
  col_sum_cpu(temp_data, temp_temp_temp_data, batch_size_, channels_, 1);
  add_row_vector_cpu<T>(bottom_diff, temp_temp_temp_data, channels_,
                        spatial_size_, batch_size_, -1);
  multiply_two_vec_cpu<T>(temp_temp_data, standarded_cache_->GetCPUDataPtr(),
                          temp_temp_data, n * channels_);
  row_sum_cpu(temp_temp_data, temp_data, channels_, spatial_size_, batch_size_);
  col_sum_cpu(temp_data, temp_temp_temp_data, batch_size_, channels_, 1);
  multiply_row_vector_cpu<T>(standarded_cache_->GetCPUDataPtr(),
                             temp_temp_temp_data, channels_, spatial_size_,
                             batch_size_);
  add_two_vec_cpu<T>(bottom_diff, standarded_cache_->GetCPUDataPtr(), -1,
                     n * channels_);
  divide_row_vector_cpu<T>(bottom_diff, variance_data, channels_, spatial_size_,
                           batch_size_, 1e-5);
  scale_cpu<T>(bottom_diff, n * channels_, 1.0f / n);
}

template class BatchNorm<float>;

}  // namespace my_tensor
