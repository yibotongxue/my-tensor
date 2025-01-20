// Copyright 2025 yibotongxue

#include "batchnorm.hpp"

#include <memory>
#include <vector>

#include "blas.hpp"
#include "error.hpp"
#include "memory-util.hpp"

namespace my_tensor {

template <Arithmetic T>
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

template <Arithmetic T>
void BatchNorm<T>::Reshape(const std::vector<TensorPtr<T>>& bottom,
                           const std::vector<TensorPtr<T>>& top) const {
  top[0]->Resize(bottom[0]->GetShape());
}

template <Arithmetic T>
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
  if (MyTensorContext::on_cpu()) {
    Fill_CPU<T>(mean_cache_->GetCPUDataPtr(), channels_, 0);
    Fill_CPU<T>(sqrt_variance_cache_->GetCPUDataPtr(), channels_, 0);
  } else {
    Fill_GPU<T>(mean_cache_->GetGPUDataPtr(), channels_, 0);
    Fill_GPU<T>(sqrt_variance_cache_->GetGPUDataPtr(), channels_, 0);
  }
  mean_ = std::make_shared<Tensor<T>>(shape);
  sqrt_variance_ = std::make_shared<Tensor<T>>(shape);
  standarded_cache_ = std::make_shared<Tensor<T>>(bottom[0]->GetShape());
  move_scale_factor_ = param->move_scale_factor_;
  batch_size_ = bottom[0]->GetShape()[0];
  spatial_size_ = bottom[0]->GetShape()[2] * bottom[0]->GetShape()[3];
  temp_cache_ = std::make_shared<Tensor<T>>(shape);
  temp_cache1_ = std::make_shared<Tensor<T>>(
      std::vector<int>{batch_size_, channels_, 1, 1});
  temp_cache2_ = std::make_shared<Tensor<T>>(bottom[0]->GetShape());
  int one_size = std::max(batch_size_, spatial_size_);
  all_ones_ = std::make_shared<Tensor<T>>(std::vector<int>{1, one_size, 1, 1});
  Fill_CPU<T>(all_ones_->GetCPUDataPtr(), one_size, 1);
}

template <Arithmetic T>
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
    row_sum_cpu(bottom_data, temp_cache1_->GetCPUDataPtr(), channels_,
                spatial_size_, batch_size_);
    col_sum_cpu(temp_cache1_->GetCPUDataPtr(), mean_data, batch_size_,
                channels_, 1);
    scale_cpu(
        mean_data, channels_,
        static_cast<T>(1.0) / static_cast<T>(batch_size_ * spatial_size_));
    MyMemcpyCPU2CPU(standarded_cache_->GetCPUDataPtr(), bottom_data,
                    bottom[0]->GetSize() * sizeof(T));
    add_row_vector_cpu<T>(standarded_cache_->GetCPUDataPtr(), mean_data,
                          channels_, spatial_size_, batch_size_, -1);
    square_cpu(standarded_cache_->GetCPUDataPtr(),
               temp_cache2_->GetCPUDataPtr(), standarded_cache_->GetSize());
    row_sum_cpu(temp_cache2_->GetCPUDataPtr(), temp_cache1_->GetCPUDataPtr(),
                channels_, spatial_size_, batch_size_);
    col_sum_cpu(temp_cache1_->GetCPUDataPtr(), variance_data, batch_size_,
                channels_, 1);
    scale_cpu(
        variance_data, channels_,
        static_cast<T>(1.0) / static_cast<T>(batch_size_ * spatial_size_));
    sqrt_cpu(variance_data, variance_data, channels_);
    scale_cpu(mean_cache_->GetCPUDataPtr(), channels_, move_scale_factor_);
    add_two_vec_cpu(mean_cache_->GetCPUDataPtr(), mean_data,
                    static_cast<T>(1.0 - move_scale_factor_), channels_);
    scale_cpu(sqrt_variance_cache_->GetCPUDataPtr(), channels_,
              move_scale_factor_);
    add_two_vec_cpu(sqrt_variance_cache_->GetCPUDataPtr(), variance_data,
                    static_cast<T>(1.0 - move_scale_factor_), channels_);
    divide_row_vector_cpu<T>(standarded_cache_->GetCPUDataPtr(), variance_data,
                             channels_, spatial_size_, batch_size_,
                             static_cast<T>(1e-5));
  } else {
    MyMemcpyCPU2CPU(mean_data, mean_cache_->GetCPUDataPtr(),
                    mean_cache_->GetSize());
    MyMemcpyCPU2CPU(variance_data, sqrt_variance_cache_->GetCPUDataPtr(),
                    sqrt_variance_cache_->GetSize());
    MyMemcpyCPU2CPU(standarded_cache_->GetCPUDataPtr(), bottom_data,
                    bottom[0]->GetSize() * sizeof(T));
    add_row_vector_cpu<T>(standarded_cache_->GetCPUDataPtr(), mean_data,
                          channels_, spatial_size_, batch_size_, -1);
    divide_row_vector_cpu<T>(standarded_cache_->GetCPUDataPtr(), variance_data,
                             channels_, spatial_size_, batch_size_,
                             static_cast<T>(1e-5));
  }
  MyMemcpyCPU2CPU(top_data, standarded_cache_->GetCPUDataPtr(),
                  standarded_cache_->GetSize() * sizeof(T));
  multiply_row_vector_cpu<T>(top_data, gama_data, channels_, spatial_size_,
                             batch_size_);
  add_row_vector_cpu<T>(top_data, beta_data, channels_, spatial_size_,
                        batch_size_);
}

template <Arithmetic T>
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
  row_sum_cpu(top_diff, temp_cache1_->GetCPUDataPtr(), channels_, spatial_size_,
              batch_size_);
  col_sum_cpu(temp_cache1_->GetCPUDataPtr(), beta_diff, batch_size_, channels_,
              1);
  multiply_two_vec_cpu<T>(top_diff, standarded_cache_->GetCPUDataPtr(),
                          temp_cache2_->GetCPUDataPtr(),
                          batch_size_ * channels_ * spatial_size_);
  row_sum_cpu(temp_cache2_->GetCPUDataPtr(), temp_cache1_->GetCPUDataPtr(),
              channels_, spatial_size_, batch_size_);
  col_sum_cpu(temp_cache1_->GetCPUDataPtr(), gama_diff, batch_size_, channels_,
              1);
  int n = batch_size_ * spatial_size_;
  MyMemcpyCPU2CPU(temp_cache2_->GetCPUDataPtr(), top_diff,
                  sizeof(T) * n * channels_);
  multiply_row_vector_cpu<T>(temp_cache2_->GetCPUDataPtr(), gama_data,
                             channels_, spatial_size_, batch_size_);
  row_sum_cpu(temp_cache2_->GetCPUDataPtr(), temp_cache1_->GetCPUDataPtr(),
              channels_, spatial_size_, batch_size_);
  MyMemcpyCPU2CPU(bottom_diff, temp_cache2_->GetCPUDataPtr(),
                  sizeof(T) * n * channels_);
  scale_cpu<T>(bottom_diff, n * channels_, n);
  col_sum_cpu(temp_cache1_->GetCPUDataPtr(), temp_cache_->GetCPUDataPtr(),
              batch_size_, channels_, 1);
  add_row_vector_cpu<T>(bottom_diff, temp_cache_->GetCPUDataPtr(), channels_,
                        spatial_size_, batch_size_, -1);
  multiply_two_vec_cpu<T>(temp_cache2_->GetCPUDataPtr(),
                          standarded_cache_->GetCPUDataPtr(),
                          temp_cache2_->GetCPUDataPtr(), n * channels_);
  row_sum_cpu(temp_cache2_->GetCPUDataPtr(), temp_cache1_->GetCPUDataPtr(),
              channels_, spatial_size_, batch_size_);
  col_sum_cpu(temp_cache1_->GetCPUDataPtr(), temp_cache_->GetCPUDataPtr(),
              batch_size_, channels_, 1);
  multiply_row_vector_cpu<T>(standarded_cache_->GetCPUDataPtr(),
                             temp_cache_->GetCPUDataPtr(), channels_,
                             spatial_size_, batch_size_);
  add_two_vec_cpu<T>(bottom_diff, standarded_cache_->GetCPUDataPtr(), -1,
                     n * channels_);
  divide_row_vector_cpu<T>(bottom_diff, variance_data, channels_, spatial_size_,
                           batch_size_, 1e-5);
  scale_cpu<T>(bottom_diff, n * channels_, 1.0f / n);
}

template class BatchNorm<float>;

}  // namespace my_tensor
