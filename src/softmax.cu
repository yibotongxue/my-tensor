// Copyright 2024 yibotongxue

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include "error.h"
#include "layer-parameter.h"
#include "softmax.cuh"
#include "tensor.cuh"

namespace my_tensor {

template <typename T>
void Softmax<T>::CheckTensorCount(const std::vector<TensorPtr<T>>& bottom,
                                  const std::vector<TensorPtr<T>>& top) const {
  if (bottom.size() != 1) {
    throw SoftmaxError(
        "The bottom of convolution layer should have one tensor.");
  }
  if (top.size() != 1) {
    throw SoftmaxError("The top of convolution layer should have one tensor.");
  }
}

template <typename T>
void Softmax<T>::Reshape(const std::vector<TensorPtr<T>>& bottom,
                         const std::vector<TensorPtr<T>>& top) const {
  top[0]->Resize({batch_size_, channels_});
}

template <typename T>
void Softmax<T>::LayerSetUp(const std::vector<TensorPtr<T>>& bottom,
                            const std::vector<TensorPtr<T>>& top) {
  if (bottom[0]->GetShape().size() != 2) {
    throw SoftmaxError("The bottom of softmax should be two dimension.");
  }
  std::shared_ptr<SoftmaxParameter> param =
      std::dynamic_pointer_cast<SoftmaxParameter>(this->layer_param_);
  channels_ = param->channels_;
  if (bottom[0]->GetShape()[1] != channels_) {
    throw SoftmaxError(
        "The channels of bottom of softmax not match the layer.");
  }
  batch_size_ = bottom[0]->GetShape()[0];
  const std::vector<int> predict_shape{batch_size_};
  predict_.reset();
  predict_ = std::make_shared<Tensor<T>>(predict_shape);
}

template <typename T>
void Softmax<T>::ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                            const std::vector<TensorPtr<T>>& top) {
  CheckShape(bottom[0], top[0]);
  const auto& bottom_data = bottom[0]->GetCPUData();
  auto& top_data = top[0]->GetCPUData();
  auto& predict_data = predict_->GetCPUData();
  auto bottom_view = std::views::all(bottom_data);
  for (int i = 0; i < batch_size_; i++) {  // for each row
    auto sub_view = bottom_view | std::views::drop(i * channels_) |
                    std::views::take(channels_);
    auto max_postion = std::ranges::max_element(sub_view);
    predict_data[i] = std::distance(sub_view.begin(), max_postion);
    T max_value = *max_postion;
    auto exp_view = sub_view | std::views::transform([max_value](T val) -> T {
                      return static_cast<T>(std::exp(val - max_value));
                    });
    T sum_value =
        std::accumulate(exp_view.begin(), exp_view.end(), T(0), std::plus<T>());
    auto norm_view = exp_view | std::views::transform([sum_value](T val) -> T {
                       return static_cast<T>(val / sum_value);
                     });
    std::ranges::copy(norm_view, top_data.begin() + i * channels_);
  }
}

namespace {
template <typename T>
__global__ void GetMaxPerRow(const T* data, const int n, const int c,
                             T* postions, T* output) {
  CUDA_KERNEL_LOOP(i, n) {
    data = data + i * c;
    T max_val = static_cast<T>(-__FLT_MAX__);
    T max_pos = -1;
    for (int j = 0; j < c; j++) {
      if (data[j] >= max_val) {
        max_val = data[j];
        max_pos = j;
      }
    }
    postions[i] = max_pos;
    output[i] = max_val;
  }
}
}  // namespace

template <typename T>
void Softmax<T>::ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                            const std::vector<TensorPtr<T>>& top) {
  const auto& bottom_data = bottom[0]->GetGPUData();
  const T* bottom_ptr = bottom[0]->GetGPUDataPtr();
  auto& top_data = top[0]->GetGPUData();
  T* predict_ptr = predict_->GetGPUDataPtr();
  thrust::device_vector<int> keys(batch_size_ * channels_);
  int channels = channels_;
  // generate key
  thrust::transform(
      thrust::counting_iterator(0),
      thrust::counting_iterator(batch_size_ * channels_), keys.begin(),
      [channels] __device__(int i) -> int { return (i / channels) + 1; });
  thrust::device_vector<int> output_keys(batch_size_);
  thrust::device_vector<T> max_values(batch_size_);
  T* max_ptr = RAW_PTR(max_values);
  GetMaxPerRow<T><<<CudaGetBlocks(batch_size_), kCudaThreadNum>>>(
      bottom_ptr, batch_size_, channels_, predict_ptr, max_ptr);
  // substract the max element
  thrust::transform(
      thrust::counting_iterator(0),
      thrust::counting_iterator(batch_size_ * channels_), bottom_data.begin(),
      top_data.begin(), [max_ptr, channels] __device__(int i, T val) -> T {
        return static_cast<T>(std::exp(val - max_ptr[i / channels]));
      });
  // compute normalization factor
  thrust::reduce_by_key(keys.begin(), keys.end(), top_data.begin(),
                        output_keys.begin(), max_values.begin(),
                        thrust::equal_to<int>(), thrust::plus<T>());
  // noramlization
  thrust::transform(thrust::counting_iterator(0),
                    thrust::counting_iterator(batch_size_ * channels_),
                    top_data.begin(), top_data.begin(),
                    [max_ptr, channels] __device__(int i, T val) -> T {
                      return static_cast<T>(val / max_ptr[i / channels]);
                    });
}

template <typename T>
void Softmax<T>::CheckShape(const TensorPtr<T> bottom,
                            const TensorPtr<T> top) const {
#ifdef DEBUG
  if (bottom->GetShape().size() != 2) {
    throw SoftmaxError(
        "The bottom of softmax layer should be a two dimension tensor.");
  }
  if (top->GetShape().size() != 2) {
    throw SoftmaxError(
        "The top of softmax layer should be a two dimension tensor.");
  }
  CHECK_SAME_SHAPE(bottom, top)
  if (bottom->GetShape()[0] != batch_size_) {
    throw SoftmaxError(
        "The batch size of bottom of softmax layer not match layer.");
  }
  if (bottom->GetShape()[1] != channels_) {
    throw SoftmaxError(
        "The channels size of bottom of softmax layer not match layer.");
  }
#endif  // DEBUG
}

template class Softmax<>;

}  // namespace my_tensor
