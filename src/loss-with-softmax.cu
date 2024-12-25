// Copyright 2024 yibotongxue

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <memory>
#include <ranges>  // NOLINT
#include <vector>

#include "error.hpp"
#include "layer-factory.hpp"
#include "loss-with-softmax.hpp"
#include "softmax.hpp"

namespace my_tensor {

template <typename T>
void LossWithSoftmax<T>::ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                                    const std::vector<TensorPtr<T>>& top) {
  CheckShape(bottom[0], bottom[1], top[0]);
  softmax_->ForwardGPU(softmax_bottom_, softmax_top_);
  const T* softmax_top_data = softmax_top_[0]->GetGPUDataPtr();
  const T* label_data = bottom[1]->GetGPUDataPtr();
  auto&& top_data = PTR_CAST(top[0]->GetGPUDataPtr());
  thrust::device_vector<T> temp_data(batch_size_);
  int channels = channels_;
  thrust::transform(
      thrust::counting_iterator(0), thrust::counting_iterator(batch_size_),
      temp_data.begin(),
      [softmax_top_data, label_data, channels] __device__(int i) -> T {
        return -std::log(
            softmax_top_data[i * channels + static_cast<int>(label_data[i])]);
      });
  top_data[0] =
      thrust::reduce(temp_data.begin(), temp_data.end()) / batch_size_;
}

template <typename T>
void LossWithSoftmax<T>::BackwardGPU(const std::vector<TensorPtr<T>>& top,
                                     const std::vector<TensorPtr<T>>& bottom) {
  CheckShape(bottom[0], bottom[1], top[0]);
  auto softmax_top_data = PTR_CAST(softmax_top_[0]->GetGPUDataPtr());
  const T* label_ptr = bottom[1]->GetGPUDataPtr();
  auto bottom_diff = PTR_CAST(bottom[0]->GetGPUDiffPtr());
  T batch_size = static_cast<T>(batch_size_);
  thrust::copy(softmax_top_data, softmax_top_data + softmax_top_[0]->GetSize(),
               bottom_diff);
  std::cout << std::endl;
  T* bottom_ptr = bottom[0]->GetGPUDiffPtr();

  int channels = channels_;
  thrust::for_each(
      thrust::counting_iterator(0), thrust::counting_iterator(batch_size_),
      [label_ptr, bottom_ptr, channels] __device__(int i) -> void {
        bottom_ptr[i * channels + static_cast<int>(label_ptr[i])] -= 1.0f;
      });
  thrust::transform(
      bottom_diff, bottom_diff + bottom[0]->GetSize(), bottom_diff,
      [batch_size] __device__(T val) -> T { return val / batch_size; });
}

template class LossWithSoftmax<>;

}  // namespace my_tensor
