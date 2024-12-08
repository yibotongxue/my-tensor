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

#include "error.hpp"
#include "layer-parameter.hpp"
#include "softmax.hpp"
#include "tensor.hpp"

namespace my_tensor {

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

template class Softmax<>;

}  // namespace my_tensor
