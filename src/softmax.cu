// Copyright 2024 yibotongxue

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
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

template <Arithmetic T>
void Softmax<T>::ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                            const std::vector<TensorPtr<T>>& top) {
  auto bottom_data = PTR_CAST(bottom[0]->GetGPUDataPtr());
  auto top_data = PTR_CAST(top[0]->GetGPUDataPtr());
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
  // compute row max element
  thrust::reduce_by_key(keys.begin(), keys.end(), bottom_data,
                        output_keys.begin(), max_values.begin(),
                        thrust::equal_to<int>(), thrust::maximum<T>());
  // substract the max element
  thrust::transform(
      thrust::counting_iterator(0),
      thrust::counting_iterator(batch_size_ * channels_), bottom_data, top_data,
      [max_ptr, channels] __device__(int i, T val) -> T {
        return static_cast<T>(std::exp(val - max_ptr[i / channels]));
      });
  // compute normalization factor
  thrust::reduce_by_key(keys.begin(), keys.end(), top_data, output_keys.begin(),
                        max_values.begin(), thrust::equal_to<int>(),
                        thrust::plus<T>());
  // noramlization
  thrust::transform(thrust::counting_iterator(0),
                    thrust::counting_iterator(batch_size_ * channels_),
                    top_data, top_data,
                    [max_ptr, channels] __device__(int i, T val) -> T {
                      return static_cast<T>(val / max_ptr[i / channels]);
                    });
}

template class Softmax<>;

}  // namespace my_tensor
