// Copyright 2024 yibotongxue

#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <memory>
#include <ranges>  // NOLINT
#include <vector>

#include "accuracy.hpp"
#include "error.hpp"
#include "layer-factory.hpp"
#include "softmax.hpp"

namespace my_tensor {

namespace {
template <typename T>
__global__ void GetCorrect(const T* data, const T* label, const int n,
                           const int c, int* correct) {
  CUDA_KERNEL_LOOP(i, n) {
    const T* temp = data + i * c;
    T max_val = temp[0];
    T max_pos = 0;
    for (int j = 1; j < c; j++) {
      if (temp[j] > max_val) {
        max_val = temp[j];
        max_pos = j;
      }
    }
    correct[i] = label[i] == max_pos ? 1 : 0;
  }
}
}  // namespace

template <typename T>
void Accuracy<T>::ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                             const std::vector<TensorPtr<T>>& top) {
  CheckShape(bottom[0], bottom[1], top[0]);
  const T* bottom_ptr = bottom[0]->GetGPUDataPtr();
  const T* label_ptr = bottom[1]->GetGPUDataPtr();
  thrust::device_vector<int> correct_vec(batch_size_);
  int* correct_ptr = RAW_PTR(correct_vec);
  GetCorrect<T><<<CudaGetBlocks(batch_size_), kCudaThreadNum>>>(
      bottom_ptr, label_ptr, batch_size_, features_, correct_ptr);
  int correct = thrust::reduce(correct_vec.begin(), correct_vec.end(), 0,
                               thrust::plus<int>());
  thrust::device_ptr<T>(top[0]->GetGPUDataPtr())[0] =
      static_cast<T>(correct) / static_cast<T>(batch_size_);
}

template <typename T>
void Accuracy<T>::BackwardGPU(const std::vector<TensorPtr<T>>& top,
                              const std::vector<TensorPtr<T>>& bottom) {
  throw AccuracyError("Unimplemention error.");
}

template class Accuracy<>;

}  // namespace my_tensor
