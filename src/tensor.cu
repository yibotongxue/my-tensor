#include <tensor.cuh>
#include <utils.cuh>

#include <numeric>
#include <iostream>

namespace my_tensor {
Tensor::Tensor(const std::vector<int>& shape)
  : shape_(shape) {
  size_ = std::accumulate(
    shape_.begin(), shape_.end(), 1, std::multiplies<int>());
  data_ = thrust::device_vector<float>(size_);
  diff_ = thrust::device_vector<float>(size_);
}

Tensor::Tensor(const Tensor& tensor)
  : shape_(tensor.shape_), size_(tensor.size_),
    data_(tensor.data_), diff_(tensor.diff_) {
}

Tensor& Tensor::operator=(const Tensor& tensor) {
  if (this == &tensor) {
    return *this;
  }
  shape_ = tensor.shape_;
  size_ = tensor.size_;
  data_ = tensor.data_;
  diff_ = tensor.diff_;
  return *this;
}

Tensor::Tensor(Tensor&& tensor)
  : shape_(tensor.shape_), size_(tensor.size_),
    data_(std::move(tensor.data_)), diff_(std::move(tensor.diff_)) {
}

Tensor& Tensor::operator=(Tensor&& tensor) {
  shape_ = tensor.shape_;
  size_ = tensor.size_;
  data_ = std::move(tensor.data_);
  diff_ = std::move(tensor.diff_);
  return *this;
}
}  // namespace my_tensor
