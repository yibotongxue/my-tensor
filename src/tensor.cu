#include <tensor.cuh>
#include <utils.cuh>
#include <error.h>

#include <numeric>
#include <iostream>

namespace my_tensor {
template <typename T>
Tensor<T>::Tensor(const std::vector<int>& shape)
  : shape_(shape) {
  size_ = std::accumulate(
    shape_.begin(), shape_.end(), 1, std::multiplies<int>());
  data_ = thrust::device_vector<T>(size_);
  diff_ = thrust::device_vector<T>(size_);
  CheckShape();
}

template <typename T>
Tensor<T>::Tensor(const Tensor<T>& tensor)
  : shape_(tensor.shape_), size_(tensor.size_),
    data_(tensor.data_), diff_(tensor.diff_) {
  *this = tensor;
  CheckShape();
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& tensor) {
  if (this == &tensor) {
    return *this;
  }
  shape_ = tensor.shape_;
  size_ = tensor.size_;
  data_ = tensor.data_;
  diff_ = tensor.diff_;
  CheckShape();
  return *this;
}

template <typename T>
Tensor<T>::Tensor(Tensor<T>&& tensor)
  : shape_(std::move(tensor.shape_)), size_(tensor.size_),
    data_(std::move(tensor.data_)), diff_(std::move(tensor.diff_)) {
  tensor.Clear();
  CheckShape();
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor<T>&& tensor) {
  shape_ = std::move(tensor.shape_);
  size_ = tensor.size_;
  data_ = std::move(tensor.data_);
  diff_ = std::move(tensor.diff_);
  tensor.Clear();
  CheckShape();
  return *this;
}

template <typename T>
void Tensor<T>::SetData(const std::vector<T>& data) {
  data_ = data;
  CheckShape();
}

template <typename T>
void Tensor<T>::SetData(std::vector<T>&& data) {
  data_ = data;
  CheckShape();
}

template <typename T>
void Tensor<T>::SetDiff(const std::vector<T>& diff) {
  diff_ = diff;
  CheckShape();
}

template <typename T>
void Tensor<T>::SetDiff(std::vector<T>&& diff) {
  diff_ = diff;
  CheckShape();
}

template <typename T>
void Tensor<T>::Clear() {
  shape_ = {0};
  size_ = 0;
  data_.clear();
  diff_.clear();
  CheckShape();
}

template <typename T>
void Tensor<T>::CheckShape() const {
  auto shape_size = std::accumulate(
    shape_.begin(), shape_.end(), 1, std::multiplies<int>());
  if (shape_size != size_) {
    throw ShapeError("Size not match the shape.");
  }
  if (data_.size() != shape_size) {
    throw ShapeError("Data size not match the shape.");
  }
  if (diff_.size() != shape_size) {
    throw ShapeError("Diff size not match the shape.");
  }
}

template class Tensor<>;
template class Tensor<double>;
}  // namespace my_tensor
