// Copyright 2024 yibotongxue

#include "tensor.hpp"

#include <functional>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "error.hpp"

namespace my_tensor {
template <typename T>
Tensor<T>::Tensor()
    : shape_({0}),
      size_(0),
      data_(std::make_shared<SyncedVector<T>>()),
      diff_(nullptr) {
  CheckShape();
}

template <typename T>
Tensor<T>::Tensor(const std::vector<int>& shape) : shape_(shape) {
  size_ =
      std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
  data_ = std::make_shared<SyncedVector<T>>(size_);
  diff_ = nullptr;
  CheckShape();
}

template <typename T>
Tensor<T>::Tensor(const Tensor<T>& tensor)
    : shape_(tensor.shape_),
      size_(tensor.size_),
      data_(std::make_shared<SyncedVector<T>>(*tensor.data_)) {
  CheckShape();
  if (tensor.diff_ != nullptr) {
    diff_ = std::make_shared<SyncedVector<T>>(*tensor.diff_);
  } else {
    diff_ = nullptr;
  }
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& tensor) {
  if (this == &tensor) {
    return *this;
  }
  shape_ = tensor.shape_;
  size_ = tensor.size_;
  *data_ = *tensor.data_;
  if (tensor.diff_ != nullptr) {
    if (diff_ == nullptr) {
      diff_ = std::make_shared<SyncedVector<T>>(*tensor.diff_);
    } else {
      *diff_ = *tensor.diff_;
    }
  } else {
    diff_ = nullptr;
  }
  CheckShape();
  return *this;
}

template <typename T>
Tensor<T>::Tensor(Tensor<T>&& tensor)
    : shape_(std::move(tensor.shape_)),
      size_(tensor.size_),
      data_(std::make_shared<SyncedVector<T>>(std::move(*tensor.data_))) {
  if (tensor.diff_ != nullptr) {
    diff_ = std::make_shared<SyncedVector<T>>(std::move(*tensor.diff_));
  } else {
    diff_ = nullptr;
  }
  CheckShape();
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor<T>&& tensor) {
  shape_ = std::move(tensor.shape_);
  size_ = tensor.size_;
  *data_ = *std::move(tensor.data_);
  if (tensor.diff_ != nullptr) {
    if (diff_ == nullptr) {
      diff_ = std::make_shared<SyncedVector<T>>(std::move(*tensor.diff_));
    } else {
      *diff_ = *std::move(tensor.diff_);
    }
  } else {
    diff_ = nullptr;
  }
  CheckShape();
  return *this;
}

template <typename T>
void Tensor<T>::Reshape(const std::vector<int>& shape) {
  shape_ = shape;
  CheckShape();
}

template <typename T>
void Tensor<T>::Resize(const std::vector<int>& shape) {
  shape_ = shape;
  size_ =
      std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
  data_->Resize(size_);
  if (diff_ != nullptr) {
    diff_->Resize(size_);
  }
  CheckShape();
}

template <typename T>
void Tensor<T>::AllocateDiff() const {
  if (diff_ == nullptr) {
    diff_ = std::make_shared<SyncedVector<T>>(size_);
  }
}

template <typename T>
void Tensor<T>::CheckShape() const {
  auto shape_size =
      std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
  if (shape_size != size_) {
    throw ShapeError("Size not match the shape.");
  }
  if (data_->size() != shape_size) {
    throw ShapeError("Data size not match the shape.");
  }
  if (diff_ != nullptr && diff_->size() != shape_size) {
    throw ShapeError("Diff size not match the shape.");
  }
}

template class Tensor<>;
template class Tensor<int>;
}  // namespace my_tensor
