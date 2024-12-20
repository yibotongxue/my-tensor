// Copyright 2024 yibotongxue

#ifndef INCLUDE_DATA_LOADER_HPP_
#define INCLUDE_DATA_LOADER_HPP_

#include <array>
#include <memory>
#include <vector>

#include "dataset.hpp"

namespace my_tensor {
template <typename T>

class Tensor;

class DataLoader {
 public:
  DataLoader(DatasetPtr dataset, int batch_size)
      : dataset_(dataset), batch_size_(batch_size), index_(0) {}

  bool HasNext() const { return index_ + batch_size_ <= dataset_->GetSize(); }

  std::array<std::shared_ptr<Tensor<float>>, 2> GetNext();

  void Reset() noexcept {
    Shuffle();
    index_ = 0;
  }

  std::vector<int> GetImageShape() const {
    return {batch_size_, dataset_->GetHeight(), dataset_->GetWidth()};
  }

  std::vector<int> GetLabelShape() const { return {batch_size_}; }

  void Shuffle() noexcept {}

 private:
  DatasetPtr dataset_;
  int batch_size_;
  int index_;
};  // class DataLoader
}  // namespace my_tensor

#endif  // INCLUDE_DATA_LOADER_HPP_
