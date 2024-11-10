// Copyright 2024 yibotongxue

#ifndef INCLUDE_DATA_LOADER_CUH_
#define INCLUDE_DATA_LOADER_CUH_

#include <array>
#include <vector>

#include "dataset.h"
#include "tensor.cuh"

namespace my_tensor {
class DataLoader {
 public:
  explicit DataLoader(DatasetPtr dataset, int batch_size)
      : dataset_(dataset), batch_size_(batch_size), index_(0) {}

  bool HasNext() const { return index_ + batch_size_ <= dataset_->GetSize(); }

  std::array<TensorPtr<>, 2> GetNext();

  void Reset() { index_ = 0; }

 private:
  DatasetPtr dataset_;
  int batch_size_;
  int index_;
};  // class DataLoader
}  // namespace my_tensor

#endif  // INCLUDE_DATA_LOADER_CUH_
