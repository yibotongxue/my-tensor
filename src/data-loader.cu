// Copyright 2024 yibotongxue

#include <memory>
#include <vector>

#include "data-loader.cuh"
#include "tensor.cuh"

namespace my_tensor {
std::vector<TensorPtr<>> DataLoader::GetNext() {
  std::vector<TensorPtr<>> top(2, nullptr);
  const auto& image_data = dataset_->GetImage();
  const auto& label_data = dataset_->GetLabel();
  const int im_size = dataset_->GetHeight() * dataset_->GetWidth();
  std::vector<int> data_shape = {batch_size_, 1, dataset_->GetHeight(),
                                 dataset_->GetWidth()};
  std::vector<int> label_shape = {batch_size_};
  top[0] = std::make_shared<Tensor<>>(data_shape);
  top[0]->SetGPUData(image_data.begin() + index_ * im_size,
                     image_data.begin() + (index_ + batch_size_) * im_size);
  top[1] = std::make_shared<Tensor<>>(label_shape);
  top[1]->SetGPUData(label_data.begin() + index_,
                     label_data.begin() + index_ + batch_size_);
  index_ += batch_size_;
  return top;
}
}  // namespace my_tensor
