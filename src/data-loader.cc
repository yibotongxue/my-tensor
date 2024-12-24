// Copyright 2024 yibotongxue

#include <array>
#include <memory>
#include <vector>

#include "common.hpp"
#include "data-loader.hpp"
#include "tensor.hpp"

namespace my_tensor {
std::array<TensorPtr<>, 2> DataLoader::GetNext() {
  std::array<TensorPtr<>, 2> top;
  const auto& image_data = dataset_->GetImage();
  const auto& label_data = dataset_->GetLabel();
  const int im_size = dataset_->GetHeight() * dataset_->GetWidth();
  std::vector<int> data_shape = {batch_size_, 1, dataset_->GetHeight(),
                                 dataset_->GetWidth()};
  std::vector<int> label_shape = {batch_size_};
  top[0] = std::make_shared<Tensor<>>(data_shape);
  top[1] = std::make_shared<Tensor<>>(label_shape);
  if (MyTensorContext::on_cpu()) {
    top[0]->SetCPUData(image_data.data() + index_ * im_size, batch_size_ * im_size);
    std::vector<float> temp(batch_size_);
    for (int i = 0; i < batch_size_; i++) {
      temp[i] = static_cast<float>(label_data[index_ + i]);
    }
    top[1]->SetCPUData(temp.data(), batch_size_);
  } else {
    top[0]->SetGPUData(image_data.data() + index_ * im_size, batch_size_ * im_size);
    std::vector<float> temp(batch_size_);
    for (int i = 0; i < batch_size_; i++) {
      temp[i] = static_cast<float>(label_data[index_ + i]);
    }
    top[1]->SetGPUData(temp.data(), batch_size_);
  }
  index_ += batch_size_;
  return top;
}
}  // namespace my_tensor
