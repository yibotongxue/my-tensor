// Copyright 2024 yibotongxue

#include "data-loader.hpp"

#include <array>
#include <memory>
#include <vector>

#include "common.hpp"
#include "tensor.hpp"

namespace my_tensor {
std::array<TensorPtr<>, 2> DataLoader::GetNext() {
  std::array<TensorPtr<>, 2> top;
  const int im_size = dataset_->GetHeight() * dataset_->GetWidth();
  std::vector<int> data_shape = {batch_size_, dataset_->GetChannel(),
                                 dataset_->GetHeight(), dataset_->GetWidth()};
  std::vector<int> label_shape = {batch_size_};
  top[0] = std::make_shared<Tensor<>>(data_shape);
  top[1] = std::make_shared<Tensor<>>(label_shape);
  auto&& image_data =
      dataset_->GetImageSpanBetweenAnd(index_, index_ + batch_size_);
  auto&& label_data =
      dataset_->GetLabelSpanBetweenAnd(index_, index_ + batch_size_);
  if (MyTensorContext::on_cpu()) {
    top[0]->SetCPUData(image_data.data(), image_data.size());
    top[1]->SetCPUData(label_data.data(), batch_size_);
  } else {
    top[0]->SetGPUData(image_data.data(), image_data.size());
    top[1]->SetGPUData(label_data.data(), batch_size_);
  }
  index_ += batch_size_;
  return top;
}
}  // namespace my_tensor
