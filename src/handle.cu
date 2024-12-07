// Copyright 2024 yibotongxue

#include "handle.hpp"

namespace my_tensor {
HandlePtr Handle::handle_ = nullptr;

my_tensor::HandlePtr handle = my_tensor::Handle::GetInstance();
}  // namespace my_tensor
