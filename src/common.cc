// Copyright 2024 yibotongxue

#include "common.hpp"

#include <unistd.h>

#include <ctime>
#include <thread>  // NOLINT

#include "utils.hpp"

namespace my_tensor {
MyTensorContext::~MyTensorContext() {}

MyTensorContext& MyTensorContext::Get() {
  static thread_local MyTensorContext instance;
  return instance;
}

MyTensorContext::MyTensorContext()
    : device_type_(CPU), random_engine_(std::random_device{}()) {  // NOLINT
  // from https://github.com/BVLC/caffe.git
  // Try to create a cublas handler, and report an error if failed (but we will
  // keep the program running as one might just want to run CPU code).
}

}  // namespace my_tensor
