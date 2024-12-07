// Copyright 2024 yibotongxue

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <vector>

#include "tensor-facade.hpp"
#include "tensor.hpp"

namespace py = pybind11;

TensorFacade TensorFacade::FromNumpy(const py::array_t<float>& data) {
  py::buffer_info info = data.request();
  std::vector<int> shape(info.shape.begin(), info.shape.end());

  // 创建并初始化 Tensor
  TensorFacade tensor(shape);

  // 设置数据
  tensor.tensor_->SetCPUData(static_cast<float*>(info.ptr),
                             static_cast<float*>(info.ptr) + info.size);
  return tensor;
}

py::array_t<float> TensorFacade::GetData() const {
  const auto& shape = GetShape();
  const auto& stride = GetByteStride();

  return py::array_t<float>(shape, stride, tensor_->GetCPUDataPtr(),
                            py::cast(this));
}

py::array_t<float> TensorFacade::GetGrad() const {
  const auto& shape = GetShape();
  const auto& stride = GetByteStride();

  return py::array_t<float>(shape, stride, tensor_->GetCPUDiffPtr(),
                            py::cast(this));
}

std::vector<int> TensorFacade::GetByteStride() const {
  const auto& shape = GetShape();
  std::vector<int> stride(shape.size(), sizeof(float));
  for (int i = shape.size() - 2; i >= 0; --i) {
    stride[i] = stride[i + 1] * shape[i + 1];
  }
  return stride;
}
