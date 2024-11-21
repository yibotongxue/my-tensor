#include "tensor.cuh"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "tensor-facade.cuh"

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

std::vector<int> TensorFacade::GetByteStride() const {
  const auto& shape = GetShape();
  std::vector<int> stride(shape.size(), sizeof(float));
  for (int i = shape.size() - 2; i >= 0; --i) {
      stride[i] = stride[i + 1] * shape[i + 1];
  }
  return stride;
}
