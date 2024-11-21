#ifndef PYTHON_TENSOR_FACADE_CUH_
#define PYTHON_TENSOR_FACADE_CUH_

#include "tensor.cuh"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class TensorFacade {
 public:
  enum DeviceType { CPU, GPU };
  TensorFacade() : tensor_(std::make_shared<my_tensor::Tensor<float>>()), type_(DeviceType::CPU) {}
  explicit TensorFacade(const std::vector<int>& shape) 
      : tensor_(std::make_shared<my_tensor::Tensor<float>>(shape)), type_(DeviceType::CPU) {}

  void Reshape(const std::vector<int>& shape) {
      tensor_->Reshape(shape); 
  }

  void SetData(const std::vector<float>& data) {
    if (OnCPU()) {
      tensor_->SetCPUData(data);
    } else {
      tensor_->SetGPUData(data);
    }
  }

  void SetGrad(const std::vector<float>& diff) {
    if (OnCPU()) {
      tensor_->SetCPUDiff(diff);
    } else {
      tensor_->SetGPUDiff(diff);
    }
  }

  static TensorFacade FromNumpy(const py::array_t<float>& data);

  py::array_t<float> GetData() const;
  py::array_t<float> GetGrad() const;

  float* data() {
    return tensor_->GetCPUDataPtr();
  }

  const std::vector<int>& GetShape() const {
    return tensor_->GetShape();
  }

  std::vector<int> GetByteStride() const;

  my_tensor::TensorPtr<float> GetTensor() noexcept { return tensor_; }

  void ToCPU() noexcept { type_ = DeviceType::CPU; }
  void ToGPU() noexcept { type_ = DeviceType::GPU; }
  bool OnCPU() const noexcept { return type_ == DeviceType::CPU; }
  bool OnGPU() const noexcept { return type_ == DeviceType::GPU; }

 private:
  my_tensor::TensorPtr<float> tensor_;
  DeviceType type_;
};

#endif  // PYTHON_TENSOR_FACADE_CUH_
