#include "tensor.cuh"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class TensorFacade {
 public:
  TensorFacade() : tensor_(std::make_shared<my_tensor::Tensor<float>>()) {}
  explicit TensorFacade(const std::vector<int>& shape) 
      : tensor_(std::make_shared<my_tensor::Tensor<float>>(shape)) {}

  void Reshape(const std::vector<int>& shape) {
      tensor_->Reshape(shape); 
  }

  void SetData(const std::vector<float>& data) {
    tensor_->SetCPUData(data);
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

 private:
  my_tensor::TensorPtr<float> tensor_;
};
