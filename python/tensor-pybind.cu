#include "tensor.cuh"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <algorithm>

namespace py = pybind11;

class TensorFacade {
 public:
  explicit TensorFacade(const std::vector<int>& shape) : tensor_(std::make_shared<my_tensor::Tensor<float>>(shape)) {}

  void Reshape(const std::vector<int>& shape) { tensor_->Reshape(shape); }

  // void SetData(const std::vector<float>& data) {
  //   tensor_->SetCPUData(data);
  // }

  void SetData(const py::array_t<float>& data) {
    py::buffer_info info = data.request();
    // Flatten the NumPy array and copy the data
    tensor_->SetCPUData(static_cast<float*>(info.ptr), 
                        static_cast<float*>(info.ptr) + info.size);
  }

  float* data() {
    return tensor_->GetCPUDataPtr();
  }

  const std::vector<int>& GetShape() const {
    return tensor_->GetShape();
  }

  std::vector<int> GetByteStride() const {
    const std::vector<int> shape = GetShape();
    std::vector<int> result(shape.size(), 1);
    for (int i = 1; i < shape.size(); i++) {
      result[i] = result[i - 1] * shape[shape.size() - i];
    }
    std::reverse(result.begin(), result.end());
    for (int i = 0; i < result.size(); i++) {
      result[i] *= sizeof(float);
    }
    return result;
  }

 private:
  my_tensor::TensorPtr<float> tensor_;
};

PYBIND11_MODULE(mytensor, m) {
    py::class_<TensorFacade>(m, "Tensor", py::buffer_protocol())
        .def(py::init<const std::vector<int>&>(), py::arg("shape"))
        .def("reshape", &TensorFacade::Reshape, py::arg("shape"))
        .def("set_data", &TensorFacade::SetData, py::arg("data"))
        .def("shape", &TensorFacade::GetShape)
        .def_buffer([](TensorFacade &m) -> py::buffer_info {
          return py::buffer_info(
            m.data(),                               /* Pointer to buffer */
            sizeof(float),                          /* Size of one scalar */
            py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
            m.GetShape().size(),                                      /* Number of dimensions */
            m.GetShape(),                 /* Buffer dimensions */
            m.GetByteStride()
          );
        });
}
