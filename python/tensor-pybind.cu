#include "tensor-facade.cuh"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // For automatic conversion of std::vector to Python lists

namespace py = pybind11;

PYBIND11_MODULE(mytensor, m) {
    py::class_<my_tensor::TensorFacade<float>>(m, "Tensor")
        .def(py::init<const std::vector<int>&>(), py::arg("shape"))
        .def("reshape", &my_tensor::TensorFacade<float>::Reshape, py::arg("shape"))
        .def("set_data", &my_tensor::TensorFacade<float>::SetData, py::arg("data"));
}
