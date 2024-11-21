#include "tensor.cuh"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include "tensor-facade.cuh"

namespace py = pybind11;

PYBIND11_MODULE(mytensor, m) {
    py::class_<TensorFacade>(m, "Tensor", py::buffer_protocol())
        .def(py::init<const std::vector<int>&>(), py::arg("shape"))
        .def("reshape", &TensorFacade::Reshape, py::arg("shape"))
        .def("set_data", &TensorFacade::SetData, py::arg("data"))
        .def("data", &TensorFacade::GetData)
        .def("grad", &TensorFacade::GetGrad)
        .def_static("from_numpy", &TensorFacade::FromNumpy, py::arg("data"))
        .def("shape", &TensorFacade::GetShape)
        .def_buffer([](TensorFacade &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                              /* Pointer to buffer */
                sizeof(float),                         /* Size of one scalar */
                py::format_descriptor<float>::format(),/* Python struct-style format descriptor */
                m.GetShape().size(),                   /* Number of dimensions */
                m.GetShape(),                          /* Buffer dimensions */
                m.GetByteStride()                      /* Strides (in bytes) */
            );
        });
}
