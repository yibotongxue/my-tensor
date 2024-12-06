// Copyright 2024 yibotongxue

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "dataset.h"
#include "layer-facade.cuh"
#include "tensor-facade.cuh"
#include "tensor.cuh"

namespace py = pybind11;

PYBIND11_MODULE(mytensor, m) {
  py::class_<TensorFacade>(m, "Tensor", py::buffer_protocol())
      .def(py::init<const std::vector<int> &>(), py::arg("shape"))
      .def("reshape", &TensorFacade::Reshape, py::arg("shape"))
      .def("set_data", &TensorFacade::SetData, py::arg("data"))
      .def("set_grad", &TensorFacade::SetGrad, py::arg("grad"))
      .def("data", &TensorFacade::GetData)
      .def("grad", &TensorFacade::GetGrad)
      .def("to_cpu", &TensorFacade::ToCPU)
      .def("to_gpu", &TensorFacade::ToGPU)
      .def("copy", &TensorFacade::Copy)
      .def_static("from_numpy", &TensorFacade::FromNumpy, py::arg("data"))
      .def("shape", &TensorFacade::GetShape)
      .def_buffer([](TensorFacade &m) -> py::buffer_info {
        return py::buffer_info(
            m.data(), sizeof(float), py::format_descriptor<float>::format(),
            m.GetShape().size(), m.GetShape(), m.GetByteStride());
      });
  py::class_<ReluFacade>(m, "Relu")
      .def(py::init<>())
      .def("forward", &ReluFacade::Forward, py::arg("input"),
           "Perform forward propagation with ReLU")
      .def("backward", &ReluFacade::Backward, py::arg("output"),
           "Perform backward propagation with ReLU");
  py::class_<SigmoidFacade>(m, "Sigmoid")
      .def(py::init<>())
      .def("forward", &SigmoidFacade::Forward, py::arg("input"),
           "Perform forward propagation with Sigmoid")
      .def("backward", &SigmoidFacade::Backward, py::arg("output"),
           "Perform backward propagation with Sigmoid");
  py::class_<LinearFacade>(m, "Linear")
      .def(py::init<int, int>())
      .def("forward", &LinearFacade::Forward, py::arg("input"),
           "Perform forward propagation with Linear")
      .def("backward", &LinearFacade::Backward, py::arg("output"),
           "Perform backward propagation with Linear")
      .def("weight", &LinearFacade::GetWeight)
      .def("bias", &LinearFacade::GetBias)
      .def("set_weight", &LinearFacade::SetWeight, py::arg("weight"))
      .def("set_bias", &LinearFacade::SetBias, py::arg("bias"));
  py::class_<ConvolutionFacade>(m, "Conv")
      .def(py::init<int, int, int>())
      .def("forward", &ConvolutionFacade::Forward, py::arg("input"),
           "Perform forward propagation with Convolutionolution")
      .def("backward", &ConvolutionFacade::Backward, py::arg("output"),
           "Perform backward propagation with Convolutionolution")
      .def("kernel", &ConvolutionFacade::GetKernel)
      .def("bias", &ConvolutionFacade::GetBias)
      .def("set_kernel", &ConvolutionFacade::SetKernel, py::arg("kernel"))
      .def("set_bias", &ConvolutionFacade::SetBias, py::arg("bias"));
  py::class_<PoolingFacade>(m, "Pooling")
      .def(py::init<int, int, int>())
      .def("forward", &PoolingFacade::Forward, py::arg("input"),
           "Perform forward propagation with Pooling")
      .def("backward", &PoolingFacade::Backward, py::arg("output"),
           "Perform backward propagation with Pooling");
  py::class_<SoftmaxFacade>(m, "Softmax")
      .def(py::init<int>())
      .def("forward", &SoftmaxFacade::Forward, py::arg("input"),
           "Perform forward propagation with Softmax")
      .def("backward", &SoftmaxFacade::Backward, py::arg("output"),
           "Perform backward propagation with Softmax");
  py::class_<CrossEntropyLossFacade>(m, "CrossEntropyLoss")
      .def(py::init<int>())
      .def("forward", &CrossEntropyLossFacade::Forward, py::arg("input"),
           py::arg("label"),
           "Perform forward propagation with CrossEntropyLoss")
      .def("backward", &CrossEntropyLossFacade::Backward, py::arg("output"),
           "Perform backward propagation with CrossEntropyLoss");
  py::class_<my_tensor::MnistDataset, std::shared_ptr<my_tensor::MnistDataset>>(
      m, "MnistDataset")
      .def(py::init<const std::string &, const std::string &>())
      .def("load_data", &my_tensor::MnistDataset::LoadData)
      .def("get_height", &my_tensor::MnistDataset::GetHeight)
      .def("get_width", &my_tensor::MnistDataset::GetWidth)
      .def("get_image", &my_tensor::MnistDataset::GetImage)
      .def("get_label", &my_tensor::MnistDataset::GetLabel)
      .def("get_size", &my_tensor::MnistDataset::GetSize);
}
