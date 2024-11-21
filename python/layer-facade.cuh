#ifndef PYTHON_LAYER_FACADE_CUH_
#define PYTHON_LAYER_FACADE_CUH_

#include "layer.cuh"
#include "relu.cuh"
#include "tensor-facade.cuh"
#include "layer-parameter.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

class ReluFacade {
 public:
  ReluFacade() : relu_(nullptr) {
    my_tensor::LayerParameterPtr param = std::make_shared<my_tensor::ReluParameter>();
    relu_.reset(new my_tensor::Relu<float>(param));
  }

  TensorFacade Forward(TensorFacade input) {
    input_cache_ = input;
    TensorFacade output;
    relu_->SetUp({input.GetTensor()}, {output.GetTensor()});
    relu_->ForwardGPU({input.GetTensor()}, {output.GetTensor()});
    return output;
  }

  TensorFacade Backward(TensorFacade output) {
    relu_->BackwardGPU({output.GetTensor()}, {input_cache_.GetTensor()});
    return input_cache_;
  }

 private:
  std::shared_ptr<my_tensor::Relu<float>> relu_;
  TensorFacade input_cache_;
};

class SigmoidFacade {

};

#endif  // PYTHON_LAYER_FACADE_CUH_
