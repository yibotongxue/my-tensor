#ifndef PYTHON_LAYER_FACADE_CUH_
#define PYTHON_LAYER_FACADE_CUH_

#include "layer.cuh"
#include "relu.cuh"
#include "sigmoid.cuh"
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
    if (input.OnCPU()) {
      relu_->ForwardCPU({input.GetTensor()}, {output.GetTensor()});
    } else {
      relu_->ForwardGPU({input.GetTensor()}, {output.GetTensor()});
    }
    return output;
  }

  TensorFacade Backward(TensorFacade output) {
    if (output.OnCPU()) {
      relu_->BackwardGPU({output.GetTensor()}, {input_cache_.GetTensor()});
    } else {
      relu_->BackwardGPU({output.GetTensor()}, {input_cache_.GetTensor()});
    }
    return input_cache_;
  }

 private:
  std::shared_ptr<my_tensor::Relu<float>> relu_;
  TensorFacade input_cache_;
};

class SigmoidFacade {
 public:
  SigmoidFacade() : sigmoid_(nullptr) {
    my_tensor::LayerParameterPtr param = std::make_shared<my_tensor::SigmoidParameter>();
    sigmoid_.reset(new my_tensor::Sigmoid<float>(param));
  }

  TensorFacade Forward(TensorFacade input) {
    input_cache_ = input;
    TensorFacade output;
    sigmoid_->SetUp({input.GetTensor()}, {output.GetTensor()});
    if (input.OnCPU()) {
      sigmoid_->ForwardCPU({input.GetTensor()}, {output.GetTensor()});
    } else {
      sigmoid_->ForwardGPU({input.GetTensor()}, {output.GetTensor()});
    }
    return output;
  }

  TensorFacade Backward(TensorFacade output) {
    if (output.OnCPU()) {
      sigmoid_->BackwardGPU({output.GetTensor()}, {input_cache_.GetTensor()});
    } else {
      sigmoid_->BackwardGPU({output.GetTensor()}, {input_cache_.GetTensor()});
    }
    return input_cache_;
  }

 private:
  std::shared_ptr<my_tensor::Sigmoid<float>> sigmoid_;
  TensorFacade input_cache_;
};

#endif  // PYTHON_LAYER_FACADE_CUH_
