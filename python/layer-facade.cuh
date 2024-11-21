#ifndef PYTHON_LAYER_FACADE_CUH_
#define PYTHON_LAYER_FACADE_CUH_

#include "layer.cuh"
#include "relu.cuh"
#include "sigmoid.cuh"
#include "linear.cuh"
#include "tensor-facade.cuh"
#include "layer-parameter.h"
#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

class ReluFacade {
 public:
  ReluFacade() : relu_(nullptr) {
    my_tensor::LayerParameterPtr param = std::make_shared<my_tensor::ReluParameter>();
    relu_.reset(new my_tensor::Relu<float>(param));
  }

  TensorFacade Forward(TensorFacade input);

  TensorFacade Backward(TensorFacade output);

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

  TensorFacade Forward(TensorFacade input);

  TensorFacade Backward(TensorFacade output);

 private:
  std::shared_ptr<my_tensor::Sigmoid<float>> sigmoid_;
  TensorFacade input_cache_;
};

class LinearFacade {
 public:
  LinearFacade(int input_feature, int output_feature) : linear_(nullptr), param_set_(false) {
    auto param = std::make_shared<my_tensor::LinearParameter>();
    param->input_feature_ = input_feature;
    param->output_feature_ = output_feature;
    auto weight_param = std::make_shared<my_tensor::XavierFillerParameter>();
    auto bias_param = std::make_shared<my_tensor::ZeroFillerParameter>();
    weight_param->n_in_ = input_feature;
    weight_param->n_out_ = output_feature;
    param->weight_filler_parameter_ = weight_param;
    param->bias_filler_parameter_ = bias_param;
    linear_.reset(new my_tensor::Linear<float>(param));
  }

  TensorFacade Forward(TensorFacade input);

  TensorFacade Backward(TensorFacade output);

  const TensorFacade& GetWeight() const {
    return weight_cache_;
  }
  void SetWeight(const TensorFacade& weight) {
    weight_cache_ = weight;
    param_set_ = true;
    // linear_->GetWeight()->SetCPUData(weight.GetTensor()->GetCPUData().begin(), weight.GetTensor()->GetCPUData().end());
  }

  const TensorFacade& GetBias() const {
    return bias_cache_;
  }
  void SetBias(const TensorFacade& bias) {
    bias_cache_ = bias;
    param_set_ = true;
    // linear_->GetBias()->SetCPUData(bias.GetTensor()->GetCPUData().begin(), bias.GetTensor()->GetCPUData().end());
  }

 private:
  std::shared_ptr<my_tensor::Linear<float>> linear_;
  TensorFacade input_cache_;
  TensorFacade weight_cache_;
  TensorFacade bias_cache_;
  bool param_set_;
};

#endif  // PYTHON_LAYER_FACADE_CUH_
