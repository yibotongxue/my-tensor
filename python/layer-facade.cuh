// Copyright 2024 yibotongxue

#ifndef PYTHON_LAYER_FACADE_CUH_
#define PYTHON_LAYER_FACADE_CUH_

#include <pybind11/pybind11.h>

#include <iostream>
#include <memory>

#include "conv.cuh"
#include "layer-parameter.h"
#include "layer.cuh"
#include "linear.cuh"
#include "loss-with-softmax.cuh"
#include "pooling.cuh"
#include "relu.cuh"
#include "sigmoid.cuh"
#include "softmax.cuh"
#include "tensor-facade.cuh"

namespace py = pybind11;

class ReluFacade {
 public:
  ReluFacade() : relu_(nullptr) {
    my_tensor::LayerParameterPtr param =
        std::make_shared<my_tensor::ReluParameter>();
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
    my_tensor::LayerParameterPtr param =
        std::make_shared<my_tensor::SigmoidParameter>();
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
  LinearFacade(int input_feature, int output_feature)
      : linear_(nullptr), param_set_(false) {
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

  const TensorFacade& GetWeight() const { return weight_cache_; }
  void SetWeight(const TensorFacade& weight) {
    weight_cache_ = weight;
    param_set_ = true;
  }

  const TensorFacade& GetBias() const { return bias_cache_; }
  void SetBias(const TensorFacade& bias) {
    bias_cache_ = bias;
    param_set_ = true;
  }

 private:
  std::shared_ptr<my_tensor::Linear<float>> linear_;
  TensorFacade input_cache_;
  TensorFacade weight_cache_;
  TensorFacade bias_cache_;
  bool param_set_;
};

class ConvolutionFacade {
 public:
  ConvolutionFacade(int input_channel, int output_channel, int kernel_size)
      : conv_(nullptr), param_set_(false) {
    auto param = std::make_shared<my_tensor::ConvolutionParameter>();
    param->input_channels_ = input_channel;
    param->output_channels_ = output_channel;
    param->kernel_size_ = kernel_size;
    auto kernel_param = std::make_shared<my_tensor::HeFillerParameter>();
    auto bias_param = std::make_shared<my_tensor::ZeroFillerParameter>();
    kernel_param->n_ = input_channel * kernel_size * kernel_size;
    param->kernel_filler_parameter_ = kernel_param;
    param->bias_filler_parameter_ = bias_param;
    conv_.reset(new my_tensor::Convolution<float>(param));
  }

  TensorFacade Forward(TensorFacade input);

  TensorFacade Backward(TensorFacade output);

  const TensorFacade& GetKernel() const { return kernel_cache_; }
  void SetKernel(const TensorFacade& kernel) {
    kernel_cache_ = kernel;
    param_set_ = true;
  }

  const TensorFacade& GetBias() const { return bias_cache_; }
  void SetBias(const TensorFacade& bias) {
    bias_cache_ = bias;
    param_set_ = true;
  }

 private:
  std::shared_ptr<my_tensor::Convolution<float>> conv_;
  TensorFacade input_cache_;
  TensorFacade kernel_cache_;
  TensorFacade bias_cache_;
  bool param_set_;
};

class PoolingFacade {
 public:
  PoolingFacade(int input_channel, int kernel_size, int stride)
      : pooling_(nullptr) {
    auto param = std::make_shared<my_tensor::PoolingParameter>();
    param->input_channels_ = input_channel;
    param->kernel_h_ = kernel_size;
    param->kernel_w_ = kernel_size;
    param->stride_h_ = stride;
    param->stride_w_ = stride;
    pooling_.reset(new my_tensor::Pooling<float>(param));
  }

  TensorFacade Forward(TensorFacade input);

  TensorFacade Backward(TensorFacade output);

 private:
  std::shared_ptr<my_tensor::Pooling<float>> pooling_;
  TensorFacade input_cache_;
};

class SoftmaxFacade {
 public:
  explicit SoftmaxFacade(int channel) : softmax_(nullptr) {
    auto param = std::make_shared<my_tensor::SoftmaxParameter>();
    param->channels_ = channel;
    softmax_.reset(new my_tensor::Softmax<float>(param));
  }

  TensorFacade Forward(TensorFacade input);

  TensorFacade Backward(TensorFacade output);

 private:
  std::shared_ptr<my_tensor::Softmax<float>> softmax_;
  TensorFacade input_cache_;
};

class CrossEntropyLossFacade {
 public:
  explicit CrossEntropyLossFacade(int channel) : loss_(nullptr) {
    auto param = std::make_shared<my_tensor::LossWithSoftmaxParameter>();
    param->channels_ = channel;
    loss_.reset(new my_tensor::LossWithSoftmax<float>(param));
  }

  TensorFacade Forward(TensorFacade input, TensorFacade label);

  TensorFacade Backward(TensorFacade output);

 private:
  std::shared_ptr<my_tensor::LossWithSoftmax<float>> loss_;
  TensorFacade input_cache_;
  TensorFacade label_cache_;
};

#endif  // PYTHON_LAYER_FACADE_CUH_
