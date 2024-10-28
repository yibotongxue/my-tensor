// Copyright 2024 yibotongxue

#ifndef INCLUDE_LAYER_PARAMETER_HPP_
#define INCLUDE_LAYER_PARAMETER_HPP_

#include <string>

#include "nlohmann/json.hpp"

namespace my_tensor {

enum class ParamType {
  kRelu,
  kSigmoid,
  kLinear,
  kConvolution
};  // enum class ParamType

enum class InitMode { kXavier, kConstant };  // enum class InitMode

class LayerParameter {
 public:
  std::string name_;
  ParamType type_;

  explicit LayerParameter(const std::string& name, const ParamType type)
      : name_(name), type_(type) {}

  virtual ~LayerParameter() = default;
};  // class LayerParameter

class ReluParamter : public LayerParameter {
 public:
  explicit ReluParamter(const std::string& name)
      : LayerParameter(name, ParamType::kRelu) {}
};  // class ReluParameter

class SigmoidParameter : public LayerParameter {
 public:
  explicit SigmoidParameter(const std::string& name)
      : LayerParameter(name, ParamType::kSigmoid) {}
};  // class SigmoidParameter

class LinearParameter : public LayerParameter {
 public:
  int input_feature_;
  int output_feature_;
  InitMode weight_init_mode_;
  int weight_conval_{0};
  InitMode bias_init_mode_;
  int bias_conval_{0};

  explicit LinearParameter(const std::string& name)
      : LayerParameter(name, ParamType::kLinear) {}
};  // class LinearParameter

class ConvolutionParameter : public LayerParameter {
 public:
  int input_channels_;
  int output_channels_;
  int kernel_size_;
  InitMode kernel_init_mode_;
  int kernel_conval_{0};
  InitMode bias_init_mode_;
  int bias_conval_{0};

  explicit ConvolutionParameter(const std::string& name)
      : LayerParameter(name, ParamType::kConvolution) {}
};  // class ConvolutionParameter

}  // namespace my_tensor

#endif  // INCLUDE_LAYER_PARAMETER_HPP_
