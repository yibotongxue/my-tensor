// Copyright 2024 yibotongxue

#ifndef INCLUDE_LAYER_PARAMETER_HPP_
#define INCLUDE_LAYER_PARAMETER_HPP_

#include <memory>
#include <string>

#include "filler-parameter.hpp"
#include "nlohmann/json.hpp"

namespace my_tensor {

enum class ParamType {
  kRelu,
  kSigmoid,
  kLinear,
  kConvolution
};  // enum class ParamType

class LayerParameter;
using LayerParameterPtr = std::shared_ptr<LayerParameter>;

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
  FillerParameterPtr weight_filler_parameter_;
  FillerParameterPtr bias_filler_parameter_;

  explicit LinearParameter(const std::string& name)
      : LayerParameter(name, ParamType::kLinear) {}
};  // class LinearParameter

class ConvolutionParameter : public LayerParameter {
 public:
  int input_channels_;
  int output_channels_;
  int kernel_size_;
  FillerParameterPtr kernel_filler_parameter_;
  FillerParameterPtr bias_filler_parameter_;

  explicit ConvolutionParameter(const std::string& name)
      : LayerParameter(name, ParamType::kConvolution) {}
};  // class ConvolutionParameter

}  // namespace my_tensor

#endif  // INCLUDE_LAYER_PARAMETER_HPP_
