// Copyright 2024 yibotongxue

#ifndef INCLUDE_LAYER_PARAMETER_H_
#define INCLUDE_LAYER_PARAMETER_H_

#include <memory>
#include <string>

#include "filler-parameter.hpp"
#include "nlohmann/json.hpp"

namespace my_tensor {

enum class ParamType {
  kRelu,
  kSigmoid,
  kLinear,
  kConvolution,
  kPooling,
  kSoftmax
};  // enum class ParamType

class LayerParameter;
using LayerParameterPtr = std::shared_ptr<LayerParameter>;

class LayerParameter {
 public:
  std::string name_;
  ParamType type_;

  explicit LayerParameter(const std::string& name, const ParamType type)
      : name_(name), type_(type) {}

  virtual void Deserialize(const nlohmann::json& js) {}

  virtual ~LayerParameter() = default;
};  // class LayerParameter

class ReluParamter final : public LayerParameter {
 public:
  explicit ReluParamter(const std::string& name)
      : LayerParameter(name, ParamType::kRelu) {}
};  // class ReluParameter

class SigmoidParameter : public LayerParameter {
 public:
  explicit SigmoidParameter(const std::string& name)
      : LayerParameter(name, ParamType::kSigmoid) {}
};  // class SigmoidParameter

class LinearParameter final : public LayerParameter {
 public:
  int input_feature_;
  int output_feature_;
  FillerParameterPtr weight_filler_parameter_;
  FillerParameterPtr bias_filler_parameter_;

  explicit LinearParameter(const std::string& name)
      : LayerParameter(name, ParamType::kLinear) {}

  void Deserialize(const nlohmann::json& js) override;
};  // class LinearParameter

class ConvolutionParameter final : public LayerParameter {
 public:
  int input_channels_;
  int output_channels_;
  int kernel_size_;
  FillerParameterPtr kernel_filler_parameter_;
  FillerParameterPtr bias_filler_parameter_;

  explicit ConvolutionParameter(const std::string& name)
      : LayerParameter(name, ParamType::kConvolution) {}

  void Deserialize(const nlohmann::json& js) override;
};  // class ConvolutionParameter

class PoolingParameter final : public LayerParameter {
 public:
  int input_channels_;
  int kernel_h_;
  int kernel_w_;
  int stride_h_;
  int stride_w_;

  explicit PoolingParameter(const std::string& name)
      : LayerParameter(name, ParamType::kPooling) {}

  // void Deserialize(const nlohmann::json& js) override;
};  // class PoolingParameter

class SoftmaxParameter final : public LayerParameter {
 public:
  int channels_;

  explicit SoftmaxParameter(const std::string& name)
      : LayerParameter(name, ParamType::kSoftmax) {}

  void Deserialize(const nlohmann::json& js) override;
};  // class SoftmaxParameter

}  // namespace my_tensor

#endif  // INCLUDE_LAYER_PARAMETER_H_
