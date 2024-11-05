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

  explicit LayerParameter(const ParamType type) : type_(type) {}

  void Deserialize(const nlohmann::json& js) {
    ParseName(js);
    ParseSettingParameter(js);
    ParseFillingParameter(js);
  }

  virtual ~LayerParameter() = default;

 protected:
  void ParseName(const nlohmann::json& js) {
    if (!js.contains("name") || !js["name"].is_string()) {
      throw FileError("Layer object in layers object should contain key name");
    }
    name_ = js["name"].get<std::string>();
  }

  virtual void ParseSettingParameter(const nlohmann::json& js) {}

  virtual void ParseFillingParameter(const nlohmann::json& js) {}
};  // class LayerParameter

class ReluParamter final : public LayerParameter {
 public:
  explicit ReluParamter() : LayerParameter(ParamType::kRelu) {}
};  // class ReluParameter

class SigmoidParameter : public LayerParameter {
 public:
  explicit SigmoidParameter() : LayerParameter(ParamType::kSigmoid) {}

  virtual ~SigmoidParameter() = default;
};  // class SigmoidParameter

class LinearParameter final : public LayerParameter {
 public:
  int input_feature_;
  int output_feature_;
  FillerParameterPtr weight_filler_parameter_;
  FillerParameterPtr bias_filler_parameter_;

  explicit LinearParameter() : LayerParameter(ParamType::kLinear) {}

 private:
  void ParseSettingParameter(const nlohmann::json& js) override;
  void ParseFillingParameter(const nlohmann::json& js) override;
};  // class LinearParameter

class ConvolutionParameter final : public LayerParameter {
 public:
  int input_channels_;
  int output_channels_;
  int kernel_size_;
  FillerParameterPtr kernel_filler_parameter_;
  FillerParameterPtr bias_filler_parameter_;

  explicit ConvolutionParameter() : LayerParameter(ParamType::kConvolution) {}

 private:
  void ParseSettingParameter(const nlohmann::json& js) override;
  void ParseFillingParameter(const nlohmann::json& js) override;
};  // class ConvolutionParameter

class PoolingParameter final : public LayerParameter {
 public:
  int input_channels_;
  int kernel_h_;
  int kernel_w_;
  int stride_h_;
  int stride_w_;

  explicit PoolingParameter() : LayerParameter(ParamType::kPooling) {}

 private:
  void ParseSettingParameter(const nlohmann::json& js) override;
};  // class PoolingParameter

class SoftmaxParameter final : public LayerParameter {
 public:
  int channels_;

  explicit SoftmaxParameter() : LayerParameter(ParamType::kSoftmax) {}

 private:
  void ParseSettingParameter(const nlohmann::json& js) override;
};  // class SoftmaxParameter

inline LayerParameterPtr CreateLayerParameter(const std::string& type) {
  if (type == "Relu") {
    return std::make_shared<ReluParamter>();
  } else if (type == "Sigmoid") {
    return std::make_shared<SigmoidParameter>();
  } else if (type == "Linear") {
    return std::make_shared<LinearParameter>();
  } else if (type == "Convolution") {
    return std::make_shared<ConvolutionParameter>();
  } else if (type == "Pooling") {
    return std::make_shared<PoolingParameter>();
  } else if (type == "Softmax") {
    return std::make_shared<SoftmaxParameter>();
  } else {
    throw LayerError("Unimplemented layer type.");
  }
}

}  // namespace my_tensor

#endif  // INCLUDE_LAYER_PARAMETER_H_
