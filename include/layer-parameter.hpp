// Copyright 2024 yibotongxue

#ifndef INCLUDE_LAYER_PARAMETER_HPP_
#define INCLUDE_LAYER_PARAMETER_HPP_

#include <format>
#include <memory>
#include <string>
#include <vector>

#include "filler-parameter.hpp"
#include "nlohmann/json.hpp"

namespace my_tensor {

enum class ParamType {
  kRelu,
  kSigmoid,
  kFlatten,
  kLinear,
  kConvolution,
  kPooling,
  kSoftmax,
  kLossWithSoftmax,
  kAccuracy
};  // enum class ParamType

class LayerParameter;
using LayerParameterPtr = std::shared_ptr<LayerParameter>;

class LayerParameter {
 public:
  std::string name_;
  std::vector<std::string> bottoms_;
  std::vector<std::string> tops_;
  ParamType type_;

  explicit LayerParameter(const ParamType type) : type_(type) {}

  void Deserialize(const nlohmann::json& js) {
    ParseName(js);
    ParseBottom(js);
    ParseTop(js);
    ParseSettingParameter(js);
    ParseFillingParameter(js);
  }

  virtual ~LayerParameter() = default;

 protected:
  void ParseName(const nlohmann::json& js) { name_ = ParseNameInFields(js); }

  void ParseBottom(const nlohmann::json& js) {
    bottoms_ = ParseNameVectorInFields(js, "bottom");
  }

  void ParseTop(const nlohmann::json& js) {
    tops_ = ParseNameVectorInFields(js, "top");
  }

  virtual void ParseSettingParameter(const nlohmann::json& js) {}

  virtual void ParseFillingParameter(const nlohmann::json& js) {}

 private:
  static std::string ParseNameInFields(const nlohmann::json& js) {
    if (!js.contains("name") || !js["name"].is_string()) {
      throw FileError("Layer object in layers object should contain key name");
    }
    return js["name"].get<std::string>();
  }

  static std::vector<std::string> ParseNameVectorInFields(
      const nlohmann::json& js, const std::string& field) {
    if (!js.contains(field) || !js[field].is_array()) {
      throw FileError(std::format(
          "Layer object in layers object should contain key {}", field));
    }
    std::vector<std::string> result;
    for (auto&& js_in_field : js[field]) {
      result.emplace_back(js_in_field.get<std::string>());
    }
    return result;
  }
};  // class LayerParameter

class ReluParameter final : public LayerParameter {
 public:
  explicit ReluParameter() : LayerParameter(ParamType::kRelu) {}
};  // class ReluParameter

class SigmoidParameter : public LayerParameter {
 public:
  SigmoidParameter() : LayerParameter(ParamType::kSigmoid) {}

  virtual ~SigmoidParameter() = default;
};  // class SigmoidParameter

class FlattenParameter : public LayerParameter {
 public:
  FlattenParameter() : LayerParameter(ParamType::kFlatten) {}

  bool inplace_;

  virtual ~FlattenParameter() = default;

 private:
  void ParseSettingParameter(const nlohmann::json& js) override;
};  // class FlattenParameter

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

class LossWithSoftmaxParameter final : public LayerParameter {
 public:
  int channels_;

  explicit LossWithSoftmaxParameter()
      : LayerParameter(ParamType::kLossWithSoftmax) {}

 private:
  void ParseSettingParameter(const nlohmann::json& js) override;
};  // class LossWithSoftmaxParameter

class AccuracyParameter final : public LayerParameter {
 public:
  int features_;

  explicit AccuracyParameter() : LayerParameter(ParamType::kAccuracy) {}

 private:
  void ParseSettingParameter(const nlohmann::json& js) override;
};  // class AccuracyParameter

inline LayerParameterPtr CreateLayerParameter(const std::string& type) {
  if (type == "Relu") {
    return std::make_shared<ReluParameter>();
  } else if (type == "Sigmoid") {
    return std::make_shared<SigmoidParameter>();
  } else if (type == "Flatten") {
    return std::make_shared<FlattenParameter>();
  } else if (type == "Linear") {
    return std::make_shared<LinearParameter>();
  } else if (type == "Convolution") {
    return std::make_shared<ConvolutionParameter>();
  } else if (type == "Pooling") {
    return std::make_shared<PoolingParameter>();
  } else if (type == "Softmax") {
    return std::make_shared<SoftmaxParameter>();
  } else if (type == "LossWithSoftmax") {
    return std::make_shared<LossWithSoftmaxParameter>();
  } else if (type == "Accuracy") {
    return std::make_shared<AccuracyParameter>();
  } else {
    throw LayerError("Unimplemented layer type.");
  }
}

}  // namespace my_tensor

#endif  // INCLUDE_LAYER_PARAMETER_HPP_
