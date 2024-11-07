// Copyright 2024 yibotongxue

#include "layer-parameter.h"

#include <memory>
#include <string>

#include "error.h"
#include "nlohmann/json.hpp"

namespace my_tensor {

void LinearParameter::ParseSettingParameter(const nlohmann::json& js) {
  if (!js.contains("input_feature") ||
      !js["input_feature"].is_number_integer()) {
    throw FileError(
        "Linear layer object should have an integer as input feature");
  }
  if (!js.contains("output_feature") ||
      !js["output_feature"].is_number_integer()) {
    throw FileError(
        "Linear layer object should have an integer as output feature");
  }
  input_feature_ = js["input_feature"].get<int>();
  output_feature_ = js["output_feature"].get<int>();
  if (input_feature_ <= 0) {
    throw FileError("Linear input feature should be greater than zero");
  }
  if (output_feature_ <= 0) {
    throw FileError("Linear output feature should be greater than zero");
  }
}

void LinearParameter::ParseFillingParameter(const nlohmann::json& js) {
  if (!js.contains("params") || !js["params"].is_array()) {
    throw FileError("Linear layer object should have a array as params");
  }
  auto params = js["params"];
  for (auto&& linear_param : params) {
    if (!linear_param.contains("name")) {
      throw FileError("Invalid linear param error for name.");
    }
    if (!linear_param.contains("init")) {
      throw FileError("Invalid linear param error for init.");
    }
    auto &&name = linear_param["name"].get<std::string>(),
         &&init_name = linear_param["init"].get<std::string>();
    if (name == "linear_weight") {
      weight_filler_parameter_ = CreateFillerParameter(init_name);
      if (auto temp = std::dynamic_pointer_cast<ConstantFillerParameter>(
              weight_filler_parameter_)) {
        if (linear_param.contains("conval")) {
          temp->val_ = linear_param.at("conval").get<int>();
        } else {
          throw FileError("Missing 'conval' in linear_param.");
        }
      } else if (auto temp = std::dynamic_pointer_cast<XavierFillerParameter>(
                     weight_filler_parameter_)) {
        temp->n_in_ = input_feature_;
        temp->n_out_ = output_feature_;
      } else if (auto temp = std::dynamic_pointer_cast<HeFillerParameter>(
                     weight_filler_parameter_)) {
        temp->n_ = input_feature_;
      } else {
        throw FileError("Unsurported init mode for weight.");
      }
    } else if (name == "linear_bias") {
      bias_filler_parameter_ = CreateFillerParameter(init_name);
      if (auto temp = std::dynamic_pointer_cast<ConstantFillerParameter>(
              bias_filler_parameter_)) {
        if (linear_param.contains("conval")) {
          temp->val_ = linear_param.at("conval").get<int>();
        } else {
          throw FileError("Missing 'conval' in linear_param.");
        }
      } else if (!std::dynamic_pointer_cast<ZeroFillerParameter>(
                     bias_filler_parameter_)) {
        throw FileError("Unsurported init mode for bias.");
      }
    }
  }
}

void ConvolutionParameter::ParseSettingParameter(const nlohmann::json& js) {
  if (!js.contains("input_channels") ||
      !js["input_channels"].is_number_integer()) {
    throw FileError(
        "Convolution layer should have an integer object as input_channels");
  }
  if (!js.contains("output_channels") ||
      !js["output_channels"].is_number_integer()) {
    throw FileError(
        "Convolution layer should have an integer object as output_channels");
  }
  if (!js.contains("kernel_size") || !js["kernel_size"].is_number_integer()) {
    throw FileError(
        "Convolution layer should have an integer object as kernel_size");
  }
  input_channels_ = js["input_channels"].get<int>();
  output_channels_ = js["output_channels"].get<int>();
  kernel_size_ = js["kernel_size"].get<int>();
  if (input_channels_ <= 0) {
    throw FileError(
        "The input channels of convolution layer should be greater than "
        "zero.");
  }
  if (output_channels_ <= 0) {
    throw FileError(
        "The output channels of convolution layer should be greater than "
        "zero.");
  }
  if (kernel_size_ <= 0) {
    throw FileError(
        "The kernel size of convolution layer should be greater than zero.");
  }
}

void ConvolutionParameter::ParseFillingParameter(const nlohmann::json& js) {
  if (!js.contains("params") || !js["params"].is_array()) {
    throw FileError("Linear layer object should have a array as params");
  }
  auto params = js["params"];
  for (auto&& conv_param : params) {
    if (!conv_param.contains("name")) {
      throw FileError("Invalid conv param error for name.");
    }
    if (!conv_param.contains("init")) {
      throw FileError("Invalid conv param error for init.");
    }
    auto &&name = conv_param["name"].get<std::string>(),
         &&init_name = conv_param["init"].get<std::string>();
    if (name == "conv_kernel") {
      kernel_filler_parameter_ = CreateFillerParameter(init_name);
      if (auto temp = std::dynamic_pointer_cast<ConstantFillerParameter>(
              kernel_filler_parameter_)) {
        if (conv_param.contains("conval")) {
          temp->val_ = conv_param.at("conval").get<int>();
        } else {
          throw FileError("Missing 'conval' in conv_param.");
        }
      } else if (auto temp = std::dynamic_pointer_cast<XavierFillerParameter>(
                     kernel_filler_parameter_)) {
        temp->n_in_ = input_channels_ * kernel_size_ * kernel_size_;
        temp->n_out_ = output_channels_;
      } else if (auto temp = std::dynamic_pointer_cast<HeFillerParameter>(
                     kernel_filler_parameter_)) {
        temp->n_ = input_channels_ * kernel_size_ * kernel_size_;
      } else {
        throw FileError("Unsurported init mode for kernel.");
      }
    } else if (name == "conv_bias") {
      bias_filler_parameter_ = CreateFillerParameter(init_name);
      if (auto temp = std::dynamic_pointer_cast<ConstantFillerParameter>(
              bias_filler_parameter_)) {
        if (conv_param.contains("conval")) {
          temp->val_ = conv_param.at("conval").get<int>();
        } else {
          throw FileError("Missing 'conval' in conv_param.");
        }
      } else if (!std::dynamic_pointer_cast<ZeroFillerParameter>(
                     bias_filler_parameter_)) {
        throw FileError("Unsurported init mode for bias.");
      }
    }
  }
}

void PoolingParameter::ParseSettingParameter(const nlohmann::json& js) {
  if (!js.contains("input_channels") ||
      !js["input_channels"].is_number_integer()) {
    throw FileError(
        "Pooling layer object should have an integer as input channels");
  }
  input_channels_ = js["input_channels"].get<int>();
  if (!js.contains("kernel_w") || !js["kernel_w"].is_number_integer()) {
    throw FileError(
        "Pooling layer object should have an integer as kernel width.");
  }
  kernel_w_ = js["kernel_w"].get<int>();

  if (!js.contains("kernel_h") || !js["kernel_h"].is_number_integer()) {
    throw FileError(
        "Pooling layer object should have an integer as kernel height.");
  }
  kernel_h_ = js["kernel_h"].get<int>();

  if (!js.contains("stride_w") || !js["stride_w"].is_number_integer()) {
    throw FileError(
        "Pooling layer object should have an integer as stride width.");
  }
  stride_w_ = js["stride_w"].get<int>();

  if (!js.contains("stride_h") || !js["stride_h"].is_number_integer()) {
    throw FileError(
        "Pooling layer object should have an integer as stride height.");
  }
  stride_h_ = js["stride_h"].get<int>();
}

void SoftmaxParameter::ParseSettingParameter(const nlohmann::json& js) {
  if (!js.contains("channels") || !js["channels"].is_number_integer()) {
    throw FileError(
        "The softmax layer should have an integer number as channels.");
  }
  channels_ = js["channels"].get<int>();
}

void LossWithSoftmaxParameter::ParseSettingParameter(const nlohmann::json& js) {
  if (!js.contains("channels") || !js["channels"].is_number_integer()) {
    throw FileError(
        "The softmax layer should have an integer number as channels.");
  }
  channels_ = js["channels"].get<int>();
}

}  // namespace my_tensor
