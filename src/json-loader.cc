// Copyright 2024 yibotongxue

#include "json-loader.h"

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "error.h"
#include "filler-parameter.hpp"
#include "layer-parameter.hpp"

namespace my_tensor {

std::unordered_map<std::string, InitMode> JsonLoader::mode_map_ = {
    {"xavier", InitMode::kXavier},
    {"constant", InitMode::kConstant},
    {"he", InitMode::kHe},
    {"zero", InitMode::kZero}};

JsonLoader::JsonLoader(const std::string& json_file_path) {
  std::ifstream fs(json_file_path);
  if (fs.fail()) {
    throw FileError("File fail.");
  }
  if (!fs.is_open()) {
    throw FileError("File not open.");
  }
  nlohmann::json js = nlohmann::json::parse(fs);
  if (!js.contains("layers")) {
    throw FileError(
        "The input json file is invalid. Valid json should contain key "
        "Layers.");
  }
  layers_ = js["layers"];
  if (!layers_.is_array()) {
    throw FileError("Json object layers should be an array.");
  }
}

std::vector<LayerParameterPtr> JsonLoader::Load() {
  std::vector<LayerParameterPtr> result;
  for (auto&& layer : layers_) {
    result.push_back(LoadLayerParam(layer));
  }
  return result;
}

FillerParameterPtr JsonLoader::LoadFillerParam(const nlohmann::json& js) {
  if (!js.contains("init")) {
    throw FileError("Invalid linear param error.");
  }
  auto&& init = js["init"].get<std::string>();
  if (!mode_map_.contains(init)) {
    throw FileError("Invalid param init mode.");
  }
  return CreateFillerParameter(mode_map_[init]);
}

LayerParameterPtr JsonLoader::LoadLayerParam(const nlohmann::json& js) {
  if (!js.contains("name") || !js["name"].is_string()) {
    throw FileError("Layer object in layers object should contain key name");
  }
  if (!js.contains("type") || !js["type"].is_string()) {
    throw FileError("Layer object in layers object should contain key type");
  }
  auto &&name = js["name"].get<std::string>(),
       &&type = js["type"].get<std::string>();
  if (type == "Relu") {
    return std::make_shared<ReluParamter>(name);
  } else if (type == "Sigmoid") {
    return std::make_shared<SigmoidParameter>(name);
  } else if (type == "Linear") {
    auto param = std::make_shared<LinearParameter>(name);
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
    param->input_feature_ = js["input_feature"].get<int>();
    param->output_feature_ = js["output_feature"].get<int>();
    if (param->input_feature_ <= 0) {
      throw FileError("Linear input feature should be greater than zero");
    }
    if (param->output_feature_ <= 0) {
      throw FileError("Linear output feature should be greater than zero");
    }
    if (!js.contains("params") || !js["params"].is_array()) {
      throw FileError("Linear layer object should have a array as params");
    }
    auto params = js["params"];
    for (auto&& linear_param : params) {
      if (!linear_param.contains("name")) {
        throw FileError("Invalid linear param error.");
      }
      auto&& name = linear_param["name"].get<std::string>();
      if (name == "linear_weight") {
        param->weight_filler_parameter_ = LoadFillerParam(linear_param);
        if (auto temp = std::dynamic_pointer_cast<ConstantFillerParameter>(
                param->weight_filler_parameter_)) {
          if (linear_param.contains("conval")) {
            temp->val_ = linear_param.at("conval").get<int>();
          } else {
            throw FileError("Missing 'conval' in linear_param.");
          }
        } else if (auto temp = std::dynamic_pointer_cast<XavierFillerParameter>(
                       param->weight_filler_parameter_)) {
          temp->n_in_ = param->input_feature_;
          temp->n_out_ = param->output_feature_;
        } else if (auto temp = std::dynamic_pointer_cast<HeFillerParameter>(
                       param->weight_filler_parameter_)) {
          temp->n_ = param->input_feature_;
        } else {
          throw FileError("Unsurported init mode for weight.");
        }
      } else if (name == "linear_bias") {
        param->bias_filler_parameter_ = LoadFillerParam(linear_param);
        if (auto temp = std::dynamic_pointer_cast<ConstantFillerParameter>(
                param->bias_filler_parameter_)) {
          if (linear_param.contains("conval")) {
            temp->val_ = linear_param.at("conval").get<int>();
          } else {
            throw FileError("Missing 'conval' in linear_param.");
          }
        } else if (!std::dynamic_pointer_cast<ZeroFillerParameter>(
                       param->bias_filler_parameter_)) {
          throw FileError("Unsurported init mode for bias.");
        }
      }
    }
    return param;
  } else if (type == "Convolution") {
    auto param = std::make_shared<ConvolutionParameter>(name);
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
    param->input_channels_ = js["input_channels"].get<int>();
    param->output_channels_ = js["output_channels"].get<int>();
    param->kernel_size_ = js["kernel_size"].get<int>();
    if (param->input_channels_ <= 0) {
      throw FileError(
          "The input channels of convolution layer should be greater than "
          "zero.");
    }
    if (param->output_channels_ <= 0) {
      throw FileError(
          "The output channels of convolution layer should be greater than "
          "zero.");
    }
    if (param->kernel_size_ <= 0) {
      throw FileError(
          "The kernel size of convolution layer should be greater than zero.");
    }
    auto params = js["params"];
    for (auto&& conv_param : params) {
      if (!conv_param.contains("name")) {
        throw FileError("Invalid conv param error.");
      }
      auto&& name = conv_param["name"].get<std::string>();
      if (name == "conv_kernel") {
        param->kernel_filler_parameter_ = LoadFillerParam(conv_param);
        if (auto temp = std::dynamic_pointer_cast<ConstantFillerParameter>(
                param->kernel_filler_parameter_)) {
          if (conv_param.contains("conval")) {
            temp->val_ = conv_param.at("conval").get<int>();
          } else {
            throw FileError("Missing 'conval' in conv_param.");
          }
        } else if (auto temp = std::dynamic_pointer_cast<XavierFillerParameter>(
                       param->kernel_filler_parameter_)) {
          temp->n_in_ = param->input_channels_ * param->kernel_size_ *
                        param->kernel_size_;
          temp->n_out_ = param->output_channels_;
        } else if (auto temp = std::dynamic_pointer_cast<HeFillerParameter>(
                       param->kernel_filler_parameter_)) {
          temp->n_ = param->input_channels_ * param->kernel_size_ *
                     param->kernel_size_;
        } else {
          throw FileError("Unsurported init mode for kernel.");
        }
      } else if (name == "conv_bias") {
        param->bias_filler_parameter_ = LoadFillerParam(conv_param);
        if (auto temp = std::dynamic_pointer_cast<ConstantFillerParameter>(
                param->bias_filler_parameter_)) {
          if (conv_param.contains("conval")) {
            temp->val_ = conv_param.at("conval").get<int>();
          } else {
            throw FileError("Missing 'conval' in conv_param.");
          }
        } else if (!std::dynamic_pointer_cast<ZeroFillerParameter>(
                       param->bias_filler_parameter_)) {
          throw FileError("Unsurported init mode for bias.");
        }
      }
    }
    return param;
  } else {
    throw FileError("Unimplemented layer type.");
  }
}

}  // namespace my_tensor
