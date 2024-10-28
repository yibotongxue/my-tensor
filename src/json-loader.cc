// Copyright 2024 yibotongxue

#include "json-loader.h"

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "error.h"
#include "layer-parameter.hpp"

namespace my_tensor {

std::unordered_map<std::string, InitMode> JsonLoader::mode_map_ = {
    {"xavier", InitMode::kXavier}, {"constant", InitMode::kConstant}};

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

std::vector<ParamPtr> JsonLoader::Load() {
  std::vector<ParamPtr> result;
  for (auto&& layer : layers_) {
    result.push_back(LoadParam(layer));
  }
  return std::move(result);
}

ParamPtr JsonLoader::LoadParam(const nlohmann::json& js) {
  if (!js.contains("name") || !js["name"].is_string()) {
    throw FileError("Layer object in layers object should contain key name");
  }
  if (!js.contains("type") || !js["type"].is_string()) {
    throw FileError("Layer object in layers object should contain key type");
  }
  auto &&name = js["name"].get<std::string>(),
       &&type = js["type"].get<std::string>();
  if (type == "Relu") {
    return std::move(std::make_unique<ReluParamter>(name));
  } else if (type == "Sigmoid") {
    return std::move(std::make_unique<SigmoidParameter>(name));
  } else if (type == "Linear") {
    auto param = std::make_unique<LinearParameter>(name);
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
      if (!linear_param.contains("name") || !linear_param.contains("init")) {
        throw FileError("Invalid linear param error.");
      }
      if (!linear_param["name"].is_string() ||
          !linear_param["init"].is_string()) {
        throw FileError("Invalid linear param error.");
      }
      auto &&name = linear_param["name"].get<std::string>(),
           &&init = linear_param["init"].get<std::string>();
      if (name == "linear_weight") {
        if (!mode_map_.contains(init)) {
          throw FileError("Invalid linear param init mode.");
        }
        param->weight_init_mode_ = mode_map_[init];
        if (linear_param.contains("conval")) {
          if (!linear_param["conval"].is_number_integer()) {
            throw FileError("Invalid linear param conval");
          }
          param->weight_conval_ = linear_param["conval"].get<int>();
        }
      } else if (name == "linear_bias") {
        if (!mode_map_.contains(init)) {
          throw FileError("Invalid linear param init mode.");
        }
        param->bias_init_mode_ = mode_map_[init];
        if (linear_param.contains("conval")) {
          if (!linear_param["conval"].is_number_integer()) {
            throw FileError("Invalid linear param conval");
          }
          param->bias_conval_ = linear_param["conval"].get<int>();
        }
      }
    }
    return std::move(param);
  } else if (type == "Convolution") {
    auto param = std::make_unique<ConvolutionParameter>(name);
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
      if (!conv_param.contains("name") || !conv_param.contains("init")) {
        throw FileError("Invalid conv param error.");
      }
      if (!conv_param["name"].is_string() || !conv_param["init"].is_string()) {
        throw FileError("Invalid conv param error.");
      }
      auto &&name = conv_param["name"].get<std::string>(),
           &&init = conv_param["init"].get<std::string>();
      if (name == "conv_kernel") {
        if (!mode_map_.contains(init)) {
          throw FileError("Invalid conv param init mode.");
        }
        param->kernel_init_mode_ = mode_map_[init];
        if (conv_param.contains("conval")) {
          if (!conv_param["conval"].is_number_integer()) {
            throw FileError("Invalid conv param conval");
          }
          param->kernel_conval_ = conv_param["conval"].get<int>();
        }
      } else if (name == "conv_bias") {
        if (!mode_map_.contains(init)) {
          throw FileError("Invalid conv param init mode.");
        }
        param->bias_init_mode_ = mode_map_[init];
        if (conv_param.contains("conval")) {
          if (!conv_param["conval"].is_number_integer()) {
            throw FileError("Invalid conv param conval");
          }
          param->bias_conval_ = conv_param["conval"].get<int>();
        }
      }
    }
    return std::move(param);
  } else {
    throw FileError("Unimplemented layer type.");
  }
}

}  // namespace my_tensor
