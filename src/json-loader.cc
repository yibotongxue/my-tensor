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
#include "layer-parameter.h"

namespace my_tensor {

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
  if (!js.contains("batch_size")) {
    throw FileError(
        "The input json file is invalid. Valid json should contain key "
        "batch_size.");
  }
  if (!js["batch_size"].is_number_integer()) {
    throw FileError("The batch_size should be an integer.");
  }
  batch_size_ = js["batch_size"].get<int>();
  if (!js.contains("learning_rate")) {
    throw FileError(
        "The input json file is invalid. Valid json should contain key "
        "learning_rate.");
  }
  if (!js["learning_rate"].is_number_float()) {
    throw FileError("The learning_rate should be a float.");
  }
  learning_rate_ = js["learning_rate"].get<float>();
}

std::vector<LayerParameterPtr> JsonLoader::LoadLayers() {
  std::vector<LayerParameterPtr> result;
  for (auto&& layer : layers_) {
    result.push_back(LoadLayerParam(layer));
  }
  return result;
}

LayerParameterPtr JsonLoader::LoadLayerParam(const nlohmann::json& js) {
  if (!js.contains("type") || !js["type"].is_string()) {
    throw FileError("Layer object in layers object should contain key type");
  }
  auto&& type = js["type"].get<std::string>();
  auto param = CreateLayerParameter(type);
  param->Deserialize(js);
  return param;
}

}  // namespace my_tensor
