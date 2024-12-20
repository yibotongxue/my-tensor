// Copyright 2024 yibotongxue

#include "json-loader.hpp"

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "error.hpp"
#include "filler-parameter.hpp"
#include "layer-parameter.hpp"

namespace my_tensor {

std::vector<LayerParameterPtr> JsonLoader::LoadLayers() {
  if (!js.contains("layers")) {
    throw FileError(
        "The input json file is invalid. Valid json should contain key "
        "Layers.");
  }
  auto layers_ = js["layers"];
  if (!layers_.is_array()) {
    throw FileError("Json object layers should be an array.");
  }
  std::vector<LayerParameterPtr> result;
  for (auto&& layer : layers_) {
    result.push_back(LoadLayerParam(layer));
  }
  return result;
}

DataParameterPtr JsonLoader::LoadDataParameter() {
  return std::make_shared<DataParameter>(LoadDataType(), LoadImageFilePath(),
                                         LoadLabelFilePath(), LoadBatchSize());
}

NetParameterPtr JsonLoader::LoadNet() {
  return std::make_shared<NetParameter>(LoadNetName(), LoadDataParameter(),
                                        LoadLayers());
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

nlohmann::json JsonLoader::LoadJsonObject(const std::string& json_file_path) {
  std::ifstream fs(json_file_path);
  if (fs.fail()) {
    throw FileError("File fail.");
  }
  if (!fs.is_open()) {
    throw FileError("File not open.");
  }
  return nlohmann::json::parse(fs);
}

}  // namespace my_tensor
