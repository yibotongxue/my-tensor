// Copyright 2024 yibotongxue

#ifndef INCLUDE_JSON_LOADER_HPP_
#define INCLUDE_JSON_LOADER_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "data-parameter.hpp"
#include "filler-parameter.hpp"
#include "layer-parameter.hpp"
#include "net-parameter.hpp"
#include "nlohmann/json.hpp"

namespace my_tensor {

class JsonLoader {
 public:
  explicit JsonLoader(const std::string& json_file_path)
      : js(LoadJsonObject(json_file_path)) {}

  int LoadBatchSize() const { return LoadWithKey<int>("batch_size"); }
  float LoadLearningRate() const { return LoadWithKey<float>("learning_rate"); }
  float LoadL2() const { return LoadWithKey<float>("l2"); }
  std::string LoadDataType() const {
    return LoadWithKey<std::string>("data_type");
  }
  std::string LoadImageFilePath() const {
    return LoadWithKey<std::string>("image_file_path");
  }
  std::string LoadLabelFilePath() const {
    return LoadWithKey<std::string>("label_file_path");
  }
  std::string LoadNetName() const { return LoadWithKey<std::string>("name"); }
  std::vector<LayerParameterPtr> LoadLayers();
  DataParameterPtr LoadDataParameter();
  NetParameterPtr LoadNet();

 private:
  nlohmann::json js;

  LayerParameterPtr LoadLayerParam(const nlohmann::json& js);

  template <typename T>
  T LoadWithKey(const std::string& key) const;

  static nlohmann::json LoadJsonObject(const std::string& json_file_path);
};

}  // namespace my_tensor

#endif  // INCLUDE_JSON_LOADER_HPP_
