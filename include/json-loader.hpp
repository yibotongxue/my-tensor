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
  std::string LoadTrainImageFilePath() const {
    return LoadWithKey<std::string>("train_image_file_path");
  }
  std::string LoadTrainLabelFilePath() const {
    return LoadWithKey<std::string>("train_label_file_path");
  }
  std::string LoadTestImageFilePath() const {
    return LoadWithKey<std::string>("test_image_file_path");
  }
  std::string LoadTestLabelFilePath() const {
    return LoadWithKey<std::string>("test_label_file_path");
  }
  std::string LoadNetName() const { return LoadWithKey<std::string>("name"); }
  std::vector<LayerParameterPtr> LoadLayers();
  DataParameterPtr LoadTrainDataParameter();
  DataParameterPtr LoadTestDataParameter();
  NetParameterPtr LoadNet();

 private:
  nlohmann::json js;

  LayerParameterPtr LoadLayerParam(const nlohmann::json& js);

  template <typename T>
  T LoadWithKey(const std::string& key) const {
    if (!js.contains(key)) {
      throw FileError(
          std::format("The json object should contain {} key", key));
    }
    try {
      return js[key].get<T>();
    } catch (...) {
      throw FileError(std::format(
          "Unknown error thrown from line {} of file {}", __LINE__, __FILE__));
    }
  }

  static nlohmann::json LoadJsonObject(const std::string& json_file_path);
};

}  // namespace my_tensor

#endif  // INCLUDE_JSON_LOADER_HPP_
