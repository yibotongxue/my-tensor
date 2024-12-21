// Copyright 2024 yibotongxue

#ifndef INCLUDE_JSON_LOADER_HPP_
#define INCLUDE_JSON_LOADER_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "data-parameter.hpp"
#include "filler-parameter.hpp"
#include "json-utils.hpp"
#include "layer-parameter.hpp"
#include "net-parameter.hpp"
#include "nlohmann/json.hpp"
#include "scheduler-parameter.hpp"
#include "solver-parameter.hpp"

namespace my_tensor {

class JsonLoader {
 public:
  explicit JsonLoader(const std::string& json_file_path)
      : js(LoadJsonObject(json_file_path)) {}

  int LoadBatchSize() const { return LoadWithKey<int>(js, "batch_size"); }
  float LoadLearningRate() const {
    return LoadWithKey<float>(js, "learning_rate");
  }
  float LoadL2() const { return LoadWithKey<float>(js, "l2"); }
  std::string LoadDataType() const {
    return LoadWithKey<std::string>(js, "data_type");
  }
  std::string LoadTrainImageFilePath() const {
    return LoadWithKey<std::string>(js, "train_image_file_path");
  }
  std::string LoadTrainLabelFilePath() const {
    return LoadWithKey<std::string>(js, "train_label_file_path");
  }
  std::string LoadTestImageFilePath() const {
    return LoadWithKey<std::string>(js, "test_image_file_path");
  }
  std::string LoadTestLabelFilePath() const {
    return LoadWithKey<std::string>(js, "test_label_file_path");
  }
  std::string LoadNetName() const {
    return LoadWithKey<std::string>(js, "name");
  }
  std::vector<LayerParameterPtr> LoadLayers();
  DataParameterPtr LoadTrainDataParameter();
  DataParameterPtr LoadTestDataParameter();
  NetParameterPtr LoadNet();
  SchedulerParameterPtr LoadScheduler();
  SolverParameterPtr LoadSolver();

 private:
  nlohmann::json js;

  LayerParameterPtr LoadLayerParam(const nlohmann::json& js);

  static nlohmann::json LoadJsonObject(const std::string& json_file_path);
};

}  // namespace my_tensor

#endif  // INCLUDE_JSON_LOADER_HPP_
