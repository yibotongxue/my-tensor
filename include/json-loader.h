// Copyright 2024 yibotongxue

#ifndef INCLUDE_JSON_LOADER_H_
#define INCLUDE_JSON_LOADER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "filler-parameter.hpp"
#include "layer-parameter.h"
#include "nlohmann/json.hpp"

namespace my_tensor {

class JsonLoader {
 public:
  explicit JsonLoader(const std::string& json_file_path);

  std::vector<LayerParameterPtr> LoadLayers();
  int LoadBatchSize() const { return batch_size_; }
  float LoadLearningRate() const { return learning_rate_; }
  float LoadL2() const { return l2_; }

 private:
  nlohmann::json layers_;
  int batch_size_;
  float learning_rate_;
  float l2_;

  LayerParameterPtr LoadLayerParam(const nlohmann::json& js);
};

}  // namespace my_tensor

#endif  // INCLUDE_JSON_LOADER_H_
