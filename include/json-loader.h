// Copyright 2024 yibotongxue

#ifndef INCLUDE_JSON_LOADER_H_
#define INCLUDE_JSON_LOADER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "layer-parameter.hpp"
#include "nlohmann/json.hpp"

namespace my_tensor {

class JsonLoader {
 public:
  explicit JsonLoader(const std::string& json_file_path);

  std::vector<ParamPtr> Load();

 private:
  nlohmann::json layers_;

  ParamPtr LoadParam(const nlohmann::json& js);

  static std::unordered_map<std::string, InitMode> mode_map_;
};

}  // namespace my_tensor

#endif  // INCLUDE_JSON_LOADER_H_
