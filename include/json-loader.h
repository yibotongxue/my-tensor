// Copyright 2024 yibotongxue

#ifndef INCLUDE_JSON_LOADER_H_
#define INCLUDE_JSON_LOADER_H_

#include <string>
#include <vector>

#include "nlohmann/json.hpp"

using nlohmann::json;

namespace my_tensor {

class JsonLoader {
 public:
  explicit JsonLoader(const std::string& json_file_path) {}

 private:
  std::vector<json> layer_json_;
};

}  // namespace my_tensor

#endif  // INCLUDE_JSON_LOADER_H_
