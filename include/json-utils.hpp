// Copyright 2024 yibotongxue

#ifndef INCLUDE_JSON_UTILS_HPP_
#define INCLUDE_JSON_UTILS_HPP_

#include <format>
#include <string>

#include "error.h"
#include "nlohmann/json.hpp"

namespace my_tensor {

template <typename T>
inline T LoadWithKey(const nlohmann::json &js, const std::string &key) {
  if (!js.contains(key)) {
    throw FileError(std::format("The json object should contain {} key", key));
  }
  try {
    return js[key].get<T>();
  } catch (...) {
    throw FileError(std::format("Unknown error thrown from line {} of file {}",
                                __LINE__, __FILE__));
  }
}

}  // namespace my_tensor

#endif  // INCLUDE_JSON_UTILS_HPP_
