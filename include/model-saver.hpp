// Copyright 2024 yibotongxue

#ifndef INCLUDE_MODEL_SAVER_HPP_
#define INCLUDE_MODEL_SAVER_HPP_

#include <string>
#include <vector>

namespace my_tensor {

class ModelSaver {
 public:
  template <typename T>
  static void Save(const std::vector<std::vector<T>>& data,
                   const std::string& file_path);

  template <typename T>
  static std::vector<std::vector<T>> Load(const std::string& file_path);
};  // class ModelSaver

}  // namespace my_tensor

#endif  // INCLUDE_MODEL_SAVER_HPP_
