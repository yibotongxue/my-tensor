// Copyright 2024 yibotongxue

#ifndef INCLUDE_MODEL_SAVER_HPP_
#define INCLUDE_MODEL_SAVER_HPP_

#include <string>
#include <vector>

namespace my_tensor {

namespace ModelSaver {
template <typename T>
void Save(const std::vector<std::vector<T>>& data,
          const std::string& file_path);

template <typename T>
std::vector<std::vector<T>> Load(const std::string& file_path);

extern template void Save<float>(const std::vector<std::vector<float>>& data,
                                 const std::string& file_path);

extern template std::vector<std::vector<float>> Load<float>(
    const std::string& file_path);
}  // namespace ModelSaver

}  // namespace my_tensor

#endif  // INCLUDE_MODEL_SAVER_HPP_
