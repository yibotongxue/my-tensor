// Copyright 2024 yibotongxue

#ifndef INCLUDE_DATA_PARAMETER_HPP_
#define INCLUDE_DATA_PARAMETER_HPP_

#include <memory>
#include <string>

#include "data-loader.hpp"
#include "dataset.hpp"

namespace my_tensor {

class DataParameter {
 public:
  std::string dataset_type_;
  std::string image_file_path_;
  std::string label_file_path_;
  int batch_size_;

  DataParameter(const std::string& dataset_type,
                const std::string& image_file_path,
                const std::string& label_file_path, int batch_size)
      : dataset_type_(dataset_type),
        image_file_path_(image_file_path),
        label_file_path_(label_file_path),
        batch_size_(batch_size) {}
};  // class DataParameter

using DataParameterPtr = std::shared_ptr<DataParameter>;

inline std::shared_ptr<DataLoader> CreateDataLoader(
    DataParameterPtr data_parameter) {
  return std::make_shared<DataLoader>(
      GetDatasetCreater(data_parameter->dataset_type_)(
          data_parameter->image_file_path_, data_parameter->label_file_path_),
      data_parameter->batch_size_);
}

}  // namespace my_tensor

#endif  // INCLUDE_DATA_PARAMETER_HPP_
