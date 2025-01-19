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
  std::string data_files_root_;
  int batch_size_;

  DataParameter(const std::string& dataset_type,
                const std::string& data_files_root, int batch_size)
      : dataset_type_(dataset_type),
        data_files_root_(data_files_root),
        batch_size_(batch_size) {}
};  // class DataParameter

using DataParameterPtr = std::shared_ptr<DataParameter>;

inline std::shared_ptr<DataLoader> CreateDataLoader(
    DataParameterPtr data_parameter, bool is_train) {
  auto dataset = GetDatasetCreater(data_parameter->dataset_type_)(
      data_parameter->data_files_root_, is_train);
  dataset->LoadData();
  return std::make_shared<DataLoader>(dataset, data_parameter->batch_size_);
}

}  // namespace my_tensor

#endif  // INCLUDE_DATA_PARAMETER_HPP_
