// Copyright 2024 yibotongxue

#ifndef INCLUDE_NET_PARAMETER_HPP_
#define INCLUDE_NET_PARAMETER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "data-parameter.hpp"
#include "layer-parameter.hpp"

namespace my_tensor {

class NetParameter final {
 public:
  std::string name_;
  DataParameterPtr train_data_parameter_;
  DataParameterPtr test_data_parameter_;
  std::vector<LayerParameterPtr> layer_params_;

  NetParameter(const std::string& name, DataParameterPtr train_data_parameter,
               DataParameterPtr test_data_parameter,
               const std::vector<LayerParameterPtr> layer_params)
      : name_(name),
        train_data_parameter_(train_data_parameter),
        test_data_parameter_(test_data_parameter),
        layer_params_(layer_params) {}
};  // class NetParameter

using NetParameterPtr = std::shared_ptr<NetParameter>;

}  // namespace my_tensor

#endif  // INCLUDE_NET_PARAMETER_HPP_
