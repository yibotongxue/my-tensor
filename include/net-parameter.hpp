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
  DataParameterPtr data_parameter_;
  std::vector<LayerParameterPtr> layer_params_;

  NetParameter(DataParameterPtr data_parameter,
               const std::vector<LayerParameterPtr> layer_params)
      : data_parameter_(data_parameter), layer_params_(layer_params) {}
};  // class NetParameter

using NetParameterPtr = std::shared_ptr<NetParameter>;

}  // namespace my_tensor

#endif  // INCLUDE_NET_PARAMETER_HPP_
