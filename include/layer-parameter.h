// Copyright 2024 yibotongxue

#ifndef INCLUDE_LAYER_PARAMETER_H_
#define INCLUDE_LAYER_PARAMETER_H_

#include <string>

#include "nlohmann/json.hpp"

namespace my_tensor {

class LayerParameter {
 public:
  std::string name_;
  std::string type_;

  explicit LayerParameter(const std::string& name, const std::string& type)
      : name_(name), type_(type) {}
};  // class LayerParameter

class ReluParamter : public LayerParameter {
 public:
  explicit ReluParamter(const std::string& name)
      : LayerParameter(name, "Relu") {}
};  // class ReluParameter

class SigmoidParameter : public LayerParameter {
 public:
  explicit SigmoidParameter(const std::string& name)
      : LayerParameter(name, "Sigmoid") {}
};  // class SigmoidParameter

}  // namespace my_tensor

#endif  // INCLUDE_LAYER_PARAMETER_H_
