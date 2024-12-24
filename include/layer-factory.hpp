// Copyright 2024 yibotongxue

#ifndef INCLUDE_LAYER_FACTORY_HPP_
#define INCLUDE_LAYER_FACTORY_HPP_

#include <memory>

// #include "accuracy.hpp"
#include "conv.hpp"
#include "error.hpp"
#include "flatten.hpp"
// #include "layer-parameter.hpp"
#include "layer.hpp"
#include "linear.hpp"
// #include "loss-with-softmax.hpp"
#include "pooling.hpp"
#include "relu.hpp"
#include "sigmoid.hpp"
#include "softmax.hpp"

namespace my_tensor {
template <typename T = float>
inline LayerPtr<T> CreateLayer(const LayerParameterPtr param) {
  switch (param->type_) {
    case ParamType::kRelu:
      return std::make_shared<Relu<T>>(param);
    case ParamType::kSigmoid:
      return std::make_shared<Sigmoid<T>>(param);
    case ParamType::kFlatten:
      return std::make_shared<Flatten<T>>(param);
    case ParamType::kLinear:
      return std::make_shared<Linear<T>>(param);
    case ParamType::kConvolution:
      return std::make_shared<Convolution<T>>(param);
    case ParamType::kPooling:
      return std::make_shared<Pooling<T>>(param);
    case ParamType::kSoftmax:
      return std::make_shared<Softmax<T>>(param);
    // case ParamType::kLossWithSoftmax:
    //   return std::make_shared<LossWithSoftmax<T>>(param);
    // case ParamType::kAccuracy:
    //   return std::make_shared<Accuracy<T>>(param);
    default:
      throw LayerError("Unimplemented layer.");
  }
}

template LayerPtr<> CreateLayer<>(const LayerParameterPtr param);
}  // namespace my_tensor

#endif  // INCLUDE_LAYER_FACTORY_HPP_
