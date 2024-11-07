// Copyright 2024 yibotongxue

#ifndef INCLUDE_LAYER_FACTORY_CUH_
#define INCLUDE_LAYER_FACTORY_CUH_

#include <memory>

#include "conv.cuh"
#include "error.h"
#include "layer-parameter.h"
#include "layer.cuh"
#include "linear.cuh"
#include "loss-with-softmax.cuh"
#include "pooling.cuh"
#include "relu.cuh"
#include "sigmoid.cuh"
#include "softmax.cuh"

namespace my_tensor {
template <typename T = float>
inline LayerPtr<T> CreateLayer(const LayerParameterPtr param) {
  switch (param->type_) {
    case ParamType::kRelu:
      return std::make_shared<Relu<T>>(param);
    case ParamType::kSigmoid:
      return std::make_shared<Sigmoid<T>>(param);
    case ParamType::kLinear:
      return std::make_shared<Linear<T>>(param);
    case ParamType::kConvolution:
      return std::make_shared<Convolution<T>>(param);
    case ParamType::kPooling:
      return std::make_shared<Pooling<T>>(param);
    case ParamType::kSoftmax:
      return std::make_shared<Softmax<T>>(param);
    case ParamType::kLossWithSoftmax:
      return std::make_shared<LossWithSoftmax<T>>(param);
    default:
      throw LayerError("Unimplemented layer.");
  }
}

template LayerPtr<> CreateLayer<>(const LayerParameterPtr param);
}  // namespace my_tensor

#endif  // INCLUDE_LAYER_FACTORY_CUH_
