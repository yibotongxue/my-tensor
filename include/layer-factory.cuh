// Copyright 2024 yibotongxue

#ifndef INCLUDE_LAYER_FACTORY_CUH_
#define INCLUDE_LAYER_FACTORY_CUH_

#include <memory>

#include "error.h"
#include "layer.cuh"
#include "relu.cuh"
#include "sigmoid.cuh"
#include "tensor.cuh"

namespace my_tensor {

template <typename T = float>
class LayerFactory {
 protected:
  LayerFactory() = default;

  LayerFactory(const LayerFactory<T>&) = delete;
  LayerFactory<T>& operator=(const LayerFactory<T>&) = delete;

  virtual ~LayerFactory() = default;

  virtual LayerPtr<T> CreateLayer() = 0;
};

template <typename T>
using LayerFactoryPtr = std::shared_ptr<LayerFactory<T>>;

template <typename T = float>
class ReluFactory final : public LayerFactory<T> {
 public:
  LayerPtr<T> CreateLayer() override { return std::make_shared<Relu<T>>(); }
};

template <typename T = float>
class SigmoidFactory final : public LayerFactory<T> {
 public:
  LayerPtr<T> CreateLayer() override { return std::make_shared<Sigmoid<T>>(); }
};

template <typename T = float>
class LinearFactory final : public LayerFactory<T> {
 public:
  LayerPtr<T> CreateLayer() override {
    throw std::runtime_error("Unimplemention error.");
  }
};

}  // namespace my_tensor

#endif  // INCLUDE_LAYER_FACTORY_CUH_
