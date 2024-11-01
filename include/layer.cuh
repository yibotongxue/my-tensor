// Copyright 2024 yibotongxue

#ifndef INCLUDE_LAYER_CUH_
#define INCLUDE_LAYER_CUH_

#include <memory>
#include <vector>

#include "layer-parameter.hpp"
#include "tensor.cuh"
#include "utils.cuh"

namespace my_tensor {
// Layer abstract class.
template <typename T = float>
class Layer {
 public:
  // Default constructor.
  explicit Layer(LayerParameterPtr param) : layer_param_(param) {}

  virtual void SetUp(const TensorPtr<T> bottom) {}

  // The layer can not be copied or moved.
  DISABLE_LAYER_COPY(Layer)

  virtual ~Layer() = default;

  // Pure virtual methods, forward and backward.
  // CPU
  virtual void ForwardCPU(const TensorPtr<T> bottom, TensorPtr<T> top) = 0;
  virtual void BackwardCPU(const TensorPtr<T> top, TensorPtr<T> bottome) = 0;
  // GPU
  virtual void ForwardGPU(const TensorPtr<T> bottom, TensorPtr<T> top) = 0;
  virtual void BackwardGPU(const TensorPtr<T> top, TensorPtr<T> bottome) = 0;

 protected:
  LayerParameterPtr layer_param_;
};

// Layer pointer.
template <typename T = float>
using LayerPtr = std::shared_ptr<my_tensor::Layer<T>>;

extern template class my_tensor::Layer<>;
}  // namespace my_tensor

#endif  // INCLUDE_LAYER_CUH_
