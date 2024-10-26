// Copyright 2024 yibotongxue

#ifndef INCLUDE_RELU_CUH_
#define INCLUDE_RELU_CUH_

#include <layer.cuh>
#include <tensor.cuh>

#include <memory>

namespace my_tensor {
// Relu class, implements Layer.
template <typename T = float>
class Relu final : public Layer<T> {
 public:
  Relu() : Layer<T>({}) {}

  DISABLE_LAYER_COPY(Relu)

  virtual ~Relu() = default;

  // Override forward and backward methods of Layer class.
  // CPU
  void ForwardCPU(const TensorPtr<T> bottom, TensorPtr<T> top) override;
  void BackwardCPU(const TensorPtr<T> top, TensorPtr<T> bottom) override;
  // GPU
  void ForwardGPU(const TensorPtr<T> bottom, TensorPtr<T> top) override;
  void BackwardGPU(const TensorPtr<T> top, TensorPtr<T> bottom) override;
};

extern template class my_tensor::Relu<>;
}  // namespace my_tensor


#endif  // INCLUDE_RELU_CUH_
