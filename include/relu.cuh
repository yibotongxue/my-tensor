// Copyright 2024 yibotongxue

#ifndef INCLUDE_RELU_CUH_
#define INCLUDE_RELU_CUH_

#include <memory>
#include <vector>

#include "layer-parameter.hpp"
#include "layer.cuh"
#include "tensor.cuh"

namespace my_tensor {
// Relu class, implements Layer.
template <typename T = float>
class Relu final : public Layer<T> {
 public:
  explicit Relu(LayerParameterPtr param) : Layer<T>(param) {}

  DISABLE_LAYER_COPY(Relu)

  void CheckTensorCount(const std::vector<TensorPtr<T>>& bottom,
                        const std::vector<TensorPtr<T>>& top) const override;

  virtual ~Relu() = default;

  // Override forward and backward methods of Layer class.
  // CPU
  void ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top) override;
  void BackwardCPU(const std::vector<TensorPtr<T>>& top,
                   const std::vector<TensorPtr<T>>& bottom) override;
  // GPU
  void ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top) override;
  void BackwardGPU(const std::vector<TensorPtr<T>>& top,
                   const std::vector<TensorPtr<T>>& bottom) override;
};

extern template class my_tensor::Relu<>;
}  // namespace my_tensor

#endif  // INCLUDE_RELU_CUH_
