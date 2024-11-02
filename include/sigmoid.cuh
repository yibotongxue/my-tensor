// Copyright 2024 yibotongxue

#ifndef INCLUDE_SIGMOID_CUH_
#define INCLUDE_SIGMOID_CUH_

#include <memory>
#include <vector>

#include "layer-parameter.hpp"
#include "layer.cuh"
#include "tensor.cuh"

namespace my_tensor {
// Sigmoid class, implements Layer class.
template <typename T = float>
class Sigmoid final : public Layer<T> {
 public:
  explicit Sigmoid(LayerParameterPtr param) : Layer<T>(param) {}

  DISABLE_LAYER_COPY(Sigmoid)

  void CheckTensorCount(const std::vector<TensorPtr<T>>& bottom,
                        const std::vector<TensorPtr<T>>& top) const override;

  ~Sigmoid() = default;

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

extern template class my_tensor::Sigmoid<>;
}  // namespace my_tensor

#endif  // INCLUDE_SIGMOID_CUH_
