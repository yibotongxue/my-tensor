// Copyright 2024 yibotongxue

#ifndef INCLUDE_FLATTEN_HPP_
#define INCLUDE_FLATTEN_HPP_

#include <vector>

#include "layer-parameter.hpp"
#include "layer.hpp"

namespace my_tensor {

template <Arithmetic T = float>
class Flatten : public Layer<T> {
 public:
  explicit Flatten(LayerParameterPtr param) : Layer<T>(param) {}

  DISABLE_LAYER_COPY(Flatten)

  void CheckTensorCount(const std::vector<TensorPtr<T>>& bottom,
                        const std::vector<TensorPtr<T>>& top) const override;
  void Reshape(const std::vector<TensorPtr<T>>& bottom,
               const std::vector<TensorPtr<T>>& top) const override;

  void LayerSetUp(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top);

  virtual ~Flatten() = default;

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

 private:
  std::vector<int> bottom_shape_;
  std::vector<int> top_shape_;
  bool inplace_;
};  // class Flatten

extern template class Flatten<>;

}  // namespace my_tensor

#endif  // INCLUDE_FLATTEN_HPP_
