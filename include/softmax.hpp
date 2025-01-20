// Copyright 2024 yibotongxue

#ifndef INCLUDE_SOFTMAX_HPP_
#define INCLUDE_SOFTMAX_HPP_

#include <vector>

#include "error.hpp"
#include "layer-parameter.hpp"
#include "layer.hpp"
#include "utils.hpp"

namespace my_tensor {

template <Arithmetic T = float>
class Softmax final : public Layer<T> {
 public:
  explicit Softmax(LayerParameterPtr param) : Layer<T>(param) {}

  void CheckTensorCount(const std::vector<TensorPtr<T>>& bottom,
                        const std::vector<TensorPtr<T>>& top) const override;
  void Reshape(const std::vector<TensorPtr<T>>& bottom,
               const std::vector<TensorPtr<T>>& top) const override;

  void LayerSetUp(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top) override;
  DISABLE_LAYER_COPY(Softmax)

  void ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top) override;

  // Don't reference this function. Use the backward in softmax with loss layer
  // instead.
  void BackwardCPU(const std::vector<TensorPtr<T>>& top,
                   const std::vector<TensorPtr<T>>& bottom) {
    throw SoftmaxError("Unimplemention error.");
  }

  void ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top) override;

  // Don't reference this function. Use the backward in softmax with loss layer
  // instead.
  void BackwardGPU(const std::vector<TensorPtr<T>>& top,
                   const std::vector<TensorPtr<T>>& bottom) {
    throw SoftmaxError("Unimplemention error.");
  }

 private:
  int batch_size_;
  int channels_;

  void CheckShape(const TensorPtr<T> bottom, const TensorPtr<T> top) const;
};  // class Softmax

extern template class Softmax<>;

}  // namespace my_tensor

#endif  // INCLUDE_SOFTMAX_HPP_
