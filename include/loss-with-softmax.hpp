// Copyright 2024 yibotongxue

#ifndef INCLUDE_LOSS_WITH_SOFTMAX_HPP_
#define INCLUDE_LOSS_WITH_SOFTMAX_HPP_

#include <iostream>
#include <vector>

#include "layer-parameter.hpp"
#include "layer.hpp"
#include "utils.hpp"

namespace my_tensor {

template <typename T = float>
class LossWithSoftmax final : public Layer<T> {
 public:
  explicit LossWithSoftmax(LayerParameterPtr param) : Layer<T>(param) {}

  void CheckTensorCount(const std::vector<TensorPtr<T>>& bottom,
                        const std::vector<TensorPtr<T>>& top) const override;
  void Reshape(const std::vector<TensorPtr<T>>& bottom,
               const std::vector<TensorPtr<T>>& top) const override;

  void LayerSetUp(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top) override;

  DISABLE_LAYER_COPY(LossWithSoftmax)

  void ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top) override;
  void BackwardCPU(const std::vector<TensorPtr<T>>& top,
                   const std::vector<TensorPtr<T>>& bottom) override;
  void ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top) override;
  void BackwardGPU(const std::vector<TensorPtr<T>>& top,
                   const std::vector<TensorPtr<T>>& bottom) override;

  float GetAccuracy(const TensorPtr<T> label) const;

 private:
  LayerPtr<T> softmax_;
  int channels_;
  int batch_size_;
  std::vector<TensorPtr<T>> softmax_bottom_;
  std::vector<TensorPtr<T>> softmax_top_;

  void CheckShape(const TensorPtr<T> input, const TensorPtr<T> label,
                  const TensorPtr<T> output) const;
};  // class LossWithSoftmax

extern template class LossWithSoftmax<>;

}  // namespace my_tensor

#endif  // INCLUDE_LOSS_WITH_SOFTMAX_HPP_
