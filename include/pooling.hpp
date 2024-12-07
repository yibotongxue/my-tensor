// Copyright 2024 yibotongxue

#ifndef INCLUDE_POOLING_HPP_
#define INCLUDE_POOLING_HPP_

#include <vector>

#include "layer-parameter.hpp"
#include "layer.hpp"
#include "tensor.hpp"
#include "utils.hpp"

namespace my_tensor {

template <typename T = float>
class Pooling final : public Layer<T> {
 public:
  explicit Pooling(LayerParameterPtr param) : Layer<T>(param) {}

  void CheckTensorCount(const std::vector<TensorPtr<T>>& bottom,
                        const std::vector<TensorPtr<T>>& top) const override;
  void Reshape(const std::vector<TensorPtr<T>>& bottom,
               const std::vector<TensorPtr<T>>& top) const override;

  void LayerSetUp(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top) override;

  DISABLE_LAYER_COPY(Pooling)

  // Only for test
  const TensorPtr<int> GetMask() const { return mask_; }
  TensorPtr<int> GetMask() { return mask_; }

  void ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top) override;
  void BackwardCPU(const std::vector<TensorPtr<T>>& top,
                   const std::vector<TensorPtr<T>>& bottom) override;
  void ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top) override;
  void BackwardGPU(const std::vector<TensorPtr<T>>& top,
                   const std::vector<TensorPtr<T>>& bottom) override;

 private:
  int batch_size_;
  int input_channels_;
  int input_height_;
  int input_width_;
  int kernel_w_;
  int kernel_h_;
  int stride_w_;
  int stride_h_;
  int output_height_;
  int output_width_;
  TensorPtr<int> mask_;

  void CheckShape(const TensorPtr<T> bottom, const TensorPtr<T> top) const;
};  // class PoolingLayer

extern template class Pooling<>;

}  // namespace my_tensor

#endif  // INCLUDE_POOLING_HPP_
