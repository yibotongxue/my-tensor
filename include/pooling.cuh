// Copyright 2024 yibotongxue

#ifndef INCLUDE_POOLING_CUH_
#define INCLUDE_POOLING_CUH_

#include "layer-parameter.hpp"
#include "layer.cuh"
#include "tensor.cuh"
#include "utils.cuh"

namespace my_tensor {

template <typename T = float>
class Pooling final : public Layer<T> {
 public:
  explicit Pooling(LayerParameterPtr param) : Layer<T>(param) {}

  void SetUp(const TensorPtr<T> bottom) override;

  DISABLE_LAYER_COPY(Pooling)

// Only for test
#ifdef DEBUG
  const TensorPtr<int> GetMask() const { return mask_; }
  TensorPtr<int> GetMask() { return mask_; }
#endif  // DEBUG

  void ForwardCPU(const TensorPtr<T> bottom, TensorPtr<T> top) override;
  void ForwardGPU(const TensorPtr<T> bottom, TensorPtr<T> top) override;
  void BackwardCPU(const TensorPtr<T> top, TensorPtr<T> bottom) override;
  void BackwardGPU(const TensorPtr<T> top, TensorPtr<T> bottom) override;

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

#endif  // INCLUDE_POOLING_CUH_
