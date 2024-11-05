// Copyright 2024 yibotongxue

#ifndef INCLUDE_CONV_CUH_
#define INCLUDE_CONV_CUH_

#include <vector>

#include "layer-parameter.h"
#include "layer.cuh"
#include "tensor.cuh"
#include "utils.cuh"

namespace my_tensor {

template <typename T = float>
class Convolution final : public Layer<T> {
 public:
  explicit Convolution(LayerParameterPtr param) : Layer<T>(param) {}

  void CheckTensorCount(const std::vector<TensorPtr<T>>& bottom,
                        const std::vector<TensorPtr<T>>& top) const override;

  void LayerSetUp(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top) override;

  DISABLE_LAYER_COPY(Convolution)

  const TensorPtr<T> GetKernel() const { return kernel_; }
  TensorPtr<T> GetKernel() { return kernel_; }
  const TensorPtr<T> GetBias() const { return bias_; }
  TensorPtr<T> GetBias() { return bias_; }

  void ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top) override;
  void BackwardCPU(const std::vector<TensorPtr<T>>& top,
                   const std::vector<TensorPtr<T>>& bottom) override;
  void ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top) override;
  void BackwardGPU(const std::vector<TensorPtr<T>>& top,
                   const std::vector<TensorPtr<T>>& bottom) override;

 private:
  TensorPtr<T> kernel_;
  TensorPtr<T> bias_;
  TensorPtr<T> col_cache_;
  int input_channels_;
  int output_channels_;
  int kernel_height_;
  int kernel_width_;
  int height_;
  int width_;
  int batch_size_;

  void CheckShape(const TensorPtr<T> bottom, const TensorPtr<T> top) const;
};  // class Convolution

extern template class Convolution<>;

}  // namespace my_tensor

#endif  // INCLUDE_CONV_CUH_
