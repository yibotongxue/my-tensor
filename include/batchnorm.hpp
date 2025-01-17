// Copyright 2025 yibotongxue

#ifndef INCLUDE_BATCHNORM_HPP_
#define INCLUDE_BATCHNORM_HPP_

#include <memory>
#include <vector>

#include "layer-parameter.hpp"
#include "layer.hpp"
#include "tensor.hpp"

namespace my_tensor {
template <typename T = float>
class BatchNorm final : public Layer<T> {
 public:
  explicit BatchNorm(LayerParameterPtr param) : Layer<T>(param) {}

  DISABLE_LAYER_COPY(BatchNorm)

  void CheckTensorCount(const std::vector<TensorPtr<T>>& bottom,
                        const std::vector<TensorPtr<T>>& top) const override;
  void Reshape(const std::vector<TensorPtr<T>>& bottom,
               const std::vector<TensorPtr<T>>& top) const override;

  void LayerSetUp(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top) override;

  virtual ~BatchNorm() = default;

  void ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top) override;
  void BackwardCPU(const std::vector<TensorPtr<T>>& top,
                   const std::vector<TensorPtr<T>>& bottom) override;

  void ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top) override;
  void BackwardGPU(const std::vector<TensorPtr<T>>& top,
                   const std::vector<TensorPtr<T>>& bottom) override;

 private:
  int channels_;
  TensorPtr<T> gama_;
  TensorPtr<T> beta_;
  TensorPtr<T> mean_cache_;
  TensorPtr<T> variance_cache_;

  int batch_cnt_;
};
}  // namespace my_tensor

#endif  // INCLUDE_BATCHNORM_HPP_
