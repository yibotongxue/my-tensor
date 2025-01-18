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

  std::vector<TensorPtr<T>> GetLearnableParameters() override {
    return {gama_, beta_};
  }

  virtual ~BatchNorm() = default;

  inline const TensorPtr<T> GetGama() const { return gama_; }
  inline TensorPtr<T> GetGama() { return gama_; }
  inline const TensorPtr<T> GetBeta() const { return beta_; }
  inline TensorPtr<T> GetBeta() { return beta_; }

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
  TensorPtr<T> mean_;
  TensorPtr<T> sqrt_variance_;
  TensorPtr<T> mean_cache_;
  TensorPtr<T> sqrt_variance_cache_;
  TensorPtr<T> standarded_cache_;

  T scale_factor_;
  T move_scale_factor_;
  T temp_cache_;
  int batch_size_;
  int spatial_size_;
};

extern template class BatchNorm<float>;
}  // namespace my_tensor

#endif  // INCLUDE_BATCHNORM_HPP_
