// Copyright 2024 yibotongxue

#ifndef INCLUDE_LINEAR_CUH_
#define INCLUDE_LINEAR_CUH_

#include <vector>

#include "layer-parameter.hpp"
#include "layer.cuh"
#include "tensor.cuh"

namespace my_tensor {

template <typename T = float>
class Linear final : public Layer<T> {
 public:
  explicit Linear(LayerParameterPtr param) : Layer<T>(param) {}

  void CheckTensorCount(const std::vector<TensorPtr<T>>& bottom,
                        const std::vector<TensorPtr<T>>& top) const override;

  void LayerSetUp(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top) override;

  DISABLE_LAYER_COPY(Linear)

  virtual ~Linear() = default;

  inline const TensorPtr<T> GetWeight() const { return weight_; }
  inline TensorPtr<T> GetWeight() { return weight_; }
  inline const TensorPtr<T> GetBias() const { return bias_; }
  inline TensorPtr<T> GetBias() { return bias_; }

  void ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top) override;
  void BackwardCPU(const std::vector<TensorPtr<T>>& top,
                   const std::vector<TensorPtr<T>>& bottom) override;
  void ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top) override;
  void BackwardGPU(const std::vector<TensorPtr<T>>& top,
                   const std::vector<TensorPtr<T>>& bottom) override;

 private:
  TensorPtr<T> weight_;
  TensorPtr<T> bias_;
  int m;
  int k;
  int n;

  void CheckShape(const TensorPtr<T> bottom, const TensorPtr<T> top) const;
};  // class Linear

extern template class Linear<>;

}  // namespace my_tensor

#endif  // INCLUDE_LINEAR_CUH_
