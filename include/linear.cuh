// Copyright 2024 yibotongxue

#ifndef INCLUDE_LINEAR_CUH_
#define INCLUDE_LINEAR_CUH_

#include <tensor.cuh>
#include <layer.cuh>

#include <vector>

namespace my_tensor {

template <typename T = float>
class Linear final : public Layer<T> {
 public:
  Linear(const std::vector<TensorPtr<T>>& params);

  DISABLE_LAYER_COPY(Linear)

  virtual ~Linear() = default;

  inline const TensorPtr<T> GetWeight() const { return this->params_[0]; }
  inline TensorPtr<T> GetWeight() { return this->params_[0]; }
  inline const TensorPtr<T> GetBias() const { return this->params_[1]; }
  inline TensorPtr<T> GetBias() { return this->params_[1]; }

  void ForwardCPU(const TensorPtr<T> bottom, TensorPtr<T> top) override;
  void BackwardCPU(const TensorPtr<T> top, TensorPtr<T> bottom) override;
  void ForwardGPU(const TensorPtr<T> bottom, TensorPtr<T> top) override;
  void BackwardGPU(const TensorPtr<T> top, TensorPtr<T> bottom) override;

 private:
  void CheckShape(const TensorPtr<T> bottom, const TensorPtr<T> top) const;
};  // class Linear

extern template class Linear<>;

}  // namespace my_tensor

#endif  // INCLUDE_LINEAR_CUH_
