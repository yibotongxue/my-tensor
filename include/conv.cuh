// Copyright 2024 yibotongxue

#ifndef INCLUDE_CONV_CUH_
#define INCLUDE_CONV_CUH_

#include <layer.cuh>
#include <tensor.cuh>
#include <utils.cuh>

#include <vector>

namespace my_tensor {

template <typename T = float>
class Convolution final : public Layer<T> {
 public:
  Convolution(const std::vector<TensorPtr<T>>& params);

  DISABLE_LAYER_COPY(Convolution)

  void ForwardCPU(const TensorPtr<T> bottom, TensorPtr<T> top) override;
  void BackwardCPU(const TensorPtr<T> top, TensorPtr<T> bottom) override;
  void ForwardGPU(const TensorPtr<T> bottom, TensorPtr<T> top) override;
  void BackwardGPU(const TensorPtr<T> top, TensorPtr<T> bottom) override;

 private:
  void CheckShape(const TensorPtr<T> bottom, const TensorPtr<T> top) const;
};  // class Convolution

extern template class Convolution<>;

}  // namespace my_tensor

#endif  // INCLUDE_CONV_CUH_
