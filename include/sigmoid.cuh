#ifndef INCLUDE_SIGMOID_CUH_
#define INCLUDE_SIGMOID_CUH_

#include <tensor.cuh>
#include <layer.cuh>

#include <memory>

namespace my_tensor {
// Sigmoid class, implements Layer class.
template <typename T=float>
class Sigmoid final : public Layer<T> {
 public:
  Sigmoid() = default;

  DISABLE_LAYER_COPY(Sigmoid)

  ~Sigmoid() = default;

  // Override forward and backward methods of Layer class.
  // CPU
  void ForwardCPU(const TensorPtr<T>& bottom, TensorPtr<T>& top) override;
  void BackwardCPU(const TensorPtr<T>& top, TensorPtr<T>& bottom) override;
  // GPU
  void ForwardGPU(const TensorPtr<T>& bottom, TensorPtr<T>& top) override;
  void BackwardGPU(const TensorPtr<T>& top, TensorPtr<T>& bottom) override;
};

extern template class my_tensor::Sigmoid<>;
}  // namespace my_tensor


#endif  // INCLUDE_SIGMOID_CUH_
