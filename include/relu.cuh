#ifndef INCLUDE_RELU_CUH_
#define INCLUDE_RELU_CUH_

#include <layer.cuh>
#include <tensor.cuh>

#include <memory>

namespace my_tensor {
// Relu class, implements Layer.
template <typename T=float>
class Relu : public Layer<T> {
 public:
  Relu() = default;

  Relu(const Relu<T>&) = delete;
  Relu<T>& operator=(const Relu<T>&) = delete;
  Relu(Relu<T>&&) = delete;
  Relu<T>& operator=(Relu<T>&&) = delete;

  virtual ~Relu() = default;

  // Override forward and backward methods of Layer class.
  void Forward(const TensorPtr<T>& bottom, TensorPtr<T>& top) override;
  void Backward(const TensorPtr<T>& top, TensorPtr<T>& bottom) override;
};

extern template class my_tensor::Relu<>;
}  // namespace my_tensor


#endif  // INCLUDE_RELU_CUH_
