#ifndef INCLUDE_RELU_CUH_
#define INCLUDE_RELU_CUH_

#include <layer.cuh>
#include <tensor.cuh>

#include <memory>

namespace my_tensor {
// Relu class, implements Layer.
class Relu : public Layer {
 public:
  Relu() = default;

  Relu(const Relu&) = delete;
  Relu& operator=(const Relu&) = delete;
  Relu(Relu&&) = delete;
  Relu& operator=(Relu&&) = delete;

  virtual ~Relu() = default;

  // Override forward and backward methods of Layer class.
  void Forward(const TensorPtr& bottom, TensorPtr& top) override;
  void Backward(const TensorPtr& top, TensorPtr& bottom) override;
};
}  // namespace my_tensor


#endif  // INCLUDE_RELU_CUH_
