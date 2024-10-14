#ifndef INCLUDE_SIGMOID_CUH_
#define INCLUDE_SIGMOID_CUH_

#include <tensor.cuh>
#include <layer.cuh>

#include <memory>

namespace my_tensor {
// Sigmoid class, implements Layer class.
class Sigmoid : public Layer {
 public:
  Sigmoid() = default;

  Sigmoid(const Sigmoid&) = delete;
  Sigmoid& operator=(const Sigmoid&) = delete;
  Sigmoid(Sigmoid&&) = delete;
  Sigmoid& operator=(Sigmoid&&) = delete;

  virtual ~Sigmoid() = default;

  // Override forward and backward methods of Layer class.
  void Forward(const TensorPtr& bottom, TensorPtr& top) override;
  void Backward(const TensorPtr& top, TensorPtr& bottom) override;
};
}  // namespace my_tensor


#endif  // INCLUDE_SIGMOID_CUH_
