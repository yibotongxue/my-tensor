#ifndef MYTENSOR_INCLUDE_SIGMOID_CUH_
#define MYTENSOR_INCLUDE_SIGMOID_CUH_

#include <tensor.cuh>
#include <layer.cuh>

#include <memory>

namespace my_tensor {
class Sigmoid : public Layer {
public:
  Sigmoid() = default;

  Sigmoid(const Sigmoid&) = delete;
  Sigmoid& operator=(const Sigmoid&) = delete;
  Sigmoid(Sigmoid&&) = delete;
  Sigmoid& operator=(Sigmoid&&) = delete;

  virtual ~Sigmoid() = default;

  virtual void Forward(
    const std::shared_ptr<Tensor> bottom, std::shared_ptr<Tensor> top) override;
  virtual void Backward(
    const std::shared_ptr<Tensor> top, std::shared_ptr<Tensor> bottom) override;
}; // class Sigmoid
} // namespace my_tensor


#endif // MYTENSOR_INCLUDE_SIGMOID_CUH_
