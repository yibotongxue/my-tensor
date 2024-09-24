#ifndef MYTENSOR_INCLUDE_RELU_CUH_
#define MYTENSOR_INCLUDE_RELU_CUH_

#include <layer.cuh>
#include <tensor.cuh>

#include <memory>

namespace my_tensor {
class Relu : public Layer {
public:
  Relu() = default;

  Relu(const Relu&) = delete;
  Relu& operator=(const Relu&) = delete;
  Relu(Relu&&) = delete;
  Relu& operator=(Relu&&) = delete;

  virtual ~Relu() = default;

  virtual void Forward(
    const std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output) override;
  virtual void Backward(
    const std::shared_ptr<Tensor> top, std::shared_ptr<Tensor> bottom) override;
};
} // namespace my_tensor


#endif // MYTENSOR_INCLUDE_RELU_CUH_
