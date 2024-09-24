#ifndef MYTENSOR_INCLUDE_LAYER_CUH_
#define MYTENSOR_INCLUDE_LAYER_CUH_

#include <tensor.cuh>

#include <memory>

namespace my_tensor {
class Layer {
public:
  Layer() = default;

  Layer(const Layer&) = delete;
  Layer& operator=(const Layer&) = delete;
  Layer(Layer&&) = delete;
  Layer& operator=(Layer&&) = delete;

  virtual ~Layer() = default;

  virtual void Forward(const std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output) = 0;
  virtual void Backward(const std::shared_ptr<Tensor> top, std::shared_ptr<Tensor> bottome) = 0;
};
} // namespace my_tensor


#endif // MYTENSOR_INCLUDE_LAYER_CUH_
