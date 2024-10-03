#ifndef INCLUDE_LAYER_CUH_
#define INCLUDE_LAYER_CUH_

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

  virtual void Forward(
    const std::shared_ptr<Tensor> bottom, std::shared_ptr<Tensor> top) = 0;
  virtual void Backward(
    const std::shared_ptr<Tensor> top, std::shared_ptr<Tensor> bottome) = 0;
};

using LayerPtr = std::unique_ptr<Layer>;
}  // namespace my_tensor


#endif  // INCLUDE_LAYER_CUH_
