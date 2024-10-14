#ifndef INCLUDE_LAYER_CUH_
#define INCLUDE_LAYER_CUH_

#include <tensor.cuh>

#include <memory>

namespace my_tensor {
// Layer abstract class.
class Layer {
 public:
  // Default constructor.
  Layer() = default;

  // The layer can not be copied or moved.
  Layer(const Layer&) = delete;
  Layer& operator=(const Layer&) = delete;
  Layer(Layer&&) = delete;
  Layer& operator=(Layer&&) = delete;

  virtual ~Layer() = default;

  // Pure virtual methods, forward and backward.
  virtual void Forward(const TensorPtr& bottom, TensorPtr& top) = 0;
  virtual void Backward(const TensorPtr& top, TensorPtr& bottome) = 0;

 protected:
  std::vector<TensorPtr> params;
};

// Layer pointer.
using LayerPtr = std::unique_ptr<Layer>;
}  // namespace my_tensor


#endif  // INCLUDE_LAYER_CUH_
