#ifndef INCLUDE_LAYER_CUH_
#define INCLUDE_LAYER_CUH_

#include <tensor.cuh>

#include <memory>

namespace my_tensor {
// Layer abstract class.
template <typename T=float>
class Layer {
 public:
  // Default constructor.
  Layer() = default;

  // The layer can not be copied or moved.
  Layer(const Layer<T>&) = delete;
  Layer<T>& operator=(const Layer<T>&) = delete;
  Layer(Layer<T>&&) = delete;
  Layer<T>& operator=(Layer<T>&&) = delete;

  virtual ~Layer() = default;

  // Pure virtual methods, forward and backward.
  virtual void Forward(const TensorPtr<T>& bottom, TensorPtr<T>& top) = 0;
  virtual void Backward(const TensorPtr<T>& top, TensorPtr<T>& bottome) = 0;

 protected:
  std::vector<TensorPtr<T>> params;
};

// Layer pointer.
template <typename T>
using LayerPtr = std::unique_ptr<my_tensor::Layer<T>>;
}  // namespace my_tensor


#endif  // INCLUDE_LAYER_CUH_
