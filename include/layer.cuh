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
  // CPU
  virtual void ForwardCPU(const TensorPtr<T>& bottom, TensorPtr<T>& top) = 0;
  virtual void BackwardCPU(const TensorPtr<T>& top, TensorPtr<T>& bottome) = 0;
  // GPU
  virtual void ForwardGPU(const TensorPtr<T>& bottom, TensorPtr<T>& top) = 0;
  virtual void BackwardGPU(const TensorPtr<T>& top, TensorPtr<T>& bottome) = 0;

 protected:
  std::vector<TensorPtr<T>> params;
};

// Layer pointer.
template <typename T = float>
using LayerPtr = std::unique_ptr<my_tensor::Layer<T>>;

extern template class my_tensor::Layer<>;
}  // namespace my_tensor


#endif  // INCLUDE_LAYER_CUH_
