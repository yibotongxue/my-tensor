#ifndef INCLUDE_LAYER_CUH_
#define INCLUDE_LAYER_CUH_

#include <tensor.cuh>
#include <utils.cuh>

#include <memory>

namespace my_tensor {
// Layer abstract class.
template <typename T=float>
class Layer {
 public:
  // Default constructor.
  Layer(const std::vector<TensorPtr<T>>& params) {
    params_.resize(params.size());
    for (int i = 0; i < params_.size(); i++) {
      params_[i] = std::make_shared<Tensor<T>>(*params[i]);
    }
  }

  // The layer can not be copied or moved.
  DISABLE_LAYER_COPY(Layer)

  virtual ~Layer() = default;

  // Pure virtual methods, forward and backward.
  // CPU
  virtual void ForwardCPU(const TensorPtr<T>& bottom, TensorPtr<T>& top) = 0;
  virtual void BackwardCPU(const TensorPtr<T>& top, TensorPtr<T>& bottome) = 0;
  // GPU
  virtual void ForwardGPU(const TensorPtr<T>& bottom, TensorPtr<T>& top) = 0;
  virtual void BackwardGPU(const TensorPtr<T>& top, TensorPtr<T>& bottome) = 0;

 protected:
  std::vector<TensorPtr<T>> params_;
};

// Layer pointer.
template <typename T = float>
using LayerPtr = std::shared_ptr<my_tensor::Layer<T>>;

extern template class my_tensor::Layer<>;
}  // namespace my_tensor


#endif  // INCLUDE_LAYER_CUH_
