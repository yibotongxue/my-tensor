#ifndef INCLUDE_LINEAR_CUH_
#define INCLUDE_LINEAR_CUH_

#include <tensor.cuh>
#include <layer.cuh>

namespace my_tensor {

template <typename T>
class Linear final : public Layer<T> {
 public:
  Linear() = default;

  DISABLE_LAYER_COPY(Linear)

  virtual ~Linear() = default;

  void ForwardCPU(const TensorPtr<T>& bottom, TensorPtr<T>& top) override;
  void BackwardCPU(const TensorPtr<T>& top, TensorPtr<T>& bottom) override;
  void ForwardGPU(const TensorPtr<T>& bottom, TensorPtr<T>& top) override;
  void BackwardGPU(const TensorPtr<T>& top, TensorPtr<T>& bottom) override;

private:
  int M_;
  int K_;
  int N_;
};  // class Linear

}  // namespace my_tensor

#endif  // INCLUDE_LINEAR_CUH_
