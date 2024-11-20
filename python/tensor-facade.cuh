#ifndef PYTHON_TENSOR_FACADE_CUH_
#define PYTHON_TENSOR_FACADE_CUH_

#include "tensor.cuh"
#include <vector>

namespace my_tensor {
template <typename T>
class TensorFacade {
 public:
  explicit TensorFacade(const std::vector<int>& shape);

  void Reshape(const std::vector<int>& shape);

  void SetData(const std::vector<T>& data);

 private:
  TensorPtr<T> tensor_;
};

extern template class TensorFacade<float>;
}

#endif  // PYTHON_TENSOR_FACADE_CUH_
