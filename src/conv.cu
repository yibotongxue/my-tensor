#include <conv.cuh>

namespace my_tensor {

template <typename T>
Convolution<T>::Convolution(const std::vector<TensorPtr<T>>& params) : Layer<T>(params) {}

}  // namespace my_tensor
