#include <relu.cuh>
#include <utils.cuh>
#include <thrust/transform.h>

namespace my_tensor {

namespace {
template <typename T>
struct ReluOperator {
  __device__ __host__ T operator()(T x) {
    return x > 0 ? x : 0;
  }
};

template <typename T>
struct ReluGradOperator {
  __device__ __host__ T operator()(T data, T diff) {
    return data > 0 ? diff : 0;
  }
};
}  // namespace

template <typename T>
void Relu<T>::Forward(const TensorPtr<T>& bottom, TensorPtr<T>& top) {
  thrust::transform(bottom->GetData().begin(), bottom->GetData().end(),
    top->GetMutableData().begin(), ReluOperator<T>());
}

template <typename T>
void Relu<T>::Backward(const TensorPtr<T>& top, TensorPtr<T>& bottom) {
  thrust::transform(bottom->GetData().begin(), bottom->GetData().end(),
    top->GetDiff().begin(), bottom->GetMutableDiff().begin(), ReluGradOperator<T>());
}

template class Layer<>;
template class Relu<>;
}  // namespace my_tensor
