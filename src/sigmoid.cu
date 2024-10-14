#include <sigmoid.cuh>
#include <utils.cuh>
#include <thrust/transform.h>

namespace my_tensor {
namespace {
template <typename T>
struct SigmoidOperator {
  __host__ __device__ T operator()(T x) {
    return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
  }
};

template <typename T>
struct SigmoidGradOperator {
  __host__ __device__ T operator()(T top_diff, T top_data) {
    return top_diff * top_data * (1 - top_data);
  }
};
}  // namespace

template <typename T>
void Sigmoid<T>::Forward(const TensorPtr<T>& bottom, TensorPtr<T>& top) {
  thrust::transform(bottom->GetData().begin(), bottom->GetData().end(),
    top->GetMutableData().begin(), SigmoidOperator<T>());
}

template <typename T>
void Sigmoid<T>::Backward(const TensorPtr<T>& top, TensorPtr<T>& bottom) {
  thrust::transform(top->GetDiff().begin(), top->GetDiff().end(),
    top->GetData().begin(), bottom->GetMutableDiff().begin(), SigmoidGradOperator<T>());
}
}  // namespace my_tensor
