#include <sigmoid.cuh>
#include <utils.cuh>
#include <thrust/transform.h>

namespace my_tensor {
namespace {
struct SigmoidOperator {
  __host__ __device__ float operator()(float x) {
    return 1.0f / (1.0f + std::exp(-x));
  }
};

struct SigmoidGradOperator {
  __host__ __device__ float operator()(float top_diff, float top_data) {
    return top_diff * top_data * (1 - top_data);
  }
};
}  // namespace

void Sigmoid::Forward(const TensorPtr& bottom, TensorPtr& top) {
  thrust::transform(bottom->GetData().begin(), bottom->GetData().end(),
    top->GetMutableData().begin(), SigmoidOperator());
}

void Sigmoid::Backward(const TensorPtr& top, TensorPtr& bottom) {
  thrust::transform(top->GetDiff().begin(), top->GetDiff().end(),
    top->GetData().begin(), bottom->GetMutableDiff().begin(), SigmoidGradOperator());
}
}  // namespace my_tensor
