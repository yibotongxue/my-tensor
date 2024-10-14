#include <relu.cuh>
#include <utils.cuh>
#include <thrust/transform.h>

namespace my_tensor {

namespace {
struct ReluOperator {
  __device__ __host__ float operator()(float x) {
    return x > 0 ? x : 0;
  }
};

struct ReluGradOperator {
  __device__ __host__ float operator()(float data, float diff) {
    return data > 0 ? diff : 0;
  }
};
}  // namespace

void Relu::Forward(const TensorPtr& bottom, TensorPtr& top) {
  thrust::transform(bottom->GetData().begin(), bottom->GetData().end(),
    top->GetMutableData().begin(), ReluOperator());
}

void Relu::Backward(const TensorPtr& top, TensorPtr& bottom) {
  thrust::transform(bottom->GetData().begin(), bottom->GetData().end(),
    top->GetDiff().begin(), bottom->GetMutableDiff().begin(), ReluGradOperator());
}

}  // namespace my_tensor
