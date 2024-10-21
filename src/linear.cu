#include <linear.cuh>
#include <blas.cuh>
#include <error.h>

#include <thrust/fill.h>

namespace my_tensor {

template <typename T>
Linear<T>::Linear(const std::vector<TensorPtr<T>>& params) : Layer<T>(params) {
  if (Layer<T>::params_.size() != 2) {
    throw LinearError("Params size not equals to 2 in linear layer.");
  }
  const std::vector<int>& weight_shape = Layer<T>::params_[0]->GetShape();
  const std::vector<int>& bias_shape = Layer<T>::params_[1]->GetShape();
  if (weight_shape.size() != 2) {
    throw LinearError("Weight shape not be two dimension.");
  }
  if (bias_shape.size() != 1) {
    throw LinearError("Bias shape not be one dimension.");
  }
  if (bias_shape[0] != weight_shape[0]) {
    throw LinearError("Weight shape not matches bias.");
  }
}

template <typename T>
void Linear<T>::ForwardCPU(const TensorPtr<T>& bottom, TensorPtr<T>& top) {
  CheckShape(bottom, top);
  const auto& weight = Layer<T>::params_[0]->GetCPUData();
  const auto& bias = Layer<T>::params_[1]->GetCPUData();
  const auto& bottom_data = bottom->GetCPUData();
  auto& top_data = top->GetCPUData();
  const std::vector<int>& weight_shape = Layer<T>::params_[0]->GetShape();
  const std::vector<int>& bottom_shape = bottom->GetShape();
  const std::vector<int>& top_shape = top->GetShape();
  int m = weight_shape[0];
  int k = bottom_shape[0];
  int n = bottom_shape[1];
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float temp = bias[i];
      for (int l = 0; l < k; l++) {
        temp += weight[i * k + l] * bottom_data[l * n + j];
      }
      top_data[i * n + j] = temp;
    }
  }
}

template <typename T>
void Linear<T>::BackwardCPU(const TensorPtr<T>& top, TensorPtr<T>& bottom) {
  CheckShape(bottom, top);
  const auto& weight = Layer<T>::params_[0]->GetCPUData();
  const auto& bias = Layer<T>::params_[1]->GetCPUData();
  const auto& top_diff = top->GetCPUDiff();
}

template <typename T>
void Linear<T>::ForwardGPU(const TensorPtr<T>& bottom, TensorPtr<T>& top) {
  CheckShape(bottom, top);
  // *top = add_vector(matmul(*Layer<T>::params_[0], *bottom), *Layer<T>::params_[1]);
}

template <typename T>
void Linear<T>::BackwardGPU(const TensorPtr<T>& top, TensorPtr<T>& bottom) {
  CheckShape(bottom, top);
  // *bottom = transpose_matmul(*Layer<T>::params_[0], *top, true);
  // *Layer<T>::params_[0] = matmul_transpose(*top, *bottom, true);
  // *Layer<T>::params_[1] = row_sum(*top, true);
}

template <typename T>
void Linear<T>::CheckShape(const TensorPtr<T>& bottom, const TensorPtr<T>& top) const {
  const std::vector<int>& weight_shape = Layer<T>::params_[0]->GetShape();
  const std::vector<int>& bottom_shape = bottom->GetShape();
  const std::vector<int>& top_shape = top->GetShape();
  if (weight_shape[1] != bottom_shape[0]) {
    throw LinearError("Matmul weight and bottom shapes not match.");
  }
  if (weight_shape[0] != top_shape[0]) {
    throw LinearError("Matmul weight and top shapes not match.");
  }
  if (bottom_shape[1] != top_shape[1]) {
    throw LinearError("Matmul bottom and top shapes not match.");
  }
}

}  // namespace my_tensor
