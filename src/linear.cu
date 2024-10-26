// Copyright 2024 yibotongxue

#include <error.h>
#include <blas.cuh>
#include <linear.cuh>

#include <thrust/fill.h>
#include <vector>

namespace my_tensor {

#define DEFINE_MKN               \
  int m = bottom->GetShape()[0]; \
  int k = bottom->GetShape()[1]; \
  int n = this->params_[0]->GetShape()[1];

template <typename T>
Linear<T>::Linear(const std::vector<TensorPtr<T>>& params) : Layer<T>(params) {
  if (this->params_.size() != 2) {
    throw LinearError("Params size not equals to 2 in linear layer.");
  }
  const std::vector<int>& weight_shape = this->params_[0]->GetShape();
  const std::vector<int>& bias_shape = this->params_[1]->GetShape();
  if (weight_shape.size() != 2) {
    throw LinearError("Weight shape not be two dimension.");
  }
  if (bias_shape.size() != 1) {
    throw LinearError("Bias shape not be one dimension.");
  }
  if (bias_shape[0] != weight_shape[1]) {
    throw LinearError("Weight shape not matches bias.");
  }
}

template <typename T>
void Linear<T>::ForwardCPU(const TensorPtr<T> bottom, TensorPtr<T> top) {
  CheckShape(bottom, top);
  const auto& weight_data = this->params_[0]->GetCPUData();
  const auto& bias_data = this->params_[1]->GetCPUData();
  const auto& bottom_data = bottom->GetCPUData();
  auto& top_data = top->GetCPUData();
  DEFINE_MKN
  // bottom m * k
  // weight k * n
  // top    m * n
  // bias   n * 1
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float temp = bias_data[j];
      for (int l = 0; l < k; l++) {
        temp += bottom_data[i * k + l] * weight_data[l * n + j];
      }
      top_data[i * n + j] = temp;
    }
  }
}

template <typename T>
void Linear<T>::BackwardCPU(const TensorPtr<T> top, TensorPtr<T> bottom) {
  CheckShape(bottom, top);
  DEFINE_MKN
  const auto& weight = this->params_[0]->GetCPUData();
  const auto& bias = this->params_[1]->GetCPUData();
  const auto& top_diff = top->GetCPUDiff();
  const auto& bottom_data = bottom->GetCPUData();
  auto& weight_diff = this->params_[0]->GetCPUDiff();
  auto& bias_diff = this->params_[1]->GetCPUDiff();
  auto& bottom_diff = bottom->GetCPUDiff();
  // bottom m * k
  // weight k * n
  // top    m * n
  // bias   n * 1
  // partial x
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      T temp{0};
      for (int l = 0; l < n; l++) {
        temp += top_diff[i * n + l] * weight[j * n + l];
      }
      bottom_diff[i * k + j] = temp;
    }
  }
  // bottom m * k
  // weight k * n
  // top    m * n
  // bias   n * 1
  // partial weight
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      T temp{0};
      for (int l = 0; l < m; l++) {
        temp += bottom_data[l * k + i] * top_diff[l * n + j];
      }
      weight_diff[i * n + j] = temp;
    }
  }
  // partial bias
  for (int i = 0; i < n; i++) {
    T temp{0};
    for (int j = 0; j < m; j++) {
      temp += top_diff[j * n + i];
    }
    bias_diff[i] = temp;
  }
}

template <typename T>
void Linear<T>::ForwardGPU(const TensorPtr<T> bottom, TensorPtr<T> top) {
  CheckShape(bottom, top);
  DEFINE_MKN
  // bottom m * k
  // weight k * n
  // top    m * n
  // bias   n * 1
  matmul(bottom->GetGPUDataPtr(), this->params_[0]->GetGPUDataPtr(),
         top->GetGPUDataPtr(), m, k, n);
  add_col_vector(top->GetGPUDataPtr(), this->params_[1]->GetGPUDataPtr(), m, n);
}

template <typename T>
void Linear<T>::BackwardGPU(const TensorPtr<T> top, TensorPtr<T> bottom) {
  CheckShape(bottom, top);
  DEFINE_MKN
  // bottom m * k
  // weight k * n
  // top    m * n
  // bias   n * 1
  matmul_transpose(top->GetGPUDiffPtr(), this->GetWeight()->GetGPUDataPtr(),
                   bottom->GetGPUDiffPtr(), m, n, k);
  transpose_matmul(bottom->GetGPUDataPtr(), top->GetGPUDiffPtr(),
                   this->GetWeight()->GetGPUDiffPtr(), k, m, n);
  col_sum(top->GetGPUDiffPtr(), this->GetBias()->GetGPUDiffPtr(), m, n);
  // *bottom = transpose_matmul(*this->params_[0], *top, true);
  // *this->params_[0] = matmul_transpose(*top, *bottom, true);
  // *this->params_[1] = row_sum(*top, true);
}

template <typename T>
void Linear<T>::CheckShape(const TensorPtr<T> bottom,
                           const TensorPtr<T> top) const {
  const std::vector<int>& weight_shape = this->params_[0]->GetShape();
  const std::vector<int>& bottom_shape = bottom->GetShape();
  const std::vector<int>& top_shape = top->GetShape();
  if (weight_shape[0] != bottom_shape[1]) {
    throw LinearError("Matmul weight and bottom shapes not match.");
  }
  if (weight_shape[1] != top_shape[1]) {
    throw LinearError("Matmul weight and top shapes not match.");
  }
  if (bottom_shape[0] != top_shape[0]) {
    throw LinearError("Matmul bottom and top shapes not match.");
  }
}

#undef DEFINE_MKN

template class Linear<>;

}  // namespace my_tensor
