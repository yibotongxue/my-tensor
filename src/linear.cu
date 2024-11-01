// Copyright 2024 yibotongxue

#include <thrust/fill.h>

#include <memory>
#include <vector>

#include "blas.cuh"
#include "error.h"
#include "linear.cuh"

namespace my_tensor {

template <typename T>
void Linear<T>::SetUp(const TensorPtr<T> bottom) {
  if (this->layer_param_->type_ != ParamType::kLinear) {
    throw LayerError("Layer type not match.");
  }
  if (bottom->GetShape().size() != 2) {
    throw LinearError("Input of linear layer should be two dimesion tensor.");
  }
  std::shared_ptr<LinearParameter> param =
      std::dynamic_pointer_cast<LinearParameter>(this->layer_param_);
  assert(param.get() != nullptr);
  weight_.reset();
  bias_.reset();
  k = param->input_feature_;
  n = param->output_feature_;
  m = bottom->GetShape()[0];
  const std::vector<int> weight_shape = {k, n};
  const std::vector<int> bias_shape = {n};
  weight_ = std::make_shared<Tensor<T>>(weight_shape);
  bias_ = std::make_shared<Tensor<T>>(bias_shape);
}

template <typename T>
void Linear<T>::ForwardCPU(const TensorPtr<T> bottom, TensorPtr<T> top) {
  CheckShape(bottom, top);
  const auto& weight_data = weight_->GetCPUData();
  const auto& bias_data = bias_->GetCPUData();
  const auto& bottom_data = bottom->GetCPUData();
  auto& top_data = top->GetCPUData();
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
  const auto& weight_data = weight_->GetCPUData();
  const auto& bias_data = bias_->GetCPUData();
  const auto& top_diff = top->GetCPUDiff();
  const auto& bottom_data = bottom->GetCPUData();
  auto& weight_diff = weight_->GetCPUDiff();
  auto& bias_diff = bias_->GetCPUDiff();
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
        temp += top_diff[i * n + l] * weight_data[j * n + l];
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
  // bottom m * k
  // weight k * n
  // top    m * n
  // bias   n * 1
  matmul(bottom->GetGPUDataPtr(), weight_->GetGPUDataPtr(),
         top->GetGPUDataPtr(), m, k, n);
  add_col_vector(top->GetGPUDataPtr(), bias_->GetGPUDataPtr(), m, n);
}

template <typename T>
void Linear<T>::BackwardGPU(const TensorPtr<T> top, TensorPtr<T> bottom) {
  CheckShape(bottom, top);
  // bottom m * k
  // weight k * n
  // top    m * n
  // bias   n * 1
  matmul_transpose(top->GetGPUDiffPtr(), this->GetWeight()->GetGPUDataPtr(),
                   bottom->GetGPUDiffPtr(), m, n, k);
  transpose_matmul(bottom->GetGPUDataPtr(), top->GetGPUDiffPtr(),
                   this->GetWeight()->GetGPUDiffPtr(), k, m, n);
  col_sum(top->GetGPUDiffPtr(), this->GetBias()->GetGPUDiffPtr(), m, n);
  // *bottom = transpose_matmul(*weight_, *top, true);
  // *weight_ = matmul_transpose(*top, *bottom, true);
  // *bias_ = row_sum(*top, true);
}

template <typename T>
void Linear<T>::CheckShape(const TensorPtr<T> bottom,
                           const TensorPtr<T> top) const {
  const std::vector<int>& bottom_shape = bottom->GetShape();
  const std::vector<int>& top_shape = top->GetShape();
  if (bottom_shape[0] != m) {
    throw LinearError("Input not match input features.");
  }
  if (bottom_shape[1] != k) {
    throw LinearError("Matmul weight and bottom shapes not match.");
  }
  if (top_shape[1] != n) {
    throw LinearError("Matmul weight and top shapes not match.");
  }
  if (bottom_shape[0] != top_shape[0]) {
    throw LinearError("Matmul bottom and top shapes not match.");
  }
}

template class Linear<>;

}  // namespace my_tensor
