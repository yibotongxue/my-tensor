#include <linear.cuh>
#include <blas.cuh>
#include <error.h>

#include <thrust/fill.h>

namespace my_tensor {

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
  if (bias_shape[0] != weight_shape[0]) {
    throw LinearError("Weight shape not matches bias.");
  }
}

template <typename T>
void Linear<T>::ForwardCPU(const TensorPtr<T>& bottom, TensorPtr<T>& top) {
  CheckShape(bottom, top);
  const auto& weight_data = this->params_[0]->GetCPUData();
  const auto& bias_data = this->params_[1]->GetCPUData();
  const auto& bottom_data = bottom->GetCPUData();
  auto& top_data = top->GetCPUData();
  int m = this->params_[0]->GetShape()[0];
  int k = this->params_[0]->GetShape()[1];
  int n = bottom->GetShape()[1];
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float temp = bias_data[i];
      for (int l = 0; l < k; l++) {
        temp += weight_data[i * k + l] * bottom_data[l * n + j];
      }
      top_data[i * n + j] = temp;
    }
  }
}

template <typename T>
void Linear<T>::BackwardCPU(const TensorPtr<T>& top, TensorPtr<T>& bottom) {
  CheckShape(bottom, top);
  int m = this->params_[0]->GetShape()[0];
  int k = this->params_[0]->GetShape()[1];
  int n = bottom->GetShape()[1];
  const auto& weight = this->params_[0]->GetCPUData();
  const auto& bias = this->params_[1]->GetCPUData();
  const auto& top_diff = top->GetCPUDiff();
  const auto& bottom_data = bottom->GetCPUData();
  auto& weight_diff = this->params_[0]->GetCPUDiff();
  auto& bias_diff = this->params_[1]->GetCPUDiff();
  auto& bottom_diff = bottom->GetCPUDiff();
  // partial x
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      T temp {0};
      for (int l = 0; l < m; l++) {
        temp += weight[l * k + i] * top_diff[l * n + j];
      }
      bottom_diff[i * n + k] = temp;
    }
  }
  // partial weight
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      T temp {0};
      for (int l = 0; l < n; l++) {
        temp += top_diff[i * n + l] * bottom_data[i * k + j];
      }
      weight_diff[i * k + j] = temp;
    }
  }
  // partial bias
  for (int i = 0; i < m; i++) {
    T temp {0};
    for (int j = 0; j < n; j++) {
      temp += top_diff[i * n + j];
    }
    bias_diff[i] = temp;
  }
}

template <typename T>
void Linear<T>::ForwardGPU(const TensorPtr<T>& bottom, TensorPtr<T>& top) {
  CheckShape(bottom, top);
  int m = this->params_[0]->GetShape()[0];
  int k = this->params_[0]->GetShape()[1];
  int n = bottom->GetShape()[1];
  matmul(this->params_[0]->GetGPUDataPtr(),
         bottom->GetGPUDataPtr(),
         top->GetGPUDataPtr(),
         m, k, n);
  add_vector(top->GetGPUDataPtr(), this->params_[1]->GetGPUDataPtr(), m, n);
}

template <typename T>
void Linear<T>::BackwardGPU(const TensorPtr<T>& top, TensorPtr<T>& bottom) {
  CheckShape(bottom, top);
  int m = this->params_[0]->GetShape()[0];
  int k = this->params_[0]->GetShape()[1];
  int n = bottom->GetShape()[1];
  transpose_matmul(this->params_[0]->GetGPUDiffPtr(),
                   top->GetGPUDiffPtr(),
                   bottom->GetGPUDiffPtr(),
                   m, k, n);
  matmul_transpose(top->GetGPUDiffPtr(),
                   bottom->GetGPUDataPtr(),
                   this->params_[0]->GetGPUDiffPtr(),
                   m, n, k);
  row_sum(top->GetGPUDiffPtr(), this->params_[1]->GetGPUDiffPtr(), m, n);
  // *bottom = transpose_matmul(*this->params_[0], *top, true);
  // *this->params_[0] = matmul_transpose(*top, *bottom, true);
  // *this->params_[1] = row_sum(*top, true);
}

template <typename T>
void Linear<T>::CheckShape(const TensorPtr<T>& bottom, const TensorPtr<T>& top) const {
  const std::vector<int>& weight_shape = this->params_[0]->GetShape();
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

template class Linear<>;

}  // namespace my_tensor
