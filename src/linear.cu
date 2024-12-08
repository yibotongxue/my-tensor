// Copyright 2024 yibotongxue

#include <thrust/fill.h>

#include <memory>
#include <vector>

#include "blas.hpp"
#include "error.hpp"
#include "filler-parameter.hpp"
#include "filler.hpp"
#include "linear.hpp"

namespace my_tensor {

template <typename T>
void Linear<T>::ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                           const std::vector<TensorPtr<T>>& top) {
  CheckShape(bottom[0], top[0]);
  // bottom m * k
  // weight k * n
  // top    m * n
  // bias   n * 1
  matmul(bottom[0]->GetGPUDataPtr(), weight_->GetGPUDataPtr(),
         top[0]->GetGPUDataPtr(), m, k, n);
  add_col_vector(top[0]->GetGPUDataPtr(), bias_->GetGPUDataPtr(), m, n);
}

template <typename T>
void Linear<T>::BackwardGPU(const std::vector<TensorPtr<T>>& top,
                            const std::vector<TensorPtr<T>>& bottom) {
  CheckShape(bottom[0], top[0]);
  // bottom m * k
  // weight k * n
  // top    m * n
  // bias   n * 1
  matmul_transpose(top[0]->GetGPUDiffPtr(), this->GetWeight()->GetGPUDataPtr(),
                   bottom[0]->GetGPUDiffPtr(), m, n, k);
  transpose_matmul(bottom[0]->GetGPUDataPtr(), top[0]->GetGPUDiffPtr(),
                   this->GetWeight()->GetGPUDiffPtr(), k, m, n);
  col_sum(top[0]->GetGPUDiffPtr(), this->GetBias()->GetGPUDiffPtr(), m, n);
  // *bottom = transpose_matmul(*weight_, *top, true);
  // *weight_ = matmul_transpose(*top, *bottom, true);
  // *bias_ = row_sum(*top, true);
}

template class Linear<>;

}  // namespace my_tensor
