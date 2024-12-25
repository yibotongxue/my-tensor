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
  matmul_gpu(bottom[0]->GetGPUDataPtr(), weight_->GetGPUDataPtr(),
             top[0]->GetGPUDataPtr(), m, k, n);
  add_col_vector_gpu(top[0]->GetGPUDataPtr(), bias_->GetGPUDataPtr(), m, n);
}

template <typename T>
void Linear<T>::BackwardGPU(const std::vector<TensorPtr<T>>& top,
                            const std::vector<TensorPtr<T>>& bottom) {
  // std::cout << this->layer_param_->name_ << " BackwardGPU" << std::endl;
  // std::ranges::copy(SPAN_DATA(weight_, T),
  //                   std::ostream_iterator<T>(std::cout, " "));
  // std::cout << std::endl;
  CheckShape(bottom[0], top[0]);
  // bottom m * k
  // weight k * n
  // top    m * n
  // bias   n * 1
  matmul_transpose_gpu(top[0]->GetGPUDiffPtr(),
                       this->GetWeight()->GetGPUDataPtr(),
                       bottom[0]->GetGPUDiffPtr(), m, n, k);
  transpose_matmul_gpu(bottom[0]->GetGPUDataPtr(), top[0]->GetGPUDiffPtr(),
                       this->GetWeight()->GetGPUDiffPtr(), k, m, n);
  col_sum_gpu(top[0]->GetGPUDiffPtr(), this->GetBias()->GetGPUDiffPtr(), m, n);
  // *bottom = transpose_matmul(*weight_, *top, true);
  // *weight_ = matmul_transpose(*top, *bottom, true);
  // *bias_ = row_sum(*top, true);
  // if (this->layer_param_->name_ == "linear3") {
  //   std::ranges::copy(SPAN_DIFF(top[0], T),
  //   std::ostream_iterator<T>(std::cout, " ")); std::cout << std::endl;
  // }
}

template class Linear<>;

}  // namespace my_tensor
