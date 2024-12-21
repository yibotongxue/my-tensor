// Copyright 2024 yibotongxue

#include "sgd-solver.hpp"

#include "blas.hpp"

namespace my_tensor {

template <typename T>
void SgdSolver<T>::UpdateParam() {
  for (auto&& param : this->net_->GetLearnableParams()) {
    add_two_vec<float>(param->GetDataPtr(), param->GetDiffPtr(),
                       -this->GetLearningRate(), param->GetSize());
  }
}

template class SgdSolver<float>;

}  // namespace my_tensor
