// Copyright 2024 yibotongxue

#include "sgd-solver.hpp"

#include "blas.hpp"

namespace my_tensor {

template <typename T>
void SgdSolver<T>::UpdateParam() {
  for (auto&& param : this->net_->GetLearnableParams()) {
    add_two_vec(param->GetDataPtr(), param->GetDiffPtr(),
                -this->GetLearningRate(), param->GetSize());
  }
}

}  // namespace my_tensor
