// Copyright 2024 yibotongxue

#include "sgd-with-momentum-solver.hpp"

#include "blas.hpp"

namespace my_tensor {

template <typename T>
void SgdWithMomentumSolver<T>::UpdateParam() {
  const auto& learnable_params = this->net_->GetLearnableParams();
  for (int i = 0; i < learnable_params.size(); i++) {
    auto& param = learnable_params[i];
    auto& history = history_data_[i];
    add_two_vec<T>(param->GetDiffPtr(), param->GetDataPtr(), 2.0f * this->l2_,
                   param->GetSize());
    scale<T>(history->GetDataPtr(), history->GetSize(), momentum_);
    add_two_vec<T>(history->GetDataPtr(), param->GetDiffPtr(), 1.0f,
                   param->GetSize());
    add_two_vec<T>(param->GetDataPtr(), history->GetDataPtr(),
                   -this->GetLearningRate(), param->GetSize());
  }
}

template <typename T>
void SgdWithMomentumSolver<T>::SpecialSetUp() {
  const auto& learnable_params = this->net_->GetLearnableParams();
  history_data_ = std::vector<TensorPtr<T>>(learnable_params.size());
  for (int i = 0; i < history_data_.size(); i++) {
    history_data_[i] =
        std::make_shared<Tensor<T>>(learnable_params[i]->GetShape());
  }
  auto sgd_with_momentum_param =
      std::dynamic_pointer_cast<SgdWithMomentumSolverParameter>(this->param_);
  momentum_ = sgd_with_momentum_param->momentum_;
}

template class SgdWithMomentumSolver<float>;

}  // namespace my_tensor
