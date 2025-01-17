// Copyright 2024 yibotongxue

#ifndef INCLUDE_ADAMW_SOLVER_HPP_
#define INCLUDE_ADAMW_SOLVER_HPP_

#include <vector>

#include "solver.hpp"

namespace my_tensor {

template <typename T>
class AdamWSolver final : public Solver<T> {
 public:
  explicit AdamWSolver(SolverParameterPtr param) : Solver<T>(param) {}

 private:
  std::vector<TensorPtr<T>> m_data_;
  std::vector<TensorPtr<T>> v_data_;
  T beta1_;
  T beta2_;
  T epsilon_;
  int time_step_;

  void UpdateParam() override;

  void SpecialSetUp() override;
};  // class AdamWSolver

extern template class AdamWSolver<float>;

}  // namespace my_tensor

#endif  // INCLUDE_ADAMW_SOLVER_HPP_
