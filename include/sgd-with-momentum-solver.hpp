// Copyright 2024 yibotongxue

#ifndef INCLUDE_SGD_WITH_MOMENTUM_SOLVER_HPP_
#define INCLUDE_SGD_WITH_MOMENTUM_SOLVER_HPP_

#include <vector>

#include "solver.hpp"

namespace my_tensor {

template <typename T>
class SgdWithMomentumSolver final : public Solver<T> {
 public:
  explicit SgdWithMomentumSolver(SolverParameterPtr param) : Solver<T>(param) {}

 private:
  std::vector<TensorPtr<T>> history_data_;
  T momentum_;

  void UpdateParam() override;

  void SpecialSetUp() override;
};  // class SgdWithMomentumSolver

extern template class SgdWithMomentumSolver<float>;

}  // namespace my_tensor

#endif  // INCLUDE_SGD_WITH_MOMENTUM_SOLVER_HPP_
