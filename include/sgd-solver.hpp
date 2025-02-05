// Copyright 2024 yibotongxue

#ifndef INCLUDE_SGD_SOLVER_HPP_
#define INCLUDE_SGD_SOLVER_HPP_

#include "solver.hpp"

namespace my_tensor {

template <Arithmetic T>
class SgdSolver final : public Solver<T> {
 public:
  explicit SgdSolver(SolverParameterPtr param) : Solver<T>(param) {}

 private:
  void UpdateParam() override;
};  // class SgdSolver

extern template class SgdSolver<float>;

}  // namespace my_tensor

#endif  // INCLUDE_SGD_SOLVER_HPP_
