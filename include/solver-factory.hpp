// Copyright 2024 yibotongxue

#ifndef INCLUDE_SOLVER_FACTORY_HPP_
#define INCLUDE_SOLVER_FACTORY_HPP_

#include <memory>

#include "sgd-solver.hpp"
#include "sgd-with-momentum-solver.hpp"
#include "solver.hpp"

namespace my_tensor {

template <typename T>
inline SolverPtr<T> CreateSolver(SolverParameterPtr param) {
  if (param->type_ == SolverType::kSgd) {
    return std::make_shared<SgdSolver<T>>(param);
  } else if (param->type_ == SolverType::kSgdWithMomentum) {
    return std::make_shared<SgdWithMomentumSolver<T>>(param);
  } else {
    // TODO(yibotongxue) Add specific exception type and description.
    throw std::runtime_error("");
  }
}

}  // namespace my_tensor

#endif  // INCLUDE_SOLVER_FACTORY_HPP_
