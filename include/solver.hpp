// Copyright 2024 yibotongxue

#ifndef INCLUDE_SOLVER_HPP_
#define INCLUDE_SOLVER_HPP_

#include <memory>

#include "net.hpp"
#include "scheduler.hpp"
#include "solver-parameter.hpp"

namespace my_tensor {

template <typename T>
class Solver {
 public:
  explicit Solver(SolverParameterPtr param)
      : param_(param), training_iter_(0) {}

  void SetUp();

  virtual ~Solver() = default;

  void Solve();

  float Test();

 protected:
  NetPtr<T> net_;
  SolverParameterPtr param_;
  int training_iter_;
  lr_scheduler scheduler_;
  int max_iter_;
  int current_epoch_;
  float base_lr_;
  float l2_;
  int test_step_;

  void CommonSetUp();
  virtual void SpecialSetUp() {}

  void Step();

  virtual void UpdateParam() = 0;

  float GetLearningRate() { return scheduler_(base_lr_, current_epoch_); }
};

extern template class Solver<float>;

template <typename T>
using SolverPtr = std::shared_ptr<Solver<T>>;

}  // namespace my_tensor

#endif  // INCLUDE_SOLVER_HPP_
