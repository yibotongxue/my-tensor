// Copyright 2024 yibotongxue

#include "solver.hpp"

#include <iostream>

#include "scheduler-parameter.hpp"

namespace my_tensor {

template <typename T>
void Solver<T>::SetUp() {
  CommonSetUp();
  SpecialSetUp();
}

template <typename T>
void Solver<T>::CommonSetUp() {
  training_iter_ = 0;
  max_iter_ = param_->max_iter_;
  base_lr_ = param_->base_lr_;
  current_epoch_ = 0;
  l2_ = param_->l2_;
  test_step_ = param_->test_step_;
  scheduler_ = CreateScheduler(param_->scheduler_param_);
  net_ = std::make_shared<Net<T>>(param_->net_param_);
  net_->SetUp();
  net_->SetTrain();
}

template <typename T>
void Solver<T>::Solve() {
  while (training_iter_ < max_iter_) {
    Step();
    UpdateParam();
  }
}

template <typename T>
void Solver<T>::Step() {
  if (net_->RefetchData()) {
    current_epoch_++;
  }
  net_->Forward();
  net_->Backward();
  training_iter_++;
  std::cout << std::format("loss = {}", net_->GetOutput()) << std::endl;
  if (training_iter_ % test_step_ == 0) {
    net_->SetTest();
    net_->RefetchData();
    net_->Forward();
    std::cout << std::format("accuracy = {}", net_->GetOutput()) << std::endl;
    net_->SetTrain();
  }
}

template class Solver<float>;

}  // namespace my_tensor
