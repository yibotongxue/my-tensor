// Copyright 2024 yibotongxue

#include "solver.hpp"

#include <spdlog/spdlog.h>

#include "model-saver.hpp"
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
  save_step_ = param_->save_step_;
  save_model_path_ = param_->save_model_path_;
  load_model_path_ = param_->load_model_path_;
  scheduler_ = CreateScheduler(param_->scheduler_param_);
  net_ = std::make_shared<Net<T>>(param_->net_param_);
  net_->SetUp();
  net_->SetTrain();
  if (!load_model_path_.empty()) {
    LoadModel(load_model_path_);
  } else {
    spdlog::info("No model loaded");
  }
}

template <typename T>
void Solver<T>::Solve() {
  while (training_iter_ < max_iter_) {
    Step();
    UpdateParam();
    if (training_iter_ % save_step_ == 0) {
      SaveModel(save_model_path_);
    }
  }
  SaveModel(save_model_path_);
}

template <typename T>
float Solver<T>::Test() {
  net_->Reset();
  net_->SetTest();
  int total = 0;
  float accuracy = 0.0f;
  while (!net_->RefetchData()) {
    net_->Forward();
    accuracy += net_->GetOutput();
    total++;
  }
  return accuracy / static_cast<float>(total);
}

template <typename T>
void Solver<T>::SaveModel(const std::string& model_path) {
  try {
    ModelSaver::Save<T>(this->net_->GetModelData(), model_path);
    spdlog::info("Model saved to {}", model_path);
  } catch (const ModelError& e) {
    spdlog::error("{}", e.what());
  }
}

template <typename T>
void Solver<T>::LoadModel(const std::string& model_path) {
  try {
    this->net_->SetModelData(ModelSaver::Load<T>(model_path));
    spdlog::info("Model loaded from {}", model_path);
  } catch (const ModelError& e) {
    spdlog::error("{}", e.what());
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
  spdlog::info("loss = {}", net_->GetOutput());
  if (training_iter_ % test_step_ == 0) {
    net_->SetTest();
    net_->RefetchData();
    net_->Forward();
    spdlog::info("test accuracy = {}", net_->GetOutput());
    net_->SetTrain();
  }
}

template class Solver<float>;

}  // namespace my_tensor
