// Copyright 2024 yibotongxue

#include <iostream>
#include <memory>
#include <vector>

#include "common.hpp"
#include "data-loader.hpp"
#include "dataset.hpp"
#include "json-loader.hpp"
#include "layer-factory.hpp"
#include "layer-parameter.hpp"
#include "net-parameter.hpp"
#include "net.hpp"
#include "sgd-solver.hpp"
#include "solver-factory.hpp"
#include "solver.hpp"
#include "tensor.hpp"

int main(int argc, char** argv) {
  if (argc != 2) {
    throw std::runtime_error("");
  }
  my_tensor::MyTensorContext::set_device_type(
      my_tensor::MyTensorContext::DeviceType::GPU);
  my_tensor::JsonLoader loader(argv[1]);
  my_tensor::SolverPtr<float> solver =
      my_tensor::CreateSolver<float>(loader.LoadSolver());
  solver->SetUp();
  solver->Solve();
  std::cout << "Final test accuracy = " << solver->Test() << std::endl;
  return 0;
}
