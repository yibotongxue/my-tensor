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
#include "solver.hpp"
#include "tensor.hpp"

int main() {
  my_tensor::MyTensorContext::set_device_type(
      my_tensor::MyTensorContext::DeviceType::GPU);
  my_tensor::JsonLoader loader("../test/json-test/mnist.json");
  my_tensor::SolverPtr<float> solver =
      std::make_shared<my_tensor::SgdSolver<float>>(loader.LoadSolver());
  solver->SetUp();
  solver->Solve();
  return 0;
}
