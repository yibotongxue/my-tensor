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
#include "tensor.hpp"

int main() {
  my_tensor::MyTensorContext::set_device_type(
      my_tensor::MyTensorContext::DeviceType::GPU);
  my_tensor::JsonLoader loader("../test/json-test/mnist.json");
  my_tensor::NetPtr<float> net =
      std::make_shared<my_tensor::Net<float>>(loader.LoadNet());
  net->SetUp();
  float learning_rate = loader.LoadLearningRate();
  float l2 = loader.LoadL2();
  net->SetTrain();
  for (int i = 0; i < 100; i++) {
    net->RefetchData();
    net->Forward();
    std::cout << std::format("loss = {}", net->GetOutput()) << std::endl;
    net->Backward();
    for (auto&& param : net->GetLearnableParams()) {
      thrust::transform(
          param->GetGPUData().begin(), param->GetGPUData().end(),
          param->GetGPUDiff().begin(), param->GetGPUData().begin(),
          [learning_rate, l2] __device__(float val, float grad) -> float {
            return val - grad * learning_rate - 2 * l2 * learning_rate * val;
          });
    }
    if (i % 10 == 0) {
      net->SetTest();
      net->RefetchData();
      net->Forward();
      std::cout << std::format("accuracy = {}", net->GetOutput()) << std::endl;
      net->SetTrain();
    }
  }
  net->SetTest();
  net->RefetchData();
  net->Forward();
  std::cout << std::format("accuracy = {}", net->GetOutput()) << std::endl;
  return 0;
}
