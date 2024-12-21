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
  int batch_size = loader.LoadBatchSize();
  float learning_rate = loader.LoadLearningRate();
  float l2 = loader.LoadL2();
  my_tensor::DatasetPtr train_dataset =
      std::make_shared<my_tensor::MnistDataset>(
          "../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte");
  my_tensor::DatasetPtr test_dataset =
      std::make_shared<my_tensor::MnistDataset>(
          "../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte");
  train_dataset->LoadData();
  test_dataset->LoadData();
  my_tensor::DataLoader train_data_loader(train_dataset, batch_size);
  my_tensor::DataLoader test_data_loader(test_dataset, batch_size);
  std::vector<my_tensor::TensorPtr<>> train_image_vec, train_label_vec;
  while (train_data_loader.HasNext()) {
    auto [image, label] = train_data_loader.GetNext();
    train_image_vec.push_back(image);
    train_label_vec.push_back(label);
  }
  std::vector<my_tensor::TensorPtr<>> test_image_vec, test_label_vec;
  while (test_data_loader.HasNext()) {
    auto [image, label] = test_data_loader.GetNext();
    test_image_vec.push_back(image);
    test_label_vec.push_back(label);
  }
  int train_batch_num = train_image_vec.size();
  int test_batch_num = test_image_vec.size();
  const auto& layer_parameters = loader.LoadLayers();
  std::vector<my_tensor::LayerPtr<>> layers(layer_parameters.size(), nullptr);
  for (int i = 0; i < layer_parameters.size(); i++) {
    layers[i] = my_tensor::CreateLayer<>(layer_parameters[i]);
  }
  std::vector<std::vector<my_tensor::TensorPtr<>>> inputs;
  inputs.push_back({train_image_vec[0]});
  for (int i = 0; i < layers.size(); i++) {
    inputs.push_back({std::make_shared<my_tensor::Tensor<>>()});
  }
  for (int i = 0; i < layers.size() - 1; i++) {
    layers[i]->SetUp(inputs[i], inputs[i + 1]);
  }
  my_tensor::TensorPtr<> label = train_label_vec[0];
  layers[layers.size() - 1]->SetUp(
      {inputs[layers.size() - 1][0], train_label_vec[0]},
      {inputs[layers.size()]});
  std::vector<my_tensor::TensorPtr<>> learnable_parameters;
  for (auto&& layer : layers) {
    for (auto&& param : layer->GetLearnableParameters()) {
      learnable_parameters.push_back(param);
    }
  }
  float best_accuracy = 0;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < train_batch_num; j++) {
      inputs[0][0] = train_image_vec[j];
      for (int k = 0; k < layers.size() - 1; k++) {
        layers[k]->ForwardGPU(inputs[k], inputs[k + 1]);
      }
      layers[layers.size() - 1]->ForwardGPU(
          {inputs[layers.size() - 1][0], train_label_vec[j]},
          inputs[layers.size()]);
      std::cout << "Epoch " << i << ", Batch " << j
                << ", loss = " << inputs.back()[0]->GetCPUData()[0]
                << ", accuracy = "
                << std::dynamic_pointer_cast<my_tensor::LossWithSoftmax<>>(
                       layers.back())
                       ->GetAccuracy(train_label_vec[j])
                << std::endl;
      layers[layers.size() - 1]->BackwardGPU(
          inputs[layers.size()],
          {inputs[layers.size() - 1][0], train_label_vec[j]});
      for (int k = layers.size() - 2; k >= 0; k--) {
        layers[k]->BackwardGPU(inputs[k + 1], inputs[k]);
      }
      bool flag = true;
      for (auto&& param : learnable_parameters) {
        if (flag) {
          thrust::transform(
              param->GetGPUData().begin(), param->GetGPUData().end(),
              param->GetGPUDiff().begin(), param->GetGPUData().begin(),
              [learning_rate, l2] __device__(float val, float grad) -> float {
                return val - grad * learning_rate -
                       2 * l2 * learning_rate * val;
              });
        } else {
          thrust::transform(
              param->GetGPUData().begin(), param->GetGPUData().end(),
              param->GetGPUDiff().begin(), param->GetGPUData().begin(),
              [learning_rate] __device__(float val, float grad) -> float {
                return val - grad * learning_rate;
              });
        }
        flag = !flag;
      }
    }
    float test_accuracy = 0.0f;
    for (int j = 0; j < test_batch_num; j++) {
      inputs[0][0] = test_image_vec[j];
      for (int k = 0; k < layers.size() - 1; k++) {
        layers[k]->ForwardGPU(inputs[k], inputs[k + 1]);
      }
      layers[layers.size() - 1]->ForwardGPU(
          {inputs[layers.size() - 1][0], test_label_vec[j]},
          inputs[layers.size()]);
      test_accuracy +=
          std::dynamic_pointer_cast<my_tensor::LossWithSoftmax<>>(layers.back())
              ->GetAccuracy(test_label_vec[j]);
    }
    std::cout << "Epoch " << i
              << ": test_accuracy = " << test_accuracy / test_batch_num
              << std::endl;
    if (test_accuracy > best_accuracy) {
      best_accuracy = test_accuracy;
    }
  }
  std::cout << "best_test_accuracy = " << best_accuracy / test_batch_num
            << std::endl;
}
