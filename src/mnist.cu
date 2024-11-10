// Copyright 2024 yibotongxue

#include <iostream>
#include <memory>
#include <vector>

#include "data-loader.cuh"
#include "dataset.h"
#include "json-loader.h"
#include "layer-factory.cuh"
#include "layer-parameter.h"
#include "tensor.cuh"

int main() {
  my_tensor::JsonLoader loader("../test/json-test/mnist.json");
  int batch_size = loader.LoadBatchSize();
  float learning_rate = loader.LoadLearningRate();
  my_tensor::DatasetPtr dataset = std::make_shared<my_tensor::MnistDataset>(
      "../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte");
  dataset->LoadData();
  my_tensor::DataLoader data_loader(dataset, batch_size);
  std::vector<my_tensor::TensorPtr<>> image_vec, label_vec;
  while (data_loader.HasNext()) {
    auto&& data = data_loader.GetNext();
    image_vec.push_back(data[0]);
    label_vec.push_back(data[1]);
  }
  int batch_num = image_vec.size();
  const auto& layer_parameters = loader.LoadLayers();
  std::vector<my_tensor::LayerPtr<>> layers(layer_parameters.size(), nullptr);
  for (int i = 0; i < layer_parameters.size(); i++) {
    layers[i] = my_tensor::CreateLayer<>(layer_parameters[i]);
  }
  const std::vector<int> conv_top_shape1{batch_size, 3, 28, 28};     // 0
  const std::vector<int> pooling_top_shape1{batch_size, 3, 14, 14};  // 1
  const std::vector<int> relu_top_shape1{batch_size, 3, 14, 14};     // 2
  const std::vector<int> conv_top_shape2{batch_size, 10, 14, 14};    // 3
  const std::vector<int> pooling_top_shape2{batch_size, 10, 7, 7};   // 4
  const std::vector<int> sigmoid_top_shape{batch_size, 10, 7, 7};    // 5
  const std::vector<int> flatten_top_shape{batch_size, 490};         // 6
  const std::vector<int> linear_top_shape1{batch_size, 120};         // 7
  const std::vector<int> relu_top_shape2{batch_size, 120};           // 8
  const std::vector<int> linear_top_shape2{batch_size, 64};          // 9
  const std::vector<int> relu_top_shape3{batch_size, 64};            // 10
  const std::vector<int> linear_top_shape3{batch_size, 10};          // 11
  const std::vector<int> loss_top_shape{1};                          // 12
  const std::vector<std::vector<int>> shapes{
      conv_top_shape1,   pooling_top_shape1, relu_top_shape1,
      conv_top_shape2,   pooling_top_shape2, sigmoid_top_shape,
      flatten_top_shape, linear_top_shape1,  relu_top_shape2,
      linear_top_shape2, relu_top_shape3,    linear_top_shape3,
      loss_top_shape};
  std::vector<std::vector<my_tensor::TensorPtr<>>> inputs;
  inputs.push_back({image_vec[0]});
  for (int i = 0; i < layers.size(); i++) {
    inputs.push_back({std::make_shared<my_tensor::Tensor<>>(shapes[i])});
  }
  for (int i = 0; i < layers.size() - 1; i++) {
    layers[i]->SetUp(inputs[i], inputs[i + 1]);
  }
  my_tensor::TensorPtr<> label = label_vec[0];
  layers[layers.size() - 1]->SetUp({inputs[layers.size() - 1][0], label_vec[0]},
                                   {inputs[layers.size()]});
  std::vector<my_tensor::TensorPtr<>> learnable_parameters;
  for (auto&& layer : layers) {
    for (auto&& param : layer->GetLearnableParameters()) {
      learnable_parameters.push_back(param);
    }
  }
  for (int i = 0; i < 20; i++) {
    for (int j = 0; j < batch_num; j++) {
      inputs[0][0] = image_vec[j];
      for (int k = 0; k < layers.size() - 1; k++) {
        layers[k]->ForwardGPU(inputs[k], inputs[k + 1]);
        const auto& data = inputs[k + 1][0]->GetCPUData();
        // std::ranges::copy(data, std::ostream_iterator<float>(std::cout, "
        // "));
      }
      // std::cout << std::endl;
      layers[layers.size() - 1]->ForwardGPU(
          {inputs[layers.size() - 1][0], label_vec[j]}, inputs[layers.size()]);
      std::cout << "Epoch " << i << ", Batch " << j
                << ", loss = " << inputs.back()[0]->GetCPUData()[0]
                << ", accuracy = "
                << std::dynamic_pointer_cast<my_tensor::LossWithSoftmax<>>(
                       layers.back())
                       ->GetAccuracy(label_vec[j])
                << std::endl;
      layers[layers.size() - 1]->BackwardGPU(
          inputs[layers.size()], {inputs[layers.size() - 1][0], label_vec[j]});
      for (int k = layers.size() - 2; k >= 0; k--) {
        layers[k]->BackwardGPU(inputs[k + 1], inputs[k]);
      }
      for (auto&& param : learnable_parameters) {
        thrust::transform(
            param->GetGPUData().begin(), param->GetGPUData().end(),
            param->GetGPUDiff().begin(), param->GetGPUData().begin(),
            [learning_rate] __device__(float val, float grad) -> float {
              return val - grad * learning_rate;
            });
      }
    }
  }
}
