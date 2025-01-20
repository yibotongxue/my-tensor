// Copyright 2024 yibotongxue

#include "sigmoid.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <random>
#include <ranges>  //NOLINT
#include <vector>

#include "json-loader.hpp"
#include "layer-factory.hpp"
#include "layer-parameter.hpp"
#include "layer.hpp"
#include "layer/layer-utils.hpp"

#define SIGMOID_TEST_CLASS(device)                                             \
  class Sigmoid##device##Test : public ::testing::Test {                       \
   protected:                                                                  \
    void SetUp() override {                                                    \
      my_tensor::JsonLoader loader("../../../../test/json-test/sigmoid.json"); \
      auto&& layer_parameters = loader.LoadLayers();                           \
      data.resize(30000);                                                      \
      diff.resize(30000);                                                      \
      std::random_device rd;                                                   \
      std::mt19937 gen(rd());                                                  \
      std::uniform_real_distribution<float> dis(-3.0f, 3.0f);                  \
      for (int i = 0; i < 30000; i++) {                                        \
        data[i] = dis(gen);                                                    \
      }                                                                        \
      for (int i = 0; i < 30000; i++) {                                        \
        diff[i] = dis(gen);                                                    \
      }                                                                        \
      sigmoid.reset();                                                         \
      bottom.reset();                                                          \
      top.reset();                                                             \
      sigmoid = my_tensor::CreateLayer<>(layer_parameters[0]);                 \
      bottom = std::make_shared<my_tensor::Tensor<>>(shape);                   \
      top = std::make_shared<my_tensor::Tensor<>>(shape);                      \
      bottom->Set##device##Data(data.data(), data.size());                     \
      top->Set##device##Diff(diff.data(), diff.size());                        \
      bottom_vec.clear();                                                      \
      top_vec.clear();                                                         \
      bottom_vec.push_back(bottom);                                            \
      top_vec.push_back(top);                                                  \
      sigmoid->SetUp(bottom_vec, top_vec);                                     \
    }                                                                          \
    const std::vector<int> shape{10000, 3};                                    \
    std::vector<float> data;                                                   \
    std::vector<float> diff;                                                   \
    my_tensor::LayerPtr<> sigmoid;                                             \
    my_tensor::TensorPtr<> bottom;                                             \
    my_tensor::TensorPtr<> top;                                                \
    std::vector<my_tensor::TensorPtr<>> bottom_vec;                            \
    std::vector<my_tensor::TensorPtr<>> top_vec;                               \
  };

SIGMOID_TEST_CLASS(CPU)
SIGMOID_TEST_CLASS(GPU)

#define SIGMOID_FORWARD_TEST(device)                         \
  TEST_F(Sigmoid##device##Test, Forward_Data) {              \
    sigmoid->Forward##device(bottom_vec, top_vec);           \
    std::ranges::transform(data, data.begin(), [](float x) { \
      return 1.0f / (1.0f + std::exp(-x));                   \
    });                                                      \
    for (int i = 0; i < 30000; i++) {                        \
      EXPECT_NEAR(top->Get##device##Data(i), data[i], 0.01); \
    }                                                        \
  }

SIGMOID_FORWARD_TEST(CPU)
SIGMOID_FORWARD_TEST(GPU)

BACKWARD_TEST(Sigmoid, sigmoid, CPU)
BACKWARD_TEST(Sigmoid, sigmoid, GPU)

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
