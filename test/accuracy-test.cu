// Copyright 2024 yibotongxue

#include <gtest/gtest.h>
#include <thrust/tabulate.h>

#include <algorithm>
#include <memory>
#include <random>
#include <ranges>  //NOLINT
#include <vector>

#include "accuracy.hpp"
#include "json-loader.hpp"
#include "layer-factory.hpp"
#include "layer.hpp"
#include "tensor.hpp"

// TEST(TrivialTest, always_succeed) {
//   EXPECT_TRUE(true);
// }

// TEST(TrivialTest, always_fail) {
//   EXPECT_TRUE(false);
// }

#define ACCURACY_TEST(device)                                          \
  TEST(AccuracyTest, Forward##device##Test) {                          \
    my_tensor::JsonLoader loader("../test/json-test/accuracy.json");   \
    auto accuracy = my_tensor::CreateLayer<>(loader.LoadLayers()[0]);  \
    const std::vector<int> bottom_shape{1024, 10};                     \
    const std::vector<int> label_shape{1024};                          \
    const std::vector<int> top_shape{1};                               \
    auto bottom = std::make_shared<my_tensor::Tensor<>>(bottom_shape); \
    auto label = std::make_shared<my_tensor::Tensor<>>(label_shape);   \
    auto top = std::make_shared<my_tensor::Tensor<>>(top_shape);       \
    std::vector<float> label_data(1024);                               \
    std::vector<float> bottom_data(10240);                             \
    std::random_device rd;                                             \
    std::mt19937 gen(rd());                                            \
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);          \
    auto random_func = [&gen, &dis]() -> float { return dis(gen); };   \
    std::ranges::generate(bottom_data, random_func);                   \
    std::uniform_int_distribution<int> label_dis(0, 9);                \
    auto label_random = [&gen, &label_dis]() -> int {                  \
      return label_dis(gen);                                           \
    };                                                                 \
    std::ranges::generate(label_data, label_random);                   \
    accuracy->SetUp({bottom, label}, {top});                           \
    bottom->Set##device##Data(bottom_data);                            \
    label->Set##device##Data(label_data);                              \
    accuracy->Forward##device({bottom, label}, {top});                 \
    int correct = 0;                                                   \
    for (int i = 0; i < 1024; i++) {                                   \
      int predict = 0;                                                 \
      float max_val = bottom_data[i * 10];                             \
      for (int j = 1; j < 10; j++) {                                   \
        if (bottom_data[i * 10 + j] > max_val) {                       \
          predict = j;                                                 \
          max_val = bottom_data[i * 10 + j];                           \
        }                                                              \
      }                                                                \
      if (predict == label_data[i]) {                                  \
        correct++;                                                     \
      }                                                                \
    }                                                                  \
    float actual = static_cast<float>(correct) / 1024.0f;              \
    ASSERT_NEAR(top->Get##device##Data()[0], actual, 0.001);           \
  }

ACCURACY_TEST(CPU)
ACCURACY_TEST(GPU)

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
