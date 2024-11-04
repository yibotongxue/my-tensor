// Copyright 2024 yibotongxue

#include <gtest/gtest.h>
#include <thrust/tabulate.h>

#include <algorithm>
#include <memory>
#include <random>
#include <ranges>
#include <vector>

#include "json-loader.h"
#include "layer.cuh"
#include "softmax.cuh"
#include "tensor.cuh"

// TEST(TrivialTest, always_succeed) {
//   EXPECT_TRUE(true);
// }

// TEST(TrivialTest, always_fail) {
//   EXPECT_TRUE(false);
// }

#define SOFTMAX_TEST(device)                                                 \
  TEST(SoftmaxTest, Forward##device##Test) {                                 \
    my_tensor::JsonLoader loader("../test/json-test/softmax.json");          \
    auto softmax = std::make_shared<my_tensor::Softmax<>>(loader.Load()[0]); \
    const std::vector<int> bottom_shape{1024, 10};                           \
    auto bottom = std::make_shared<my_tensor::Tensor<>>(bottom_shape);       \
    auto top = std::make_shared<my_tensor::Tensor<>>(bottom_shape);          \
    softmax->SetUp({bottom}, {top});                                         \
    std::vector<float> bottom_data(10240);                                   \
    std::random_device rd;                                                   \
    std::mt19937 gen(rd());                                                  \
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);                \
    auto random_func = [&gen, &dis]() -> float { return dis(gen); };         \
    std::ranges::generate(bottom_data, random_func);                         \
    bottom->Set##device##Data(bottom_data);                                  \
    softmax->Forward##device({bottom}, {top});                               \
    std::vector<float> actual(top->Get##device##Data().begin(),              \
                              top->Get##device##Data().end());               \
    std::vector<float> max_values(1024, -11);                                \
    for (int i = 0; i < 1024; i++) {                                         \
      for (int j = 0; j < 10; j++) {                                         \
        max_values[i] = std::max(bottom_data[i * 10 + j], max_values[i]);    \
      }                                                                      \
    }                                                                        \
    for (int i = 0; i < 1024; i++) {                                         \
      for (int j = 0; j < 10; j++) {                                         \
        bottom_data[i * 10 + j] -= max_values[i];                            \
      }                                                                      \
    }                                                                        \
    std::ranges::transform(bottom_data, bottom_data.begin(),                 \
                           [](float val) { return std::exp(val); });         \
    for (int i = 0; i < 1024; i++) {                                         \
      max_values[i] = 0;                                                     \
      for (int j = 0; j < 10; j++) {                                         \
        max_values[i] += bottom_data[i * 10 + j];                            \
      }                                                                      \
    }                                                                        \
    for (int i = 0; i < 1024; i++) {                                         \
      for (int j = 0; j < 10; j++) {                                         \
        float expect = bottom_data[i * 10 + j] / max_values[i];              \
        ASSERT_NEAR(actual[i * 10 + j], expect, 0.01);                       \
      }                                                                      \
    }                                                                        \
  }

SOFTMAX_TEST(CPU)
SOFTMAX_TEST(GPU)

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
