// Copyright 2024 yibotongxue

#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <vector>

#include "json-loader.h"
#include "layer-parameter.h"
#include "layer.cuh"
#include "layer/layer-utils.cuh"
#include "relu.cuh"

// TEST(TrivialTest, always_succeed) {
//   EXPECT_TRUE(true);
// }

// TEST(TrivialTest, always_fail) {
//   EXPECT_TRUE(false);
// }

#define RELU_TEST_CLASS(device)                                        \
  class Relu##device##Test : public ::testing::Test {                  \
   protected:                                                          \
    void SetUp() override {                                            \
      my_tensor::JsonLoader loader("../test/json-test/relu.json");     \
      auto&& layer_parameters = loader.Load();                         \
      data.resize(30000);                                              \
      diff.resize(30000);                                              \
      std::random_device rd;                                           \
      std::mt19937 gen(rd());                                          \
      std::uniform_real_distribution<float> dis(-3.0f, 3.0f);          \
      for (int i = 0; i < 30000; i++) {                                \
        data[i] = dis(gen);                                            \
        if (data[i] >= -0.001 && data[i] <= 0) {                       \
          data[i] = 0.001;                                             \
        }                                                              \
      }                                                                \
      for (int i = 0; i < 30000; i++) {                                \
        diff[i] = dis(gen);                                            \
      }                                                                \
      relu.reset();                                                    \
      bottom.reset();                                                  \
      top.reset();                                                     \
      relu = std::make_shared<my_tensor::Relu<>>(layer_parameters[0]); \
      bottom = std::make_shared<my_tensor::Tensor<>>(shape);           \
      top = std::make_shared<my_tensor::Tensor<>>(shape);              \
      bottom->Set##device##Data(data);                                 \
      top->Set##device##Diff(diff);                                    \
      bottom_vec.clear();                                              \
      top_vec.clear();                                                 \
      bottom_vec.push_back(bottom);                                    \
      top_vec.push_back(top);                                          \
      relu->SetUp(bottom_vec, top_vec);                                \
    }                                                                  \
    const std::vector<int> shape{10000, 3};                            \
    std::vector<float> data;                                           \
    std::vector<float> diff;                                           \
    my_tensor::LayerPtr<> relu;                                        \
    my_tensor::TensorPtr<> bottom;                                     \
    my_tensor::TensorPtr<> top;                                        \
    std::vector<my_tensor::TensorPtr<>> bottom_vec;                    \
    std::vector<my_tensor::TensorPtr<>> top_vec;                       \
  };

RELU_TEST_CLASS(CPU)
RELU_TEST_CLASS(GPU)

#define RELU_FORWARD_TEST(device)                                       \
  TEST_F(Relu##device##Test, Forward_Data) {                            \
    relu->Forward##device(bottom_vec, top_vec);                         \
    const std::vector<float> top_data(top->Get##device##Data().begin(), \
                                      top->Get##device##Data().end());  \
    for (int i = 0; i < 30000; i++) {                                   \
      if (data[i] > 0) {                                                \
        EXPECT_EQ(top_data[i], data[i]);                                \
      } else {                                                          \
        EXPECT_EQ(top_data[i], 0);                                      \
      }                                                                 \
    }                                                                   \
  }

RELU_FORWARD_TEST(CPU)
RELU_FORWARD_TEST(GPU)
BACKWARD_TEST(Relu, relu, CPU)
BACKWARD_TEST(Relu, relu, GPU)

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
