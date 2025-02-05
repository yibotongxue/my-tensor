// Copyright 2024 yibotongxue

#include "linear.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <random>
#include <ranges>  //NOLINT
#include <vector>

#include "json-loader.hpp"
#include "layer-factory.hpp"
#include "layer-parameter.hpp"
#include "layer/layer-utils.hpp"

// TEST(TrivialTest, always_succeed) {
//   EXPECT_TRUE(true);
// }

// TEST(TrivialTest, always_fail) {
//   EXPECT_TRUE(false);
// }

#define LINEAR_TEST(device)                                                    \
  class Linear##device##Test : public ::testing::Test {                        \
   protected:                                                                  \
    void SetUp() override {                                                    \
      my_tensor::JsonLoader loader("../../../../test/json-test/linear.json");  \
      auto&& layer_parameters = loader.LoadLayers();                           \
      weight_data.resize(80000);                                               \
      x_data.resize(60000);                                                    \
      bias_data.resize(400);                                                   \
      y_diff.resize(120000);                                                   \
      std::random_device rd;                                                   \
      std::mt19937 gen(rd());                                                  \
      std::uniform_real_distribution<float> dis(-10.0f, 10.0f);                \
      auto random_func = [&gen, &dis]() -> float { return dis(gen); };         \
      std::ranges::generate(weight_data, random_func);                         \
      std::ranges::generate(x_data, random_func);                              \
      std::ranges::generate(bias_data, random_func);                           \
      std::ranges::generate(y_diff, random_func);                              \
      X.reset();                                                               \
      weight.reset();                                                          \
      bias.reset();                                                            \
      Y.reset();                                                               \
      X = std::make_shared<my_tensor::Tensor<float>>(x_shape);                 \
      Y = std::make_shared<my_tensor::Tensor<float>>(y_shape);                 \
      X->Set##device##Data(x_data.data(), x_data.size());                      \
      linear.reset();                                                          \
      linear = my_tensor::CreateLayer<float>(layer_parameters[0]);             \
      auto temp = std::dynamic_pointer_cast<my_tensor::Linear<float>>(linear); \
      bottom.clear();                                                          \
      top.clear();                                                             \
      bottom.push_back(X);                                                     \
      top.push_back(Y);                                                        \
      linear->SetUp(bottom, top);                                              \
      weight = temp->GetWeight();                                              \
      bias = temp->GetBias();                                                  \
      weight->Set##device##Data(weight_data.data(), weight_data.size());       \
      bias->Set##device##Data(bias_data.data(), bias_data.size());             \
      linear->Forward##device(bottom, top);                                    \
      Y->Set##device##Diff(y_diff.data(), y_diff.size());                      \
      linear->Backward##device(top, bottom);                                   \
    }                                                                          \
    const std::vector<int> weight_shape{200, 400};                             \
    const std::vector<int> x_shape{300, 200};                                  \
    const std::vector<int> bias_shape{400};                                    \
    const std::vector<int> y_shape{300, 400};                                  \
    std::vector<float> weight_data;                                            \
    std::vector<float> x_data;                                                 \
    std::vector<float> bias_data;                                              \
    std::vector<float> y_diff;                                                 \
    my_tensor::TensorPtr<float> X;                                             \
    my_tensor::TensorPtr<float> weight;                                        \
    my_tensor::TensorPtr<float> bias;                                          \
    my_tensor::TensorPtr<float> Y;                                             \
    std::vector<my_tensor::TensorPtr<float>> bottom;                           \
    std::vector<my_tensor::TensorPtr<float>> top;                              \
    my_tensor::LayerPtr<float> linear;                                         \
    int m = 300;                                                               \
    int k = 200;                                                               \
    int n = 400;                                                               \
  };

LINEAR_TEST(CPU)
LINEAR_TEST(GPU)

#define LINEAR_FORWARD_TEST(device)                                 \
  TEST_F(Linear##device##Test, Linear_Forward##device##Test) {      \
    for (int i = 0; i < 120000; i++) {                              \
      int row = i / 400;                                            \
      int col = i % 400;                                            \
      float temp = bias_data[col];                                  \
      for (int j = 0; j < 200; j++) {                               \
        temp += x_data[row * 200 + j] * weight_data[j * 400 + col]; \
      }                                                             \
      ASSERT_NEAR(temp, Y->Get##device##Data(i), 0.01);             \
    }                                                               \
  }

LINEAR_FORWARD_TEST(CPU)
LINEAR_FORWARD_TEST(GPU)

#define LINEAR_BACKWARD_BOTTOM_TEST(device)                           \
  TEST_F(Linear##device##Test, Linear_BackwardBottom##device##Test) { \
    for (int i = 0; i < 60000; i++) {                                 \
      int row = i / 200;                                              \
      int col = i % 200;                                              \
      float expect{0.0f};                                             \
      for (int j = 0; j < 400; j++) {                                 \
        expect += weight_data[col * 400 + j] * y_diff[row * 400 + j]; \
      }                                                               \
      ASSERT_NEAR(X->Get##device##Diff(i), expect, 0.01);             \
    }                                                                 \
  }

LINEAR_BACKWARD_BOTTOM_TEST(CPU)
LINEAR_BACKWARD_BOTTOM_TEST(GPU)

#define LINEAR_BACKWARD_WEIGHT_TEST(device)                           \
  TEST_F(Linear##device##Test, Linear_BackwardWeight##device##Test) { \
    for (int i = 0; i < 80000; i++) {                                 \
      int row = i / 400;                                              \
      int col = i % 400;                                              \
      float expect{0.0f};                                             \
      for (int j = 0; j < 300; j++) {                                 \
        expect += y_diff[j * 400 + col] * x_data[j * 200 + row];      \
      }                                                               \
      ASSERT_NEAR(weight->Get##device##Diff(i), expect, 0.01);        \
    }                                                                 \
  }

LINEAR_BACKWARD_WEIGHT_TEST(CPU)
LINEAR_BACKWARD_WEIGHT_TEST(GPU)

#define LINEAR_BACKWARD_BIAS_TEST(device)                           \
  TEST_F(Linear##device##Test, Linear_BackwardBias##device##Test) { \
    for (int i = 0; i < 400; i++) {                                 \
      float expect{0.0f};                                           \
      for (int j = 0; j < 300; j++) {                               \
        expect += y_diff[j * 400 + i];                              \
      }                                                             \
      ASSERT_NEAR(bias->Get##device##Diff(i), expect, 0.01);        \
    }                                                               \
  }

LINEAR_BACKWARD_BIAS_TEST(CPU)
LINEAR_BACKWARD_BIAS_TEST(GPU)

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  ::testing::GTEST_FLAG(catch_exceptions) = false;
  return RUN_ALL_TESTS();
}
