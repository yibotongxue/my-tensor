// Copyright 2024 yibotongxue

#include "loss-with-softmax.hpp"

#include <algorithm>
#include <memory>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "json-loader.hpp"
#include "layer-factory.hpp"
#include "layer-parameter.hpp"
#include "tensor.hpp"

// TEST(TrivialTest, always_succeed) {
//   EXPECT_TRUE(true);
// }

// TEST(TrivialTest, always_fail) {
//   EXPECT_TRUE(false);
// }

#define LOSS_WITH_SOFTMAX_TEST_CLASS(device)                                \
  class LossWithSoftmax##device##Test : public ::testing::Test {            \
   protected:                                                               \
    void SetUp() override {                                                 \
      my_tensor::JsonLoader loader(                                         \
          "../../../../test/json-test/"                                     \
          "loss-with-softmax.json");                                        \
      loss_with_softmax.reset();                                            \
      loss_with_softmax = my_tensor::CreateLayer<>(loader.LoadLayers()[0]); \
      const std::vector<int> input_shape{1024, 10};                         \
      const std::vector<int> label_shape{1024};                             \
      const std::vector<int> loss_shape{1};                                 \
      input_data.resize(10240);                                             \
      label_data.resize(1024);                                              \
      input.reset();                                                        \
      label.reset();                                                        \
      loss.reset();                                                         \
      std::random_device rd;                                                \
      std::mt19937 gen(rd());                                               \
      std::uniform_real_distribution<float> dis(-10.0f, 10.0f);             \
      auto random_func = [&gen, &dis]() -> float { return dis(gen); };      \
      std::ranges::generate(input_data, random_func);                       \
      std::uniform_int_distribution<int> label_dis(0, 9);                   \
      auto label_random = [&gen, &label_dis]() -> int {                     \
        return label_dis(gen);                                              \
      };                                                                    \
      std::ranges::generate(label_data, label_random);                      \
      input = std::make_shared<my_tensor::Tensor<>>(input_shape);           \
      label = std::make_shared<my_tensor::Tensor<>>(label_shape);           \
      loss = std::make_shared<my_tensor::Tensor<>>(loss_shape);             \
      input->Set##device##Data(input_data.data(), input_data.size());       \
      label->Set##device##Data(label_data.data(), label_data.size());       \
      loss_with_softmax->SetUp({input, label}, {loss});                     \
    }                                                                       \
    std::vector<float> input_data;                                          \
    std::vector<float> label_data;                                          \
    my_tensor::TensorPtr<> input;                                           \
    my_tensor::TensorPtr<> label;                                           \
    my_tensor::TensorPtr<> loss;                                            \
    my_tensor::LayerPtr<> loss_with_softmax;                                \
  };

LOSS_WITH_SOFTMAX_TEST_CLASS(CPU)
LOSS_WITH_SOFTMAX_TEST_CLASS(GPU)

#define LOSS_WITH_SOFTMAX_TEST_FORWARD_LOSS(device)                           \
  TEST_F(LossWithSoftmax##device##Test, ForwardLoss) {                        \
    loss_with_softmax->Forward##device({input, label}, {loss});               \
    float actual = loss->Get##device##Data(0);                                \
    std::vector<float> max_values(1024, -11);                                 \
    for (int i = 0; i < 1024; i++) {                                          \
      for (int j = 0; j < 10; j++) {                                          \
        max_values[i] = std::max(input_data[i * 10 + j], max_values[i]);      \
      }                                                                       \
    }                                                                         \
    for (int i = 0; i < 1024; i++) {                                          \
      for (int j = 0; j < 10; j++) {                                          \
        input_data[i * 10 + j] -= max_values[i];                              \
      }                                                                       \
    }                                                                         \
    std::ranges::transform(input_data, input_data.begin(),                    \
                           [](float val) { return std::exp(val); });          \
    for (int i = 0; i < 1024; i++) {                                          \
      max_values[i] = 0;                                                      \
      for (int j = 0; j < 10; j++) {                                          \
        max_values[i] += input_data[i * 10 + j];                              \
      }                                                                       \
    }                                                                         \
    float expect{0};                                                          \
    for (int i = 0; i < 1024; i++) {                                          \
      expect -= std::log(input_data[i * 10 + label_data[i]] / max_values[i]); \
    }                                                                         \
    expect /= 1024;                                                           \
    ASSERT_NEAR(actual, expect, 0.01);                                        \
  }

LOSS_WITH_SOFTMAX_TEST_FORWARD_LOSS(CPU)
LOSS_WITH_SOFTMAX_TEST_FORWARD_LOSS(GPU)

#define LOSS_WITH_SOFTMAX_TEST_BACKWARD_BOTTOM(device)                   \
  TEST_F(LossWithSoftmax##device##Test, BackwardBottom) {                \
    loss_with_softmax->Forward##device({input, label}, {loss});          \
    loss_with_softmax->Backward##device({loss}, {input, label});         \
    std::vector<float> max_values(1024, -11);                            \
    for (int i = 0; i < 1024; i++) {                                     \
      for (int j = 0; j < 10; j++) {                                     \
        max_values[i] = std::max(input_data[i * 10 + j], max_values[i]); \
      }                                                                  \
    }                                                                    \
    for (int i = 0; i < 1024; i++) {                                     \
      for (int j = 0; j < 10; j++) {                                     \
        input_data[i * 10 + j] -= max_values[i];                         \
      }                                                                  \
    }                                                                    \
    std::ranges::transform(input_data, input_data.begin(),               \
                           [](float val) { return std::exp(val); });     \
    for (int i = 0; i < 1024; i++) {                                     \
      max_values[i] = 0;                                                 \
      for (int j = 0; j < 10; j++) {                                     \
        max_values[i] += input_data[i * 10 + j];                         \
      }                                                                  \
    }                                                                    \
    for (int i = 0; i < 1024; i++) {                                     \
      for (int j = 0; j < 10; j++) {                                     \
        float expect = input_data[i * 10 + j] / max_values[i];           \
        if (label_data[i] == j) {                                        \
          expect -= 1;                                                   \
        }                                                                \
        expect /= 1024;                                                  \
        ASSERT_NEAR(input->Get##device##Diff(i * 10 + j), expect, 0.01); \
      }                                                                  \
    }                                                                    \
  }

LOSS_WITH_SOFTMAX_TEST_BACKWARD_BOTTOM(CPU)
LOSS_WITH_SOFTMAX_TEST_BACKWARD_BOTTOM(GPU)

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
