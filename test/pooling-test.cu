// Copyright 2024 yibotongxue

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <random>
#include <ranges>
#include <vector>

#include "json-loader.h"
#include "layer.cuh"
#include "pooling.cuh"
#include "tensor.cuh"

// TEST(TrivialTest, always_succeed) {
//   EXPECT_TRUE(true);
// }

// TEST(TrivialTest, always_fail) {
//   EXPECT_TRUE(false);
// }

#define POOLING_TEST_CLASS(device)                                           \
  class Pooling##device##Test : public ::testing::Test {                     \
   protected:                                                                \
    void SetUp() override {                                                  \
      my_tensor::JsonLoader loader("../test/json-test/pooling.json");        \
      auto&& layer_parameters = loader.Load();                               \
      bottom_data.resize(59520);                                             \
      top_diff.resize(14400);                                                \
      std::random_device rd;                                                 \
      std::mt19937 gen(rd());                                                \
      std::uniform_real_distribution<float> dis(-10.0f, 10.0f);              \
      auto random_func = [&gen, &dis]() -> float { return dis(gen); };       \
      std::ranges::generate(bottom_data, random_func);                       \
      std::ranges::generate(top_diff, random_func);                          \
      bottom.reset();                                                        \
      top.reset();                                                           \
      bottom = std::make_shared<my_tensor::Tensor<>>(bottom_shape);          \
      top = std::make_shared<my_tensor::Tensor<>>(top_shape);                \
      bottom->Set##device##Data(bottom_data);                                \
      top->Set##device##Diff(top_diff);                                      \
      pooling.reset();                                                       \
      pooling = std::make_shared<my_tensor::Pooling<>>(layer_parameters[0]); \
      pooling->SetUp(bottom);                                                \
      pooling->Forward##device(bottom, top);                                 \
    }                                                                        \
    std::vector<float> bottom_data;                                          \
    std::vector<float> top_diff;                                             \
    const std::vector<int> bottom_shape = {10, 3, 31, 64};                   \
    const std::vector<int> top_shape = {10, 3, 15, 32};                      \
    my_tensor::TensorPtr<> bottom;                                           \
    my_tensor::TensorPtr<> top;                                              \
    my_tensor::LayerPtr<> pooling;                                           \
  };

POOLING_TEST_CLASS(CPU)
POOLING_TEST_CLASS(GPU)

#define POOLING_FORWARD_TOP_TEST(device)                                       \
  TEST_F(Pooling##device##Test, ForwardTop) {                                  \
    std::vector<float> actual(top->Get##device##Data().begin(),                \
                              top->Get##device##Data().end());                 \
    for (int i = 0; i < 14400; i++) {                                          \
      int t = i / 480;                                                         \
      int row = (i % 480) / 32;                                                \
      int col = i % 32;                                                        \
      int input_row = row * 2;                                                 \
      int input_col = col * 2;                                                 \
      float expect = -__FLT_MAX__;                                             \
      for (int x = 0; x < 2; x++) {                                            \
        for (int y = 0; y < 2; y++) {                                          \
          expect = std::max(                                                   \
              expect,                                                          \
              bottom_data[t * 1984 + (input_row + x) * 64 + (input_col + y)]); \
        }                                                                      \
      }                                                                        \
      ASSERT_EQ(actual[i], expect);                                            \
    }                                                                          \
  }

POOLING_FORWARD_TOP_TEST(CPU)
POOLING_FORWARD_TOP_TEST(GPU)

#define POOLING_FORWARD_MASK_TEST(device)                                   \
  TEST_F(Pooling##device##Test, ForwardMask) {                              \
    auto pooling_ptr =                                                      \
        std::dynamic_pointer_cast<my_tensor::Pooling<>>(pooling);           \
    std::vector<int> actual(                                                \
        pooling_ptr->GetMask()->Get##device##Data().begin(),                \
        pooling_ptr->GetMask()->Get##device##Data().end());                 \
    for (int i = 0; i < 14400; i++) {                                       \
      int t = i / 480;                                                      \
      int row = (i % 480) / 32;                                             \
      int col = i % 32;                                                     \
      int input_row = row * 2;                                              \
      int input_col = col * 2;                                              \
      int expect = -1;                                                      \
      float temp = -__FLT_MAX__;                                            \
      for (int x = 0; x < 2; x++) {                                         \
        for (int y = 0; y < 2; y++) {                                       \
          int temp_idx = t * 1984 + (input_row + x) * 64 + (input_col + y); \
          if (temp < bottom_data[temp_idx]) {                               \
            temp = bottom_data[temp_idx];                                   \
            expect = temp_idx;                                              \
          }                                                                 \
        }                                                                   \
      }                                                                     \
      ASSERT_EQ(actual[i], expect);                                         \
    }                                                                       \
  }

POOLING_FORWARD_MASK_TEST(CPU)
POOLING_FORWARD_MASK_TEST(GPU)

#define POOLING_BACKWARD_BOTTOM_TEST(device)                                \
  TEST_F(Pooling##device##Test, BackwardBottom) {                           \
    pooling->Backward##device(top, bottom);                                 \
    std::vector<float> actual(bottom->Get##device##Diff().begin(),          \
                              bottom->Get##device##Diff().end());           \
    std::vector<float> expect(59520, 0.0f);                                 \
    for (int t = 0; t < 30; t++) {                                          \
      for (int i = 0; i < 15; i++) {                                        \
        for (int j = 0; j < 32; j++) {                                      \
          int h_start = i * 2;                                              \
          int w_start = j * 2;                                              \
          int idx = -1;                                                     \
          float temp = -__FLT_MAX__;                                        \
          for (int x = 0; x < 2; x++) {                                     \
            for (int y = 0; y < 2; y++) {                                   \
              int temp_idx = t * 1984 + (h_start + x) * 64 + (w_start + y); \
              if (temp < bottom_data[temp_idx]) {                           \
                temp = bottom_data[temp_idx];                               \
                idx = temp_idx;                                             \
              }                                                             \
            }                                                               \
          }                                                                 \
          expect[idx] = top_diff[t * 480 + i * 32 + j];                     \
        }                                                                   \
      }                                                                     \
    }                                                                       \
    for (int i = 0; i < 59520; i++) {                                       \
      ASSERT_EQ(actual[i], expect[i]);                                      \
    }                                                                       \
  }

POOLING_BACKWARD_BOTTOM_TEST(CPU)
POOLING_BACKWARD_BOTTOM_TEST(GPU)

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
