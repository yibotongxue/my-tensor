// Copyright 2024 yibotongxue

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <random>
#include <ranges>  //NOLINT
#include <vector>

#include "json-loader.hpp"
#include "layer-factory.hpp"
#include "layer.hpp"
#include "pooling.hpp"
#include "tensor.hpp"

// TEST(TrivialTest, always_succeed) {
//   EXPECT_TRUE(true);
// }

// TEST(TrivialTest, always_fail) {
//   EXPECT_TRUE(false);
// }

#define POOLING_TEST_CLASS(device)                                     \
  class Pooling##device##Test : public ::testing::Test {               \
   protected:                                                          \
    void SetUp() override {                                            \
      my_tensor::JsonLoader loader("../test/json-test/pooling.json");  \
      auto&& layer_parameters = loader.LoadLayers();                   \
      bottom_data.resize(59520);                                       \
      top_diff.resize(14400);                                          \
      std::random_device rd;                                           \
      std::mt19937 gen(rd());                                          \
      std::uniform_real_distribution<float> dis(-10.0f, 10.0f);        \
      auto random_func = [&gen, &dis]() -> float { return dis(gen); }; \
      std::ranges::generate(bottom_data, random_func);                 \
      std::ranges::generate(top_diff, random_func);                    \
      bottom.reset();                                                  \
      top.reset();                                                     \
      bottom = std::make_shared<my_tensor::Tensor<>>(bottom_shape);    \
      top = std::make_shared<my_tensor::Tensor<>>(top_shape);          \
      bottom->Set##device##Data(bottom_data.data(), bottom_data.size());                          \
      top->Set##device##Diff(top_diff.data(), top_diff.size());                                \
      bottom_vec.clear();                                              \
      top_vec.clear();                                                 \
      bottom_vec.push_back(bottom);                                    \
      top_vec.push_back(top);                                          \
      pooling.reset();                                                 \
      pooling = my_tensor::CreateLayer<>(layer_parameters[0]);         \
      pooling->SetUp(bottom_vec, top_vec);                             \
      pooling->Forward##device(bottom_vec, top_vec);                   \
    }                                                                  \
    std::vector<float> bottom_data;                                    \
    std::vector<float> top_diff;                                       \
    const std::vector<int> bottom_shape = {10, 3, 31, 64};             \
    const std::vector<int> top_shape = {10, 3, 15, 32};                \
    my_tensor::TensorPtr<> bottom;                                     \
    my_tensor::TensorPtr<> top;                                        \
    std::vector<my_tensor::TensorPtr<>> bottom_vec;                    \
    std::vector<my_tensor::TensorPtr<>> top_vec;                       \
    my_tensor::LayerPtr<> pooling;                                     \
  };

POOLING_TEST_CLASS(CPU)
POOLING_TEST_CLASS(GPU)

#define POOLING_FORWARD_TOP_TEST(device)                                       \
  TEST_F(Pooling##device##Test, ForwardTop) {                                  \
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
      ASSERT_EQ(top->Get##device##Data(i), expect);                                            \
    }                                                                          \
  }

POOLING_FORWARD_TOP_TEST(CPU)
POOLING_FORWARD_TOP_TEST(GPU)

#define POOLING_FORWARD_MASK_TEST(device)                                   \
  TEST_F(Pooling##device##Test, ForwardMask) {                              \
    auto pooling_ptr =                                                      \
        std::dynamic_pointer_cast<my_tensor::Pooling<>>(pooling);           \
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
      ASSERT_EQ(pooling_ptr->GetMask()->Get##device##Data(i), expect);                                         \
    }                                                                       \
  }

POOLING_FORWARD_MASK_TEST(CPU)
POOLING_FORWARD_MASK_TEST(GPU)

#define POOLING_BACKWARD_BOTTOM_TEST(device)                                \
  TEST_F(Pooling##device##Test, BackwardBottom) {                           \
    pooling->Backward##device(top_vec, bottom_vec);                         \
    std::vector<float> expect(59520, 0.0f);                                 \
    for (int t = 0; t < 30; t++) {                                          \
      for (int i = 0; i < 15; i++) {                                        \
        for (int j = 0; j < 32; j++) {                                      \
          int h_start = i * 2;                                              \
          int w_start = j * 2;                                              \
          int idx = -1;                                                     \
          float temp = -11;                                                 \
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
      ASSERT_EQ(bottom->Get##device##Diff(i), expect[i]);                                      \
    }                                                                       \
  }

POOLING_BACKWARD_BOTTOM_TEST(CPU)
POOLING_BACKWARD_BOTTOM_TEST(GPU)

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
