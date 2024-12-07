// Copyright 2024 yibotongxue

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <random>
#include <ranges>  //NOLINT
#include <vector>

#include "im2col.hpp"
#include "tensor.hpp"

// TEST(TrivialTest, always_succeed) {
//   EXPECT_TRUE(true);
// }

// TEST(TrivialTest, always_fail) {
//   EXPECT_TRUE(false);
// }

#define TEST_IM2COL(device)                                                  \
  TEST(Im2col##device##Test, test_one_channel) {                             \
    const std::vector<int> im_shape = {1, 8, 6};                             \
    auto im_tensor = std::make_shared<my_tensor::Tensor<>>(im_shape);        \
    std::vector<float> data(48);                                             \
    float i = 0.0f;                                                          \
    auto func = [&i]() -> float {                                            \
      i++;                                                                   \
      return i;                                                              \
    };                                                                       \
    std::ranges::generate(data, func);                                       \
    im_tensor->Set##device##Data(data);                                      \
                                                                             \
    const std::vector<int> col_shape = {1, 48, 9};                           \
    auto col_tensor = std::make_shared<my_tensor::Tensor<>>(col_shape);      \
    ASSERT_NO_THROW(my_tensor::Im2col_##device(                              \
        1, im_tensor->Get##device##DataPtr(), 1, 6, 8, 3, 3,                 \
        col_tensor->Get##device##DataPtr()));                                \
    const std::vector<float>                                                 \
        expect{                                                              \
            0,  0,  0,  0,  1,  2,  0,  9,  10, 0,  0,  0,  1,  2,  3,       \
            9,  10, 11, 0,  0,  0,  2,  3,  4,  10, 11, 12, 0,  0,  0,       \
            3,  4,  5,  11, 12, 13, 0,  0,  0,  4,  5,  6,  12, 13, 14,      \
            0,  0,  0,  5,  6,  7,  13, 14, 15, 0,  0,  0,  6,  7,  8,       \
            14, 15, 16, 0,  0,  0,  7,  8,  0,  15, 16, 0,                   \
                                                                             \
            0,  1,  2,  0,  9,  10, 0,  17, 18, 1,  2,  3,  9,  10, 11,      \
            17, 18, 19, 2,  3,  4,  10, 11, 12, 18, 19, 20, 3,  4,  5,       \
            11, 12, 13, 19, 20, 21, 4,  5,  6,  12, 13, 14, 20, 21, 22,      \
            5,  6,  7,  13, 14, 15, 21, 22, 23, 6,  7,  8,  14, 15, 16,      \
            22, 23, 24, 7,  8,  0,  15, 16, 0,  23, 24, 0,                   \
                                                                             \
            0,  9,  10, 0,  17, 18, 0,  25, 26, 9,  10, 11, 17, 18, 19,      \
            25, 26, 27, 10, 11, 12, 18, 19, 20, 26, 27, 28, 11, 12, 13,      \
            19, 20, 21, 27, 28, 29, 12, 13, 14, 20, 21, 22, 28, 29, 30,      \
            13, 14, 15, 21, 22, 23, 29, 30, 31, 14, 15, 16, 22, 23, 24,      \
            30, 31, 32, 15, 16, 0,  23, 24, 0,  31, 32, 0,                   \
                                                                             \
            0,  17, 18, 0,  25, 26, 0,  33, 34, 17, 18, 19, 25, 26, 27,      \
            33, 34, 35, 18, 19, 20, 26, 27, 28, 34, 35, 36, 19, 20, 21,      \
            27, 28, 29, 35, 36, 37, 20, 21, 22, 28, 29, 30, 36, 37, 38,      \
            21, 22, 23, 29, 30, 31, 37, 38, 39, 22, 23, 24, 30, 31, 32,      \
            38, 39, 40, 23, 24, 0,  31, 32, 0,  39, 40, 0,                   \
                                                                             \
            0,  25, 26, 0,  33, 34, 0,  41, 42, 25, 26, 27, 33, 34, 35,      \
            41, 42, 43, 26, 27, 28, 34, 35, 36, 42, 43, 44, 27, 28, 29,      \
            35, 36, 37, 43, 44, 45, 28, 29, 30, 36, 37, 38, 44, 45, 46,      \
            29, 30, 31, 37, 38, 39, 45, 46, 47, 30, 31, 32, 38, 39, 40,      \
            46, 47, 48, 31, 32, 0,  39, 40, 0,  47, 48, 0,                   \
                                                                             \
            0,  33, 34, 0,  41, 42, 0,  0,  0,  33, 34, 35, 41, 42, 43,      \
            0,  0,  0,  34, 35, 36, 42, 43, 44, 0,  0,  0,  35, 36, 37,      \
            43, 44, 45, 0,  0,  0,  36, 37, 38, 44, 45, 46, 0,  0,  0,       \
            37, 38, 39, 45, 46, 47, 0,  0,  0,  38, 39, 40, 46, 47, 48,      \
            0,  0,  0,  39, 40, 0,  47, 48, 0,  0,  0,  0,                   \
        };                                                                   \
    const std::vector<float> actual(col_tensor->Get##device##Data().begin(), \
                                    col_tensor->Get##device##Data().end());  \
    for (int i = 0; i < 48 * 9; i++) {                                       \
      ASSERT_NEAR(expect[((i % 48) * 9 + (i / 48))], actual[i], 0.01f);      \
    }                                                                        \
  }

#define IM2COL_TEST_CLASS(device)                                      \
  class Im2col##device##Test : public ::testing::Test {                \
   protected:                                                          \
    void SetUp() override {                                            \
      im_data.resize(61440);                                           \
      col_diff.resize(552960);                                         \
      im_tensor.reset();                                               \
      im_tensor = std::make_shared<my_tensor::Tensor<>>(im_shape);     \
      col_tensor.reset();                                              \
      col_tensor = std::make_shared<my_tensor::Tensor<>>(col_shape);   \
      std::random_device rd;                                           \
      std::mt19937 gen(rd());                                          \
      std::uniform_real_distribution<float> dis(-10.0f, 10.0f);        \
      auto random_func = [&gen, &dis]() -> float { return dis(gen); }; \
      std::ranges::generate(im_data, random_func);                     \
      std::ranges::generate(col_diff, random_func);                    \
      im_tensor->Set##device##Data(im_data);                           \
      col_tensor->Set##device##Diff(col_diff);                         \
    }                                                                  \
    const std::vector<int> im_shape{10, 3, 32, 64};                    \
    const std::vector<int> col_shape{10, 27, 2048};                    \
    my_tensor::TensorPtr<> im_tensor;                                  \
    my_tensor::TensorPtr<> col_tensor;                                 \
    std::vector<float> im_data;                                        \
    std::vector<float> col_diff;                                       \
  };

IM2COL_TEST_CLASS(CPU)
IM2COL_TEST_CLASS(GPU)

#define IM2COL_TEST(device)                                                   \
  TEST_F(Im2col##device##Test, im2col) {                                      \
    my_tensor::Im2col_##device(10, im_tensor->Get##device##DataPtr(), 3, 32,  \
                               64, 3, 3, col_tensor->Get##device##DataPtr()); \
    std::vector<float> actual(col_tensor->Get##device##Data().begin(),        \
                              col_tensor->Get##device##Data().end());         \
    std::vector<float> expect(552960);                                        \
    for (int i = 0; i < 30; i++) {                                            \
      for (int j = 0; j < 9; j++) {                                           \
        int k_row = j / 3;                                                    \
        int k_col = j % 3;                                                    \
        for (int k = 0; k < 2048; k++) {                                      \
          int row = k / 64;                                                   \
          int col = k % 64;                                                   \
          int input_row = row + k_row - 1;                                    \
          int input_col = col + k_col - 1;                                    \
          expect[i * 2048 * 9 + j * 2048 + k] =                               \
              (input_row >= 0 && input_row < 32 && input_col >= 0 &&          \
               input_col < 64)                                                \
                  ? im_data[i * 2048 + input_row * 64 + input_col]            \
                  : 0;                                                        \
        }                                                                     \
      }                                                                       \
    }                                                                         \
    for (int i = 0; i < 552960; i++) {                                        \
      ASSERT_NEAR(expect[i], actual[i], 0.01);                                \
    }                                                                         \
  }

IM2COL_TEST(CPU)
IM2COL_TEST(GPU)

#define COL2IM_TEST(device)                                                   \
  TEST_F(Im2col##device##Test, col2im) {                                      \
    my_tensor::Col2im_##device(10, col_tensor->Get##device##DiffPtr(), 3, 32, \
                               64, 3, 3, im_tensor->Get##device##DiffPtr());  \
    std::vector<float> actual(im_tensor->Get##device##Diff().begin(),         \
                              im_tensor->Get##device##Diff().end());          \
    std::vector<float> expect(61440, 0.0f);                                   \
    for (int i = 0; i < 30; i++) {                                            \
      for (int j = 0; j < 9; j++) {                                           \
        int k_row = j / 3;                                                    \
        int k_col = j % 3;                                                    \
        for (int k = 0; k < 2048; k++) {                                      \
          int row = k / 64;                                                   \
          int col = k % 64;                                                   \
          int input_row = row + k_row - 1;                                    \
          int input_col = col + k_col - 1;                                    \
          if (input_row >= 0 && input_row < 32 && input_col >= 0 &&           \
              input_col < 64) {                                               \
            expect[i * 2048 + input_row * 64 + input_col] +=                  \
                col_diff[i * 2048 * 9 + j * 2048 + k];                        \
          }                                                                   \
        }                                                                     \
      }                                                                       \
    }                                                                         \
    for (int i = 0; i < 61440; i++) {                                         \
      ASSERT_NEAR(expect[i], actual[i], 0.01);                                \
    }                                                                         \
  }

COL2IM_TEST(CPU)
COL2IM_TEST(GPU)

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
