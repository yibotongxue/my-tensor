// Copyright 2024 yibotongxue

#include "blas.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <random>
#include <ranges>  //NOLINT
#include <vector>

#include "tensor.hpp"

// TEST(TrivialTest, always_succeed) {
//   EXPECT_TRUE(true);
// }

// TEST(TrivialTest, always_fail) {
//   EXPECT_TRUE(false);
// }

class BlasMatmulTest : public ::testing::Test {
 protected:
  void SetUp() override {
    lhs_data.resize(64000);
    rhs_data.resize(8192);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    auto random_func = [&gen, &dis]() -> float { return dis(gen); };
    std::ranges::generate(lhs_data, random_func);
    std::ranges::generate(rhs_data, random_func);
    lhs = std::make_shared<my_tensor::Tensor<>>(left_shape);
    rhs = std::make_shared<my_tensor::Tensor<>>(right_shape);
    result_expect.resize(32000);
    result = std::make_shared<my_tensor::Tensor<>>(result_shape);
  }

  std::vector<float> lhs_data;
  std::vector<float> rhs_data;
  std::vector<float> result_expect;
  my_tensor::TensorPtr<> lhs;
  my_tensor::TensorPtr<> rhs;
  my_tensor::TensorPtr<> result;
  const std::vector<int> left_shape{500, 128};
  const std::vector<int> right_shape{128, 64};
  const std::vector<int> result_shape{500, 64};
};

#define BLAS_MATMUL_TEST(device, device_low)                                  \
  TEST_F(BlasMatmulTest, Blas_Matmul##device##Test) {                         \
    lhs->Set##device##Data(lhs_data.data(), lhs_data.size());                 \
    rhs->Set##device##Data(rhs_data.data(), rhs_data.size());                 \
    my_tensor::matmul_##device_low(                                           \
        lhs->Get##device##DataPtr(), rhs->Get##device##DataPtr(),             \
        result->Get##device##DataPtr(), 500, 128, 64);                        \
    for (int i = 0; i < 32000; i++) {                                         \
      int row = i / 64;                                                       \
      int col = i % 64;                                                       \
      for (int k = 0; k < 128; k++) {                                         \
        result_expect[i] += lhs_data[row * 128 + k] * rhs_data[k * 64 + col]; \
      }                                                                       \
    }                                                                         \
    for (int i = 0; i < 32000; i++) {                                         \
      ASSERT_NEAR(result_expect[i], result->Get##device##Data(i), 0.01);      \
    }                                                                         \
  }

BLAS_MATMUL_TEST(GPU, gpu)
BLAS_MATMUL_TEST(CPU, cpu)

#define BLAS_MATMUL_T_TEST(device, device_low)                                 \
  TEST_F(BlasMatmulTest, Blas_MatmulTranspose##device##Test) {                 \
    rhs->Reshape({64, 128});                                                   \
    lhs->Set##device##Data(lhs_data.data(), lhs_data.size());                  \
    rhs->Set##device##Data(rhs_data.data(), rhs_data.size());                  \
    my_tensor::matmul_transpose_##device_low(                                  \
        lhs->Get##device##DataPtr(), rhs->Get##device##DataPtr(),              \
        result->Get##device##DataPtr(), 500, 128, 64);                         \
    for (int i = 0; i < 32000; i++) {                                          \
      int row = i / 64;                                                        \
      int col = i % 64;                                                        \
      for (int k = 0; k < 128; k++) {                                          \
        result_expect[i] += lhs_data[row * 128 + k] * rhs_data[col * 128 + k]; \
      }                                                                        \
    }                                                                          \
    for (int i = 0; i < 32000; i++) {                                          \
      ASSERT_NEAR(result_expect[i], result->Get##device##Data(i), 0.01);       \
    }                                                                          \
  }

BLAS_MATMUL_T_TEST(GPU, gpu)
BLAS_MATMUL_T_TEST(CPU, cpu)

#define BLAS_T_MATMUL_TEST(device, device_low)                                \
  TEST_F(BlasMatmulTest, Blas_TransposeMatmul##device##Test) {                \
    lhs->Reshape({128, 500});                                                 \
    lhs->Set##device##Data(lhs_data.data(), lhs_data.size());                 \
    rhs->Set##device##Data(rhs_data.data(), rhs_data.size());                 \
    my_tensor::transpose_matmul_##device_low(                                 \
        lhs->Get##device##DataPtr(), rhs->Get##device##DataPtr(),             \
        result->Get##device##DataPtr(), 500, 128, 64);                        \
    for (int i = 0; i < 32000; i++) {                                         \
      int row = i / 64;                                                       \
      int col = i % 64;                                                       \
      for (int k = 0; k < 128; k++) {                                         \
        result_expect[i] += lhs_data[k * 500 + row] * rhs_data[k * 64 + col]; \
      }                                                                       \
    }                                                                         \
    for (int i = 0; i < 32000; i++) {                                         \
      ASSERT_NEAR(result_expect[i], result->Get##device##Data(i), 0.01);      \
    }                                                                         \
  }

BLAS_T_MATMUL_TEST(GPU, gpu)
BLAS_T_MATMUL_TEST(CPU, cpu)

#define BLAS_T_MATMUL_T_TEST(device, device_low)                               \
  TEST_F(BlasMatmulTest, Blas_TransposeMatmulTranspose##device##Test) {        \
    lhs->Reshape({128, 500});                                                  \
    rhs->Reshape({64, 128});                                                   \
    lhs->Set##device##Data(lhs_data.data(), lhs_data.size());                  \
    rhs->Set##device##Data(rhs_data.data(), rhs_data.size());                  \
    my_tensor::transpose_matmul_transpose_##device_low(                        \
        lhs->Get##device##DataPtr(), rhs->Get##device##DataPtr(),              \
        result->Get##device##DataPtr(), 500, 128, 64);                         \
    for (int i = 0; i < 32000; i++) {                                          \
      int row = i / 64;                                                        \
      int col = i % 64;                                                        \
      for (int k = 0; k < 128; k++) {                                          \
        result_expect[i] += lhs_data[k * 500 + row] * rhs_data[col * 128 + k]; \
      }                                                                        \
    }                                                                          \
    for (int i = 0; i < 32000; i++) {                                          \
      ASSERT_NEAR(result_expect[i], result->Get##device##Data(i), 0.01);       \
    }                                                                          \
  }

BLAS_T_MATMUL_T_TEST(GPU, gpu)
BLAS_T_MATMUL_T_TEST(CPU, cpu)

class BlasMatmulBatchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    lhs_data.resize(640000);
    rhs_data.resize(81920);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    auto random_func = [&gen, &dis]() -> float { return dis(gen); };
    std::ranges::generate(lhs_data, random_func);
    std::ranges::generate(rhs_data, random_func);
    lhs = std::make_shared<my_tensor::Tensor<>>(left_shape);
    rhs = std::make_shared<my_tensor::Tensor<>>(right_shape);
    result_expect.resize(320000);
    result = std::make_shared<my_tensor::Tensor<>>(result_shape);
  }

  std::vector<float> lhs_data;
  std::vector<float> rhs_data;
  std::vector<float> result_expect;
  my_tensor::TensorPtr<> lhs;
  my_tensor::TensorPtr<> rhs;
  my_tensor::TensorPtr<> result;
  const std::vector<int> left_shape{10, 500, 128};
  const std::vector<int> right_shape{10, 128, 64};
  const std::vector<int> result_shape{10, 500, 64};
};

#define BLAS_MATMUL_BATCH_TEST(device, device_low)                       \
  TEST_F(BlasMatmulBatchTest, Blas_MatmulBatch##device##Test) {          \
    lhs->Set##device##Data(lhs_data.data(), lhs_data.size());            \
    rhs->Set##device##Data(rhs_data.data(), rhs_data.size());            \
    my_tensor::matmul_##device_low(                                      \
        lhs->Get##device##DataPtr(), rhs->Get##device##DataPtr(),        \
        result->Get##device##DataPtr(), 500, 128, 64, 10);               \
    for (int t = 0; t < 10; t++) {                                       \
      for (int i = 0; i < 32000; i++) {                                  \
        int row = i / 64;                                                \
        int col = i % 64;                                                \
        for (int k = 0; k < 128; k++) {                                  \
          result_expect[t * 32000 + i] +=                                \
              lhs_data[t * 64000 + row * 128 + k] *                      \
              rhs_data[t * 8192 + k * 64 + col];                         \
        }                                                                \
      }                                                                  \
    }                                                                    \
    for (int i = 0; i < 320000; i++) {                                   \
      ASSERT_NEAR(result_expect[i], result->Get##device##Data(i), 0.01); \
    }                                                                    \
  }

BLAS_MATMUL_BATCH_TEST(GPU, gpu)
BLAS_MATMUL_BATCH_TEST(CPU, cpu)

#define BLAS_MATMUL_T_BATCH_TEST(device, device_low)                     \
  TEST_F(BlasMatmulBatchTest, Blas_MatmulTransposeBatch##device##Test) { \
    rhs->Reshape({10, 64, 128});                                         \
    lhs->Set##device##Data(lhs_data.data(), lhs_data.size());            \
    rhs->Set##device##Data(rhs_data.data(), rhs_data.size());            \
    my_tensor::matmul_transpose_##device_low(                            \
        lhs->Get##device##DataPtr(), rhs->Get##device##DataPtr(),        \
        result->Get##device##DataPtr(), 500, 128, 64, 10);               \
    for (int t = 0; t < 10; t++) {                                       \
      for (int i = 0; i < 32000; i++) {                                  \
        int row = i / 64;                                                \
        int col = i % 64;                                                \
        for (int k = 0; k < 128; k++) {                                  \
          result_expect[t * 32000 + i] +=                                \
              lhs_data[t * 64000 + row * 128 + k] *                      \
              rhs_data[t * 8192 + col * 128 + k];                        \
        }                                                                \
      }                                                                  \
    }                                                                    \
    for (int i = 0; i < 320000; i++) {                                   \
      ASSERT_NEAR(result_expect[i], result->Get##device##Data(i), 0.01); \
    }                                                                    \
  }

BLAS_MATMUL_T_BATCH_TEST(GPU, gpu)
BLAS_MATMUL_T_BATCH_TEST(CPU, cpu)

#define BLAS_T_MATMUL_BATCH_TEST(device, device_low)                     \
  TEST_F(BlasMatmulBatchTest, Blas_TransposeMatmul##device##Test) {      \
    lhs->Reshape({10, 128, 500});                                        \
    lhs->Set##device##Data(lhs_data.data(), lhs_data.size());            \
    rhs->Set##device##Data(rhs_data.data(), rhs_data.size());            \
    my_tensor::transpose_matmul_##device_low(                            \
        lhs->Get##device##DataPtr(), rhs->Get##device##DataPtr(),        \
        result->Get##device##DataPtr(), 500, 128, 64, 10);               \
    for (int t = 0; t < 10; t++) {                                       \
      for (int i = 0; i < 32000; i++) {                                  \
        int row = i / 64;                                                \
        int col = i % 64;                                                \
        for (int k = 0; k < 128; k++) {                                  \
          result_expect[t * 32000 + i] +=                                \
              lhs_data[t * 64000 + k * 500 + row] *                      \
              rhs_data[t * 8192 + k * 64 + col];                         \
        }                                                                \
      }                                                                  \
    }                                                                    \
    for (int i = 0; i < 320000; i++) {                                   \
      ASSERT_NEAR(result_expect[i], result->Get##device##Data(i), 0.01); \
    }                                                                    \
  }

BLAS_T_MATMUL_BATCH_TEST(GPU, gpu)
BLAS_T_MATMUL_BATCH_TEST(CPU, cpu)

#define BLAS_T_MATMUL_T_BATCH_TEST(device, device_low)                       \
  TEST_F(BlasMatmulBatchTest, Blas_TransposeMatmulTranspose##device##Test) { \
    lhs->Reshape({10, 128, 500});                                            \
    rhs->Reshape({10, 64, 128});                                             \
    lhs->Set##device##Data(lhs_data.data(), lhs_data.size());                \
    rhs->Set##device##Data(rhs_data.data(), rhs_data.size());                \
    my_tensor::transpose_matmul_transpose_##device_low(                      \
        lhs->Get##device##DataPtr(), rhs->Get##device##DataPtr(),            \
        result->Get##device##DataPtr(), 500, 128, 64, 10);                   \
    for (int t = 0; t < 10; t++) {                                           \
      for (int i = 0; i < 32000; i++) {                                      \
        int row = i / 64;                                                    \
        int col = i % 64;                                                    \
        for (int k = 0; k < 128; k++) {                                      \
          result_expect[t * 32000 + i] +=                                    \
              lhs_data[t * 64000 + k * 500 + row] *                          \
              rhs_data[t * 8192 + col * 128 + k];                            \
        }                                                                    \
      }                                                                      \
    }                                                                        \
    for (int i = 0; i < 320000; i++) {                                       \
      ASSERT_NEAR(result_expect[i], result->Get##device##Data(i), 0.01);     \
    }                                                                        \
  }

BLAS_T_MATMUL_T_BATCH_TEST(GPU, gpu)
BLAS_T_MATMUL_T_BATCH_TEST(CPU, cpu)

class BlasMatmulBatchOneBroadcastTest : public ::testing::Test {
 protected:
  void SetUp() override {
    lhs_data.resize(64000);
    rhs_data.resize(81920);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    auto random_func = [&gen, &dis]() -> float { return dis(gen); };
    std::ranges::generate(lhs_data, random_func);
    std::ranges::generate(rhs_data, random_func);
    lhs = std::make_shared<my_tensor::Tensor<>>(left_shape);
    rhs = std::make_shared<my_tensor::Tensor<>>(right_shape);
    result_expect.resize(320000);
    result = std::make_shared<my_tensor::Tensor<>>(result_shape);
  }

  std::vector<float> lhs_data;
  std::vector<float> rhs_data;
  std::vector<float> result_expect;
  my_tensor::TensorPtr<> lhs;
  my_tensor::TensorPtr<> rhs;
  my_tensor::TensorPtr<> result;
  const std::vector<int> left_shape{500, 128};
  const std::vector<int> right_shape{10, 128, 64};
  const std::vector<int> result_shape{10, 500, 64};
};

#define BLAS_MATMUL_BATCH_ONE_BROADCAST_TEST(device, device_low)           \
  TEST_F(BlasMatmulBatchOneBroadcastTest, Blas_Matmul##device##Test) {     \
    lhs->Set##device##Data(lhs_data.data(), lhs_data.size());              \
    rhs->Set##device##Data(rhs_data.data(), rhs_data.size());              \
    my_tensor::matmul_##device_low(                                        \
        lhs->Get##device##DataPtr(), rhs->Get##device##DataPtr(),          \
        result->Get##device##DataPtr(), 500, 128, 64, 10, 1);              \
    for (int t = 0; t < 10; t++) {                                         \
      for (int i = 0; i < 32000; i++) {                                    \
        int row = i / 64;                                                  \
        int col = i % 64;                                                  \
        for (int k = 0; k < 128; k++) {                                    \
          result_expect[t * 32000 + i] +=                                  \
              lhs_data[row * 128 + k] * rhs_data[t * 8192 + k * 64 + col]; \
        }                                                                  \
      }                                                                    \
    }                                                                      \
    for (int i = 0; i < 320000; i++) {                                     \
      ASSERT_NEAR(result_expect[i], result->Get##device##Data(i), 0.01);   \
    }                                                                      \
  }

BLAS_MATMUL_BATCH_ONE_BROADCAST_TEST(GPU, gpu)
BLAS_MATMUL_BATCH_ONE_BROADCAST_TEST(CPU, cpu)

#define BLAS_MATMUL_T_BATCH_ONE_BROADCAST_TEST(device, device_low)          \
  TEST_F(BlasMatmulBatchOneBroadcastTest,                                   \
         Blas_MatmulTranspose##device##Test) {                              \
    rhs->Reshape({10, 64, 128});                                            \
    lhs->Set##device##Data(lhs_data.data(), lhs_data.size());               \
    rhs->Set##device##Data(rhs_data.data(), rhs_data.size());               \
    my_tensor::matmul_transpose_##device_low(                               \
        lhs->Get##device##DataPtr(), rhs->Get##device##DataPtr(),           \
        result->Get##device##DataPtr(), 500, 128, 64, 10, 1);               \
    for (int t = 0; t < 10; t++) {                                          \
      for (int i = 0; i < 32000; i++) {                                     \
        int row = i / 64;                                                   \
        int col = i % 64;                                                   \
        for (int k = 0; k < 128; k++) {                                     \
          result_expect[t * 32000 + i] +=                                   \
              lhs_data[row * 128 + k] * rhs_data[t * 8192 + col * 128 + k]; \
        }                                                                   \
      }                                                                     \
    }                                                                       \
    for (int i = 0; i < 320000; i++) {                                      \
      ASSERT_NEAR(result_expect[i], result->Get##device##Data(i), 0.01);    \
    }                                                                       \
  }

BLAS_MATMUL_T_BATCH_ONE_BROADCAST_TEST(GPU, gpu)
BLAS_MATMUL_T_BATCH_ONE_BROADCAST_TEST(CPU, cpu)

#define BLAS_T_MATMUL_BATCH_ONE_BROADCAST_TEST(device, device_low)         \
  TEST_F(BlasMatmulBatchOneBroadcastTest,                                  \
         Blas_TransposeMatmul##device##Test) {                             \
    lhs->Reshape({128, 500});                                              \
    lhs->Set##device##Data(lhs_data.data(), lhs_data.size());              \
    rhs->Set##device##Data(rhs_data.data(), rhs_data.size());              \
    my_tensor::transpose_matmul_##device_low(                              \
        lhs->Get##device##DataPtr(), rhs->Get##device##DataPtr(),          \
        result->Get##device##DataPtr(), 500, 128, 64, 10, 1);              \
    for (int t = 0; t < 10; t++) {                                         \
      for (int i = 0; i < 32000; i++) {                                    \
        int row = i / 64;                                                  \
        int col = i % 64;                                                  \
        for (int k = 0; k < 128; k++) {                                    \
          result_expect[t * 32000 + i] +=                                  \
              lhs_data[k * 500 + row] * rhs_data[t * 8192 + k * 64 + col]; \
        }                                                                  \
      }                                                                    \
    }                                                                      \
    for (int i = 0; i < 320000; i++) {                                     \
      ASSERT_NEAR(result_expect[i], result->Get##device##Data(i), 0.01);   \
    }                                                                      \
  }

BLAS_T_MATMUL_BATCH_ONE_BROADCAST_TEST(GPU, gpu)
BLAS_T_MATMUL_BATCH_ONE_BROADCAST_TEST(CPU, cpu)

#define BLAS_T_MATMUL_T_BATCH_ONE_BROADCAST_TEST(device, device_low)        \
  TEST_F(BlasMatmulBatchOneBroadcastTest,                                   \
         Blas_TransposeMatmulTranspose##device##Test) {                     \
    lhs->Reshape({128, 500});                                               \
    rhs->Reshape({10, 64, 128});                                            \
    lhs->Set##device##Data(lhs_data.data(), lhs_data.size());               \
    rhs->Set##device##Data(rhs_data.data(), rhs_data.size());               \
    my_tensor::transpose_matmul_transpose_##device_low(                     \
        lhs->Get##device##DataPtr(), rhs->Get##device##DataPtr(),           \
        result->Get##device##DataPtr(), 500, 128, 64, 10, 1);               \
    for (int t = 0; t < 10; t++) {                                          \
      for (int i = 0; i < 32000; i++) {                                     \
        int row = i / 64;                                                   \
        int col = i % 64;                                                   \
        for (int k = 0; k < 128; k++) {                                     \
          result_expect[t * 32000 + i] +=                                   \
              lhs_data[k * 500 + row] * rhs_data[t * 8192 + col * 128 + k]; \
        }                                                                   \
      }                                                                     \
    }                                                                       \
    for (int i = 0; i < 320000; i++) {                                      \
      ASSERT_NEAR(result_expect[i], result->Get##device##Data(i), 0.01);    \
    }                                                                       \
  }

BLAS_T_MATMUL_T_BATCH_ONE_BROADCAST_TEST(GPU, gpu)
BLAS_T_MATMUL_T_BATCH_ONE_BROADCAST_TEST(CPU, cpu)

class BlasMatmulBatchTwoBroadcastTest : public ::testing::Test {
 protected:
  void SetUp() override {
    lhs_data.resize(640000);
    rhs_data.resize(8192);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    auto random_func = [&gen, &dis]() -> float { return dis(gen); };
    std::ranges::generate(lhs_data, random_func);
    std::ranges::generate(rhs_data, random_func);
    lhs = std::make_shared<my_tensor::Tensor<>>(left_shape);
    rhs = std::make_shared<my_tensor::Tensor<>>(right_shape);
    result_expect.resize(320000);
    result = std::make_shared<my_tensor::Tensor<>>(result_shape);
  }

  std::vector<float> lhs_data;
  std::vector<float> rhs_data;
  std::vector<float> result_expect;
  my_tensor::TensorPtr<> lhs;
  my_tensor::TensorPtr<> rhs;
  my_tensor::TensorPtr<> result;
  const std::vector<int> left_shape{10, 500, 128};
  const std::vector<int> right_shape{128, 64};
  const std::vector<int> result_shape{10, 500, 64};
};

#define BLAS_MATMUL_BATCH_TWO_BROADCAST_TEST(device, device_low)            \
  TEST_F(BlasMatmulBatchTwoBroadcastTest, Blas_Matmul##device##Test) {      \
    lhs->Set##device##Data(lhs_data.data(), lhs_data.size());               \
    rhs->Set##device##Data(rhs_data.data(), rhs_data.size());               \
    my_tensor::matmul_##device_low(                                         \
        lhs->Get##device##DataPtr(), rhs->Get##device##DataPtr(),           \
        result->Get##device##DataPtr(), 500, 128, 64, 10, 2);               \
    for (int t = 0; t < 10; t++) {                                          \
      for (int i = 0; i < 32000; i++) {                                     \
        int row = i / 64;                                                   \
        int col = i % 64;                                                   \
        for (int k = 0; k < 128; k++) {                                     \
          result_expect[t * 32000 + i] +=                                   \
              lhs_data[t * 64000 + row * 128 + k] * rhs_data[k * 64 + col]; \
        }                                                                   \
      }                                                                     \
    }                                                                       \
    for (int i = 0; i < 320000; i++) {                                      \
      ASSERT_NEAR(result_expect[i], result->Get##device##Data(i), 0.01);    \
    }                                                                       \
  }

BLAS_MATMUL_BATCH_TWO_BROADCAST_TEST(GPU, gpu)
BLAS_MATMUL_BATCH_TWO_BROADCAST_TEST(CPU, cpu)

#define BLAS_MATMUL_T_BATCH_TWO_BROADCAST_TEST(device, device_low)           \
  TEST_F(BlasMatmulBatchTwoBroadcastTest,                                    \
         Blas_MatmulTranspose##device##Test) {                               \
    rhs->Reshape({64, 128});                                                 \
    lhs->Set##device##Data(lhs_data.data(), lhs_data.size());                \
    rhs->Set##device##Data(rhs_data.data(), rhs_data.size());                \
    my_tensor::matmul_transpose_##device_low(                                \
        lhs->Get##device##DataPtr(), rhs->Get##device##DataPtr(),            \
        result->Get##device##DataPtr(), 500, 128, 64, 10, 2);                \
    for (int t = 0; t < 10; t++) {                                           \
      for (int i = 0; i < 32000; i++) {                                      \
        int row = i / 64;                                                    \
        int col = i % 64;                                                    \
        for (int k = 0; k < 128; k++) {                                      \
          result_expect[t * 32000 + i] +=                                    \
              lhs_data[t * 64000 + row * 128 + k] * rhs_data[col * 128 + k]; \
        }                                                                    \
      }                                                                      \
    }                                                                        \
    for (int i = 0; i < 320000; i++) {                                       \
      ASSERT_NEAR(result_expect[i], result->Get##device##Data(i), 0.01);     \
    }                                                                        \
  }

BLAS_MATMUL_T_BATCH_TWO_BROADCAST_TEST(GPU, gpu)
BLAS_MATMUL_T_BATCH_TWO_BROADCAST_TEST(CPU, cpu)

#define BLAS_T_MATMUL_BATCH_TWO_BROADCAST_TEST(device, device_low)          \
  TEST_F(BlasMatmulBatchTwoBroadcastTest,                                   \
         Blas_TransposeMatmul##device##Test) {                              \
    lhs->Reshape({10, 128, 500});                                           \
    lhs->Set##device##Data(lhs_data.data(), lhs_data.size());               \
    rhs->Set##device##Data(rhs_data.data(), rhs_data.size());               \
    my_tensor::transpose_matmul_##device_low(                               \
        lhs->Get##device##DataPtr(), rhs->Get##device##DataPtr(),           \
        result->Get##device##DataPtr(), 500, 128, 64, 10, 2);               \
    for (int t = 0; t < 10; t++) {                                          \
      for (int i = 0; i < 32000; i++) {                                     \
        int row = i / 64;                                                   \
        int col = i % 64;                                                   \
        for (int k = 0; k < 128; k++) {                                     \
          result_expect[t * 32000 + i] +=                                   \
              lhs_data[t * 64000 + k * 500 + row] * rhs_data[k * 64 + col]; \
        }                                                                   \
      }                                                                     \
    }                                                                       \
    for (int i = 0; i < 320000; i++) {                                      \
      ASSERT_NEAR(result_expect[i], result->Get##device##Data(i), 0.01);    \
    }                                                                       \
  }

BLAS_T_MATMUL_BATCH_TWO_BROADCAST_TEST(GPU, gpu)
BLAS_T_MATMUL_BATCH_TWO_BROADCAST_TEST(CPU, cpu)

#define BLAS_T_MATMUL_T_BATCH_TWO_BROADCAST_TEST(device, device_low)         \
  TEST_F(BlasMatmulBatchTwoBroadcastTest,                                    \
         Blas_TransposeMatmulTranspose##device##Test) {                      \
    lhs->Reshape({10, 128, 500});                                            \
    rhs->Reshape({64, 128});                                                 \
    lhs->Set##device##Data(lhs_data.data(), lhs_data.size());                \
    rhs->Set##device##Data(rhs_data.data(), rhs_data.size());                \
    my_tensor::transpose_matmul_transpose_##device_low(                      \
        lhs->Get##device##DataPtr(), rhs->Get##device##DataPtr(),            \
        result->Get##device##DataPtr(), 500, 128, 64, 10, 2);                \
    for (int t = 0; t < 10; t++) {                                           \
      for (int i = 0; i < 32000; i++) {                                      \
        int row = i / 64;                                                    \
        int col = i % 64;                                                    \
        for (int k = 0; k < 128; k++) {                                      \
          result_expect[t * 32000 + i] +=                                    \
              lhs_data[t * 64000 + k * 500 + row] * rhs_data[col * 128 + k]; \
        }                                                                    \
      }                                                                      \
    }                                                                        \
    for (int i = 0; i < 320000; i++) {                                       \
      ASSERT_NEAR(result_expect[i], result->Get##device##Data(i), 0.01);     \
    }                                                                        \
  }

BLAS_T_MATMUL_T_BATCH_TWO_BROADCAST_TEST(GPU, gpu)
BLAS_T_MATMUL_T_BATCH_TWO_BROADCAST_TEST(CPU, cpu)

class BlasAddVecTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tensor_data.resize(2000000);
    vec_data.resize(1000);
    result_expect.resize(2000000);
    tensor = std::make_shared<my_tensor::Tensor<>>(tensor_shape);
    vec = std::make_shared<my_tensor::Tensor<>>(vec_shape);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    auto random_func = [&gen, &dis]() -> float { return dis(gen); };
    std::ranges::generate(tensor_data, random_func);
    std::ranges::generate(vec_data, random_func);
  }

  std::vector<float> tensor_data;
  std::vector<float> vec_data;
  std::vector<float> result_expect;
  my_tensor::TensorPtr<> tensor;
  my_tensor::TensorPtr<> vec;
  const std::vector<int> tensor_shape{10, 1000, 200};
  const std::vector<int> vec_shape{1000, 1};
};

#define BLAS_ADD_ROW_VEC_TEST(data_diff)                                    \
  TEST_F(BlasAddVecTest, Blas_AddRowVec##data_diff##Test) {                 \
    for (int i = 0; i < 2000000; i++) {                                     \
      result_expect[i] = tensor_data[i] + vec_data[(i % 200000) / 200];     \
    }                                                                       \
    tensor->SetGPU##data_diff(tensor_data.data(), tensor_data.size());      \
    vec->SetGPU##data_diff(vec_data.data(), vec_data.size());               \
    my_tensor::add_row_vector_gpu(tensor->GetGPU##data_diff##Ptr(),         \
                                  vec->GetGPU##data_diff##Ptr(), 1000, 200, \
                                  10);                                      \
    for (int i = 0; i < 2000000; i++) {                                     \
      ASSERT_NEAR(result_expect[i], tensor->GetCPU##data_diff(i), 0.01);    \
    }                                                                       \
  }

BLAS_ADD_ROW_VEC_TEST(Data)
BLAS_ADD_ROW_VEC_TEST(Diff)

#define BLAS_ADD_COL_VEC_TEST(data_diff)                                    \
  TEST_F(BlasAddVecTest, Blas_AddColVec##data_diff##Test) {                 \
    tensor->Reshape({10, 200, 1000});                                       \
    for (int i = 0; i < 2000000; i++) {                                     \
      result_expect[i] = tensor_data[i] + vec_data[(i % 200000) % 1000];    \
    }                                                                       \
    tensor->SetGPU##data_diff(tensor_data.data(), tensor_data.size());      \
    vec->SetGPU##data_diff(vec_data.data(), vec_data.size());               \
    my_tensor::add_col_vector_gpu(tensor->GetGPU##data_diff##Ptr(),         \
                                  vec->GetGPU##data_diff##Ptr(), 200, 1000, \
                                  10);                                      \
    for (int i = 0; i < 2000000; i++) {                                     \
      ASSERT_NEAR(result_expect[i], tensor->GetCPU##data_diff(i), 0.01);    \
    }                                                                       \
  }

BLAS_ADD_COL_VEC_TEST(Data)
BLAS_ADD_COL_VEC_TEST(Diff)

class BlasSumTest : public ::testing::Test {
 protected:
  void SetUp() override {
    data.resize(200000);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    auto random_func = [&gen, &dis]() -> float { return dis(gen); };
    std::ranges::generate(data, random_func);
    tensor = std::make_shared<my_tensor::Tensor<>>(shape);
  }
  const std::vector<int> shape{10, 100, 200};
  std::vector<float> data;
  my_tensor::TensorPtr<> tensor;
};

#define BLAS_SUM_TENSOR_SUM_TEST(data_diff)                                  \
  TEST_F(BlasSumTest, Blas_SumTensorSum##data_diff##Test) {                  \
    tensor->SetGPU##data_diff(data.data(), data.size());                     \
    float sum_actual =                                                       \
        my_tensor::tensor_sum_gpu(tensor->GetGPU##data_diff##Ptr(), 200000); \
    float sum_expect =                                                       \
        std::accumulate(data.begin(), data.end(), 0.0f, std::plus<float>()); \
    ASSERT_NEAR(sum_actual, sum_expect, 0.1);                                \
  }

BLAS_SUM_TENSOR_SUM_TEST(Data)
BLAS_SUM_TENSOR_SUM_TEST(Diff)

#define BLAS_SUM_ROW_SUM_TEST(data_diff)                                    \
  TEST_F(BlasSumTest, Blas_SumRowSum##data_diff##Test) {                    \
    tensor->SetGPU##data_diff(data.data(), data.size());                    \
    const std::vector<int> result_shape{10, 100, 1};                        \
    auto result = std::make_shared<my_tensor::Tensor<>>(result_shape);      \
    my_tensor::row_sum_gpu(tensor->GetGPU##data_diff##Ptr(),                \
                           result->GetGPU##data_diff##Ptr(), 100, 200, 10); \
    std::vector<float> result_expect(1000, 0.0f);                           \
    for (int t = 0; t < 10; t++) {                                          \
      for (int i = 0; i < 100; i++) {                                       \
        for (int j = 0; j < 200; j++) {                                     \
          result_expect[t * 100 + i] += data[t * 20000 + i * 200 + j];      \
        }                                                                   \
      }                                                                     \
    }                                                                       \
    for (int i = 0; i < 1000; i++) {                                        \
      ASSERT_NEAR(result->GetCPU##data_diff(i), result_expect[i], 0.01);    \
    }                                                                       \
  }

BLAS_SUM_ROW_SUM_TEST(Data);
BLAS_SUM_ROW_SUM_TEST(Diff);

#define BLAS_SUM_COL_SUM_TEST(data_diff)                                    \
  TEST_F(BlasSumTest, Blas_SumColSum##data_diff##Test) {                    \
    tensor->Reshape({10, 200, 100});                                        \
    tensor->SetGPU##data_diff(data.data(), data.size());                    \
    const std::vector<int> result_shape{10, 100, 1};                        \
    auto result = std::make_shared<my_tensor::Tensor<>>(result_shape);      \
    my_tensor::col_sum_gpu(tensor->GetGPU##data_diff##Ptr(),                \
                           result->GetGPU##data_diff##Ptr(), 200, 100, 10); \
    std::vector<float> result_expect(1000, 0.0f);                           \
    for (int t = 0; t < 10; t++) {                                          \
      for (int i = 0; i < 100; i++) {                                       \
        for (int j = 0; j < 200; j++) {                                     \
          result_expect[t * 100 + i] += data[t * 20000 + j * 100 + i];      \
        }                                                                   \
      }                                                                     \
    }                                                                       \
    for (int i = 0; i < 1000; i++) {                                        \
      ASSERT_NEAR(result->GetCPU##data_diff(i), result_expect[i], 0.01);    \
    }                                                                       \
  }

BLAS_SUM_COL_SUM_TEST(Data)
BLAS_SUM_COL_SUM_TEST(Diff)

TEST(SquareTest, SquareCPUTest) {
  std::vector<float> data(200000);
  std::vector<float> result_expect(200000);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
  auto random_func = [&gen, &dis]() -> float { return dis(gen); };
  std::ranges::generate(data, random_func);
  for (int i = 0; i < 200000; i++) {
    result_expect[i] = data[i] * data[i];
  }
  my_tensor::square_cpu<float>(data.data(), data.data(), 200000);
  for (int i = 0; i < 200000; i++) {
    ASSERT_NEAR(result_expect[i], data[i], 0.01);
  }
}

TEST(SquareTest, SquareGPUTest) {
  std::vector<float> data(200000);
  std::vector<float> result_expect(200000);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
  auto random_func = [&gen, &dis]() -> float { return dis(gen); };
  std::ranges::generate(data, random_func);
  for (int i = 0; i < 200000; i++) {
    result_expect[i] = data[i] * data[i];
  }
  my_tensor::Tensor<float> tensor({200000});
  tensor.SetGPUData(data.data(), data.size());
  my_tensor::square_gpu<float>(tensor.GetGPUDataPtr(), tensor.GetGPUDataPtr(),
                               200000);
  for (int i = 0; i < 200000; i++) {
    ASSERT_NEAR(result_expect[i], tensor.GetCPUData(i), 0.01);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
