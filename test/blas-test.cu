#include <gtest/gtest.h>
#include <blas.cuh>

#include <random>
#include <algorithm>
#include <ranges>
#include <tensor.cuh>

// TEST(TrivialTest, always_succeed) {
//   EXPECT_TRUE(true);
// }

// TEST(TrivialTest, always_fail) {
//   EXPECT_TRUE(false);
// }

class BlasMatmulTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    lhs_data.resize(64000);
    rhs_data.resize(8192);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    auto random_func = [&gen, &dis]() -> float
    { return dis(gen); };
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

#define BLAS_MATMUL_TEST(data_diff)                                                                                                  \
  TEST_F(BlasMatmulTest, Blas_Matmul##data_diff##Test)                                                                               \
  {                                                                                                                                  \
    lhs->SetGPU##data_diff(lhs_data);                                                                                                \
    rhs->SetGPU##data_diff(rhs_data);                                                                                                \
    my_tensor::matmul(lhs->GetGPU##data_diff##Ptr(), rhs->GetGPU##data_diff##Ptr(), result->GetGPU##data_diff##Ptr(), 500, 128, 64); \
    std::vector<float> result_actual(result->GetGPU##data_diff().begin(), result->GetGPU##data_diff().end());                        \
    for (int i = 0; i < 32000; i++)                                                                                                  \
    {                                                                                                                                \
      int row = i / 64;                                                                                                              \
      int col = i % 64;                                                                                                              \
      for (int k = 0; k < 128; k++)                                                                                                  \
      {                                                                                                                              \
        result_expect[i] += lhs_data[row * 128 + k] * rhs_data[k * 64 + col];                                                        \
      }                                                                                                                              \
    }                                                                                                                                \
    for (int i = 0; i < 32000; i++)                                                                                                  \
    {                                                                                                                                \
      ASSERT_NEAR(result_expect[i], result_actual[i], 0.01);                                                                         \
    }                                                                                                                                \
  }

BLAS_MATMUL_TEST(Data)
BLAS_MATMUL_TEST(Diff)

#define BLAS_MATMUL_T_TEST(data_diff)                                                                                                          \
  TEST_F(BlasMatmulTest, Blas_MatmulTranspose##data_diff##Test)                                                                                \
  {                                                                                                                                            \
    rhs->Reshape({64, 128});                                                                                                                   \
    lhs->SetGPU##data_diff(lhs_data);                                                                                                          \
    rhs->SetGPU##data_diff(rhs_data);                                                                                                          \
    my_tensor::matmul_transpose(lhs->GetGPU##data_diff##Ptr(), rhs->GetGPU##data_diff##Ptr(), result->GetGPU##data_diff##Ptr(), 500, 128, 64); \
    std::vector<float> result_actual(result->GetGPU##data_diff().begin(), result->GetGPU##data_diff().end());                                  \
    for (int i = 0; i < 32000; i++)                                                                                                            \
    {                                                                                                                                          \
      int row = i / 64;                                                                                                                        \
      int col = i % 64;                                                                                                                        \
      for (int k = 0; k < 128; k++)                                                                                                            \
      {                                                                                                                                        \
        result_expect[i] += lhs_data[row * 128 + k] * rhs_data[col * 128 + k];                                                                 \
      }                                                                                                                                        \
    }                                                                                                                                          \
    for (int i = 0; i < 32000; i++)                                                                                                            \
    {                                                                                                                                          \
      ASSERT_NEAR(result_expect[i], result_actual[i], 0.01);                                                                                   \
    }                                                                                                                                          \
  }

BLAS_MATMUL_T_TEST(Data)
BLAS_MATMUL_T_TEST(Diff)

#define BLAS_T_MATMUL_TEST(data_diff)                                                                                                          \
  TEST_F(BlasMatmulTest, Blas_TransposeMatmul##data_diff##Test)                                                                                \
  {                                                                                                                                            \
    lhs->Reshape({128, 500});                                                                                                                  \
    lhs->SetGPU##data_diff(lhs_data);                                                                                                          \
    rhs->SetGPU##data_diff(rhs_data);                                                                                                          \
    my_tensor::transpose_matmul(lhs->GetGPU##data_diff##Ptr(), rhs->GetGPU##data_diff##Ptr(), result->GetGPU##data_diff##Ptr(), 500, 128, 64); \
    std::vector<float> result_actual(result->GetGPU##data_diff().begin(), result->GetGPU##data_diff().end());                                  \
    for (int i = 0; i < 32000; i++)                                                                                                            \
    {                                                                                                                                          \
      int row = i / 64;                                                                                                                        \
      int col = i % 64;                                                                                                                        \
      for (int k = 0; k < 128; k++)                                                                                                            \
      {                                                                                                                                        \
        result_expect[i] += lhs_data[k * 500 + row] * rhs_data[k * 64 + col];                                                                  \
      }                                                                                                                                        \
    }                                                                                                                                          \
    for (int i = 0; i < 32000; i++)                                                                                                            \
    {                                                                                                                                          \
      ASSERT_NEAR(result_expect[i], result_actual[i], 0.01);                                                                                   \
    }                                                                                                                                          \
  }

BLAS_T_MATMUL_TEST(Data)
BLAS_T_MATMUL_TEST(Diff)

#define BLAS_T_MATMUL_T_TEST(data_diff)                                                                                                                  \
  TEST_F(BlasMatmulTest, Blas_TransposeMatmulTranspose##data_diff##Test)                                                                                 \
  {                                                                                                                                                      \
    lhs->Reshape({128, 500});                                                                                                                            \
    rhs->Reshape({64, 128});                                                                                                                             \
    lhs->SetGPU##data_diff(lhs_data);                                                                                                                    \
    rhs->SetGPU##data_diff(rhs_data);                                                                                                                    \
    my_tensor::transpose_matmul_transpose(lhs->GetGPU##data_diff##Ptr(), rhs->GetGPU##data_diff##Ptr(), result->GetGPU##data_diff##Ptr(), 500, 128, 64); \
    std::vector<float> result_actual(result->GetGPU##data_diff().begin(), result->GetGPU##data_diff().end());                                            \
    for (int i = 0; i < 32000; i++)                                                                                                                      \
    {                                                                                                                                                    \
      int row = i / 64;                                                                                                                                  \
      int col = i % 64;                                                                                                                                  \
      for (int k = 0; k < 128; k++)                                                                                                                      \
      {                                                                                                                                                  \
        result_expect[i] += lhs_data[k * 500 + row] * rhs_data[col * 128 + k];                                                                           \
      }                                                                                                                                                  \
    }                                                                                                                                                    \
    for (int i = 0; i < 32000; i++)                                                                                                                      \
    {                                                                                                                                                    \
      ASSERT_NEAR(result_expect[i], result_actual[i], 0.01);                                                                                             \
    }                                                                                                                                                    \
  }

BLAS_T_MATMUL_T_TEST(Data)
BLAS_T_MATMUL_T_TEST(Diff)

class BlasAddVecTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    tensor_data.resize(200000);
    vec_data.resize(1000);
    result_expect.resize(200000);
    tensor = std::make_shared<my_tensor::Tensor<>>(tensor_shape);
    vec = std::make_shared<my_tensor::Tensor<>>(vec_shape);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    auto random_func = [&gen, &dis]() -> float
    { return dis(gen); };
    std::ranges::generate(tensor_data, random_func);
    std::ranges::generate(vec_data, random_func);
  }

  std::vector<float> tensor_data;
  std::vector<float> vec_data;
  std::vector<float> result_expect;
  my_tensor::TensorPtr<> tensor;
  my_tensor::TensorPtr<> vec;
  const std::vector<int> tensor_shape{1000, 200};
  const std::vector<int> vec_shape{1000, 1};
};

#define BLAS_ADD_VEC_TEST(data_diff)                                                                          \
  TEST_F(BlasAddVecTest, Blas_AddVec##data_diff##Test)                                                        \
  {                                                                                                           \
    for (int i = 0; i < 200000; i++)                                                                          \
    {                                                                                                         \
      result_expect[i] = tensor_data[i] + vec_data[i / 200];                                                  \
    }                                                                                                         \
    tensor->SetGPU##data_diff(tensor_data);                                                                   \
    vec->SetGPU##data_diff(vec_data);                                                                         \
    my_tensor::add_vector(tensor->GetGPU##data_diff##Ptr(), vec->GetGPU##data_diff##Ptr(), 1000, 200);        \
    std::vector<float> result_actual(tensor->GetGPU##data_diff().begin(), tensor->GetGPU##data_diff().end()); \
    for (int i = 0; i < 200000; i++)                                                                          \
    {                                                                                                         \
      ASSERT_NEAR(result_expect[i], result_actual[i], 0.01);                                                  \
    }                                                                                                         \
  }

BLAS_ADD_VEC_TEST(Data)
BLAS_ADD_VEC_TEST(Diff)

class BlasSumTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    data.resize(20000);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    auto random_func = [&gen, &dis]() -> float
    { return dis(gen); };
    std::ranges::generate(data, random_func);
    tensor = std::make_shared<my_tensor::Tensor<>>(shape);
  }
  const std::vector<int> shape{100, 200};
  std::vector<float> data;
  my_tensor::TensorPtr<> tensor;
};

#define BLAS_SUM_TENSOR_SUM_TEST(data_diff)                                                 \
  TEST_F(BlasSumTest, Blas_SumTensorSum##data_diff##Test)                                   \
  {                                                                                         \
    tensor->SetGPU##data_diff(data);                                                        \
    float sum_actual = my_tensor::tensor_sum(tensor->GetGPU##data_diff##Ptr(), 20000);      \
    float sum_expect = std::accumulate(data.begin(), data.end(), 0.0f, std::plus<float>()); \
    ASSERT_NEAR(sum_actual, sum_expect, 0.01);                                              \
  }

BLAS_SUM_TENSOR_SUM_TEST(Data)
BLAS_SUM_TENSOR_SUM_TEST(Diff)

#define BLAS_SUM_ROW_SUM_TEST(data_diff)                                                                      \
  TEST_F(BlasSumTest, Blas_SumRowSum##data_diff##Test)                                                        \
  {                                                                                                           \
    tensor->SetGPU##data_diff(data);                                                                          \
    const std::vector<int> result_shape{100, 1};                                                              \
    auto result = std::make_shared<my_tensor::Tensor<>>(result_shape);                                        \
    my_tensor::row_sum(tensor->GetGPU##data_diff##Ptr(), result->GetGPU##data_diff##Ptr(), 100, 200);         \
    std::vector<float> result_actual(result->GetGPU##data_diff().begin(), result->GetGPU##data_diff().end()); \
    std::vector<float> result_expect(100, 0.0f);                                                              \
    for (int i = 0; i < 100; i++)                                                                             \
    {                                                                                                         \
      for (int j = 0; j < 200; j++)                                                                           \
      {                                                                                                       \
        result_expect[i] += data[i * 200 + j];                                                                \
      }                                                                                                       \
    }                                                                                                         \
    for (int i = 0; i < 100; i++)                                                                             \
    {                                                                                                         \
      ASSERT_NEAR(result_actual[i], result_expect[i], 0.01);                                                  \
    }                                                                                                         \
  }

BLAS_SUM_ROW_SUM_TEST(Data);
BLAS_SUM_ROW_SUM_TEST(Diff);

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
