#include <gtest/gtest.h>
#include <blas.cuh>

#include <algorithm>
#include <random>
#include <ranges>

// TEST(TrivialTest, always_succeed) {
//   EXPECT_TRUE(true);
// }

// TEST(TrivialTest, always_fail) {
//   EXPECT_TRUE(false);
// }

class BlasTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    lhs_data.resize(5000);
    rhs_data.resize(5000);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    auto random_func = [&gen, &dis]() -> float
    { return dis(gen); };
    std::ranges::generate(lhs_data, random_func);
    std::ranges::generate(rhs_data, random_func);
    lhs = std::make_shared<my_tensor::Tensor<>>(shape);
    rhs = std::make_shared<my_tensor::Tensor<>>(shape);
  }

  std::vector<float> lhs_data;
  std::vector<float> rhs_data;
  my_tensor::TensorPtr<> lhs;
  my_tensor::TensorPtr<> rhs;
  const std::vector<int> shape{1000, 5};
};

TEST_F(BlasTest, Blas_AddTest)
{
  lhs->SetGPUData(lhs_data);
  rhs->SetGPUData(rhs_data);
  auto result = std::make_shared<my_tensor::Tensor<>>(*lhs + *rhs);
  ASSERT_EQ(result->GetShape(), shape);
  std::vector<float> result_actual(result->GetGPUData().begin(), result->GetGPUData().end());
  std::vector<float> result_expect(5000);
  std::ranges::transform(lhs_data, rhs_data, result_expect.begin(), std::plus<float>());
  for (int i = 0; i < 5000; i++)
  {
    EXPECT_NEAR(result_expect[i], result_actual[i], 0.01);
  }
}

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

#define BLAS_ADD_VEC_TEST(data_diff, at_grad)                                                                 \
  TEST_F(BlasAddVecTest, Blas_AddVec##data_diff##Test)                                                        \
  {                                                                                                           \
    tensor->SetGPU##data_diff(tensor_data);                                                                   \
    vec->SetGPU##data_diff(vec_data);                                                                         \
    auto result = std::make_shared<my_tensor::Tensor<>>(add_vector(*tensor, *vec, at_grad));                  \
    ASSERT_EQ(result->GetShape(), tensor_shape);                                                              \
    std::vector<float> result_actual(result->GetGPU##data_diff().begin(), result->GetGPU##data_diff().end()); \
    for (int i = 0; i < 200000; i++)                                                                          \
    {                                                                                                         \
      result_expect[i] = tensor_data[i] + vec_data[i / 200];                                                  \
    }                                                                                                         \
    for (int i = 0; i < 200000; i++)                                                                          \
    {                                                                                                         \
      ASSERT_NEAR(result_expect[i], result_actual[i], 0.01);                                                  \
    }                                                                                                         \
  }

BLAS_ADD_VEC_TEST(Data, false)
BLAS_ADD_VEC_TEST(Diff, true)

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
  }

  std::vector<float> lhs_data;
  std::vector<float> rhs_data;
  std::vector<float> result_expect;
  my_tensor::TensorPtr<> lhs;
  my_tensor::TensorPtr<> rhs;
  const std::vector<int> left_shape{500, 128};
  const std::vector<int> right_shape{128, 64};
  const std::vector<int> result_shape{500, 64};
};

#define BLAS_MATMUL_TEST(data_diff, at_grad)                                                                  \
  TEST_F(BlasMatmulTest, Blas_Matmu##data_diff##lTest)                                                        \
  {                                                                                                           \
    lhs->SetGPU##data_diff(lhs_data);                                                                         \
    rhs->SetGPU##data_diff(rhs_data);                                                                         \
    auto result = std::make_shared<my_tensor::Tensor<>>(matmul(*lhs, *rhs, at_grad));                         \
    ASSERT_EQ(result->GetShape(), result_shape);                                                              \
    std::vector<float> result_actual(result->GetGPU##data_diff().begin(), result->GetGPU##data_diff().end()); \
    for (int i = 0; i < 32000; i++)                                                                           \
    {                                                                                                         \
      int row = i / 64;                                                                                       \
      int col = i % 64;                                                                                       \
      for (int k = 0; k < 128; k++)                                                                           \
      {                                                                                                       \
        result_expect[i] += lhs_data[row * 128 + k] * rhs_data[k * 64 + col];                                 \
      }                                                                                                       \
    }                                                                                                         \
    for (int i = 0; i < 32000; i++)                                                                           \
    {                                                                                                         \
      ASSERT_NEAR(result_expect[i], result_actual[i], 0.01);                                                  \
    }                                                                                                         \
  }

BLAS_MATMUL_TEST(Data, false)
BLAS_MATMUL_TEST(Diff, true)

#define BLAS_T_MATMUL_TEST(data_diff, at_grad)                                                                \
  TEST_F(BlasMatmulTest, Blas_TransposeMatmul##data_diff##Test)                                               \
  {                                                                                                           \
    lhs->SetGPU##data_diff(lhs_data);                                                                         \
    rhs->SetGPU##data_diff(rhs_data);                                                                         \
    lhs->Reshape({128, 500});                                                                                 \
    auto result = std::make_shared<my_tensor::Tensor<>>(transpose_matmul(*lhs, *rhs, at_grad));               \
    ASSERT_EQ(result->GetShape(), result_shape);                                                              \
    std::vector<float> result_actual(result->GetGPU##data_diff().begin(), result->GetGPU##data_diff().end()); \
    for (int i = 0; i < 32000; i++)                                                                           \
    {                                                                                                         \
      int row = i / 64;                                                                                       \
      int col = i % 64;                                                                                       \
      for (int k = 0; k < 128; k++)                                                                           \
      {                                                                                                       \
        result_expect[i] += lhs_data[k * 500 + row] * rhs_data[k * 64 + col];                                 \
      }                                                                                                       \
    }                                                                                                         \
    for (int i = 0; i < 32000; i++)                                                                           \
    {                                                                                                         \
      ASSERT_NEAR(result_expect[i], result_actual[i], 0.01);                                                  \
    }                                                                                                         \
  }

BLAS_T_MATMUL_TEST(Data, false)
BLAS_T_MATMUL_TEST(Diff, true)

#define BLAS_MATMUL_T_TEST(data_diff, at_grad)                                                                \
  TEST_F(BlasMatmulTest, Blas_MatmulTranspose##data_diff##Test)                                               \
  {                                                                                                           \
    lhs->SetGPU##data_diff(lhs_data);                                                                         \
    rhs->SetGPU##data_diff(rhs_data);                                                                         \
    rhs->Reshape({64, 128});                                                                                  \
    auto result = std::make_shared<my_tensor::Tensor<>>(matmul_transpose(*lhs, *rhs, at_grad));                        \
    ASSERT_EQ(result->GetShape(), result_shape);                                                              \
    std::vector<float> result_actual(result->GetGPU##data_diff().begin(), result->GetGPU##data_diff().end()); \
    for (int i = 0; i < 32000; i++)                                                                           \
    {                                                                                                         \
      int row = i / 64;                                                                                       \
      int col = i % 64;                                                                                       \
      for (int k = 0; k < 128; k++)                                                                           \
      {                                                                                                       \
        result_expect[i] += lhs_data[row * 128 + k] * rhs_data[col * 128 + k];                                \
      }                                                                                                       \
    }                                                                                                         \
    for (int i = 0; i < 32000; i++)                                                                           \
    {                                                                                                         \
      ASSERT_NEAR(result_expect[i], result_actual[i], 0.01);                                                  \
    }                                                                                                         \
  }

BLAS_MATMUL_T_TEST(Data, false)
BLAS_MATMUL_T_TEST(Diff, true)

#define BLAS_T_MATMUL_T_TEST(data_diff, at_grad)\
TEST_F(BlasMatmulTest, Blas_TransposeMatmulTranspose##data_diff##Test)\
{\
    lhs->SetGPU##data_diff(lhs_data);                                                                         \
    rhs->SetGPU##data_diff(rhs_data);                                                                         \
  lhs->Reshape({128, 500});\
  rhs->Reshape({64, 128});\
  auto result = std::make_shared<my_tensor::Tensor<>>(transpose_matmul_transpose(*lhs, *rhs, at_grad));\
  ASSERT_EQ(result->GetShape(), result_shape);\
  std::vector<float> result_actual(result->GetGPU##data_diff().begin(), result->GetGPU##data_diff().end());\
  for (int i = 0; i < 32000; i++)\
  {\
    int row = i / 64;\
    int col = i % 64;\
    for (int k = 0; k < 128; k++)\
    {\
      result_expect[i] += lhs_data[k * 500 + row] * rhs_data[col * 128 + k];\
    }\
  }\
  for (int i = 0; i < 32000; i++)\
  {\
    ASSERT_NEAR(result_expect[i], result_actual[i], 0.01);\
  }\
}

BLAS_T_MATMUL_T_TEST(Data, false)
BLAS_T_MATMUL_T_TEST(Diff, true)

class BlasTransposeTest : public ::testing::Test
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
    tensor->SetGPUData(data);
  }

  std::vector<float> data;
  my_tensor::TensorPtr<> tensor;
  const std::vector<int> shape{1000, 20};
  const std::vector<int> transpose_shape{20, 1000};
};

TEST_F(BlasTransposeTest, Blas_TransposeTest)
{
  auto result = std::make_shared<my_tensor::Tensor<>>(transpose(*tensor));
  ASSERT_EQ(result->GetShape(), transpose_shape);
  std::vector<float> result_actual(result->GetGPUData().begin(), result->GetGPUData().end());
  for (int i = 0; i < 20; i++)
  {
    for (int j = 0; j < 1000; j++)
    {
      float expect = data[j * 20 + i];
      float actual = result_actual[i * 1000 + j];
      ASSERT_NEAR(actual, expect, 0.01);
    }
  }
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
