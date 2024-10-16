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

class BlasTest : public ::testing::Test {
 protected:
  void SetUp() override {
    lhs_data.resize(5000);
    rhs_data.resize(5000);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis (-10.0f, 10.0f);
    for (int i = 0; i < 5000; i++) {
      lhs_data[i] = dis(gen);
    }
    for (int i = 0; i < 5000; i++) {
      rhs_data[i] = dis(gen);
    }
    lhs = std::make_shared<my_tensor::Tensor<>>(shape);
    rhs = std::make_shared<my_tensor::Tensor<>>(shape);
    lhs->SetGPUData(lhs_data);
    rhs->SetGPUData(rhs_data);
  }

  std::vector<float> lhs_data;
  std::vector<float> rhs_data;
  my_tensor::TensorPtr<> lhs;
  my_tensor::TensorPtr<> rhs;
  const std::vector<int> shape {1000, 5};
};

TEST_F(BlasTest, Blas_AddTest) {
  auto result = std::make_shared<my_tensor::Tensor<>>(*lhs + *rhs);
  std::vector<float> result_actual (result->GetGPUData().begin(), result->GetGPUData().end());
  std::vector<float> result_expect(5000);
  std::ranges::transform(lhs_data, rhs_data, result_expect.begin(), std::plus<float>());
  for (int i = 0; i < 5000; i++) {
    EXPECT_NEAR(result_expect[i], result_actual[i], 0.01);
  }
}

class BlasMatmulTest : public ::testing::Test {
 protected:
  void SetUp() override {
    lhs_data.resize(64000);
    rhs_data.resize(8192);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis (-10.0f, 10.0f);
    for (int i = 0; i < 64000; i++) {
      lhs_data[i] = dis(gen);
    }
    for (int i = 0; i < 8192; i++) {
      rhs_data[i] = dis(gen);
    }
    lhs = std::make_shared<my_tensor::Tensor<>>(left_shape);
    rhs = std::make_shared<my_tensor::Tensor<>>(right_shape);
    lhs->SetGPUData(lhs_data);
    rhs->SetGPUData(rhs_data);
  }

  std::vector<float> lhs_data;
  std::vector<float> rhs_data;
  my_tensor::TensorPtr<> lhs;
  my_tensor::TensorPtr<> rhs;
  const std::vector<int> left_shape {500, 128};
  const std::vector<int> right_shape {128, 64};
  const std::vector<int> result_shape {500, 64};
};

TEST_F(BlasMatmulTest, Blas_MatmulTest) {
  auto result = std::make_shared<my_tensor::Tensor<>>(matmul(*lhs, *rhs));
  std::vector<float> result_actual (result->GetGPUData().begin(), result->GetGPUData().end());
  std::vector<float> result_expect(32000, 0.0f);
  // for (int i = 0; i < 32000; i++) {
  //   int row = i / 64;
  //   int col = i % 64;
  //   for (int k = 0; k < 128; k++) {
  //     result_expect[i] += lhs_data[row * 128 + k] * rhs_data[k * 64 + col];
  //   }
  // }
  for (int i = 0; i < 500; i++) {
    for (int j = 0; j < 64; j++) {
      float temp = 0.0f;
      for (int k = 0; k < 128; k++) {
        temp += lhs_data[i * 128 + k] * rhs_data[k * 64 + j];
      }
      result_expect[i * 64 + j] = temp;
    }
  }
  for (int i = 0; i < 32000; i++) {
    ASSERT_NEAR(result_expect[i], result_actual[i], 0.01);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
