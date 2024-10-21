#include <gtest/gtest.h>
#include <linear.cuh>
#include <layer/layer-utils.cuh>

#include <random>
#include <algorithm>
#include <ranges>

// TEST(TrivialTest, always_succeed) {
//   EXPECT_TRUE(true);
// }

// TEST(TrivialTest, always_fail) {
//   EXPECT_TRUE(false);
// }

class LinearTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    x_data.resize(80000);
    weight_data.resize(60000);
    bias_data.resize(300);
    y_diff.resize(120000);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    auto random_func = [&gen, &dis]() -> float
    { return dis(gen); };
    std::ranges::generate(x_data, random_func);
    std::ranges::generate(weight_data, random_func);
    std::ranges::generate(bias_data, random_func);
    std::ranges::generate(y_diff, random_func);
    X.reset();
    X = std::make_shared<my_tensor::Tensor<>>(x_shape);
    weight.reset();
    weight = std::make_shared<my_tensor::Tensor<>>(weight_shape);
    bias.reset();
    bias = std::make_shared<my_tensor::Tensor<>>(bias_shape);
    Y.reset();
    Y = std::make_shared<my_tensor::Tensor<>>(y_shape);
  }
  const std::vector<int> x_shape{200, 400};
  const std::vector<int> weight_shape{300, 200};
  const std::vector<int> bias_shape{300};
  const std::vector<int> y_shape{300, 400};
  std::vector<float> x_data;
  std::vector<float> weight_data;
  std::vector<float> bias_data;
  std::vector<float> y_diff;
  my_tensor::TensorPtr<> X;
  my_tensor::TensorPtr<> weight;
  my_tensor::TensorPtr<> bias;
  my_tensor::TensorPtr<> Y;
  my_tensor::LayerPtr<> linear;
  int m = 300;
  int k = 200;
  int n = 400;
};

TEST_F(LinearTest, Linear_ForwardCPUTest)
{
  X->SetCPUData(x_data);
  weight->SetCPUData(weight_data);
  bias->SetCPUData(bias_data);
  const std::vector<my_tensor::TensorPtr<>> params = {weight, bias};
  linear.reset();
  linear = std::make_shared<my_tensor::Linear<>>(params);
  linear->ForwardCPU(X, Y);
  std::vector<float> result_actual(Y->GetCPUData().begin(), Y->GetCPUData().end());
  for (int i = 0; i < 1; i++) {
    int row = i / 300;
    int col = i % 400;
    float temp = bias_data[row];
    for (int j = 0; j < 200; j++) {
      temp += weight_data[row * 200 + j] * x_data[j * 400 + col];
    }
    ASSERT_NEAR(temp, result_actual[i], 0.01);
  }
}

// LINEAR_FORWARD_TEST(CPU)
// LINEAR_FORWARD_TEST(GPU)

TEST_F(LinearTest, Linear_ForwardCPUGPUTest)
{
  X->SetCPUData(x_data);
  weight->SetCPUData(weight_data);
  bias->SetCPUData(bias_data);
  const std::vector<my_tensor::TensorPtr<>> params = {weight, bias};
  linear.reset();
  linear = std::make_shared<my_tensor::Linear<>>(params);
  auto another = std::make_shared<my_tensor::Tensor<>>(*Y);
  linear->ForwardCPU(X, Y);
  linear->ForwardGPU(X, another);
  std::vector<float> result_actual(Y->GetCPUData().begin(), Y->GetCPUData().end());
  std::vector<float> result_expect(another->GetCPUData().begin(), another->GetCPUData().end());
  for (int i = 0; i < m * n; i++)
  {
    ASSERT_NEAR(result_actual[i], result_expect[i], 0.01);
  }
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
