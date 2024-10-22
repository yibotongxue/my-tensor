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

#define LINEAR_TEST(device)                                              \
  class Linear##device##Test : public ::testing::Test                    \
  {                                                                      \
  protected:                                                             \
    void SetUp() override                                                \
    {                                                                    \
      weight_data.resize(80000);                                         \
      x_data.resize(60000);                                              \
      bias_data.resize(400);                                             \
      y_diff.resize(120000);                                             \
      std::random_device rd;                                             \
      std::mt19937 gen(rd());                                            \
      std::uniform_real_distribution<float> dis(-10.0f, 10.0f);          \
      auto random_func = [&gen, &dis]() -> float                         \
      { return dis(gen); };                                              \
      std::ranges::generate(weight_data, random_func);                   \
      std::ranges::generate(x_data, random_func);                        \
      std::ranges::generate(bias_data, random_func);                     \
      std::ranges::generate(y_diff, random_func);                        \
      X.reset();                                                         \
      weight.reset();                                                    \
      bias.reset();                                                      \
      Y.reset();                                                         \
      weight = std::make_shared<my_tensor::Tensor<>>(weight_shape);      \
      X = std::make_shared<my_tensor::Tensor<>>(x_shape);                \
      bias = std::make_shared<my_tensor::Tensor<>>(bias_shape);          \
      Y = std::make_shared<my_tensor::Tensor<>>(y_shape);                \
      weight->Set##device##Data(weight_data);                            \
      X->Set##device##Data(x_data);                                      \
      bias->Set##device##Data(bias_data);                                \
      const std::vector<my_tensor::TensorPtr<>> params = {weight, bias}; \
      linear.reset();                                                    \
      linear = std::make_shared<my_tensor::Linear<>>(params);            \
      linear->Forward##device(X, Y);                                     \
      Y->Set##device##Diff(y_diff);                                      \
      linear->Backward##device(Y, X);                                    \
    }                                                                    \
    const std::vector<int> weight_shape{200, 400};                       \
    const std::vector<int> x_shape{300, 200};                            \
    const std::vector<int> bias_shape{400};                              \
    const std::vector<int> y_shape{300, 400};                            \
    std::vector<float> weight_data;                                      \
    std::vector<float> x_data;                                           \
    std::vector<float> bias_data;                                        \
    std::vector<float> y_diff;                                           \
    my_tensor::TensorPtr<> X;                                            \
    my_tensor::TensorPtr<> weight;                                       \
    my_tensor::TensorPtr<> bias;                                         \
    my_tensor::TensorPtr<> Y;                                            \
    my_tensor::LayerPtr<> linear;                                        \
    int m = 300;                                                         \
    int k = 200;                                                         \
    int n = 400;                                                         \
  };

LINEAR_TEST(CPU)
LINEAR_TEST(GPU)

#define LINEAR_FORWARD_TEST(device)                                                                 \
  TEST_F(Linear##device##Test, Linear_Forward##device##Test)                                        \
  {                                                                                                 \
    std::vector<float> result_actual(Y->Get##device##Data().begin(), Y->Get##device##Data().end()); \
    for (int i = 0; i < 120000; i++)                                                                \
    {                                                                                               \
      int row = i / 400;                                                                            \
      int col = i % 400;                                                                            \
      float temp = bias_data[col];                                                                  \
      for (int j = 0; j < 200; j++)                                                                 \
      {                                                                                             \
        temp += x_data[row * 200 + j] * weight_data[j * 400 + col];                                 \
      }                                                                                             \
      ASSERT_NEAR(temp, result_actual[i], 0.01);                                                    \
    }                                                                                               \
  }

LINEAR_FORWARD_TEST(CPU)
LINEAR_FORWARD_TEST(GPU)

#define LINEAR_BACKWARD_BOTTOM_TEST(device)                                                  \
  TEST_F(Linear##device##Test, Linear_BackwardBottom##device##Test)                          \
  {                                                                                          \
    std::vector<float> actual(X->Get##device##Diff().begin(), X->Get##device##Diff().end()); \
    for (int i = 0; i < 60000; i++)                                                          \
    {                                                                                        \
      int row = i / 200;                                                                     \
      int col = i % 200;                                                                     \
      float expect{0.0f};                                                                    \
      for (int j = 0; j < 400; j++)                                                          \
      {                                                                                      \
        expect += weight_data[col * 400 + j] * y_diff[row * 400 + j];                        \
      }                                                                                      \
      ASSERT_NEAR(actual[i], expect, 0.01);                                                  \
    }                                                                                        \
  }

LINEAR_BACKWARD_BOTTOM_TEST(CPU)
LINEAR_BACKWARD_BOTTOM_TEST(GPU)

#define LINEAR_BACKWARD_WEIGHT_TEST(device)                                                            \
  TEST_F(Linear##device##Test, Linear_BackwardWeight##device##Test)                                    \
  {                                                                                                    \
    std::vector<float> actual(weight->Get##device##Diff().begin(), weight->Get##device##Diff().end()); \
    for (int i = 0; i < 80000; i++)                                                                    \
    {                                                                                                  \
      int row = i / 400;                                                                               \
      int col = i % 400;                                                                               \
      float expect{0.0f};                                                                              \
      for (int j = 0; j < 300; j++)                                                                    \
      {                                                                                                \
        expect += y_diff[j * 400 + col] * x_data[j * 200 + row];                                       \
      }                                                                                                \
      ASSERT_NEAR(actual[i], expect, 0.01);                                                            \
    }                                                                                                  \
  }

LINEAR_BACKWARD_WEIGHT_TEST(CPU)
LINEAR_BACKWARD_WEIGHT_TEST(GPU)

#define LINEAR_BACKWARD_BIAS_TEST(device)                                                          \
  TEST_F(Linear##device##Test, Linear_BackwardBias##device##Test)                                  \
  {                                                                                                \
    std::vector<float> actual(bias->Get##device##Diff().begin(), bias->Get##device##Diff().end()); \
    for (int i = 0; i < 400; i++)                                                                  \
    {                                                                                              \
      float expect{0.0f};                                                                          \
      for (int j = 0; j < 300; j++)                                                                \
      {                                                                                            \
        expect += y_diff[j * 400 + i];                                                             \
      }                                                                                            \
      ASSERT_NEAR(actual[i], expect, 0.01);                                                        \
    }                                                                                              \
  }

LINEAR_BACKWARD_BIAS_TEST(CPU)
LINEAR_BACKWARD_BIAS_TEST(GPU)

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
