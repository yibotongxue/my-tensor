#include <gtest/gtest.h>
#include <layer.cuh>
#include <sigmoid.cuh>
#include <layer/layer-utils.cuh>

#include <random>
#include <algorithm>
#include <ranges>

#define SIGMOID_TEST_CLASS(device)                            \
  class Sigmoid##device##Test : public ::testing::Test        \
  {                                                           \
  protected:                                                  \
    void SetUp() override                                     \
    {                                                         \
      data.resize(30000);                                     \
      diff.resize(30000);                                     \
      std::random_device rd;                                  \
      std::mt19937 gen(rd());                                 \
      std::uniform_real_distribution<float> dis(-3.0f, 3.0f); \
      for (int i = 0; i < 30000; i++)                         \
      {                                                       \
        data[i] = dis(gen);                                   \
      }                                                       \
      for (int i = 0; i < 30000; i++)                         \
      {                                                       \
        diff[i] = dis(gen);                                   \
      }                                                       \
      sigmoid.reset();                                        \
      bottom.reset();                                         \
      top.reset();                                            \
      sigmoid = std::make_shared<my_tensor::Sigmoid<>>();     \
      bottom = std::make_shared<my_tensor::Tensor<>>(shape);  \
      top = std::make_shared<my_tensor::Tensor<>>(shape);     \
      bottom->Set##device##Data(data);                        \
      top->Set##device##Diff(diff);                           \
    }                                                         \
    const std::vector<int> shape{10000, 3};                   \
    std::vector<float> data;                                  \
    std::vector<float> diff;                                  \
    my_tensor::LayerPtr<> sigmoid;                            \
    my_tensor::TensorPtr<> bottom;                            \
    my_tensor::TensorPtr<> top;                               \
  };

SIGMOID_TEST_CLASS(CPU)
SIGMOID_TEST_CLASS(GPU)

#define SIGMOID_FORWARD_TEST(device)                                                                     \
  TEST_F(Sigmoid##device##Test, Forward_Data)                                                            \
  {                                                                                                      \
    sigmoid->Forward##device(bottom, top);                                                               \
    const std::vector<float> top_data(top->Get##device##Data().begin(), top->Get##device##Data().end()); \
    std::ranges::transform(data, data.begin(), [](float x) { return 1.0f / (1.0f + std::exp(-x)); });    \
    for (int i = 0; i < 30000; i++)                                                                      \
    {                                                                                                    \
      EXPECT_NEAR(top_data[i], data[i], 0.01);                                                           \
    }                                                                                                    \
  }

SIGMOID_FORWARD_TEST(CPU)
SIGMOID_FORWARD_TEST(GPU)

BACKWARD_TEST(Sigmoid, sigmoid, CPU)
BACKWARD_TEST(Sigmoid, sigmoid, GPU)

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
