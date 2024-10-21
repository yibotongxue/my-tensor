#include <gtest/gtest.h>
#include <layer.cuh>
#include <relu.cuh>
#include <layer/layer-utils.cuh>

#include <random>

// TEST(TrivialTest, always_succeed) {
//   EXPECT_TRUE(true);
// }

// TEST(TrivialTest, always_fail) {
//   EXPECT_TRUE(false);
// }

#define RELU_TEST_CLASS(device)                               \
  class Relu##device##Test : public ::testing::Test           \
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
        if (data[i] >= -0.001 && data[i] <= 0)                \
        {                                                     \
          data[i] = 0.001;                                    \
        }                                                     \
      }                                                       \
      for (int i = 0; i < 30000; i++)                         \
      {                                                       \
        diff[i] = dis(gen);                                   \
      }                                                       \
      relu.reset();                                           \
      bottom.reset();                                         \
      top.reset();                                            \
      relu = std::make_shared<my_tensor::Relu<>>();           \
      bottom = std::make_shared<my_tensor::Tensor<>>(shape);  \
      top = std::make_shared<my_tensor::Tensor<>>(shape);     \
      bottom->Set##device##Data(data);                        \
      top->Set##device##Diff(diff);                           \
    }                                                         \
    const std::vector<int> shape{10000, 3};                   \
    std::vector<float> data;                                  \
    std::vector<float> diff;                                  \
    my_tensor::LayerPtr<> relu;                               \
    my_tensor::TensorPtr<> bottom;                            \
    my_tensor::TensorPtr<> top;                               \
  };

RELU_TEST_CLASS(CPU)
RELU_TEST_CLASS(GPU)

#define RELU_FORWARD_TEST(device)                                                                        \
  TEST_F(Relu##device##Test, Forward_Data)                                                               \
  {                                                                                                      \
    relu->Forward##device(bottom, top);                                                                  \
    const std::vector<float> top_data(top->Get##device##Data().begin(), top->Get##device##Data().end()); \
    for (int i = 0; i < 30000; i++)                                                                      \
    {                                                                                                    \
      if (data[i] > 0)                                                                                   \
      {                                                                                                  \
        EXPECT_EQ(top_data[i], data[i]);                                                                 \
      }                                                                                                  \
      else                                                                                               \
      {                                                                                                  \
        EXPECT_EQ(top_data[i], 0);                                                                       \
      }                                                                                                  \
    }                                                                                                    \
  }

RELU_FORWARD_TEST(CPU)
RELU_FORWARD_TEST(GPU)
BACKWARD_TEST(Relu, relu, CPU)
BACKWARD_TEST(Relu, relu, GPU)

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
