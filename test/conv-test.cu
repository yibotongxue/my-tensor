#include <gtest/gtest.h>
#include <conv.cuh>

#include <random>
#include <algorithm>
#include <ranges>
// #include <cudnn.h>

// TEST(TrivialTest, always_succeed) {
//   EXPECT_TRUE(true);
// }

// TEST(TrivialTest, always_fail) {
//   EXPECT_TRUE(false);
// }

#define CONVOLUTION_TEST_CLASS(device)\
class Convolution##device##Test : public ::testing::Test {\
 protected:\
  void SetUp() override {\
    input_data.resize(102400);\
    output_diff.resize(184320);\
    kernels_data.resize(405);\
    std::random_device rd;\
    std::mt19937 gen(rd());\
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);\
    auto random_func = [&gen, &dis]() -> float\
    { return dis(gen); };\
    std::ranges::generate(input_data, random_func);\
    std::ranges::generate(output_diff, random_func);\
    std::ranges::generate(kernels_data, random_func);\
    input = std::make_shared<my_tensor::Tensor<>>(input_shape);\
    input->Set##device##Data(input_data);\
    output = std::make_shared<my_tensor::Tensor<>>(output_shape);\
    output->Set##device##Diff(output_diff);\
    kernels = std::make_shared<my_tensor::Tensor<>>(kernels_shape);\
    kernels->Set##device##Data(kernels_data);\
    std::vector<my_tensor::TensorPtr<>> params = {kernels};\
    conv = std::make_shared<my_tensor::Convolution<>>(params);\
  }\
  const std::vector<int> input_shape {10, 5, 32, 64};\
  const std::vector<int> output_shape {10, 9, 32, 64};\
  const std::vector<int> kernels_shape {9, 5, 3, 3};\
  std::vector<float> input_data;\
  std::vector<float> output_diff;\
  std::vector<float> kernels_data;\
  my_tensor::TensorPtr<> input;\
  my_tensor::TensorPtr<> output;\
  my_tensor::TensorPtr<> kernels;\
  my_tensor::LayerPtr<> conv;\
};

CONVOLUTION_TEST_CLASS(CPU)
CONVOLUTION_TEST_CLASS(GPU)

#define CONVOLUTION_FORWARD_TEST(device)\
TEST_F(Convolution##device##Test, ForwardTest) {\
  conv->Forward##device(input, output);\
  std::vector<float> actual (output->Get##device##Data().begin(), output->Get##device##Data().end());\
  for (int i = 0; i < 184320; i++) {\
    int n = i / 18432;\
    int c = (i % 18432) / 2048;\
    int row =  (i % 2048) / 64;\
    int col = i % 64;\
    float expect = 0.0f;\
    for (int j = 0; j < 5; j++) {\
      for (int x = 0; x < 3; x++) {\
        for (int y = 0; y < 3; y++) {\
          int input_row = row + x - 1;\
          int input_col = col + y - 1;\
          if (input_row >= 0 && input_row < 32 && input_col >= 0 && input_col < 64) {\
            expect += kernels_data[c * 5 * 9 + j * 9 + x * 3 + y] * input_data[n * 10240 + j * 2048 + input_row * 64 + input_col];\
          }\
        }\
      }\
    }\
    ASSERT_NEAR(actual[i], expect, 0.01);\
  }\
}

CONVOLUTION_FORWARD_TEST(CPU)
CONVOLUTION_FORWARD_TEST(GPU)

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}