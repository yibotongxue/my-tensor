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

#define CONVOLUTION_BACKWARD_BOTTOM(device)\
TEST_F(Convolution##device##Test, BackwardBottomTest) {\
  conv->Backward##device(output, input);\
  std::vector<float> actual(input->Get##device##Diff().begin(), input->Get##device##Diff().end());\
  for (int i = 0; i < 102400; i++) {\
    int n = i / 10240;\
    int c = (i % 10240) / 2048;\
    int row = (i % 2048) / 64;\
    int col = i % 64;\
    float expect = 0.0f;\
    for (int j = 0; j < 9; j++) {\
      for (int x = 0; x < 3; x++) {\
        for (int y = 0; y < 3; y++) {\
          int output_row = row - x + 1;\
          int output_col = col - y + 1;\
          if (output_row >= 0 && output_row < 32 && output_col >= 0 && output_col < 64) {\
            expect += kernels_data[j * 5 * 9 + c * 9 + x * 3 + y] * output_diff[n * 18432 + j * 2048 + output_row * 64 + output_col];\
          }\
        }\
      }\
    }\
    ASSERT_NEAR(actual[i], expect, 0.01);\
  }\
}

CONVOLUTION_BACKWARD_BOTTOM(CPU)
CONVOLUTION_BACKWARD_BOTTOM(GPU)

#define CONVOLUTION_BACKWARD_KERNEL(device)\
TEST_F(Convolution##device##Test, BackwardKernelTest) {\
  conv->Backward##device(output, input);\
  std::vector<float> actual(kernels->Get##device##Diff().begin(), kernels->Get##device##Diff().end());\
  for (int i = 0; i < 405; i++) {\
    int c_out = i / 45;\
    int c_in = (i % 45) / 9;\
    int k_h = (i % 9) / 3;\
    int k_w = i % 3;\
    float expect = 0.0f;\
    for (int t = 0; t < 10; t++) {\
      for (int x = 0; x < 32; x++) {\
        int input_row = x - 1 + k_h;\
        for (int y = 0; y < 64; y++) {\
        int input_col = y - 1 + k_w;\
          if (input_row >= 0 && input_row < 32 && input_col >= 0 && input_col < 64) {\
            expect += input_data[t * 10240 + c_in * 2048 + input_row * 64 + input_col]\
                * output_diff[t * 18432 + c_out * 2048 + x * 64 + y];\
          }\
          input_col++;\
        }\
        input_row++;\
      }\
    }\
    ASSERT_NEAR(actual[i], expect, 0.1);\
  }\
}

CONVOLUTION_BACKWARD_KERNEL(CPU)
CONVOLUTION_BACKWARD_KERNEL(GPU)

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}