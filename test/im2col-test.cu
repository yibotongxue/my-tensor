#include <gtest/gtest.h>
#include <im2col.cuh>
#include <tensor.cuh>

#include <algorithm>
#include <vector>
#include <ranges>

// TEST(TrivialTest, always_succeed) {
//   EXPECT_TRUE(true);
// }

// TEST(TrivialTest, always_fail) {
//   EXPECT_TRUE(false);
// }

#define TEST_IM2COL(device)\
TEST(Im2col##device##Test, test_one_channel) {\
  const std::vector<int> im_shape = {1, 8, 6};\
  auto im_tensor = std::make_shared<my_tensor::Tensor<>>(im_shape);\
  std::vector<float> data(48);\
  float i = 0.0f;\
  auto func = [&i]()->float { i++; return i; };\
  std::ranges::generate(data, func);\
  im_tensor->Set##device##Data(data);\
\
  const std::vector<int> col_shape = {1, 48, 9};\
  auto col_tensor = std::make_shared<my_tensor::Tensor<>>(col_shape);\
  ASSERT_NO_THROW(my_tensor::Im2col_##device(im_tensor->Get##device##DataPtr(), 1, 6, 8, 3, 3, col_tensor->Get##device##DataPtr()));\
  const std::vector<float> expect {\
    10, 9, 10, 2, 1, 2, 10, 9, 10,\
    9, 10, 11, 1, 2, 3, 9, 10, 11,\
    10, 11, 12, 2, 3, 4, 10, 11, 12,\
    11, 12, 13, 3, 4, 5, 11, 12, 13,\
    12, 13, 14, 4, 5, 6, 12, 13, 14,\
    13, 14, 15, 5, 6, 7, 13, 14, 15,\
    14, 15, 16, 6, 7, 8, 14, 15, 16,\
    15, 16, 15, 7, 8, 7, 15, 16, 15,\
    \
    2, 1, 2, 10, 9, 10, 18, 17, 18,\
    1, 2, 3, 9, 10, 11, 17, 18, 19,\
    2, 3, 4, 10, 11, 12, 18, 19, 20,\
    3, 4, 5, 11, 12, 13, 19, 20, 21,\
    4, 5, 6, 12, 13, 14, 20, 21, 22,\
    5, 6, 7, 13, 14, 15, 21, 22, 23,\
    6, 7, 8, 14, 15, 16, 22, 23, 24,\
    7, 8, 7, 15, 16, 15, 23, 24, 23,\
   \
    10, 9, 10, 18, 17, 18, 26, 25, 26,\
    9, 10, 11, 17, 18, 19, 25, 26, 27,\
    10, 11, 12, 18, 19, 20, 26, 27, 28,\
    11, 12, 13, 19, 20, 21, 27, 28, 29,\
    12, 13, 14, 20, 21, 22, 28, 29, 30,\
    13, 14, 15, 21, 22, 23, 29, 30, 31,\
    14, 15, 16, 22, 23, 24, 30, 31, 32,\
    15, 16, 15, 23, 24, 23, 31, 32, 31,\
    \
    18, 17, 18, 26, 25, 26, 34, 33, 34,\
    17, 18, 19, 25, 26, 27, 33, 34, 35,\
    18, 19, 20, 26, 27, 28, 34, 35, 36,\
    19, 20, 21, 27, 28, 29, 35, 36, 37,\
    20, 21, 22, 28, 29, 30, 36, 37, 38,\
    21, 22, 23, 29, 30, 31, 37, 38, 39,\
    22, 23, 24, 30, 31, 32, 38, 39, 40,\
    23, 24, 23, 31, 32, 31, 39, 40, 39,\
    \
    26, 25, 26, 34, 33, 34, 42, 41, 42,\
    25, 26, 27, 33, 34, 35, 41, 42, 43,\
    26, 27, 28, 34, 35, 36, 42, 43, 44,\
    27, 28, 29, 35, 36, 37, 43, 44, 45,\
    28, 29, 30, 36, 37, 38, 44, 45, 46,\
    29, 30, 31, 37, 38, 39, 45, 46, 47,\
    30, 31, 32, 38, 39, 40, 46, 47, 48,\
    31, 32, 31, 39, 40, 39, 47, 48, 47,\
    \
    34, 33, 34, 42, 41, 42, 34, 33, 34,\
    33, 34, 35, 41, 42, 43, 33, 34, 35,\
    34, 35, 36, 42, 43, 44, 34, 35, 36,\
    35, 36, 37, 43, 44, 45, 35, 36, 37,\
    36, 37, 38, 44, 45, 46, 36, 37, 38,\
    37, 38, 39, 45, 46, 47, 37, 38, 39,\
    38, 39, 40, 46, 47, 48, 38, 39, 40,\
    39, 40, 39, 47, 48, 47, 39, 40, 39,\
  };\
  const std::vector<float> actual(col_tensor->Get##device##Data().begin(), col_tensor->Get##device##Data().end());\
  for (int i = 0; i < 48 * 9; i++) {\
    ASSERT_NEAR(expect[i], actual[i], 0.01f);\
  }\
}

TEST_IM2COL(CPU)
TEST_IM2COL(GPU)

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
