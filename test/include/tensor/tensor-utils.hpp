// Copyright 2024 yibotongxue

#ifndef TEST_INCLUDE_TENSOR_TENSOR_UTILS_HPP_
#define TEST_INCLUDE_TENSOR_TENSOR_UTILS_HPP_

#include <iostream>
#include <memory>
#include <vector>

#define TEST_SHAPE_AND_SIZE(test_fixture)                         \
  TEST_F(test_fixture##Test, test_fixture##_ShapeMatches) {       \
    const std::vector<int> shape = {2, 3};                        \
    EXPECT_EQ(tensor->GetShape(), shape);                         \
  }                                                               \
  TEST_F(test_fixture##Test, test_fixture##_SizeMatches) {        \
    EXPECT_EQ(tensor->GetSize(), 6);                              \
  }                                                               \
  TEST_F(test_fixture##Test, test_fixture##_CPUDataSizeMatches) { \
    EXPECT_EQ(tensor->GetCPUData().size(), 6);                    \
  }                                                               \
  TEST_F(test_fixture##Test, test_fixture##_CPUDiffSizeMatches) { \
    EXPECT_EQ(tensor->GetCPUDiff().size(), 6);                    \
  }                                                               \
  TEST_F(test_fixture##Test, test_fixture##_GPUDataSizeMatches) { \
    EXPECT_EQ(tensor->GetGPUData().size(), 6);                    \
  }                                                               \
  TEST_F(test_fixture##Test, test_fixture##_GPUDiffSizeMatches) { \
    EXPECT_EQ(tensor->GetGPUDiff().size(), 6);                    \
  }

#define DEFINE_DATA_AND_DIFF(data_name, diff_name) \
  data_name.resize(6);                             \
  diff_name.resize(6);                             \
  for (int i = 0; i < 6; i++) {                    \
    data_name[i] = i + 1;                          \
  }                                                \
  diff_name[0] = 1;                                \
  diff_name[1] = 3;                                \
  diff_name[2] = 5;                                \
  diff_name[3] = 2;                                \
  diff_name[4] = 4;                                \
  diff_name[5] = 6;

#define DEFINE_TESNOR(tensor_name)                                            \
  const std::vector<int> tensor_name##_shape{2, 3};                           \
  try {                                                                       \
    tensor_name = std::make_shared<my_tensor::Tensor<>>(tensor_name##_shape); \
  } catch (my_tensor::ShapeError & e) {                                       \
    std::cerr << e.what() << std::endl;                                       \
    FAIL() << "Failed to construct tensor.";                                  \
  }

#define DATA_EQUAL_TEST(device)                       \
  for (int i = 0; i < 6; i++) {                       \
    EXPECT_EQ(tensor->Get##device##Data()[i], i + 1); \
  }

#define DIFF_EQUAL_TEST(device)                 \
  EXPECT_EQ(tensor->Get##device##Diff()[0], 1); \
  EXPECT_EQ(tensor->Get##device##Diff()[1], 3); \
  EXPECT_EQ(tensor->Get##device##Diff()[2], 5); \
  EXPECT_EQ(tensor->Get##device##Diff()[3], 2); \
  EXPECT_EQ(tensor->Get##device##Diff()[4], 4); \
  EXPECT_EQ(tensor->Get##device##Diff()[5], 6);

#define TEST_DATA(test_fixture, device)                          \
  TEST_F(test_fixture##Test, test_fixture##device##_DataEqual) { \
    DATA_EQUAL_TEST(device)                                      \
  }

#define TEST_DIFF(test_fixture, device)                          \
  TEST_F(test_fixture##Test, test_fixture##device##_DiffEqual) { \
    DIFF_EQUAL_TEST(device)                                      \
  }

#define TEST_DATA_AND_DIFF_WITH_DEVICE(test_fixture, device) \
  TEST_DATA(test_fixture, device)                            \
  TEST_DIFF(test_fixture, device)

#define TEST_DATA_AND_DIFF(test_fixture)            \
  TEST_DATA_AND_DIFF_WITH_DEVICE(test_fixture, CPU) \
  TEST_DATA_AND_DIFF_WITH_DEVICE(test_fixture, GPU)

#endif  // TEST_INCLUDE_TENSOR_TENSOR_UTILS_HPP_
