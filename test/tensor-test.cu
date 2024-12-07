// Copyright 2024 yibotongxue

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <ranges>  //NOLINT
#include <utility>
#include <vector>

#include "error.hpp"
#include "tensor.hpp"
#include "tensor/tensor-utils.hpp"

class TensorConstructNoDataTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::vector<int> shape = {2, 3};
    try {
      tensor = std::make_shared<my_tensor::Tensor<>>(shape);
    } catch (my_tensor::ShapeError &e) {
      std::cerr << e.what() << std::endl;
      FAIL() << "Failed to construct tensor.\n";
    }
  }
  my_tensor::TensorPtr<> tensor;
};

TEST_SHAPE_AND_SIZE(TensorConstructNoData)

class TensorSetMethodTest : public ::testing::Test {
 protected:
  void SetUp() override                  // NOLINT
      {DEFINE_DATA_AND_DIFF(data, diff)  // NOLINT
       DEFINE_TESNOR(tensor)}            // NOLINT

  std::vector<float> data;
  std::vector<float> diff;
  my_tensor::TensorPtr<> tensor;
};

TEST_F(TensorSetMethodTest, TensorSetCPUOnCPUData_Left) {
  tensor->SetCPUData(data);
  DATA_EQUAL_TEST(CPU)
}

TEST_F(TensorSetMethodTest, TensorSetCPUOnGPUData_Left) {
  tensor->SetCPUData(data);
  DATA_EQUAL_TEST(GPU)
}

TEST_F(TensorSetMethodTest, TensorSetGPUOnCPUData_Left) {
  tensor->SetGPUData(data);
  DATA_EQUAL_TEST(CPU)
}

TEST_F(TensorSetMethodTest, TensorSetGPUOnGPUData_Left) {
  tensor->SetCPUData(data);
  DATA_EQUAL_TEST(GPU)
}

TEST_F(TensorSetMethodTest, TensorSetCPUOnCPUDiff_Left) {
  tensor->SetCPUDiff(diff);
  DIFF_EQUAL_TEST(CPU)
}

TEST_F(TensorSetMethodTest, TensorSetCPUOnGPUDiff_Left) {
  tensor->SetCPUDiff(diff);
  DIFF_EQUAL_TEST(GPU)
}

TEST_F(TensorSetMethodTest, TensorSetGPUOnCPUDiff_Left) {
  tensor->SetGPUDiff(diff);
  DIFF_EQUAL_TEST(CPU)
}

TEST_F(TensorSetMethodTest, TensorSetGPUOnGPUDiff_Left) {
  tensor->SetCPUDiff(diff);
  DIFF_EQUAL_TEST(GPU)
}

TEST(TensorSetMethodIteratorTest, TensorSetCPUOnCPUDIterata_Data) {
  const std::vector<int> shape{3, 4};
  auto tensor = std::make_shared<my_tensor::Tensor<>>(shape);
  std::vector<float> data(14);
  float i = 0;
  std::ranges::generate(data, [&i]() -> float { return i++; });
  tensor->SetCPUData(data.begin() + 1, data.end() - 1);
  for (int j = 0; j < 12; j++) {
    EXPECT_EQ(tensor->GetCPUData()[j], static_cast<float>(j + 1));
  }
}

TEST(TensorSetMethodIteratorTest, TensorSetCPUOnGPUDIterata_Data) {
  const std::vector<int> shape{3, 4};
  auto tensor = std::make_shared<my_tensor::Tensor<>>(shape);
  std::vector<float> data(14);
  float i = 0;
  std::ranges::generate(data, [&i]() -> float { return i++; });
  tensor->SetCPUData(data.begin() + 1, data.end() - 1);
  for (int j = 0; j < 12; j++) {
    EXPECT_EQ(tensor->GetGPUData()[j], static_cast<float>(j + 1));
  }
}

TEST(TensorSetMethodIteratorTest, TensorSetGPUOnCPUDIterata_Data) {
  const std::vector<int> shape{3, 4};
  auto tensor = std::make_shared<my_tensor::Tensor<>>(shape);
  std::vector<float> data(14);
  float i = 0;
  std::ranges::generate(data, [&i]() -> float { return i++; });
  tensor->SetGPUData(data.begin() + 1, data.end() - 1);
  for (int j = 0; j < 12; j++) {
    EXPECT_EQ(tensor->GetCPUData()[j], static_cast<float>(j + 1));
  }
}

TEST(TensorSetMethodIteratorTest, TensorSetGPUOnGPUDIterata_Data) {
  const std::vector<int> shape{3, 4};
  auto tensor = std::make_shared<my_tensor::Tensor<>>(shape);
  std::vector<float> data(14);
  float i = 0;
  std::ranges::generate(data, [&i]() -> float { return i++; });
  tensor->SetGPUData(data.begin() + 1, data.end() - 1);
  for (int j = 0; j < 12; j++) {
    EXPECT_EQ(tensor->GetGPUData()[j], static_cast<float>(j + 1));
  }
}

TEST(TensorSetMethodIteratorTest, TensorSetCPUOnCPUDIterata_Diff) {
  const std::vector<int> shape{3, 4};
  auto tensor = std::make_shared<my_tensor::Tensor<>>(shape);
  std::vector<float> diff(14);
  float i = 0;
  std::ranges::generate(diff, [&i]() -> float { return i++; });
  tensor->SetCPUDiff(diff.begin() + 1, diff.end() - 1);
  for (int j = 0; j < 12; j++) {
    EXPECT_EQ(tensor->GetCPUDiff()[j], static_cast<float>(j + 1));
  }
}

TEST(TensorSetMethodIteratorTest, TensorSetCPUOnGPUDIterata_Diff) {
  const std::vector<int> shape{3, 4};
  auto tensor = std::make_shared<my_tensor::Tensor<>>(shape);
  std::vector<float> diff(14);
  float i = 0;
  std::ranges::generate(diff, [&i]() -> float { return i++; });
  tensor->SetCPUDiff(diff.begin() + 1, diff.end() - 1);
  for (int j = 0; j < 12; j++) {
    EXPECT_EQ(tensor->GetGPUDiff()[j], static_cast<float>(j + 1));
  }
}

TEST(TensorSetMethodIteratorTest, TensorSetGPUOnCPUDIterata_Diff) {
  const std::vector<int> shape{3, 4};
  auto tensor = std::make_shared<my_tensor::Tensor<>>(shape);
  std::vector<float> diff(14);
  float i = 0;
  std::ranges::generate(diff, [&i]() -> float { return i++; });
  tensor->SetGPUDiff(diff.begin() + 1, diff.end() - 1);
  for (int j = 0; j < 12; j++) {
    EXPECT_EQ(tensor->GetCPUDiff()[j], static_cast<float>(j + 1));
  }
}

TEST(TensorSetMethodIteratorTest, TensorSetGPUOnGPUDIterata_Diff) {
  const std::vector<int> shape{3, 4};
  auto tensor = std::make_shared<my_tensor::Tensor<>>(shape);
  std::vector<float> diff(14);
  float i = 0;
  std::ranges::generate(diff, [&i]() -> float { return i++; });
  tensor->SetGPUDiff(diff.begin() + 1, diff.end() - 1);
  for (int j = 0; j < 12; j++) {
    EXPECT_EQ(tensor->GetGPUDiff()[j], static_cast<float>(j + 1));
  }
}

#define TENSOR_COPY_CONSTRUCT_CLASS(device)                          \
  class TensorCopyConstruct##device##Test : public ::testing::Test { \
   protected:                                                        \
    void SetUp() override {                                          \
      const std::vector<int> shape = {2, 3};                         \
      DEFINE_DATA_AND_DIFF(data, diff)                               \
      try {                                                          \
        my_tensor::TensorPtr<> another =                             \
            std::make_shared<my_tensor::Tensor<>>(shape);            \
        another->Set##device##Data(data);                            \
        another->Set##device##Diff(diff);                            \
        tensor = std::make_shared<my_tensor::Tensor<>>(*another);    \
      } catch (my_tensor::ShapeError & e) {                          \
        std::cerr << e.what() << std::endl;                          \
        FAIL() << "Failed to construct tensor.";                     \
      }                                                              \
    }                                                                \
    std::vector<float> data;                                         \
    std::vector<float> diff;                                         \
    my_tensor::TensorPtr<> tensor;                                   \
  };

TENSOR_COPY_CONSTRUCT_CLASS(CPU)
TENSOR_COPY_CONSTRUCT_CLASS(GPU)

TEST_SHAPE_AND_SIZE(TensorCopyConstructCPU)
TEST_SHAPE_AND_SIZE(TensorCopyConstructGPU)
TEST_DATA_AND_DIFF(TensorCopyConstructCPU)
TEST_DATA_AND_DIFF(TensorCopyConstructGPU)

#define TENSOR_MOVE_CONSTRUCT_CLASS(device)                                  \
  class TensorMoveConstruct##device##Test : public ::testing::Test {         \
   protected:                                                                \
    void SetUp() override {                                                  \
      const std::vector<int> shape = {2, 3};                                 \
      DEFINE_DATA_AND_DIFF(data, diff)                                       \
      try {                                                                  \
        my_tensor::TensorPtr<> another =                                     \
            std::make_shared<my_tensor::Tensor<>>(shape);                    \
        another->Set##device##Data(data);                                    \
        another->Set##device##Diff(diff);                                    \
        tensor = std::make_shared<my_tensor::Tensor<>>(std::move(*another)); \
      } catch (my_tensor::ShapeError & e) {                                  \
        std::cerr << e.what() << std::endl;                                  \
        FAIL() << "Failed to construct tensor.";                             \
      }                                                                      \
    }                                                                        \
    std::vector<float> data;                                                 \
    std::vector<float> diff;                                                 \
    my_tensor::TensorPtr<> tensor;                                           \
  };

TENSOR_MOVE_CONSTRUCT_CLASS(CPU)
TENSOR_MOVE_CONSTRUCT_CLASS(GPU)

TEST_SHAPE_AND_SIZE(TensorMoveConstructCPU)
TEST_SHAPE_AND_SIZE(TensorMoveConstructGPU)
TEST_DATA_AND_DIFF(TensorMoveConstructCPU)
TEST_DATA_AND_DIFF(TensorMoveConstructGPU)

#define TENSOR_COPY_CLASS(device_from, device_to)                              \
  class TensorCopy##device_from##2##device_to##Test : public ::testing::Test { \
   protected:                                                                  \
    void SetUp() override {                                                    \
      DEFINE_DATA_AND_DIFF(another_data, another_diff)                         \
      DEFINE_TESNOR(another)                                                   \
      const std::vector<int> shape = {2, 3, 4};                                \
      data.resize(24);                                                         \
      diff.resize(24);                                                         \
      data = {1,  3,  5,  2,  4,  6,  7,  9,  11, 8,  10, 12,                  \
              13, 15, 17, 14, 16, 18, 19, 21, 23, 20, 22, 24};                 \
      for (int i = 0; i < 24; i++) {                                           \
        diff[i] = i + 1;                                                       \
      }                                                                        \
      try {                                                                    \
        another->Set##device_from##Data(another_data);                         \
        another->Set##device_from##Diff(another_diff);                         \
        tensor = std::make_shared<my_tensor::Tensor<>>(shape);                 \
        tensor->Set##device_to##Data(data);                                    \
        tensor->Set##device_to##Diff(diff);                                    \
        *tensor = *another;                                                    \
      } catch (my_tensor::ShapeError & e) {                                    \
        std::cerr << e.what() << std::endl;                                    \
        FAIL() << "Failed to construct tensor.";                               \
      }                                                                        \
    }                                                                          \
    std::vector<float> another_data;                                           \
    std::vector<float> another_diff;                                           \
    std::vector<float> data;                                                   \
    std::vector<float> diff;                                                   \
    my_tensor::TensorPtr<> tensor;                                             \
    my_tensor::TensorPtr<> another;                                            \
  };

TENSOR_COPY_CLASS(CPU, CPU)
TENSOR_COPY_CLASS(CPU, GPU)
TENSOR_COPY_CLASS(GPU, CPU)
TENSOR_COPY_CLASS(GPU, GPU)

TEST_SHAPE_AND_SIZE(TensorCopyCPU2CPU)
TEST_SHAPE_AND_SIZE(TensorCopyCPU2GPU)
TEST_SHAPE_AND_SIZE(TensorCopyGPU2CPU)
TEST_SHAPE_AND_SIZE(TensorCopyGPU2GPU)
TEST_DATA_AND_DIFF(TensorCopyCPU2CPU)
TEST_DATA_AND_DIFF(TensorCopyCPU2GPU)
TEST_DATA_AND_DIFF(TensorCopyGPU2CPU)
TEST_DATA_AND_DIFF(TensorCopyGPU2GPU)

#define TENSOR_MOVE_CLASS(device_from, device_to)                              \
  class TensorMove##device_from##2##device_to##Test : public ::testing::Test { \
   protected:                                                                  \
    void SetUp() override {                                                    \
      DEFINE_DATA_AND_DIFF(another_data, another_diff)                         \
      DEFINE_TESNOR(another)                                                   \
      const std::vector<int> shape = {2, 3, 4};                                \
      data.resize(24);                                                         \
      diff.resize(24);                                                         \
      data = {1,  3,  5,  2,  4,  6,  7,  9,  11, 8,  10, 12,                  \
              13, 15, 17, 14, 16, 18, 19, 21, 23, 20, 22, 24};                 \
      for (int i = 0; i < 24; i++) {                                           \
        diff[i] = i + 1;                                                       \
      }                                                                        \
      try {                                                                    \
        another->Set##device_from##Data(another_data);                         \
        another->Set##device_from##Diff(another_diff);                         \
        tensor = std::make_shared<my_tensor::Tensor<>>(shape);                 \
        tensor->Set##device_to##Data(data);                                    \
        tensor->Set##device_to##Diff(diff);                                    \
        *tensor = std::move(*another);                                         \
      } catch (my_tensor::ShapeError & e) {                                    \
        std::cerr << e.what() << std::endl;                                    \
        FAIL() << "Failed to construct tensor.";                               \
      }                                                                        \
    }                                                                          \
    std::vector<float> another_data;                                           \
    std::vector<float> another_diff;                                           \
    std::vector<float> data;                                                   \
    std::vector<float> diff;                                                   \
    my_tensor::TensorPtr<> tensor;                                             \
    my_tensor::TensorPtr<> another;                                            \
  };

TENSOR_MOVE_CLASS(CPU, CPU)
TENSOR_MOVE_CLASS(CPU, GPU)
TENSOR_MOVE_CLASS(GPU, CPU)
TENSOR_MOVE_CLASS(GPU, GPU)

TEST_SHAPE_AND_SIZE(TensorMoveCPU2CPU)
TEST_SHAPE_AND_SIZE(TensorMoveCPU2GPU)
TEST_SHAPE_AND_SIZE(TensorMoveGPU2CPU)
TEST_SHAPE_AND_SIZE(TensorMoveGPU2GPU)
TEST_DATA_AND_DIFF(TensorMoveCPU2CPU)
TEST_DATA_AND_DIFF(TensorMoveCPU2GPU)
TEST_DATA_AND_DIFF(TensorMoveGPU2CPU)
TEST_DATA_AND_DIFF(TensorMoveGPU2GPU)

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
