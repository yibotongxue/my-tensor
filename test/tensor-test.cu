#include <gtest/gtest.h>
#include <tensor.cuh>
#include <error.h>
#include <tensor/tensor-utils.cuh>
#include <iostream>

class TensorConstructNoDataTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::vector<int> shape = {2, 3};
    try {
      tensor = std::make_shared<my_tensor::Tensor<>>(shape);
    } catch (my_tensor::ShapeError& e) {
      std::cerr << e.what() << std::endl;
      FAIL() << "Failed to construct tensor.\n";
    }
  }
  
  my_tensor::TensorPtr<> tensor;
};

TEST_SHAPE_AND_SIZE(TensorConstructNoData)

class TensorConstructWithDataTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::vector<int> shape = {2, 3};
    const std::vector<float> data = {1, 2, 3, 4, 5, 6};
    try {
      tensor = std::make_shared<my_tensor::Tensor<>>(shape, data);
    } catch (my_tensor::ShapeError& e) {
      std::cerr << e.what() << std::endl;
      FAIL() << "Failed to construct tensor.";
    }
  }
  
  my_tensor::TensorPtr<> tensor;
};

TEST_SHAPE_AND_SIZE(TensorConstructWithData)
TEST_DATA(TensorConstructWithData)

class TensorCopyConstructTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::vector<int> shape = {2, 3};
    const std::vector<float> data = {1, 2, 3, 4, 5, 6};
    try {
      my_tensor::TensorPtr<> another = std::make_shared<my_tensor::Tensor<>>(shape, data);
      tensor = std::make_shared<my_tensor::Tensor<>>(*another);
    } catch (my_tensor::ShapeError& e) {
      std::cerr << e.what() << std::endl;
      FAIL() << "Failed to construct tensor.";
    }
  }

  my_tensor::TensorPtr<> tensor;
};

TEST_SHAPE_AND_SIZE(TensorCopyConstruct)
TEST_DATA(TensorCopyConstruct)

class TensorMoveConstructTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::vector<int> shape = {2, 3};
    const std::vector<float> data = {1, 2, 3, 4, 5, 6};
    try {
      my_tensor::TensorPtr<> another = std::make_shared<my_tensor::Tensor<>>(shape, data);
      tensor = std::make_shared<my_tensor::Tensor<>>(std::move(*another));
    } catch (my_tensor::ShapeError& e) {
      std::cerr << e.what() << std::endl;
      FAIL() << "Failed to construct tensor.";
    }
  }

  my_tensor::TensorPtr<> tensor;
};

TEST_SHAPE_AND_SIZE(TensorMoveConstruct)
TEST_DATA(TensorMoveConstruct)

class TensorCopyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::vector<int> shape = {2, 3};
    const std::vector<int> another_shape = {2, 3, 4};
    const std::vector<float> data = {1, 2, 3, 4, 5, 6};
    const std::vector<float> another_data = 
      {1, 3, 5, 2, 4, 6,
       7, 9, 11, 8, 10, 12,
       13, 15, 17, 14, 16, 18,
       19, 21, 23, 20, 22, 24};
    try {
      my_tensor::TensorPtr<> another = std::make_shared<my_tensor::Tensor<>>(shape, data);
      tensor = std::make_shared<my_tensor::Tensor<>>(another_shape, another_data);
      *tensor = *another;
    } catch (my_tensor::ShapeError& e) {
      std::cerr << e.what() << std::endl;
      FAIL() << "Failed to construct tensor.";
    }
  }

  my_tensor::TensorPtr<> tensor;
};

TEST_SHAPE_AND_SIZE(TensorCopy)
TEST_DATA(TensorCopy)

class TensorMoveTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::vector<int> shape = {2, 3};
    const std::vector<int> another_shape = {2, 3, 4};
    const std::vector<float> data = {1, 2, 3, 4, 5, 6};
    const std::vector<float> another_data =
      {1, 3, 5, 2, 4, 6,
       7, 9, 11, 8, 10, 12,
       13, 15, 17, 14, 16, 18,
       19, 21, 23, 20, 22, 24};
    try {
      my_tensor::TensorPtr<> another = std::make_shared<my_tensor::Tensor<>>(shape, data);
      tensor = std::make_shared<my_tensor::Tensor<>>(another_shape, another_data);
      *tensor = std::move(*another);
    } catch (my_tensor::ShapeError& e) {
      std::cerr << e.what() << std::endl;
      FAIL() << "Failed to construct tensor.";
    }
  }

  my_tensor::TensorPtr<> tensor;
};

TEST_SHAPE_AND_SIZE(TensorMove)
TEST_DATA(TensorMove)

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
