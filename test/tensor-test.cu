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

class TensorSetMethodTest : public ::testing::Test {
 protected:
  void SetUp() override {
    DEFINE_DATA_AND_DIFF(data, diff)
    DEFINE_TESNOR(tensor)
  }

  std::vector<float> data;
  std::vector<float> diff;
  my_tensor::TensorPtr<> tensor;
};

TEST_F(TensorSetMethodTest, TensorSetData_Left) {
  tensor->SetData(data);
  DATA_EQUAL_TEST
}

TEST_F(TensorSetMethodTest, TensorSetData_Right) {
  tensor->SetData(std::move(data));
  DATA_EQUAL_TEST
}

TEST_F(TensorSetMethodTest, TensorSetDiff_Left) {
  tensor->SetDiff(diff);
  DIFF_EQUAL_TEST
}

TEST_F(TensorSetMethodTest, TensorSetDiff_Right) {
  tensor->SetDiff(std::move(diff));
  DIFF_EQUAL_TEST
}

class TensorCopyConstructTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::vector<int> shape = {2, 3};
    DEFINE_DATA_AND_DIFF(data, diff)
    try {
      my_tensor::TensorPtr<> another = std::make_shared<my_tensor::Tensor<>>(shape);
      another->SetData(data);
      another->SetDiff(diff);
      tensor = std::make_shared<my_tensor::Tensor<>>(*another);
    } catch (my_tensor::ShapeError& e) {
      std::cerr << e.what() << std::endl;
      FAIL() << "Failed to construct tensor.";
    }
  }

  std::vector<float> data;
  std::vector<float> diff;
  my_tensor::TensorPtr<> tensor;
};

TEST_SHAPE_AND_SIZE(TensorCopyConstruct)
TEST_DATA_AND_DIFF(TensorCopyConstruct)

class TensorMoveConstructTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::vector<int> shape = {2, 3};
    DEFINE_DATA_AND_DIFF(data, diff)
    try {
      my_tensor::TensorPtr<> another = std::make_shared<my_tensor::Tensor<>>(shape);
      another->SetData(data);
      another->SetDiff(diff);
      tensor = std::make_shared<my_tensor::Tensor<>>(std::move(*another));
    } catch (my_tensor::ShapeError& e) {
      std::cerr << e.what() << std::endl;
      FAIL() << "Failed to construct tensor.";
    }
  }

  std::vector<float> data;
  std::vector<float> diff;
  my_tensor::TensorPtr<> tensor;
};

TEST_SHAPE_AND_SIZE(TensorMoveConstruct)
TEST_DATA_AND_DIFF(TensorMoveConstruct)

class TensorCopyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    DEFINE_DATA_AND_DIFF(another_data, another_diff)
    DEFINE_TESNOR(another)
    const std::vector<int> shape = {2, 3, 4};
    data.resize(24);
    diff.resize(24);
    data = 
      {1, 3, 5, 2, 4, 6,
       7, 9, 11, 8, 10, 12,
       13, 15, 17, 14, 16, 18,
       19, 21, 23, 20, 22, 24};
    for (int i = 0; i < 24; i++) {
      diff[i] = i + 1;
    }
    try {
      another->SetData(another_data);
      another->SetDiff(another_diff);
      tensor = std::make_shared<my_tensor::Tensor<>>(shape);
      tensor->SetData(data);
      tensor->SetDiff(diff);
      *tensor = *another;
    } catch (my_tensor::ShapeError& e) {
      std::cerr << e.what() << std::endl;
      FAIL() << "Failed to construct tensor.";
    }
  }

  std::vector<float> another_data;
  std::vector<float> another_diff;
  std::vector<float> data;
  std::vector<float> diff;
  my_tensor::TensorPtr<> tensor;
  my_tensor::TensorPtr<> another;
};

TEST_SHAPE_AND_SIZE(TensorCopy)
TEST_DATA_AND_DIFF(TensorCopy)

class TensorMoveTest : public ::testing::Test {
 protected:
  void SetUp() override {
    DEFINE_DATA_AND_DIFF(another_data, another_diff)
    DEFINE_TESNOR(another)
    const std::vector<int> shape = {2, 3, 4};
    data.resize(24);
    diff.resize(24);
    data = 
      {1, 3, 5, 2, 4, 6,
       7, 9, 11, 8, 10, 12,
       13, 15, 17, 14, 16, 18,
       19, 21, 23, 20, 22, 24};
    for (int i = 0; i < 24; i++) {
      diff[i] = i + 1;
    }
    try {
      another->SetData(another_data);
      another->SetDiff(another_diff);
      tensor = std::make_shared<my_tensor::Tensor<>>(shape);
      tensor->SetData(data);
      tensor->SetDiff(diff);
      *tensor = std::move(*another);
    } catch (my_tensor::ShapeError& e) {
      std::cerr << e.what() << std::endl;
      FAIL() << "Failed to construct tensor.";
    }
  }

  std::vector<float> another_data;
  std::vector<float> another_diff;
  std::vector<float> data;
  std::vector<float> diff;
  my_tensor::TensorPtr<> tensor;
  my_tensor::TensorPtr<> another;
};

TEST_SHAPE_AND_SIZE(TensorMove)
TEST_DATA_AND_DIFF(TensorMove)

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
