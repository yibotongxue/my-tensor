#include <gtest/gtest.h>
#include <tensor.cuh>

TEST(tensor_test, tensor_test_construct) {
  my_tensor::Tensor tensor = my_tensor::Tensor({1, 2, 3});
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
