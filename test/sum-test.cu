#include <gtest/gtest.h>

#include <sum.cuh>

TEST(sum_test, test_on_cpu) {
  std::vector<int> shape = { 1024, 2 };
  std::shared_ptr<my_tensor::Tensor> tensor = std::make_shared<my_tensor::Tensor>(shape);
  float *data = tensor->GetMutableData();
  for (int i = 0; i < 2048; ++i) {
    *(data + i) = i + 1;
  }
  float *result = new float;
  my_tensor::Sum(result, tensor);
  EXPECT_EQ(*result, 2098176.0f);
  delete result;
}

TEST(sum_test, test_on_gpu) {
  std::vector<int> shape = { 1023, 2 };
  std::shared_ptr<my_tensor::Tensor> tensor = std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::GPU);
  float *data = reinterpret_cast<float*>(malloc(2046 * sizeof(float)));
  for (int i = 0; i < 2046; ++i) {
    *(data + i) = i + 1;
  }
  cudaMemcpy(tensor->GetMutableData(), data, 2046 * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  free(data);
  float *result = nullptr;
  cudaMalloc(&result, sizeof(float));
  my_tensor::Sum(result, tensor);
  float* cpu_result = new float;
  cudaMemcpy(cpu_result, result, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(result);
  result = nullptr;
  EXPECT_EQ(*cpu_result, 2094081.0f);
  delete cpu_result;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
