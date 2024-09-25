#include <gtest/gtest.h>
#include <sigmoid.cuh>
#include <tensor.cuh>
#include <memory>
#include <vector>
#include <random>

TEST(sigmoid_forward_test, random_on_cpu) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0f, 1.0f);
  std::vector<int> shape = { 10000 };
  std::shared_ptr<my_tensor::Tensor> bottom = std::make_shared<my_tensor::Tensor>(shape);
  float *bottom_data = bottom->GetMutableData();
  for (int i = 0; i < 10000; i++) {
    *(bottom_data + i) = dis(gen);
  }
  std::shared_ptr<my_tensor::Tensor> top = std::make_shared<my_tensor::Tensor>(shape);
  my_tensor::Sigmoid sigmoid;
  EXPECT_NO_THROW(sigmoid.Forward(bottom, top));
  const float *top_data = top->GetData();
  for (int i = 0; i < 10000; i++) {
    const float actual = *(top_data + i);
    const float expect = 1.0f / (1.0f + std::exp(-*(bottom_data + i)));
    const float loss =std::abs((actual - expect) / max(std::abs(expect), (float)1e-6));
    EXPECT_LT(loss, 1e-2);
  }
  bottom_data = nullptr;
  top_data = nullptr;
}

TEST(sigmoid_forward_test, random_on_gpu) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0f, 1.0f);
  std::vector<int> shape = { 10000 };
  std::shared_ptr<my_tensor::Tensor> bottom =
    std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::GPU);
  float *bottom_data = reinterpret_cast<float *>(malloc(10000 * sizeof(float)));
  for (int i = 0; i < 10000; i++) {
    *(bottom_data + i) = dis(gen);
  }
  cudaMemcpy(bottom->GetMutableData(), bottom_data, 10000 * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  std::shared_ptr<my_tensor::Tensor> top =
    std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::GPU);
  my_tensor::Sigmoid sigmoid;
  EXPECT_NO_THROW(sigmoid.Forward(bottom, top));
  float *top_data = reinterpret_cast<float *>(malloc(10000 * sizeof(float)));
  cudaMemcpy(top_data, top->GetData(), 10000 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < 10000; i++) {
    const float actual = *(top_data + i);
    const float expect = 1.0f / (1.0f + std::exp(-*(bottom_data + i)));
    const float loss = std::abs((actual - expect) / max(std::abs(expect), (float)1e-6));
    EXPECT_LT(loss, 1e-2);
  }
  free(bottom_data);
  free(top_data);
  bottom_data = nullptr;
  top_data = nullptr;
}

TEST(sigmoid_backward_test, random_on_cpu) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0f, 1.0f);
  std::vector<int> shape = { 10000 };
  std::shared_ptr<my_tensor::Tensor> bottom = std::make_shared<my_tensor::Tensor>(shape);
  float *bottom_data = bottom->GetMutableData();
  for (int i = 0; i < 10000; i++) {
    *(bottom_data + i) = dis(gen);
  }
  std::shared_ptr<my_tensor::Tensor> top = std::make_shared<my_tensor::Tensor>(shape);
  my_tensor::Sigmoid sigmoid;
  EXPECT_NO_THROW(sigmoid.Forward(bottom, top));
  bottom_data = nullptr;
  const float *top_data = top->GetData();
  float* top_diff = top->GetMutableDiff();
  for (int i = 0; i < 10000; i++) {
    *(top_diff + i) = dis(gen);
  }
  EXPECT_NO_THROW(sigmoid.Backward(top, bottom));
  const float *bottom_diff = bottom->GetDiff();
  for (int i = 0; i < 10000; i++) {
    const float actual = *(bottom_diff + i);
    const float expect = *(top_diff + i) * *(top_data + i) * (1 - *(top_data + i));
    const float loss = std::abs((actual - expect) / max(std::abs(expect), (float)1e-6));
    EXPECT_LT(loss, 1e-2);
  }
  top_data = nullptr;
  bottom_diff = nullptr;
}

TEST(sigmoid_backward_test, random_on_gpu) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0f, 1.0f);
  std::vector<int> shape = { 10000 };
  std::shared_ptr<my_tensor::Tensor> bottom =
    std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::GPU);
  float *bottom_data = reinterpret_cast<float *>(malloc(10000 * sizeof(float)));
  for (int i = 0; i < 10000; i++) {
    *(bottom_data + i) = dis(gen);
  }
  cudaMemcpy(bottom->GetMutableData(), bottom_data, 10000 * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  free(bottom_data);
  bottom_data = nullptr;
  std::shared_ptr<my_tensor::Tensor> top =
    std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::GPU);
  my_tensor::Sigmoid sigmoid;
  EXPECT_NO_THROW(sigmoid.Forward(bottom, top));
  float *top_data = reinterpret_cast<float *>(malloc(10000 * sizeof(float)));
  cudaMemcpy(top_data, top->GetData(), 10000 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  float *top_diff = reinterpret_cast<float *>(malloc(10000 * sizeof(float)));
  for (int i = 0; i < 10000; i++) {
    *(top_diff + i) = dis(gen);
  }
  cudaMemcpy(top->GetMutableDiff(), top_diff, 10000 * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  EXPECT_NO_THROW(sigmoid.Backward(top, bottom));
  float *bottom_diff = reinterpret_cast<float *>(malloc(10000 * sizeof(float)));
  cudaMemcpy(bottom_diff, bottom->GetDiff(), 10000 * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10000; i++) {
    const float actual = *(bottom_diff + i);
    const float expect = *(top_diff + i) * *(top_data + i) * (1 - *(top_data + i));
    const float loss = std::abs((actual - expect) / max(std::abs(expect), (float)1e-6));
    EXPECT_LT(loss, 1e-2);
  }
  free(top_data);
  free(top_diff);
  free(bottom_diff);
  top_data = nullptr;
  top_diff = nullptr;
  bottom_diff = nullptr;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}