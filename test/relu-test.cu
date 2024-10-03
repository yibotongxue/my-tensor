#include <gtest/gtest.h>
#include <relu.cuh>
#include <tensor.cuh>
#include <memory>
#include <vector>
#include <random>

TEST(relu_forward_test, random_on_cpu) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0f, 1.0f);
  std::vector<int> shape = { 10000 };
  std::shared_ptr<my_tensor::Tensor> bottom = std::make_shared<my_tensor::Tensor>(shape);
  float *data = bottom->GetMutableData();
  for (int i = 0; i < 10000; i++) {
    *(data + i) = dis(gen);
  }
  my_tensor::Relu relu;
  std::shared_ptr<my_tensor::Tensor> top = std::make_shared<my_tensor::Tensor>(shape);
  EXPECT_NO_THROW(relu.Forward(bottom, top));
  const float *top_data = top->GetData();
  for (int i = 0; i < 10000; i++) {
    EXPECT_EQ(*(top_data + i), max(*(data + i), 0.0f));
  }
}

TEST(relu_forward_test, random_on_gpu) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0f, 1.0f);
  std::vector<int> shape = { 10000 };
  std::shared_ptr<my_tensor::Tensor> bottom =
    std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::GPU);
  float *data = reinterpret_cast<float *>(malloc(10000 * sizeof(float)));
  for (int i = 0; i < 10000; i++) {
    *(data + i) = dis(gen);
  }
  cudaMemcpy(bottom->GetMutableData(), data, 10000 * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  my_tensor::Relu relu;
  std::shared_ptr<my_tensor::Tensor> top =
    std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::GPU);
  EXPECT_NO_THROW(relu.Forward(bottom, top));
  float *top_data = reinterpret_cast<float *>(malloc(10000 * sizeof(float)));
  cudaMemcpy(top_data, top->GetData(), 10000 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < 10000; i++) {
    EXPECT_EQ(*(top_data + i), max(*(data + i), 0.0f));
  }
  free(data);
  free(top_data);
  data = nullptr;
  top_data = nullptr;
}

TEST(relu_backward_test, random_on_cpu) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0f, 1.0f);
  std::vector<int> shape = { 10000 };
  std::shared_ptr<my_tensor::Tensor> bottom = std::make_shared<my_tensor::Tensor>(shape);
  float *data = bottom->GetMutableData();
  for (int i = 0; i < 10000; i++) {
    *(data + i) = dis(gen);
  }
  my_tensor::Relu relu;
  std::shared_ptr<my_tensor::Tensor> top = std::make_shared<my_tensor::Tensor>(shape);
  EXPECT_NO_THROW(relu.Forward(bottom, top));
  float *diff = top->GetMutableDiff();
  for (int i = 0; i < 10000; i++) {
    *(diff + i) = dis(gen);
  }
  EXPECT_NO_THROW(relu.Backward(top, bottom));
  const float *bottom_diff = bottom->GetDiff();
  for (int i = 0; i < 10000; i++) {
    if (*(data + i) > 0) {
      EXPECT_EQ(*(bottom_diff + i), *(diff + i));
    } else {
      EXPECT_EQ(*(bottom_diff + i), 0);
    }
  }
}

TEST(relu_backward_test, random_on_gpu) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0f, 1.0f);
  std::vector<int> shape = { 10000 };
  std::shared_ptr<my_tensor::Tensor> bottom =
    std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::GPU);
  float *data = reinterpret_cast<float *>(malloc(10000 * sizeof(float)));
  for (int i = 0; i < 10000; i++) {
    *(data + i) = dis(gen);
  }
  cudaMemcpy(bottom->GetMutableData(), data, 10000 * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  my_tensor::Relu relu;
  std::shared_ptr<my_tensor::Tensor> top =
    std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::GPU);
  EXPECT_NO_THROW(relu.Forward(bottom, top));
  float *diff = reinterpret_cast<float *>(malloc(10000 * sizeof(float)));
  for (int i = 0; i < 10000; i++) {
    *(diff + i) = dis(gen);
  }
  cudaMemcpy(top->GetMutableDiff(), diff, 10000 * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  EXPECT_NO_THROW(relu.Backward(top, bottom));
  float *bottom_diff = reinterpret_cast<float *>(malloc(10000 * sizeof(float)));
  cudaMemcpy(bottom_diff, bottom->GetDiff(), 10000 * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10000; i++) {
    if (*(data + i) > 0) {
      EXPECT_EQ(*(bottom_diff + i), *(diff + i));
    } else {
      EXPECT_EQ(*(bottom_diff + i), 0);
    }
  }
  free(data);
  free(diff);
  free(bottom_diff);
  data = nullptr;
  diff = nullptr;
  bottom_diff = nullptr;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
