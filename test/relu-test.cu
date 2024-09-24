#include <gtest/gtest.h>
#include <relu.cuh>
#include <tensor.cuh>
#include <memory>
#include <vector>
#include <random>

TEST(relu_forward_test, all_positive_on_cpu) {
  std::vector<int> shape {1, 2, 3, 4};
  std::shared_ptr<my_tensor::Tensor> input = std::make_shared<my_tensor::Tensor>(shape);
  float *data = input->GetMutableData();
  for (int i = 0; i < 24; i++) {
    *(data + i) = i + 1;
  }
  data = nullptr;
  my_tensor::Relu relu;
  std::shared_ptr<my_tensor::Tensor> output = std::make_shared<my_tensor::Tensor>(shape);
  relu.Forward(input, output);
  const float *output_data = output->GetData();
  for (int i = 0; i < 24; i++) {
    EXPECT_EQ(*(output_data + i), i + 1);
  }
}

TEST(relu_forward_test, all_positive_on_gpu) {
  std::vector<int> shape {1, 2, 3, 4};
  std::shared_ptr<my_tensor::Tensor> input =
    std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::GPU);
  float *data = (float*) malloc(24 * sizeof(float));
  for (int i = 0; i < 24; i++) {
    *(data + i) = i + 1;
  }
  cudaMemcpy(input->GetMutableData(), data, 24 * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  free(data);
  data = nullptr;
  my_tensor::Relu relu;
  std::shared_ptr<my_tensor::Tensor> output =
    std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::GPU);
  relu.Forward(input, output);
  float *output_data = (float*) malloc(24 * sizeof(float));
  cudaMemcpy(output_data, output->GetData(), 24 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < 24; i++) {
    EXPECT_EQ(*(output_data + i), i + 1);
  }
  free(output_data);
  output_data = nullptr;
}

TEST(relu_forward_test, all_negative_on_cpu) {
  std::vector<int> shape {1, 2, 3, 4};
  std::shared_ptr<my_tensor::Tensor> input = std::make_shared<my_tensor::Tensor>(shape);
  float *data = input->GetMutableData();
  for (int i = 0; i < 24; i++) {
    *(data + i) = -(i + 1);
  }
  data = nullptr;
  my_tensor::Relu relu;
  std::shared_ptr<my_tensor::Tensor> output = std::make_shared<my_tensor::Tensor>(shape);
  relu.Forward(input, output);
  const float *output_data = output->GetData();
  for (int i = 0; i < 24; i++) {
    EXPECT_EQ(*(output_data + i), 0);
  }
}

TEST(relu_forward_test, all_negative_on_gpu) {
  std::vector<int> shape {1, 2, 3, 4};
  std::shared_ptr<my_tensor::Tensor> input =
    std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::GPU);
  float *data = (float*) malloc(24 * sizeof(float));
  for (int i = 0; i < 24; i++) {
    *(data + i) = -(i + 1);
  }
  cudaMemcpy(input->GetMutableData(), data, 24 * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  free(data);
  data = nullptr;
  my_tensor::Relu relu;
  std::shared_ptr<my_tensor::Tensor> output =
    std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::GPU);
  relu.Forward(input, output);
  float *output_data = (float*) malloc(24 * sizeof(float));
  cudaMemcpy(output_data, output->GetData(), 24 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < 24; i++) {
    EXPECT_EQ(*(output_data + i), 0);
  }
  free(output_data);
  output_data = nullptr;
}

TEST(relu_forward_test, random_on_cpu) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0f, 1.0f);
  std::vector<int> shape = { 1000 };
  std::shared_ptr<my_tensor::Tensor> input = std::make_shared<my_tensor::Tensor>(shape);
  float *data = input->GetMutableData();
  for (int i = 0; i < 1000; i++) {
    *(data + i) = dis(gen);
  }
  my_tensor::Relu relu;
  std::shared_ptr<my_tensor::Tensor> output = std::make_shared<my_tensor::Tensor>(shape);
  relu.Forward(input, output);
  const float *output_data = output->GetData();
  for (int i = 0; i < 1000; i++) {
    EXPECT_EQ(*(output_data + i), max(*(data + i), 0.0f));
  }
}

TEST(relu_forward_test, random_on_gpu) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0f, 1.0f);
  std::vector<int> shape = { 1000 };
  std::shared_ptr<my_tensor::Tensor> input =
    std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::GPU);
  float *data = (float*) malloc(1000 * sizeof(float));
  for (int i = 0; i < 1000; i++) {
    *(data + i) = dis(gen);
  }
  cudaMemcpy(input->GetMutableData(), data, 1000 * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  my_tensor::Relu relu;
  std::shared_ptr<my_tensor::Tensor> output =
    std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::GPU);
  relu.Forward(input, output);
  float *output_data = (float*) malloc(1000 * sizeof(float));
  cudaMemcpy(output_data, output->GetData(), 1000 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < 1000; i++) {
    EXPECT_EQ(*(output_data + i), max(*(data + i), 0.0f));
  }
  free(data);
  free(output_data);
  data = nullptr;
  output_data = nullptr;
}

TEST(relu_backward_test, all_positive_on_cpu) {
  std::vector<int> shape {1, 2, 3, 4};
  std::shared_ptr<my_tensor::Tensor> bottom = std::make_shared<my_tensor::Tensor>(shape);
  float *data = bottom->GetMutableData();
  for (int i = 0; i < 24; i++) {
    *(data + i) = i + 1;
  }
  data = nullptr;
  my_tensor::Relu relu;
  std::shared_ptr<my_tensor::Tensor> top = std::make_shared<my_tensor::Tensor>(shape);
  relu.Forward(bottom, top);
  float *diff = top->GetMutableDiff();
  for (int i = 0; i < 24; i++) {
    *(diff + i) = i - 12;
  }
  diff = nullptr;
  relu.Backward(top, bottom);
  const float *bottom_diff = bottom->GetDiff();
  for (int i = 0; i < 24; i++) {
    EXPECT_EQ(*(bottom_diff + i), i - 12);
  }
  bottom_diff = nullptr;
}

TEST(relu_backward_test, all_positive_on_gpu) {
  std::vector<int> shape {1, 2, 3, 4};
  std::shared_ptr<my_tensor::Tensor> bottom =
    std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::GPU);
  float *data = (float*) malloc(24 * sizeof(float));
  for (int i = 0; i < 24; i++) {
    *(data + i) = i + 1;
  }
  cudaMemcpy(bottom->GetMutableData(), data, 24 * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  free(data);
  data = nullptr;
  my_tensor::Relu relu;
  std::shared_ptr<my_tensor::Tensor> top =
    std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::GPU);
  relu.Forward(bottom, top);
  float *diff = (float*) malloc(24 * sizeof(float));
  for (int i = 0; i < 24; i++) {
    *(diff + i) = i - 12;
  }
  cudaMemcpy(top->GetMutableDiff(), diff, 24 * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  free(diff);
  diff = nullptr;
  relu.Backward(top, bottom);
  float *bottom_diff = (float*) malloc(24 * sizeof(float));
  cudaMemcpy(bottom_diff, bottom->GetDiff(), 24 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < 24; i++) {
    EXPECT_EQ(*(bottom_diff + i), i - 12);
  }
  free(bottom_diff);
  bottom_diff = nullptr;
}

TEST(relu_backward_test, all_negative_on_cpu) {
  std::vector<int> shape {1, 2, 3, 4};
  std::shared_ptr<my_tensor::Tensor> bottom = std::make_shared<my_tensor::Tensor>(shape);
  float *data = bottom->GetMutableData();
  for (int i = 0; i < 24; i++) {
    *(data + i) = -(i + 1);
  }
  data = nullptr;
  my_tensor::Relu relu;
  std::shared_ptr<my_tensor::Tensor> top = std::make_shared<my_tensor::Tensor>(shape);
  relu.Forward(bottom, top);
  float *diff = top->GetMutableDiff();
  for (int i = 0; i < 24; i++) {
    *(diff + i) = i - 12;
  }
  diff = nullptr;
  relu.Backward(top, bottom);
  const float *bottom_diff = bottom->GetDiff();
  for (int i = 0; i < 24; i++) {
    EXPECT_EQ(*(bottom_diff + i), 0);
  }
  bottom_diff = nullptr;
}

TEST(relu_backward_test, all_negative_on_gpu) {
  std::vector<int> shape {1, 2, 3, 4};
  std::shared_ptr<my_tensor::Tensor> bottom =
    std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::GPU);
  float *data = (float*) malloc(24 * sizeof(float));
  for (int i = 0; i < 24; i++) {
    *(data + i) = -(i + 1);
  }
  cudaMemcpy(bottom->GetMutableData(), data, 24 * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  free(data);
  data = nullptr;
  my_tensor::Relu relu;
  std::shared_ptr<my_tensor::Tensor> top =
    std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::GPU);
  relu.Forward(bottom, top);
  float *diff = (float*) malloc(24 * sizeof(float));
  for (int i = 0; i < 24; i++) {
    *(diff + i) = i - 12;
  }
  cudaMemcpy(top->GetMutableDiff(), diff, 24 * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  free(diff);
  diff = nullptr;
  relu.Backward(top, bottom);
  float *bottom_diff = (float*) malloc(24 * sizeof(float));
  cudaMemcpy(bottom_diff, bottom->GetDiff(), 24 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < 24; i++) {
    EXPECT_EQ(*(bottom_diff + i), 0);
  }
  free(bottom_diff);
  bottom_diff = nullptr;
}

TEST(relu_backward_test, random_on_cpu) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0f, 1.0f);
  std::vector<int> shape = { 1000 };
  std::shared_ptr<my_tensor::Tensor> bottom = std::make_shared<my_tensor::Tensor>(shape);
  float *data = bottom->GetMutableData();
  for (int i = 0; i < 1000; i++) {
    *(data + i) = dis(gen);
  }
  my_tensor::Relu relu;
  std::shared_ptr<my_tensor::Tensor> top = std::make_shared<my_tensor::Tensor>(shape);
  relu.Forward(bottom, top);
  float *diff = top->GetMutableDiff();
  for (int i = 0; i < 1000; i++) {
    *(diff + i) = dis(gen);
  }
  relu.Backward(top, bottom);
  const float *bottom_diff = bottom->GetDiff();
  for (int i = 0; i < 1000; i++) {
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
  std::vector<int> shape = { 1000 };
  std::shared_ptr<my_tensor::Tensor> bottom =
    std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::GPU);
  float *data = (float*) malloc(1000 * sizeof(float));
  for (int i = 0; i < 1000; i++) {
    *(data + i) = dis(gen);
  }
  cudaMemcpy(bottom->GetMutableData(), data, 1000 * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  my_tensor::Relu relu;
  std::shared_ptr<my_tensor::Tensor> top =
    std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::GPU);
  relu.Forward(bottom, top);
  float *diff = (float*) malloc(1000 * sizeof(float));
  for (int i = 0; i < 1000; i++) {
    *(diff + i) = dis(gen);
  }
  cudaMemcpy(top->GetMutableDiff(), diff, 1000 * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  relu.Backward(top, bottom);
  float *bottom_diff = (float*) malloc(1000 * sizeof(float));
  cudaMemcpy(bottom_diff, bottom->GetDiff(), 1000 * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 1000; i++) {
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
