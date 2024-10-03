#include <gtest/gtest.h>
#include <relu.cuh>
#include <tensor.cuh>
#include <memory>
#include <vector>
#include <random>

#define TENSOR_CONSTRUCT_ON_CPU(shape_vec, tensor_name) \
  auto tensor_name = std::make_shared<my_tensor::Tensor>(shape_vec);

#define TENSOR_CONSTRUCT_ON_GPU(shape_vec, tensor_name) \
  auto tensor_name = std::make_shared<my_tensor::Tensor>( \
    shape_vec, my_tensor::DeviceType::GPU);

#define DEFINE_DATA_ON_CPU(data_ptr, n, func) \
  float *data_ptr = reinterpret_cast<float*>(malloc(n * sizeof(float))); \
  for (int i = 0; i < n; i++) { \
    data_ptr[i] = func(i); \
  }

#define SET_DATA_ON_CPU(data_ptr, n, func) \
  for (int i = 0; i < n; i++) { \
    data_ptr[i] = func(i); \
  }

#define SET_DATA_ON_GPU(data_ptr, n, func) \
do { \
  DEFINE_DATA_ON_CPU(temp_data, n, func); \
  cudaMemcpy(data_ptr, temp_data, n * sizeof(float), cudaMemcpyHostToDevice); \
  free(temp_data); \
} while (0);

#define DEFINE_DATA_ON_GPU_FROM_CPU(data_ptr_gpu, data_ptr_cpu, n) \
  float *data_ptr_gpu = nullptr; \
  cudaMalloc(&data_ptr_gpu, n * sizeof(float)); \
  cudaMemcpy(data_ptr_gpu, data_ptr_cpu, n * sizeof(float), cudaMemcpyHostToDevice);

#define DEFINE_DATA_ON_CPU_FROM_GPU(data_ptr_cpu, data_ptr_gpu, n) \
  float *data_ptr_cpu = reinterpret_cast<float*>(malloc(n * sizeof(float))); \
  cudaMemcpy(data_ptr_cpu, data_ptr_gpu, n * sizeof(float), cudaMemcpyDeviceToHost);

#define SETUP_FOR_RELU_POSITIVE_TEST(device) \
  std::vector<int> shape {1, 2, 3, 4}; \
  TENSOR_CONSTRUCT_ON_##device(shape, bottom); \
  float *data = bottom->GetMutableData(); \
  auto func = [](int x) -> int { return x + 1; }; \
  SET_DATA_ON_##device(data, 24, func); \
  LayerPtr relu = std::make_unique<Relu>(); \
  TENSOR_CONSTRUCT_ON_##device(shape, top); \
  EXPECT_NO_THROW(relu->Forward(bottom, top)); \
  const float *top_data = top->GetData();

TEST(relu_forward_test, all_positive_on_cpu) {
  std::vector<int> shape {1, 2, 3, 4};
  TENSOR_CONSTRUCT_ON_CPU(shape, bottom);
  float *data = bottom->GetMutableData();
  auto func = [](int x) -> int { return x + 1; };
  SET_DATA_ON_CPU(data, 24, func);
  my_tensor::Relu relu;
  TENSOR_CONSTRUCT_ON_CPU(shape, top);
  EXPECT_NO_THROW(relu.Forward(bottom, top));
  const float *top_data = top->GetData();
  for (int i = 0; i < 24; i++) {
    EXPECT_EQ(*(top_data + i), i + 1);
  }
}

TEST(relu_forward_test, all_positive_on_gpu) {
  std::vector<int> shape {1, 2, 3, 4};
  std::shared_ptr<my_tensor::Tensor> bottom =
    std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::GPU);
  float *data = reinterpret_cast<float *>(malloc(24 * sizeof(float)));
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
  EXPECT_NO_THROW(relu.Forward(bottom, top));
  float *top_data = reinterpret_cast<float *>(malloc(24 * sizeof(float)));
  cudaMemcpy(top_data, top->GetData(), 24 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < 24; i++) {
    EXPECT_EQ(*(top_data + i), i + 1);
  }
  free(top_data);
  top_data = nullptr;
}

TEST(relu_forward_test, all_negative_on_cpu) {
  std::vector<int> shape {1, 2, 3, 4};
  std::shared_ptr<my_tensor::Tensor> bottom = std::make_shared<my_tensor::Tensor>(shape);
  float *data = bottom->GetMutableData();
  for (int i = 0; i < 24; i++) {
    *(data + i) = -(i + 1);
  }
  data = nullptr;
  my_tensor::Relu relu;
  std::shared_ptr<my_tensor::Tensor> top = std::make_shared<my_tensor::Tensor>(shape);
  EXPECT_NO_THROW(relu.Forward(bottom, top));
  const float *top_data = top->GetData();
  for (int i = 0; i < 24; i++) {
    EXPECT_EQ(*(top_data + i), 0);
  }
}

TEST(relu_forward_test, all_negative_on_gpu) {
  std::vector<int> shape {1, 2, 3, 4};
  std::shared_ptr<my_tensor::Tensor> bottom =
    std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::GPU);
  float *data = reinterpret_cast<float *>(malloc(24 * sizeof(float)));
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
  EXPECT_NO_THROW(relu.Forward(bottom, top));
  float *top_data = reinterpret_cast<float *>(malloc(24 * sizeof(float)));
  cudaMemcpy(top_data, top->GetData(), 24 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < 24; i++) {
    EXPECT_EQ(*(top_data + i), 0);
  }
  free(top_data);
  top_data = nullptr;
}

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
  EXPECT_NO_THROW(relu.Forward(bottom, top));
  float *diff = top->GetMutableDiff();
  for (int i = 0; i < 24; i++) {
    *(diff + i) = i - 12;
  }
  diff = nullptr;
  EXPECT_NO_THROW(relu.Backward(top, bottom));
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
  float *data = reinterpret_cast<float *>(malloc(24 * sizeof(float)));
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
  EXPECT_NO_THROW(relu.Forward(bottom, top));
  float *diff = reinterpret_cast<float *>(malloc(24 * sizeof(float)));
  for (int i = 0; i < 24; i++) {
    *(diff + i) = i - 12;
  }
  cudaMemcpy(top->GetMutableDiff(), diff, 24 * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  free(diff);
  diff = nullptr;
  EXPECT_NO_THROW(relu.Backward(top, bottom));
  float *bottom_diff = reinterpret_cast<float *>(malloc(24 * sizeof(float)));
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
  EXPECT_NO_THROW(relu.Forward(bottom, top));
  float *diff = top->GetMutableDiff();
  for (int i = 0; i < 24; i++) {
    *(diff + i) = i - 12;
  }
  diff = nullptr;
  EXPECT_NO_THROW(relu.Backward(top, bottom));
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
  float *data = reinterpret_cast<float *>(malloc(24 * sizeof(float)));
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
  EXPECT_NO_THROW(relu.Forward(bottom, top));
  float *diff = reinterpret_cast<float *>(malloc(24 * sizeof(float)));
  for (int i = 0; i < 24; i++) {
    *(diff + i) = i - 12;
  }
  cudaMemcpy(top->GetMutableDiff(), diff, 24 * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  free(diff);
  diff = nullptr;
  EXPECT_NO_THROW(relu.Backward(top, bottom));
  float *bottom_diff = reinterpret_cast<float *>(malloc(24 * sizeof(float)));
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
