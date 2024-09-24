#include <gtest/gtest.h>
#include <tensor.cuh>


/*************************TENSOR_TEST_CONSTRUCT**************************** */
TEST(tensor_test_construct, tensor_test_construct_shape_cpu) {
  std::vector<int> shape {1, 2, 3};
  my_tensor::Tensor tensor { shape };
  EXPECT_EQ(tensor.GetShape(), shape);
}

TEST(tensor_test_construct, tensor_test_construct_shape_gpu) {
  std::vector<int> shape {1, 2, 3};
  my_tensor::Tensor tensor { shape, my_tensor::DeviceType::GPU };
  EXPECT_EQ(tensor.GetShape(), shape);
}

TEST(tensor_test_construct, tensor_test_construct_device_cpu) {
  std::vector<int> shape {1, 2, 3};
  my_tensor::Tensor tensor { shape };
  EXPECT_TRUE(tensor.OnCPU());
  EXPECT_FALSE(tensor.OnGPU());
}

TEST(tensor_test_construct, tensor_test_construct_device_explicit_cpu) {
  std::vector<int> shape {1, 2, 3};
  my_tensor::Tensor tensor { shape, my_tensor::DeviceType::CPU };
  EXPECT_TRUE(tensor.OnCPU());
  EXPECT_FALSE(tensor.OnGPU());
}

TEST(tensor_test_construct, tensor_test_construct_device_gpu) {
  std::vector<int> shape {1, 2, 3};
  my_tensor::Tensor tensor { shape, my_tensor::DeviceType::GPU };
  EXPECT_FALSE(tensor.OnCPU());
  EXPECT_TRUE(tensor.OnGPU());
}

TEST(tensor_test_construct, tensor_test_construct_data_position_cpu) {
  std::vector<int> shape { 1, 3, 2 };
  my_tensor::Tensor tensor { shape };
  float* data = nullptr;
  cudaMalloc(&data, 6 * sizeof(float));
  cudaError_t error =
    cudaMemcpy(data, tensor.GetData(), 6 * sizeof(float), cudaMemcpyHostToDevice);
  EXPECT_EQ(error, cudaSuccess);
  cudaFree(data);
}

TEST(tensor_test_construct, tensor_test_construct_data_position_gpu) {
  std::vector<int> shape { 1, 3, 2 };
  my_tensor::Tensor tensor { shape, my_tensor::DeviceType::GPU };
  float* data = nullptr;
  cudaMalloc(&data, 6 * sizeof(float));
  cudaError_t error = 
    cudaMemcpy(data, tensor.GetData(), 6 * sizeof(float), cudaMemcpyDeviceToDevice);
  EXPECT_EQ(error, cudaSuccess);
  cudaFree(data);
}
/*************************TENSOR_TEST_CONSTRUCT**************************** */



/**********************TENSOR_TEST_COPY_CONSTRUCT************************** */
TEST(tensor_test_copy_construct, tensor_test_copy_construct_shape_cpu) {
  std::vector<int> shape { 1, 2, 3 };
  my_tensor::Tensor tensor { shape };
  my_tensor::Tensor another { tensor };
  EXPECT_EQ(another.GetShape(), shape);
}

TEST(tensor_test_copy_construct, tensor_test_copy_construct_shape_gpu) {
  std::vector<int> shape { 1, 2, 3 };
  my_tensor::Tensor tensor { shape, my_tensor::DeviceType::GPU };
  my_tensor::Tensor another { tensor };
  EXPECT_EQ(another.GetShape(), shape);
}

TEST(tensor_test_copy_construct, tensor_test_copy_construct_device_cpu) {
  std::vector<int> shape { 1, 2, 3 };
  my_tensor::Tensor tensor { shape };
  my_tensor::Tensor another { tensor };
  EXPECT_TRUE(another.OnCPU());
  EXPECT_FALSE(another.OnGPU());
}

TEST(tensor_test_copy_construct, tensor_test_copy_construct_device_explicit_cpu) {
  std::vector<int> shape { 1, 2, 3 };
  my_tensor::Tensor tensor { shape, my_tensor::DeviceType::CPU };
  my_tensor::Tensor another { tensor };
  EXPECT_TRUE(another.OnCPU());
  EXPECT_FALSE(another.OnGPU());
}

TEST(tensor_test_copy_construct, tensor_test_copy_construct_device_gpu) {
  std::vector<int> shape { 1, 2, 3 };
  my_tensor::Tensor tensor { shape, my_tensor::DeviceType::GPU };
  my_tensor::Tensor another { tensor };
  EXPECT_FALSE(another.OnCPU());
  EXPECT_TRUE(another.OnGPU());
}

TEST(tensor_test_copy_construct, tensor_test_copy_construct_data_position_cpu) {
  std::vector<int> shape { 1, 2, 3 };
  my_tensor::Tensor tensor { shape };
  my_tensor::Tensor another { tensor };
  float* data = nullptr;
  cudaMalloc(&data, 6 * sizeof(float));
  cudaError_t error =
    cudaMemcpy(data, another.GetData(), 6 * sizeof(float), cudaMemcpyHostToDevice);
  EXPECT_EQ(error, cudaSuccess);
  cudaFree(data);
}

TEST(tensor_test_copy_construct, tensor_test_copy_construct_data_position_gpu) {
  std::vector<int> shape { 1, 2, 3 };
  my_tensor::Tensor tensor { shape, my_tensor::DeviceType::GPU };
  my_tensor::Tensor another { tensor };
  float* data = nullptr;
  cudaMalloc(&data, 6 * sizeof(float));
  cudaError_t error = 
    cudaMemcpy(data, another.GetData(), 6 * sizeof(float), cudaMemcpyDeviceToDevice);
  EXPECT_EQ(error, cudaSuccess);
  cudaFree(data);
}
/**********************TENSOR_TEST_COPY_CONSTRUCT************************** */



/****************************TENSOR_TEST_COPY****************************** */
TEST(tensor_test_copy, tensor_test_copy_shape) {
  std::vector<int> shape { 1, 2, 3 };
  std::vector<int> another_shape { 3, 2, 4 };
  my_tensor::Tensor tensor { shape };
  my_tensor::Tensor another { another_shape };
  another = tensor;
  EXPECT_EQ(another.GetShape(), shape);
}

TEST(tensor_test_copy, tensor_test_copy_device_cpu2cpu) {
  std::vector<int> shape { 1, 2, 3 };
  std::vector<int> another_shape { 3, 2, 4 };
  my_tensor::Tensor tensor { shape };
  my_tensor::Tensor another { another_shape };
  another = tensor;
  EXPECT_TRUE(another.OnCPU());
  EXPECT_FALSE(another.OnGPU());
  float* data = nullptr;
  cudaMalloc(&data, 6 * sizeof(float));
  cudaError_t error =
    cudaMemcpy(data, another.GetData(), 6 * sizeof(float), cudaMemcpyHostToDevice);
  EXPECT_EQ(error, cudaSuccess);
  cudaFree(data);
}

TEST(tensor_test_copy, tensor_test_copy_device_cpu2gpu) {
  std::vector<int> shape { 1, 2, 3 };
  std::vector<int> another_shape { 3, 2, 4 };
  my_tensor::Tensor tensor { shape };
  my_tensor::Tensor another { another_shape, my_tensor::DeviceType::GPU };
  another = tensor;
  EXPECT_TRUE(another.OnCPU());
  EXPECT_FALSE(another.OnGPU());
  float* data = nullptr;
  cudaMalloc(&data, 6 * sizeof(float));
  cudaError_t error =
    cudaMemcpy(data, another.GetData(), 6 * sizeof(float), cudaMemcpyHostToDevice);
  EXPECT_EQ(error, cudaSuccess);
  cudaFree(data);
}

TEST(tensor_test_copy, tensor_test_copy_device_gpu2cpu) {
  std::vector<int> shape { 1, 2, 3 };
  std::vector<int> another_shape { 3, 2, 4 };
  my_tensor::Tensor tensor { shape, my_tensor::DeviceType::GPU };
  my_tensor::Tensor another { another_shape };
  another = tensor;
  EXPECT_TRUE(another.OnGPU());
  EXPECT_FALSE(another.OnCPU());
  float* data = nullptr;
  cudaMalloc(&data, 6 * sizeof(float));
  cudaError_t error = 
    cudaMemcpy(data, another.GetData(), 6 * sizeof(float), cudaMemcpyDeviceToDevice);
  EXPECT_EQ(error, cudaSuccess);
  cudaFree(data);
}

TEST(tensor_test_copy, tensor_test_copy_device_gpu2gpu) {
  std::vector<int> shape { 1, 2, 3 };
  std::vector<int> another_shape { 3, 2, 4 };
  my_tensor::Tensor tensor { shape, my_tensor::DeviceType::GPU };
  my_tensor::Tensor another { another_shape, my_tensor::DeviceType::GPU };
  another = tensor;
  EXPECT_TRUE(another.OnGPU());
  EXPECT_FALSE(another.OnCPU());
  float* data = nullptr;
  cudaMalloc(&data, 6 * sizeof(float));
  cudaError_t error = 
    cudaMemcpy(data, another.GetData(), 6 * sizeof(float), cudaMemcpyDeviceToDevice);
  EXPECT_EQ(error, cudaSuccess);
  cudaFree(data);
}

TEST(tensor_test_copy, tensor_test_copy_data_cpu2cpu) {
  std::vector<int> shape { 1, 2, 3 };
  std::vector<int> another_shape { 3, 2, 4 };
  my_tensor::Tensor tensor { shape };
  float* data = tensor.GetMutableData();
  for (int i = 0; i < 6; ++i) {
    *(data + i) = static_cast<float>(i);
  }
  my_tensor::Tensor another { another_shape };
  another = tensor;
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(*(another.GetData() + i), static_cast<float>(i));
  }
}

TEST(tensor_test_copy, tensor_test_copy_data_cpu2gpu) {
  std::vector<int> shape { 1, 2, 3 };
  std::vector<int> another_shape { 3, 2, 4 };
  my_tensor::Tensor tensor { shape };
  float* data = tensor.GetMutableData();
  for (int i = 0; i < 6; ++i) {
    *(data + i) = static_cast<float>(i);
  }
  my_tensor::Tensor another { another_shape, my_tensor::DeviceType::GPU };
  another = tensor;
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(*(another.GetData() + i), static_cast<float>(i));
  }
}

TEST(tensor_test_copy, tensor_test_copy_data_gpu2cpu) {
  std::vector<int> shape { 1, 2, 3 };
  std::vector<int> another_shape { 3, 2, 4 };
  my_tensor::Tensor tensor { shape, my_tensor::DeviceType::GPU };
  float* data = (float*) malloc(6 * sizeof(float));
  for (int i = 0; i < 6; ++i) {
    *(data + i) = (float)i;
  }
  cudaMemcpy(tensor.GetMutableData(), data, 6 * sizeof(float), cudaMemcpyHostToDevice);
  free(data);
  data = nullptr;
  my_tensor::Tensor another { another_shape };
  another = tensor;
  float *another_data = (float*) malloc(6 * sizeof(float));
  cudaMemcpy(another_data, another.GetData(), 6 * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(*(another_data + i), static_cast<float>(i));
  }
  free(another_data);
}

TEST(tensor_test_copy, tensor_test_copy_data_gpu2gpu) {
  std::vector<int> shape { 1, 2, 3 };
  std::vector<int> another_shape { 3, 2, 4 };
  my_tensor::Tensor tensor { shape, my_tensor::DeviceType::GPU };
  float* data = (float*) malloc(6 * sizeof(float));
  for (int i = 0; i < 6; ++i) {
    *(data + i) = (float)i;
  }
  cudaMemcpy(tensor.GetMutableData(), data, 6 * sizeof(float), cudaMemcpyHostToDevice);
  free(data);
  data = nullptr;
  my_tensor::Tensor another { another_shape, my_tensor::DeviceType::GPU };
  another = tensor;
  float *another_data = (float*) malloc(6 * sizeof(float));
  cudaMemcpy(another_data, another.GetData(), 6 * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(*(another_data + i), static_cast<float>(i));
  }
  free(another_data);
}
/****************************TENSOR_TEST_COPY****************************** */



int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
