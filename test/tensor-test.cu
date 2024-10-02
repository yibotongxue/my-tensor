#include <gtest/gtest.h>
#include <tensor.cuh>

#define TENSOR_CONSTRUCT_ON_CPU(shape_vec, tensor_name) \
  auto tensor_name = std::make_shared<my_tensor::Tensor>(shape_vec);

#define TENSOR_CONSTRUCT_ON_GPU(shape_vec, tensor_name) \
  auto tensor_name = std::make_shared<my_tensor::Tensor>( \
    shape_vec, my_tensor::DeviceType::GPU);

#define TENSOR_CONSTRUCTOR_COPY(tensor_dst, tensor_src) \
  std::shared_ptr<my_tensor::Tensor> tensor_dst = std::make_shared<my_tensor::Tensor>(*tensor_src);

#define TENSOR_CONSTRUCTOR_MOVE(tensor_dst, tensor_src) \
  std::shared_ptr<my_tensor::Tensor> tensor_dst = std::make_shared<my_tensor::Tensor>(std::move(*tensor_src));

#define TENSOR_EXPECT_SHAPE(tensor_ptr, shape_vec) \
do { \
  EXPECT_EQ(tensor_ptr->GetShape(), shape_vec); \
} while (0);

#define TENSOR_EXPECT_ON_CPU(tensor_ptr) \
do { \
  EXPECT_TRUE(tensor_ptr->OnCPU()); \
  EXPECT_FALSE(tensor_ptr->OnGPU()); \
} while (0);

#define TENSOR_EXPECT_ON_GPU(tensor_ptr) \
do { \
  EXPECT_TRUE(tensor_ptr->OnGPU()); \
  EXPECT_FALSE(tensor_ptr->OnCPU()); \
} while (0);

#define TENSOR_DATA_ON_CPU(tensor_ptr) \
do { \
  std::size_t byte_size = tensor_ptr->GetByteSize(); \
  float* data = nullptr; \
  cudaMalloc(&data, byte_size); \
  cudaError_t error = \
    cudaMemcpy(data, tensor_ptr->GetData(), byte_size, cudaMemcpyHostToDevice); \
  EXPECT_EQ(error, cudaSuccess); \
  cudaFree(data); \
} while (0);

#define TENSOR_DATA_ON_GPU(tensor_ptr) \
do { \
  std::size_t byte_size = tensor_ptr->GetByteSize(); \
  float* data = nullptr; \
  cudaMalloc(&data, byte_size); \
  cudaError_t error = \
    cudaMemcpy(data, tensor_ptr->GetData(), byte_size, cudaMemcpyDeviceToDevice); \
  EXPECT_EQ(error, cudaSuccess); \
  cudaFree(data); \
} while (0);

#define DATA_EXPECT_EQ(data1, data2, n) \
do { \
  for (int i = 0; i < n; i++) { \
    EXPECT_EQ(data1[i], data2[i]); \
  } \
} while (0);

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

#define TENSOR_EXPECT_EQ_DATA_CPU_CPU(tensor_this, tensor_that) \
do { \
  int n = tensor_this->GetSize(); \
  EXPECT_EQ(tensor_that->GetSize(), n); \
  DATA_EXPECT_EQ(tensor_this->GetData(), tensor_that->GetData(), n); \
} while (0);

#define TENSOR_EXPECT_EQ_DATA_CPU_GPU(tensor_this, tensor_that) \
do { \
  int n = tensor_this->GetSize(); \
  EXPECT_EQ(tensor_that->GetSize(), n); \
  DEFINE_DATA_ON_CPU_FROM_GPU(data_that, tensor_that->GetData(), n); \
  DATA_EXPECT_EQ(tensor_this->GetData(), data_that, n); \
  free(data_that); \
} while (0);

#define TENSOR_EXPECT_EQ_DATA_GPU_CPU(tensor_this, tensor_that) \
do { \
  int n = tensor_this->GetSize(); \
  EXPECT_EQ(tensor_that->GetSize(), n); \
  DEFINE_DATA_ON_CPU_FROM_GPU(data_this, tensor_this->GetData(), n); \
  DATA_EXPECT_EQ(data_this, tensor_that->GetData(), n); \
  free(data_this); \
} while (0);

#define TENSOR_EXPECT_EQ_DATA_GPU_GPU(tensor_this, tensor_that) \
do { \
  int n = tensor_this->GetSize(); \
  EXPECT_EQ(tensor_that->GetSize(), n); \
  DEFINE_DATA_ON_CPU_FROM_GPU(data_this, tensor_this->GetData(), n); \
  DEFINE_DATA_ON_CPU_FROM_GPU(data_that, tensor_that->GetData(), n); \
  DATA_EXPECT_EQ(data_this, data_that, n); \
  free(data_this); \
  free(data_that); \
} while (0);

/*************************TENSOR_TEST_CONSTRUCT**************************** */
#define SET_UP_SIX_ELEMENTS_FOR_TEST_CONSTRUCT(device) \
  std::vector<int> shape {1, 2, 3}; \
  TENSOR_CONSTRUCT_ON_##device(shape, tensor);

TEST(tensor_test_construct, tensor_test_construct_shape_cpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_CONSTRUCT(CPU);
  TENSOR_EXPECT_SHAPE(tensor, shape);
}

TEST(tensor_test_construct, tensor_test_construct_shape_gpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_CONSTRUCT(GPU);
  TENSOR_EXPECT_SHAPE(tensor, shape);
}

TEST(tensor_test_construct, tensor_test_construct_device_cpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_CONSTRUCT(CPU);
  TENSOR_EXPECT_ON_CPU(tensor);
}

TEST(tensor_test_construct, tensor_test_construct_device_gpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_CONSTRUCT(GPU);
  TENSOR_EXPECT_ON_GPU(tensor);
}

TEST(tensor_test_construct, tensor_test_construct_data_position_cpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_CONSTRUCT(CPU);
  TENSOR_DATA_ON_CPU(tensor);
}

TEST(tensor_test_construct, tensor_test_construct_data_position_gpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_CONSTRUCT(GPU);
  TENSOR_DATA_ON_GPU(tensor);
}
/*************************TENSOR_TEST_CONSTRUCT**************************** */



/**********************TENSOR_TEST_COPY_CONSTRUCT************************** */
#define SET_UP_SIX_ELEMENTS_COPY_CONSTRUCT(device) \
  std::vector<int> shape { 1, 2, 3 }; \
  TENSOR_CONSTRUCT_ON_##device(shape, tensor); \
  float* data = tensor->GetMutableData(); \
  auto func = [](int x) -> float { return static_cast<float>(x); }; \
  SET_DATA_ON_##device(data, 6, func); \
  TENSOR_CONSTRUCTOR_COPY(another, tensor);

TEST(tensor_test_copy_construct, tensor_test_copy_construct_shape_cpu) {
  SET_UP_SIX_ELEMENTS_COPY_CONSTRUCT(CPU);
  TENSOR_EXPECT_SHAPE(another, shape);
}

TEST(tensor_test_copy_construct, tensor_test_copy_construct_shape_gpu) {
  SET_UP_SIX_ELEMENTS_COPY_CONSTRUCT(GPU);
  TENSOR_EXPECT_SHAPE(another, shape);
}

TEST(tensor_test_copy_construct, tensor_test_copy_construct_device_cpu) {
  SET_UP_SIX_ELEMENTS_COPY_CONSTRUCT(CPU);
  TENSOR_EXPECT_ON_CPU(another);
}

TEST(tensor_test_copy_construct, tensor_test_copy_construct_device_gpu) {
  SET_UP_SIX_ELEMENTS_COPY_CONSTRUCT(GPU);
  TENSOR_EXPECT_ON_GPU(another);
}

TEST(tensor_test_copy_construct, tensor_test_copy_construct_data_position_cpu) {
  SET_UP_SIX_ELEMENTS_COPY_CONSTRUCT(CPU);
  TENSOR_DATA_ON_CPU(another);
}

TEST(tensor_test_copy_construct, tensor_test_copy_construct_data_position_gpu) {
  SET_UP_SIX_ELEMENTS_COPY_CONSTRUCT(GPU);
  TENSOR_DATA_ON_GPU(another);
}

TEST(tensor_test_copy_construct, tensor_test_copy_construct_data_cpu2cpu) {
  SET_UP_SIX_ELEMENTS_COPY_CONSTRUCT(CPU);
  TENSOR_EXPECT_EQ_DATA_CPU_CPU(tensor, another);
}

TEST(tensor_test_copy_construct, tensor_test_copy_construct_data_gpu2gpu) {
  SET_UP_SIX_ELEMENTS_COPY_CONSTRUCT(GPU);
  TENSOR_EXPECT_EQ_DATA_GPU_GPU(tensor, another);
}
/**********************TENSOR_TEST_COPY_CONSTRUCT************************** */



/****************************TENSOR_TEST_COPY****************************** */
#define SET_UP_SIX_ELEMENTS_FOR_TEST_COPY(device_from, device_to) \
  std::vector<int> shape { 1, 2, 3 }; \
  std::vector<int> another_shape { 3, 2, 4 }; \
  TENSOR_CONSTRUCT_ON_##device_from(shape, tensor); \
  float* data = tensor->GetMutableData(); \
  auto func = [](int x) -> float { return static_cast<float>(x); }; \
  SET_DATA_ON_##device_from(data, 6, func); \
  TENSOR_CONSTRUCT_ON_##device_to(another_shape, another); \
  *another = *tensor;

TEST(tensor_test_copy, tensor_test_copy_shape) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_COPY(CPU, CPU);
  TENSOR_EXPECT_SHAPE(another, shape);
}

TEST(tensor_test_copy, tensor_test_copy_device_cpu2cpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_COPY(CPU, CPU);
  TENSOR_EXPECT_ON_CPU(another);
  TENSOR_DATA_ON_CPU(another);
}

TEST(tensor_test_copy, tensor_test_copy_device_cpu2gpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_COPY(CPU, GPU);
  TENSOR_EXPECT_ON_CPU(another);
  TENSOR_DATA_ON_CPU(another);
}

TEST(tensor_test_copy, tensor_test_copy_device_gpu2cpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_COPY(GPU, CPU);
  TENSOR_EXPECT_ON_GPU(another);
  TENSOR_DATA_ON_GPU(another);
}

TEST(tensor_test_copy, tensor_test_copy_device_gpu2gpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_COPY(GPU, GPU);
  TENSOR_EXPECT_ON_GPU(another);
  TENSOR_DATA_ON_GPU(another);
}

TEST(tensor_test_copy, tensor_test_copy_data_cpu2cpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_COPY(CPU, CPU);
  TENSOR_EXPECT_EQ_DATA_CPU_CPU(tensor, another);
}

TEST(tensor_test_copy, tensor_test_copy_data_cpu2gpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_COPY(CPU, GPU);
  TENSOR_EXPECT_EQ_DATA_CPU_CPU(tensor, another);
}

TEST(tensor_test_copy, tensor_test_copy_data_gpu2cpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_COPY(GPU, CPU);
  TENSOR_EXPECT_EQ_DATA_GPU_GPU(tensor, another);
}

TEST(tensor_test_copy, tensor_test_copy_data_gpu2gpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_COPY(GPU, GPU);
  TENSOR_EXPECT_EQ_DATA_GPU_GPU(tensor, another);
}
/****************************TENSOR_TEST_COPY****************************** */


/***********************TENSOR_TEST_MOVE_CONSTRUCT************************* */
#define SET_UP_SIX_ELEMENTS_FOR_TEST_MOVE_CONSTRUCT(device) \
  std::vector<int> shape {1, 2, 3}; \
  TENSOR_CONSTRUCT_ON_##device(shape, tensor); \
  const float *temp_data = tensor->GetData(); \
  TENSOR_CONSTRUCTOR_MOVE(another, tensor); \
  const float *tensor_data = tensor->GetData(); \
  const float *another_data = another->GetData();

#define TENSOR_MOVE_SUCCESSFULY \
do { \
  EXPECT_EQ(another_data, temp_data); \
  EXPECT_EQ(tensor_data, nullptr); \
} while (0);

TEST(tensor_test_move_construct, tensor_test_move_construct_cpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_MOVE_CONSTRUCT(CPU);
  TENSOR_MOVE_SUCCESSFULY;
}

TEST(tensor_test_move_construct, tensor_test_move_construct_gpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_MOVE_CONSTRUCT(GPU);
  TENSOR_MOVE_SUCCESSFULY;
}

TEST(tensor_test_move_construct, tensor_test_move_construct_shape_cpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_MOVE_CONSTRUCT(CPU);
  TENSOR_EXPECT_SHAPE(another, shape);
}

TEST(tensor_test_move_construct, tensor_test_move_construct_shape_gpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_MOVE_CONSTRUCT(GPU);
  TENSOR_EXPECT_SHAPE(another, shape);
}

TEST(tensor_test_move_construct, tensor_test_move_construct_device_cpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_MOVE_CONSTRUCT(CPU);
  TENSOR_DATA_ON_CPU(another);
}

TEST(tensor_test_move_construct, tensor_test_move_construct_device_gpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_MOVE_CONSTRUCT(GPU);
  TENSOR_DATA_ON_GPU(another);
}

TEST(tensor_test_move_construct, tensor_test_move_construct_data_position_cpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_MOVE_CONSTRUCT(CPU);
  TENSOR_DATA_ON_CPU(another);
}

TEST(tensor_test_move_construct, tensor_test_move_construct_data_position_gpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_MOVE_CONSTRUCT(GPU);
  TENSOR_DATA_ON_GPU(another);
}
/***********************TENSOR_TEST_MOVE_CONSTRUCT************************* */



/***************************TENSOR_TEST_MOVE******************************* */
#define SET_UP_SIX_ELEMENTS_TEST_MOVE(device_from, device_to) \
  std::vector<int> shape {1, 2, 3}; \
  std::vector<int> another_shape {2, 3, 4}; \
  TENSOR_CONSTRUCT_ON_##device_from(shape, tensor); \
  const float *temp_data = tensor->GetData(); \
  float* data = tensor->GetMutableData(); \
  auto func = [](int x) -> float { return static_cast<float>(x); }; \
  SET_DATA_ON_##device_from(data, 6, func); \
  TENSOR_CONSTRUCTOR_COPY(temp_tensor, tensor); \
  TENSOR_CONSTRUCT_ON_##device_to(another_shape, another); \
  *another = std::move(*tensor); \
  const float *tensor_data = tensor->GetData(); \
  const float *another_data = another->GetData();

TEST(tensor_test_move, tensor_test_move_cpu2cpu) {
  SET_UP_SIX_ELEMENTS_TEST_MOVE(CPU, CPU);
  TENSOR_MOVE_SUCCESSFULY;
}

TEST(tensor_test_move, tensor_test_move_cpu2gpu) {
  SET_UP_SIX_ELEMENTS_TEST_MOVE(CPU, GPU);
  TENSOR_MOVE_SUCCESSFULY;
}

TEST(tensor_test_move, tensor_test_move_gpu2cpu) {
  SET_UP_SIX_ELEMENTS_TEST_MOVE(GPU, CPU);
  TENSOR_MOVE_SUCCESSFULY;
}

TEST(tensor_test_move, tensor_test_move_gpu2gpu) {
  SET_UP_SIX_ELEMENTS_TEST_MOVE(GPU, GPU);
  TENSOR_MOVE_SUCCESSFULY;
}

TEST(tensor_test_move, tensor_test_move_shape) {
  SET_UP_SIX_ELEMENTS_TEST_MOVE(CPU, CPU);
  TENSOR_EXPECT_SHAPE(another, shape);
}

TEST(tensor_test_move, tensor_test_move_device_cpu2cpu) {
  SET_UP_SIX_ELEMENTS_TEST_MOVE(CPU, CPU);
  TENSOR_EXPECT_ON_CPU(another);
}

TEST(tensor_test_move, tensor_test_move_device_cpu2gpu) {
  SET_UP_SIX_ELEMENTS_TEST_MOVE(CPU, GPU);
  TENSOR_EXPECT_ON_CPU(another);
}

TEST(tensor_test_move, tensor_test_move_device_gpu2cpu) {
  SET_UP_SIX_ELEMENTS_TEST_MOVE(GPU, CPU);
  TENSOR_EXPECT_ON_GPU(another);
}

TEST(tensor_test_move, tensor_test_move_device_gpu2gpu) {
  SET_UP_SIX_ELEMENTS_TEST_MOVE(GPU, GPU);
  TENSOR_EXPECT_ON_GPU(another);
}

TEST(tensor_test_move, tensor_test_move_data_cpu2cpu) {
  SET_UP_SIX_ELEMENTS_TEST_MOVE(CPU, CPU);
  TENSOR_EXPECT_EQ_DATA_CPU_CPU(another, temp_tensor);
}

TEST(tensor_test_move, tensor_test_move_data_cpu2gpu) {
  SET_UP_SIX_ELEMENTS_TEST_MOVE(CPU, GPU);
  TENSOR_EXPECT_EQ_DATA_CPU_CPU(another, temp_tensor);
}

TEST(tensor_test_move, tensor_test_move_data_gpu2cpu) {
  SET_UP_SIX_ELEMENTS_TEST_MOVE(GPU, CPU);
  TENSOR_EXPECT_EQ_DATA_GPU_GPU(another, temp_tensor);
}

TEST(tensor_test_move, tensor_test_move_data_gpu2gpu) {
  SET_UP_SIX_ELEMENTS_TEST_MOVE(GPU, GPU);
  TENSOR_EXPECT_EQ_DATA_GPU_GPU(another, temp_tensor);
}
/***************************TENSOR_TEST_MOVE******************************* */



/****************************TENSOR_TEST_CPU******************************* */
#define SET_UP_SIX_ELEMENTS_FOR_TEST_CPU(device) \
  std::vector<int> shape {1, 2, 3}; \
  TENSOR_CONSTRUCT_ON_##device(shape, tensor); \
  float* data = tensor->GetMutableData(); \
  auto func = [](int x) -> float { return static_cast<float>(x); }; \
  SET_DATA_ON_##device(data, 6, func); \
  auto another = std::make_shared<my_tensor::Tensor>(tensor->cpu()); \

TEST(tensor_test_cpu, tensor_test_cpu_on_cpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_CPU(CPU);
  TENSOR_EXPECT_SHAPE(another, shape);
  TENSOR_EXPECT_ON_CPU(another);
}

TEST(tensor_test_cpu, tensor_test_cpu_on_gpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_CPU(GPU);
  TENSOR_EXPECT_SHAPE(another, shape);
  TENSOR_EXPECT_ON_CPU(another);
}

TEST(tensor_test_cpu, tensor_test_cpu_data_on_cpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_CPU(CPU);
  TENSOR_DATA_ON_CPU(another);
  TENSOR_EXPECT_EQ_DATA_CPU_CPU(another, tensor);
}

TEST(tensor_test_cpu, tensor_test_cpu_data_on_gpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_CPU(GPU);
  TENSOR_DATA_ON_CPU(another);
  TENSOR_EXPECT_EQ_DATA_CPU_GPU(another, tensor);
}
/****************************TENSOR_TEST_CPU******************************* */


/****************************TENSOR_TEST_GPU******************************* */
#define SET_UP_SIX_ELEMENTS_FOR_TEST_GPU(device) \
  std::vector<int> shape {1, 2, 3}; \
  TENSOR_CONSTRUCT_ON_##device(shape, tensor); \
  float* data = tensor->GetMutableData(); \
  auto func = [](int x) -> float { return static_cast<float>(x); }; \
  SET_DATA_ON_##device(data, 6, func); \
  auto another = std::make_shared<my_tensor::Tensor>(tensor->gpu());

TEST(tensor_test_gpu, tensor_test_gpu_on_cpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_GPU(CPU);
  TENSOR_EXPECT_SHAPE(another, shape);
  TENSOR_EXPECT_ON_GPU(another);
}

TEST(tensor_test_gpu, tensor_test_gpu_on_gpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_GPU(GPU);
  TENSOR_EXPECT_SHAPE(another, shape);
  TENSOR_EXPECT_ON_GPU(another);
}

TEST(tensor_test_gpu, tensor_test_gpu_data_on_cpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_GPU(CPU);
  TENSOR_DATA_ON_GPU(another);
  TENSOR_EXPECT_EQ_DATA_CPU_GPU(tensor, another);
}

TEST(tensor_test_gpu, tensor_test_gpu_data_on_gpu) {
  SET_UP_SIX_ELEMENTS_FOR_TEST_GPU(GPU);
  TENSOR_DATA_ON_GPU(another);
  TENSOR_EXPECT_EQ_DATA_GPU_GPU(tensor, another);
}
/****************************TENSOR_TEST_GPU******************************* */



int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
