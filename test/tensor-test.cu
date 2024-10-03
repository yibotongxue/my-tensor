#include <gtest/gtest.h>
#include <tensor.cuh>
#include <functional>

#define TENSOR_CONSTRUCT_ON_CPU(shape_vec, tensor_name) \
  auto tensor_name = std::make_unique<my_tensor::Tensor>(shape_vec);

#define TENSOR_CONSTRUCT_ON_GPU(shape_vec, tensor_name) \
  auto tensor_name = std::make_unique<my_tensor::Tensor>( \
    shape_vec, my_tensor::DeviceType::GPU);

#define TENSOR_CONSTRUCTOR_COPY(tensor_dst, tensor_src) \
  std::unique_ptr<my_tensor::Tensor> tensor_dst = std::make_unique<my_tensor::Tensor>(*tensor_src);

#define TENSOR_CONSTRUCTOR_MOVE(tensor_dst, tensor_src) \
  std::unique_ptr<my_tensor::Tensor> tensor_dst = std::make_unique<my_tensor::Tensor>(std::move(*tensor_src));

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

// define basic test of shape, oncpu and ongpu, and data position
#define TENSOR_SHAPE_TEST(common, device) \
TEST_F(common##device, shape_test) { \
  TENSOR_EXPECT_SHAPE(tensor, shape); \
}

#define TENSOR_ON_DEVICE_TEST(common, device) \
TEST_F(common##device, ondevice_test) { \
  TENSOR_EXPECT_ON_##device(tensor); \
}

#define TENSOR_DATA_ON_DEVICE_TEST(common, device) \
TEST_F(common##device, data_ondevice_##device) { \
  TENSOR_DATA_ON_##device(tensor); \
}

#define TENSOR_CONSTRUCT_BASIC_TEST_SPECIFIC_DEVICE(common, device) \
  TENSOR_SHAPE_TEST(common, device) \
  TENSOR_ON_DEVICE_TEST(common, device) \
  TENSOR_DATA_ON_DEVICE_TEST(common, device)

#define TENSOR_CONSTRUCT_BASIC_TEST(common) \
  TENSOR_CONSTRUCT_BASIC_TEST_SPECIFIC_DEVICE(common, CPU) \
  TENSOR_CONSTRUCT_BASIC_TEST_SPECIFIC_DEVICE(common, GPU)


// define basic test of shape, oncpu and ongpu, and data position of two devices
#define TENSOR_SHAPE_TWO_DEVICES_TEST(common, device_from, device_to) \
TEST_F(common##device_from##2##device_to, shape_test) { \
  TENSOR_EXPECT_SHAPE(tensor, shape); \
}

#define TENSOR_ON_DEVICE_TWO_DEVICES_TEST(common, device_from, device_to) \
TEST_F(common##device_from##2##device_to, ondevice_test) { \
  TENSOR_EXPECT_ON_##device_from(tensor); \
}

#define TENSOR_DATA_ON_DEVICE_TWO_DEVICES_TEST(common, device_from, device_to) \
TEST_F(common##device_from##2##device_to, data_ondevice) { \
  TENSOR_DATA_ON_##device_from(tensor); \
}

#define TENSOR_MOVE_OR_COPY_BASIC_TEST_SPECIFIC_DEVICES(common, device_from, device_to) \
  TENSOR_SHAPE_TWO_DEVICES_TEST(common, device_from, device_to) \
  TENSOR_ON_DEVICE_TWO_DEVICES_TEST(common, device_from, device_to) \
  TENSOR_DATA_ON_DEVICE_TWO_DEVICES_TEST(common, device_from, device_to)

#define TENSOR_MOVE_OR_COPY_BASIC_TEST(common) \
  TENSOR_MOVE_OR_COPY_BASIC_TEST_SPECIFIC_DEVICES(common, CPU, CPU) \
  TENSOR_MOVE_OR_COPY_BASIC_TEST_SPECIFIC_DEVICES(common, CPU, GPU) \
  TENSOR_MOVE_OR_COPY_BASIC_TEST_SPECIFIC_DEVICES(common, GPU, CPU) \
  TENSOR_MOVE_OR_COPY_BASIC_TEST_SPECIFIC_DEVICES(common, GPU, GPU)

/*************************TENSOR_TEST_CONSTRUCT**************************** */
#define TENSOR_CONSTRUCT_TEST_CLASS(device) \
  class TensorConstructTest##device : public ::testing::Test { \
   protected: \
    void SetUp() override { \
      tensor = \
        std::move(std::make_unique<my_tensor::Tensor>(shape, my_tensor::DeviceType::device)); \
    } \
    std::vector<int> shape {1, 2, 3}; \
    std::unique_ptr<my_tensor::Tensor> tensor; \
  };

TENSOR_CONSTRUCT_TEST_CLASS(CPU);
TENSOR_CONSTRUCT_TEST_CLASS(GPU);

TENSOR_CONSTRUCT_BASIC_TEST(TensorConstructTest)
/*************************TENSOR_TEST_CONSTRUCT**************************** */



/**********************TENSOR_TEST_COPY_CONSTRUCT************************** */
#define TENSOR_COPY_CONSTRUCT_TEST_CLASS(device) \
  class TensorCopyConstructTest##device : public ::testing::Test { \
   protected: \
    void SetUp() override { \
      another = \
        std::move(std::make_unique<my_tensor::Tensor>(shape, my_tensor::DeviceType::device)); \
      auto func = [](int x) -> float { return x; }; \
      float *data = another->GetMutableData(); \
      SET_DATA_ON_##device(data, 6, func); \
      tensor = \
        std::move(std::make_unique<my_tensor::Tensor>(*another)); \
    } \
    std::vector<int> shape {1, 2, 3}; \
    std::unique_ptr<my_tensor::Tensor> another; \
    std::unique_ptr<my_tensor::Tensor> tensor; \
  };

TENSOR_COPY_CONSTRUCT_TEST_CLASS(CPU);
TENSOR_COPY_CONSTRUCT_TEST_CLASS(GPU);

TENSOR_CONSTRUCT_BASIC_TEST(TensorCopyConstructTest)

#define TENSOR_COPY_CONSTRUCT_SUCCESSFULLY(device) \
TEST_F(TensorCopyConstructTest##device, data_copy) { \
  TENSOR_EXPECT_EQ_DATA_##device##_##device(another, tensor); \
}

TENSOR_COPY_CONSTRUCT_SUCCESSFULLY(CPU)
TENSOR_COPY_CONSTRUCT_SUCCESSFULLY(GPU)
/**********************TENSOR_TEST_COPY_CONSTRUCT************************** */


/***********************TENSOR_TEST_MOVE_CONSTRUCT************************* */
#define TENSOR_MOVE_CONSTRUCT_TEST_CLASS(device) \
  class TensorMoveConstructTest##device : public ::testing::Test { \
   protected: \
    void SetUp() override { \
      another = \
        std::move(std::make_unique<my_tensor::Tensor>(shape, my_tensor::DeviceType::device)); \
      auto func = [](int x) -> float { return x; }; \
      float *data = another->GetMutableData(); \
      SET_DATA_ON_##device(data, 6, func); \
      temp_data = another->GetData(); \
      tensor = \
        std::move(std::make_unique<my_tensor::Tensor>(std::move(*another))); \
    } \
    std::vector<int> shape {1, 2, 3}; \
    std::unique_ptr<my_tensor::Tensor> another; \
    std::unique_ptr<my_tensor::Tensor> tensor; \
    const float *temp_data; \
  };

TENSOR_MOVE_CONSTRUCT_TEST_CLASS(CPU)
TENSOR_MOVE_CONSTRUCT_TEST_CLASS(GPU)

TENSOR_CONSTRUCT_BASIC_TEST(TensorMoveConstructTest)

#define TENSOR_MOVE_CONSTRUCT_SUCCESSFULLY(device) \
TEST_F(TensorMoveConstructTest##device, data_move) { \
  EXPECT_EQ(tensor->GetData(), temp_data); \
  EXPECT_EQ(another->GetData(), nullptr); \
}

TENSOR_MOVE_CONSTRUCT_SUCCESSFULLY(CPU)
TENSOR_MOVE_CONSTRUCT_SUCCESSFULLY(GPU)

#define TENSOR_MOVE_SUCCESSFULY \
do { \
  EXPECT_EQ(another_data, temp_data); \
  EXPECT_EQ(tensor_data, nullptr); \
} while (0);
/***********************TENSOR_TEST_MOVE_CONSTRUCT************************* */



/****************************TENSOR_TEST_COPY****************************** */
#define TENSOR_COPY_TEST_CLASS(device_from, device_to) \
  class TensorCopyTest##device_from##2##device_to : public ::testing::Test { \
   protected: \
    void SetUp() override { \
      another = \
        std::move(std::make_unique<my_tensor::Tensor>(shape, my_tensor::DeviceType::device_from)); \
      auto func = [](int x) -> float { return x; }; \
      float *data = another->GetMutableData(); \
      SET_DATA_ON_##device_from(data, 6, func); \
      tensor = \
        std::move(std::make_unique<my_tensor::Tensor>(other_shape, my_tensor::DeviceType::device_to)); \
      *tensor = *another; \
    } \
    std::vector<int> shape {1, 2, 3}; \
    std::vector<int> other_shape {2, 3, 4}; \
    std::unique_ptr<my_tensor::Tensor> another; \
    std::unique_ptr<my_tensor::Tensor> tensor; \
  };

TENSOR_COPY_TEST_CLASS(CPU, CPU);
TENSOR_COPY_TEST_CLASS(CPU, GPU);
TENSOR_COPY_TEST_CLASS(GPU, CPU);
TENSOR_COPY_TEST_CLASS(GPU, GPU);

TENSOR_MOVE_OR_COPY_BASIC_TEST(TensorCopyTest)

#define TENSOR_COPY_SUCCESSFULLY(device_from, device_to) \
TEST_F(TensorCopyTest##device_from##2##device_to, data_copy) { \
  TENSOR_EXPECT_EQ_DATA_##device_from##_##device_from(another, tensor); \
}

TENSOR_COPY_SUCCESSFULLY(CPU, CPU);
TENSOR_COPY_SUCCESSFULLY(CPU, GPU);
TENSOR_COPY_SUCCESSFULLY(GPU, CPU);
TENSOR_COPY_SUCCESSFULLY(GPU, GPU);
/****************************TENSOR_TEST_COPY****************************** */



/***************************TENSOR_TEST_MOVE******************************* */
#define TENSOR_MOVE_TEST_CLASS(device_from, device_to) \
  class TensorMoveTest##device_from##2##device_to : public ::testing::Test { \
   protected: \
    void SetUp() override { \
      another = \
        std::move(std::make_unique<my_tensor::Tensor>(shape, my_tensor::DeviceType::device_from)); \
      auto func = [](int x) -> float { return x; }; \
      float *data = another->GetMutableData(); \
      SET_DATA_ON_##device_from(data, 6, func); \
      temp_data = another->GetData(); \
      tensor = \
        std::move(std::make_unique<my_tensor::Tensor>(other_shape, my_tensor::DeviceType::device_to)); \
      *tensor = std::move(*another); \
    } \
    std::vector<int> shape {1, 2, 3}; \
    std::vector<int> other_shape {2, 3, 4}; \
    std::unique_ptr<my_tensor::Tensor> another; \
    std::unique_ptr<my_tensor::Tensor> tensor; \
    const float *temp_data; \
  };

TENSOR_MOVE_TEST_CLASS(CPU, CPU)
TENSOR_MOVE_TEST_CLASS(CPU, GPU)
TENSOR_MOVE_TEST_CLASS(GPU, CPU)
TENSOR_MOVE_TEST_CLASS(GPU, GPU)

TENSOR_MOVE_OR_COPY_BASIC_TEST(TensorMoveTest)

#define TENSOR_MOVE_SUCCESSFULLY(device_from, device_to) \
TEST_F(TensorMoveTest##device_from##2##device_to, data_move) { \
  EXPECT_EQ(tensor->GetData(), temp_data); \
  EXPECT_EQ(another->GetData(), nullptr); \
}

TENSOR_MOVE_SUCCESSFULLY(CPU, CPU)
TENSOR_MOVE_SUCCESSFULLY(CPU, GPU)
TENSOR_MOVE_SUCCESSFULLY(GPU, CPU)
TENSOR_MOVE_SUCCESSFULLY(GPU, GPU)
/***************************TENSOR_TEST_MOVE******************************* */



/****************************TENSOR_TEST_CPU******************************* */
#define SET_UP_SIX_ELEMENTS_FOR_TEST_CPU(device) \
  std::vector<int> shape {1, 2, 3}; \
  TENSOR_CONSTRUCT_ON_##device(shape, tensor); \
  float* data = tensor->GetMutableData(); \
  auto func = [](int x) -> float { return static_cast<float>(x); }; \
  SET_DATA_ON_##device(data, 6, func); \
  auto another = std::make_unique<my_tensor::Tensor>(tensor->cpu()); \

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
  auto another = std::make_unique<my_tensor::Tensor>(tensor->gpu());

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
