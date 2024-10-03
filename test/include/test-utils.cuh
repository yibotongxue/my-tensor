#ifndef TEST_TEST_UTILS_CUH_
#define TEST_TEST_UTILS_CUH_

#define TENSOR_CONSTRUCT_ON_CPU(shape_vec, tensor_name) \
  auto tensor_name = std::make_unique<my_tensor::Tensor>(shape_vec);

#define TENSOR_CONSTRUCT_ON_GPU(shape_vec, tensor_name)   \
  auto tensor_name = std::make_unique<my_tensor::Tensor>( \
      shape_vec, my_tensor::DeviceType::GPU);

#define TENSOR_CONSTRUCTOR_COPY(tensor_dst, tensor_src) \
  std::unique_ptr<my_tensor::Tensor> tensor_dst = std::make_unique<my_tensor::Tensor>(*tensor_src);

#define TENSOR_CONSTRUCTOR_MOVE(tensor_dst, tensor_src) \
  std::unique_ptr<my_tensor::Tensor> tensor_dst = std::make_unique<my_tensor::Tensor>(std::move(*tensor_src));

#define TENSOR_EXPECT_SHAPE(tensor_ptr, shape_vec) \
  do                                               \
  {                                                \
    EXPECT_EQ(tensor_ptr->GetShape(), shape_vec);  \
  } while (0);

#define TENSOR_EXPECT_ON_CPU(tensor_ptr) \
  do                                     \
  {                                      \
    EXPECT_TRUE(tensor_ptr->OnCPU());    \
    EXPECT_FALSE(tensor_ptr->OnGPU());   \
  } while (0);

#define TENSOR_EXPECT_ON_GPU(tensor_ptr) \
  do                                     \
  {                                      \
    EXPECT_TRUE(tensor_ptr->OnGPU());    \
    EXPECT_FALSE(tensor_ptr->OnCPU());   \
  } while (0);

#define TENSOR_DATA_ON_CPU(tensor_ptr)                                              \
  do                                                                                \
  {                                                                                 \
    std::size_t byte_size = tensor_ptr->GetByteSize();                              \
    float *data = nullptr;                                                          \
    cudaMalloc(&data, byte_size);                                                   \
    cudaError_t error =                                                             \
        cudaMemcpy(data, tensor_ptr->GetData(), byte_size, cudaMemcpyHostToDevice); \
    EXPECT_EQ(error, cudaSuccess);                                                  \
    cudaFree(data);                                                                 \
    cudaDeviceSynchronize();                                                        \
  } while (0);

#define TENSOR_DATA_ON_GPU(tensor_ptr)                                                \
  do                                                                                  \
  {                                                                                   \
    std::size_t byte_size = tensor_ptr->GetByteSize();                                \
    float *data = nullptr;                                                            \
    cudaMalloc(&data, byte_size);                                                     \
    cudaError_t error =                                                               \
        cudaMemcpy(data, tensor_ptr->GetData(), byte_size, cudaMemcpyDeviceToDevice); \
    EXPECT_EQ(error, cudaSuccess);                                                    \
    cudaFree(data);                                                                   \
    cudaDeviceSynchronize();                                                          \
  } while (0);

#define DATA_EXPECT_EQ(data1, data2, n) \
  do                                    \
  {                                     \
    for (int i = 0; i < n; i++)         \
    {                                   \
      EXPECT_EQ(data1[i], data2[i]);    \
    }                                   \
  } while (0);

#define DEFINE_DATA_ON_CPU(data_ptr, n, func)                             \
  float *data_ptr = reinterpret_cast<float *>(malloc(n * sizeof(float))); \
  for (int i = 0; i < n; i++)                                             \
  {                                                                       \
    data_ptr[i] = func(i);                                                \
  }

#define SET_DATA_ON_CPU(data_ptr, n, func) \
  for (int i = 0; i < n; i++)              \
  {                                        \
    data_ptr[i] = func(i);                 \
  }

#define SET_DATA_ON_GPU(data_ptr, n, func)                                      \
  do                                                                            \
  {                                                                             \
    DEFINE_DATA_ON_CPU(temp_data, n, func);                                     \
    cudaMemcpy(data_ptr, temp_data, n * sizeof(float), cudaMemcpyHostToDevice); \
    cudaDeviceSynchronize();                                                    \
    free(temp_data);                                                            \
  } while (0);

#define DEFINE_DATA_ON_GPU_FROM_CPU(data_ptr_gpu, data_ptr_cpu, n)                   \
  float *data_ptr_gpu = nullptr;                                                     \
  cudaMalloc(&data_ptr_gpu, n * sizeof(float));                                      \
  cudaMemcpy(data_ptr_gpu, data_ptr_cpu, n * sizeof(float), cudaMemcpyHostToDevice); \
  cudaDeviceSynchronize();

#define DEFINE_DATA_ON_CPU_FROM_CPU(data_ptr_dst, data_ptr_src, n)            \
  float *data_ptr_dst = reinterpret_cast<float *>(malloc(n * sizeof(float))); \
  memcpy(data_ptr_dst, data_ptr_src, n * sizeof(float))

#define DEFINE_DATA_ON_CPU_FROM_GPU(data_ptr_cpu, data_ptr_gpu, n)                   \
  float *data_ptr_cpu = reinterpret_cast<float *>(malloc(n * sizeof(float)));        \
  cudaMemcpy(data_ptr_cpu, data_ptr_gpu, n * sizeof(float), cudaMemcpyDeviceToHost); \
  cudaDeviceSynchronize();

#define TENSOR_EXPECT_EQ_DATA_CPU_CPU(tensor_this, tensor_that)        \
  do                                                                   \
  {                                                                    \
    int n = tensor_this->GetSize();                                    \
    EXPECT_EQ(tensor_that->GetSize(), n);                              \
    DATA_EXPECT_EQ(tensor_this->GetData(), tensor_that->GetData(), n); \
  } while (0);

#define TENSOR_EXPECT_EQ_DATA_CPU_GPU(tensor_this, tensor_that)        \
  do                                                                   \
  {                                                                    \
    int n = tensor_this->GetSize();                                    \
    EXPECT_EQ(tensor_that->GetSize(), n);                              \
    DEFINE_DATA_ON_CPU_FROM_GPU(data_that, tensor_that->GetData(), n); \
    DATA_EXPECT_EQ(tensor_this->GetData(), data_that, n);              \
    free(data_that);                                                   \
  } while (0);

#define TENSOR_EXPECT_EQ_DATA_GPU_CPU(tensor_this, tensor_that)        \
  do                                                                   \
  {                                                                    \
    int n = tensor_this->GetSize();                                    \
    EXPECT_EQ(tensor_that->GetSize(), n);                              \
    DEFINE_DATA_ON_CPU_FROM_GPU(data_this, tensor_this->GetData(), n); \
    DATA_EXPECT_EQ(data_this, tensor_that->GetData(), n);              \
    free(data_this);                                                   \
  } while (0);

#define TENSOR_EXPECT_EQ_DATA_GPU_GPU(tensor_this, tensor_that)        \
  do                                                                   \
  {                                                                    \
    int n = tensor_this->GetSize();                                    \
    EXPECT_EQ(tensor_that->GetSize(), n);                              \
    DEFINE_DATA_ON_CPU_FROM_GPU(data_this, tensor_this->GetData(), n); \
    DEFINE_DATA_ON_CPU_FROM_GPU(data_that, tensor_that->GetData(), n); \
    DATA_EXPECT_EQ(data_this, data_that, n);                           \
    free(data_this);                                                   \
    free(data_that);                                                   \
  } while (0);

#endif // TEST_TEST_UTILS_CUH_
