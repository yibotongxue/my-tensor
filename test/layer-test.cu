#include <gtest/gtest.h>
#include <layer.cuh>
#include <utils.cuh>
#include <relu.cuh>
#include <sigmoid.cuh>
#include <tensor.cuh>
#include <test-utils.cuh>
#include <memory>
#include <vector>
#include <random>

void AddCPU(const float *data_src, float *data_dst, const float delta, int n)
{
  for (int i = 0; i < n; i++)
  {
    data_dst[i] = data_src[i] + delta;
  }
}

__global__ void CudaAdd(const float *data_src, float *data_dst, const float delta, int n)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    data_dst[i] = data_src[i] + delta;
  }
}

void AddGPU(const float *data_src, float *data_dst, const float delta, int n)
{
  CudaAdd<<<CudaGetBlocks(n), kCudaThreadNum>>>(data_src, data_dst, delta, n);
}

#define LAYER_TEST_CLASS(layer_name, device)                                                    \
  class layer_name##Test##device : public ::testing::Test                                       \
  {                                                                                             \
  protected:                                                                                    \
    void SetUp() override                                                                       \
    {                                                                                           \
      std::random_device rd;                                                                    \
      std::mt19937 gen(rd());                                                                   \
      std::uniform_real_distribution<> dis(-1.0f, 1.0f);                                        \
      std::vector<int> shape = {10000};                                                         \
      bottom = std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::device);       \
      float *data = bottom->GetMutableData();                                                   \
      auto func = [&dis, &gen](int x) -> float {                                                \
        float result = dis(gen);                                                                \
        if (result >= -0.001 && result <= 0)                                                    \
        {                                                                                       \
          result = 0.001;                                                                       \
        }                                                                                       \
        return result;                                                                          \
      };                                                                                        \
      SET_DATA_ON_##device(data, 10000, func);                                                  \
      bottom_delta = std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::device); \
      Add##device(bottom->GetData(), bottom_delta->GetMutableData(), 0.001, 10000);             \
      top_delta = std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::device);    \
      layer = std::move(std::make_unique<my_tensor::layer_name>());                             \
      top = std::make_shared<my_tensor::Tensor>(shape, my_tensor::DeviceType::device);          \
      float *diff = top->GetMutableDiff();                                                      \
      SET_DATA_ON_##device(diff, 10000, func);                                                  \
    }                                                                                           \
    my_tensor::LayerPtr layer;                                                                  \
    std::shared_ptr<my_tensor::Tensor> bottom;                                                  \
    std::shared_ptr<my_tensor::Tensor> top;                                                     \
    std::shared_ptr<my_tensor::Tensor> bottom_delta;                                            \
    std::shared_ptr<my_tensor::Tensor> top_delta;                                               \
  };

#define RELU_FORWARD_TEST(device)                                          \
  TEST_F(ReluTest##device, forward_test)                                   \
  {                                                                        \
    EXPECT_NO_THROW(layer->Forward(bottom, top));                          \
    const float *bottom_data = bottom->GetData();                          \
    DEFINE_DATA_ON_CPU_FROM_##device(cpu_bottom_data, bottom_data, 10000); \
    bottom_data = nullptr;                                                 \
    const float *top_data = top->GetData();                                \
    DEFINE_DATA_ON_CPU_FROM_##device(cpu_top_data, top_data, 10000);       \
    top_data = nullptr;                                                    \
    for (int i = 0; i < 10000; i++)                                        \
    {                                                                      \
      float actual = cpu_top_data[i];                                      \
      float expect = std::max(cpu_bottom_data[i], 0.0f);                   \
      EXPECT_NEAR(actual, expect, 0.001);                                  \
    }                                                                      \
    free(cpu_bottom_data);                                                 \
    free(cpu_top_data);                                                    \
  }

#define SIGMOID_FORWARD_TEST(device)                                       \
  TEST_F(SigmoidTest##device, forward_test)                                \
  {                                                                        \
    EXPECT_NO_THROW(layer->Forward(bottom, top));                          \
    const float *bottom_data = bottom->GetData();                          \
    DEFINE_DATA_ON_CPU_FROM_##device(cpu_bottom_data, bottom_data, 10000); \
    bottom_data = nullptr;                                                 \
    const float *top_data = top->GetData();                                \
    DEFINE_DATA_ON_CPU_FROM_##device(cpu_top_data, top_data, 10000);       \
    top_data = nullptr;                                                    \
    for (int i = 0; i < 10000; i++)                                        \
    {                                                                      \
      float actual = cpu_top_data[i];                                      \
      float expect = 1.0f / (1.0f + std::exp(-cpu_bottom_data[i]));        \
      EXPECT_NEAR(actual, expect, 0.001);                                  \
    }                                                                      \
    free(cpu_bottom_data);                                                 \
    free(cpu_top_data);                                                    \
  }

#define BACKWARD_TEST(layer_name, device)                                                                                                 \
  TEST_F(layer_name##Test##device, backward_test)                                                                                         \
  {                                                                                                                                       \
    layer->Forward(bottom, top);                                                                                                          \
    EXPECT_NO_THROW(layer->Backward(top, bottom));                                                                                        \
    const float *bottom_diff = bottom->GetDiff();                                                                                         \
    DEFINE_DATA_ON_CPU_FROM_##device(cpu_bottom_diff, bottom_diff, 10000);                                                                \
    bottom_diff = nullptr;                                                                                                                \
    const float *top_diff = top->GetDiff();                                                                                               \
    DEFINE_DATA_ON_CPU_FROM_##device(cpu_top_diff, top_diff, 10000);                                                                      \
    layer->Forward(bottom_delta, top_delta);                                                                                              \
    const float *top_data = top->GetData();                                                                                               \
    DEFINE_DATA_ON_CPU_FROM_##device(cpu_top_data, top_data, 10000);                                                                      \
    top_data = nullptr;                                                                                                                   \
    const float *top_delta_data = top_delta->GetData();                                                                                   \
    DEFINE_DATA_ON_CPU_FROM_##device(cpu_top_delta_data, top_delta_data, 10000);                                                          \
    top_delta_data = nullptr;                                                                                                             \
    for (int i = 0; i < 10000; i++)                                                                                                       \
    {                                                                                                                                     \
      float expect = ((static_cast<double>(cpu_top_delta_data[i]) - static_cast<double>(cpu_top_data[i])) / static_cast<double>(0.001)) * \
                     static_cast<double>(cpu_top_diff[i]);                                                                                \
      float actual = cpu_bottom_diff[i];                                                                                                  \
      EXPECT_NEAR(actual, expect, 0.001);                                                                                                 \
    }                                                                                                                                     \
    free(cpu_bottom_diff);                                                                                                                \
    free(cpu_top_diff);                                                                                                                   \
    free(cpu_top_data);                                                                                                                   \
    free(cpu_top_delta_data);                                                                                                             \
  }

// Define test class
LAYER_TEST_CLASS(Relu, CPU)
LAYER_TEST_CLASS(Relu, GPU)

// Relu forward test on cpu and gpu
RELU_FORWARD_TEST(CPU)
RELU_FORWARD_TEST(GPU)

// Relu backward test on cpu and gpu
BACKWARD_TEST(Relu, CPU);
BACKWARD_TEST(Relu, GPU);

// Define test class
LAYER_TEST_CLASS(Sigmoid, CPU)
LAYER_TEST_CLASS(Sigmoid, GPU)

// Sigmoid forward test on cpu and gpu
SIGMOID_FORWARD_TEST(CPU)
SIGMOID_FORWARD_TEST(GPU)

// Sigmoid forward test on cpu and gpu
BACKWARD_TEST(Sigmoid, CPU)
BACKWARD_TEST(Sigmoid, GPU)

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
