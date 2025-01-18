// Copyright 2025 yibotongxue

#include "batchnorm.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <random>
#include <ranges>  // NOLINT
#include <vector>

#include "json-loader.hpp"
#include "layer-factory.hpp"

// TEST(TrivialTest, always_succeed) {
//   EXPECT_TRUE(true);
// }

// TEST(TrivialTest, always_fail) {
//   EXPECT_TRUE(false);
// }

class BatchNormCPUTest : public ::testing::Test {
 protected:
  void SetUp() override {
    my_tensor::JsonLoader loader(
        "/home/linyibo/Code/my-tensor/test/json-test/batchnorm.json");
    auto&& layer_parameters = loader.LoadLayers();
    gama_data.resize(100);
    beta_data.resize(100);
    input_data.resize(1024000);
    output_diff.resize(1024000);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    auto random_func = [&gen, &dis]() -> float { return dis(gen); };
    std::ranges::generate(gama_data, random_func);
    std::ranges::generate(beta_data, random_func);
    std::ranges::generate(input_data, random_func);
    std::ranges::generate(output_diff, random_func);
    input.reset();
    output.reset();
    gama.reset();
    beta.reset();
    input = std::make_shared<my_tensor::Tensor<float>>(input_shape);
    output = std::make_shared<my_tensor::Tensor<float>>(output_shape);
    input->SetCPUData(input_data.data(), input_data.size());
    batch_norm = my_tensor::CreateLayer<>(layer_parameters[0]);
    auto temp = std::dynamic_pointer_cast<my_tensor::BatchNorm<>>(batch_norm);
    bottom.clear();
    top.clear();
    bottom.push_back(input);
    top.push_back(output);
    batch_norm->SetUp(bottom, top);
    gama = temp->GetGama();
    beta = temp->GetBeta();
    gama->SetCPUData(gama_data.data(), gama_data.size());
    beta->SetCPUData(beta_data.data(), beta_data.size());
    batch_norm->ForwardCPU(bottom, top);
    output->SetCPUDiff(output_diff.data(), output_diff.size());
  }

  const std::vector<int> gama_shape{1, 100, 1, 1};
  const std::vector<int> beta_shape{1, 100, 1, 1};
  std::vector<float> gama_data;
  std::vector<float> beta_data;
  const std::vector<int> input_shape{10, 100, 32, 32};
  const std::vector<int> output_shape{10, 100, 32, 32};
  std::vector<float> input_data;
  std::vector<float> output_diff;
  my_tensor::TensorPtr<> input;
  my_tensor::TensorPtr<> output;
  my_tensor::TensorPtr<> gama;
  my_tensor::TensorPtr<> beta;
  std::vector<my_tensor::TensorPtr<>> bottom;
  std::vector<my_tensor::TensorPtr<>> top;
  my_tensor::LayerPtr<> batch_norm;
  int batch_size = 10;
  int channels = 100;
  int spatial_size = 1024;
};

TEST_F(BatchNormCPUTest, BatchNormForwardTest) {
  std::vector<float> mean(100, 0.0f);
  std::vector<float> variance(100, 0.0f);
  for (int i = 0; i < batch_size * channels * spatial_size; i++) {
    mean[(i / spatial_size) % channels] += input_data[i];
  }
  for (int i = 0; i < channels; i++) {
    mean[i] /= batch_size * spatial_size;
  }
  for (int i = 0; i < batch_size * channels * spatial_size; i++) {
    variance[(i / spatial_size) % channels] +=
        (input_data[i] - mean[(i / spatial_size) % channels]) *
        (input_data[i] - mean[(i / spatial_size) % channels]);
  }
  for (int i = 0; i < channels; i++) {
    variance[i] /= batch_size * spatial_size;
  }
  for (int i = 0; i < batch_size * channels * spatial_size; i++) {
    float expect =
        gama_data[(i / spatial_size) % channels] *
            ((input_data[i] - mean[(i / spatial_size) % channels]) /
             std::sqrt(variance[(i / spatial_size) % channels] + 1e-5)) +
        beta_data[(i / spatial_size) % channels];
    EXPECT_NEAR(output->GetCPUData(i), expect, 0.01);
  }
}

TEST_F(BatchNormCPUTest, BatchNormBackwardBetaTest) {
  batch_norm->BackwardCPU(top, bottom);
  std::vector<float> beta_diff(100, 0.0f);
  for (int i = 0; i < batch_size * channels * spatial_size; i++) {
    beta_diff[(i / spatial_size) % channels] += output_diff[i];
  }
  for (int i = 0; i < channels; i++) {
    EXPECT_NEAR(beta->GetCPUDiff(i), beta_diff[i], 0.01);
  }
}

TEST_F(BatchNormCPUTest, BatchNormBackwardGamaTest) {
  std::vector<float> mean(100, 0.0f);
  std::vector<float> variance(100, 0.0f);
  for (int i = 0; i < batch_size * channels * spatial_size; i++) {
    mean[(i / spatial_size) % channels] += input_data[i];
  }
  for (int i = 0; i < channels; i++) {
    mean[i] /= batch_size * spatial_size;
  }
  for (int i = 0; i < batch_size * channels * spatial_size; i++) {
    variance[(i / spatial_size) % channels] +=
        (input_data[i] - mean[(i / spatial_size) % channels]) *
        (input_data[i] - mean[(i / spatial_size) % channels]);
  }
  for (int i = 0; i < channels; i++) {
    variance[i] /= batch_size * spatial_size;
  }
  batch_norm->BackwardCPU(top, bottom);
  std::vector<float> gama_diff(100, 0.0f);
  for (int i = 0; i < batch_size * channels * spatial_size; i++) {
    gama_diff[(i / spatial_size) % channels] +=
        output_diff[i] *
        ((input_data[i] - mean[(i / spatial_size) % channels]) /
         std::sqrt(variance[(i / spatial_size) % channels] + 1e-5));
  }
  for (int i = 0; i < channels; i++) {
    EXPECT_NEAR(gama->GetCPUDiff(i), gama_diff[i], 0.01);
  }
}

TEST_F(BatchNormCPUTest, BatchNormCPUBackwardBottomTest) {
  batch_norm->BackwardCPU(top, bottom);
  std::vector<float> mean(100, 0.0f);
  std::vector<float> variance(100, 0.0f);
  for (int i = 0; i < batch_size * channels * spatial_size; i++) {
    mean[(i / spatial_size) % channels] += input_data[i];
  }
  for (int i = 0; i < channels; i++) {
    mean[i] /= batch_size * spatial_size;
  }
  for (int i = 0; i < batch_size * channels * spatial_size; i++) {
    variance[(i / spatial_size) % channels] +=
        (input_data[i] - mean[(i / spatial_size) % channels]) *
        (input_data[i] - mean[(i / spatial_size) % channels]);
  }
  for (int i = 0; i < channels; i++) {
    variance[i] /= batch_size * spatial_size;
  }
  std::vector<float> dx_(1024000, 0.0f);
  for (int i = 0; i < batch_size * channels * spatial_size; i++) {
    dx_[i] = output_diff[i] * gama_data[(i / spatial_size) % channels];
  }
  std::vector<float> dx_sum_(100, 0.0f);
  for (int i = 0; i < batch_size * channels * spatial_size; i++) {
    dx_sum_[(i / spatial_size) % channels] += dx_[i];
  }
  std::vector<float> x_(1024000, 0.0f);
  for (int i = 0; i < batch_size * channels * spatial_size; i++) {
    x_[i] = (input_data[i] - mean[(i / spatial_size) % channels]) /
            std::sqrt(variance[(i / spatial_size) % channels] + 1e-5);
  }
  std::vector<float> dx_times_x_sum(100, 0.0f);
  for (int i = 0; i < batch_size * channels * spatial_size; i++) {
    dx_times_x_sum[(i / spatial_size) % channels] += dx_[i] * x_[i];
  }

  for (int i = 0; i < batch_size * channels * spatial_size; i++) {
    float expect = 10240 * dx_[i] - dx_sum_[(i / spatial_size) % channels] -
                   x_[i] * dx_times_x_sum[(i / spatial_size) % channels];
    expect /= 10240 * std::sqrt(variance[(i / spatial_size) % channels] + 1e-5);
    EXPECT_NEAR(input->GetCPUDiff(i), expect, 0.01);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
