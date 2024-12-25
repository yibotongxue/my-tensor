// Copyright 2024 yibotongxue

#include "json-loader.hpp"

#include <gtest/gtest.h>

#include <iostream>
#include <utility>

#include "error.hpp"

TEST(JsonTest, NotExistFile) {
  EXPECT_THROW(
      my_tensor::JsonLoader loader(
          "/home/linyibo/Code/my-tensor/test/json-test/json-loader.json"),
      my_tensor::FileError);
}

TEST(JsonTest, UnimplementedLayer) {
  my_tensor::JsonLoader loader(
      "/home/linyibo/Code/my-tensor/test/json-test/json-loader.json");
  EXPECT_THROW(auto params = loader.LoadLayers(), my_tensor::FileError);
  try {
    auto params = loader.LoadLayers();
  } catch (my_tensor::FileError& e) {
    std::cerr << e.what() << std::endl;
  }
}

TEST(JsonTest, WithoutName) {
  my_tensor::JsonLoader loader(
      "/home/linyibo/Code/my-tensor/test/json-test/json-loader.json");
  EXPECT_THROW(auto params = loader.LoadLayers(), my_tensor::FileError);
  try {
    auto params = loader.LoadLayers();
  } catch (my_tensor::FileError& e) {
    std::cerr << e.what() << std::endl;
  }
}

TEST(JsonTest, WithoutType) {
  my_tensor::JsonLoader loader(
      "/home/linyibo/Code/my-tensor/test/json-test/json-loader.json");
  EXPECT_THROW(auto params = loader.LoadLayers(), my_tensor::FileError);
  try {
    auto params = loader.LoadLayers();
  } catch (my_tensor::FileError& e) {
    std::cerr << e.what() << std::endl;
  }
}

TEST(JsonTest, ReluSuccess) {
  my_tensor::JsonLoader loader(
      "/home/linyibo/Code/my-tensor/test/json-test/json-loader.json");
  ASSERT_NO_THROW(loader.LoadLayers());
  auto params = loader.LoadLayers();
  auto param = params[2];
  ASSERT_EQ(param->name_, "relu1");
  ASSERT_EQ(param->type_, my_tensor::ParamType::kRelu);
  auto* ptr = param.get();
  ASSERT_NE(dynamic_cast<my_tensor::ReluParameter*>(ptr), nullptr);
}

TEST(JsonTest, SigmoidSuccess) {
  my_tensor::JsonLoader loader(
      "/home/linyibo/Code/my-tensor/test/json-test/json-loader.json");
  ASSERT_NO_THROW(loader.LoadLayers());
  auto params = loader.LoadLayers();
  auto param = params[5];
  ASSERT_EQ(param->name_, "sigmoid2");
  ASSERT_EQ(param->type_, my_tensor::ParamType::kSigmoid);
  auto* ptr = param.get();
  ASSERT_NE(dynamic_cast<my_tensor::SigmoidParameter*>(ptr), nullptr);
}

TEST(JsonTest, FlattenSuccess) {
  my_tensor::JsonLoader loader(
      "/home/linyibo/Code/my-tensor/test/json-test/json-loader.json");
  ASSERT_NO_THROW(loader.LoadLayers());
  auto params = loader.LoadLayers();
  auto param = params[6];
  ASSERT_EQ(param->name_, "flatten");
  ASSERT_EQ(param->type_, my_tensor::ParamType::kFlatten);
  auto* fptr = dynamic_cast<my_tensor::FlattenParameter*>(param.get());
  ASSERT_NE(fptr, nullptr);
  ASSERT_TRUE(fptr->inplace_);
}

TEST(JsonTest, LinearSuccess) {
  my_tensor::JsonLoader loader(
      "/home/linyibo/Code/my-tensor/test/json-test/json-loader.json");
  ASSERT_NO_THROW(loader.LoadLayers());
  auto params = loader.LoadLayers();
  auto param = params[7];
  ASSERT_EQ(param->name_, "linear1");
  ASSERT_EQ(param->type_, my_tensor::ParamType::kLinear);
  auto* lptr = dynamic_cast<my_tensor::LinearParameter*>(param.get());
  ASSERT_NE(lptr, nullptr);
  ASSERT_EQ(lptr->input_feature_, 490);
  ASSERT_EQ(lptr->output_feature_, 120);
  ASSERT_EQ(lptr->weight_filler_parameter_->init_mode_,
            my_tensor::InitMode::kXavier);
  ASSERT_EQ(lptr->bias_filler_parameter_->init_mode_,
            my_tensor::InitMode::kConstant);
  ASSERT_EQ(std::dynamic_pointer_cast<my_tensor::ConstantFillerParameter>(
                lptr->bias_filler_parameter_)
                ->val_,
            1);
}

TEST(JsonTest, ConvolutionSuccess) {
  my_tensor::JsonLoader loader(
      "/home/linyibo/Code/my-tensor/test/json-test/json-loader.json");
  ASSERT_NO_THROW(loader.LoadLayers());
  auto params = loader.LoadLayers();
  auto param = params[0];
  ASSERT_EQ(param->name_, "conv1");
  ASSERT_EQ(param->type_, my_tensor::ParamType::kConvolution);
  auto* cptr = dynamic_cast<my_tensor::ConvolutionParameter*>(param.get());
  ASSERT_NE(cptr, nullptr);
  ASSERT_EQ(cptr->input_channels_, 1);
  ASSERT_EQ(cptr->output_channels_, 3);
  ASSERT_EQ(cptr->kernel_size_, 3);
  ASSERT_EQ(cptr->kernel_filler_parameter_->init_mode_,
            my_tensor::InitMode::kHe);
  ASSERT_EQ(cptr->bias_filler_parameter_->init_mode_,
            my_tensor::InitMode::kZero);
}

TEST(JsonTest, PoolingSuccess) {
  my_tensor::JsonLoader loader(
      "/home/linyibo/Code/my-tensor/test/json-test/json-loader.json");
  ASSERT_NO_THROW(loader.LoadLayers());
  auto params = loader.LoadLayers();
  auto param = params[1];
  ASSERT_EQ(param->name_, "pooling1");
  ASSERT_EQ(param->type_, my_tensor::ParamType::kPooling);
  auto* pptr = dynamic_cast<my_tensor::PoolingParameter*>(param.get());
  ASSERT_NE(pptr, nullptr);
  ASSERT_EQ(pptr->input_channels_, 3);
  ASSERT_EQ(pptr->kernel_h_, 2);
  ASSERT_EQ(pptr->kernel_w_, 2);
  ASSERT_EQ(pptr->stride_h_, 2);
  ASSERT_EQ(pptr->stride_w_, 2);
}

TEST(JsonTest, SoftmaxSuccess) {
  my_tensor::JsonLoader loader(
      "/home/linyibo/Code/my-tensor/test/json-test/json-loader.json");
  ASSERT_NO_THROW(loader.LoadLayers());
  auto params = loader.LoadLayers();
  auto param = params[12];
  ASSERT_EQ(param->name_, "softmax");
  ASSERT_EQ(param->type_, my_tensor::ParamType::kSoftmax);
  auto* sptr = dynamic_cast<my_tensor::SoftmaxParameter*>(param.get());
  ASSERT_NE(sptr, nullptr);
  ASSERT_EQ(sptr->channels_, 10);
}

TEST(JsonTest, LossWithSoftmaxSuccess) {
  my_tensor::JsonLoader loader(
      "/home/linyibo/Code/my-tensor/test/json-test/json-loader.json");
  ASSERT_NO_THROW(loader.LoadLayers());
  auto params = loader.LoadLayers();
  auto param = params[13];
  ASSERT_EQ(param->name_, "loss_with_softmax");
  ASSERT_EQ(param->type_, my_tensor::ParamType::kLossWithSoftmax);
  auto* lptr = dynamic_cast<my_tensor::LossWithSoftmaxParameter*>(param.get());
  ASSERT_NE(lptr, nullptr);
  ASSERT_EQ(lptr->channels_, 10);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
