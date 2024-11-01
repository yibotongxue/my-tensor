// Copyright 2024 yibotongxue

#include "json-loader.h"

#include <gtest/gtest.h>

#include <iostream>
#include <utility>

#include "error.h"

TEST(JsonTest, NotExistFile) {
  EXPECT_THROW(my_tensor::JsonLoader loader("../test/json-test/not-exist.json"),
               my_tensor::FileError);
}

TEST(JsonTest, UnimplementedLayer) {
  my_tensor::JsonLoader loader("../test/json-test/unimplemented.json");
  EXPECT_THROW(auto params = loader.Load(), my_tensor::FileError);
  try {
    auto params = loader.Load();
  } catch (my_tensor::FileError& e) {
    std::cerr << e.what() << std::endl;
  }
}

TEST(JsonTest, WithoutName) {
  my_tensor::JsonLoader loader("../test/json-test/without-name.json");
  EXPECT_THROW(auto params = loader.Load(), my_tensor::FileError);
  try {
    auto params = loader.Load();
  } catch (my_tensor::FileError& e) {
    std::cerr << e.what() << std::endl;
  }
}

TEST(JsonTest, WithoutType) {
  my_tensor::JsonLoader loader("../test/json-test/without-type.json");
  EXPECT_THROW(auto params = loader.Load(), my_tensor::FileError);
  try {
    auto params = loader.Load();
  } catch (my_tensor::FileError& e) {
    std::cerr << e.what() << std::endl;
  }
}

TEST(JsonTest, ReluSuccess) {
  my_tensor::JsonLoader loader("../test/json-test/example.json");
  auto params = loader.Load();
  auto param = params[1];
  ASSERT_EQ(param->name_, "relu1");
  ASSERT_EQ(param->type_, my_tensor::ParamType::kRelu);
  auto* ptr = param.get();
  ASSERT_NE(dynamic_cast<my_tensor::ReluParamter*>(ptr), nullptr);
}

TEST(JsonTest, SigmoidSuccess) {
  my_tensor::JsonLoader loader("../test/json-test/example.json");
  auto params = loader.Load();
  auto param = params[3];
  ASSERT_EQ(param->name_, "sigmoid2");
  ASSERT_EQ(param->type_, my_tensor::ParamType::kSigmoid);
  auto* ptr = param.get();
  ASSERT_NE(dynamic_cast<my_tensor::SigmoidParameter*>(ptr), nullptr);
}

TEST(JsonTest, LinearSuccess) {
  my_tensor::JsonLoader loader("../test/json-test/example.json");
  auto params = loader.Load();
  auto param = params[4];
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
  my_tensor::JsonLoader loader("../test/json-test/example.json");
  auto params = loader.Load();
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

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
