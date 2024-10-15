#include <gtest/gtest.h>
#include <layer.cuh>
#include <relu.cuh>
#include <layer/layer-utils.cuh>

#include <random>

// TEST(TrivialTest, always_succeed) {
//   EXPECT_TRUE(true);
// }

// TEST(TrivialTest, always_fail) {
//   EXPECT_TRUE(false);
// }

class ReluTest : public ::testing::Test {
 protected:
  void SetUp() override {
    data.resize(30000);
    diff.resize(30000);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis (-3.0f, 3.0f);
    for (int i = 0; i < 30000; i++) {
      data[i] = dis(gen);
      if (data[i] >= -0.001 && data[i] <= 0) {
        data[i] = 0.001;
      }
    }    
    for (int i = 0; i < 30000; i++) {
      diff[i] = dis(gen);
    }
    relu.reset();
    bottom.reset();
    top.reset();
    relu = std::make_unique<my_tensor::Relu<>>();
    bottom = std::make_shared<my_tensor::Tensor<>>(shape);
    top = std::make_shared<my_tensor::Tensor<>>(shape);
    bottom->SetData(data);
    top->SetDiff(diff);
  }


  const std::vector<int> shape {10000, 3};
  std::vector<float> data;
  std::vector<float> diff;
  my_tensor::LayerPtr<> relu;
  my_tensor::TensorPtr<> bottom;
  my_tensor::TensorPtr<> top;
};

TEST_F(ReluTest, Forward_Data) {
  relu->Forward(bottom, top);
  const std::vector<float> top_data (top->GetData().begin(), top->GetData().end());
  for (int i = 0; i < 30000; i++) {
    if (data[i] > 0) {
      EXPECT_EQ(top_data[i], data[i]);
    } else {
      EXPECT_EQ(top_data[i], 0);
    }
  }
}

BACKWARD_TEST(Relu, relu)

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
