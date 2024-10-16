#include <gtest/gtest.h>
#include <linear.cuh>
#include <layer/layer-utils.cuh>

TEST(TrivialTest, always_succeed) {
  EXPECT_TRUE(true);
}

TEST(TrivialTest, always_fail) {
  EXPECT_TRUE(false);
}

// class LinearTest : public ::testing::Test {
//  protected:
//   void SetUp() override {

//   }
  
// };

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
