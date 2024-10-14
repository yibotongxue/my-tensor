#include <gtest/gtest.h>
#include <relu.cuh>

TEST(TrivialTest, always_succeed) {
  EXPECT_TRUE(true);
}

TEST(TrivialTest, always_fail) {
  EXPECT_TRUE(false);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
