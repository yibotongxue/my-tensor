#include <gtest/gtest.h>
#include <conv.cuh>

#include <random>
#include <algorithm>
#include <ranges>

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