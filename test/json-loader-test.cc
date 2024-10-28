#include "json-loader.h"

#include <gtest/gtest.h>

#include <iostream>

#include "error.h"

TEST(JsonTest, NotExistFile) {
  EXPECT_THROW(my_tensor::JsonLoader loader("../test/json-test/not-exist.json"),
               my_tensor::FileError);
}

TEST(JsonTest, UnimplementedLayer) {
  my_tensor::JsonLoader loader("../test/json-test/unimplemented.json");
  EXPECT_THROW(auto params = loader.Load(), my_tensor::FileError);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}