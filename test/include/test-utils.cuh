#ifndef TEST_INCLUDE_TEST_UTILS_CUH_
#define TEST_INCLUDE_TEST_UTILS_CUH_

#define TEST_SHAPE_AND_SIZE(test_fixture) \
  TEST_F(test_fixture##Test, test_fixture##_ShapeMatches) { \
    const std::vector<int> shape = {2, 3}; \
    EXPECT_EQ(tensor->GetShape(), shape); \
  } \
  TEST_F(test_fixture##Test, test_fixture##_SizeMatches) { \
    EXPECT_EQ(tensor->GetSize(), 6); \
  } \
  TEST_F(test_fixture##Test, test_fixture##_DataSizeMatches) { \
    EXPECT_EQ(tensor->GetData().size(), 6); \
  } \
  TEST_F(test_fixture##Test, test_fixture##_DiffSizeMatches) { \
    EXPECT_EQ(tensor->GetDiff().size(), 6); \
  } \

#define TEST_DATA(test_fixture) \
  TEST_F(test_fixture##Test, test_fixture##_DataEqual) { \
    for (int i = 0; i < 6; i++) { \
      EXPECT_EQ(tensor->GetData()[i], i + 1); \
    } \
  }

#endif  // TEST_INCLUDE_TEST_UTILS_CUH_
