#ifndef TEST_INCLUDE_TEST_UTILS_CUH_
#define TEST_INCLUDE_TEST_UTILS_CUH_

#define TEST_SHAPE_AND_SIZE(test_fixture)                    \
  TEST_F(test_fixture##Test, test_fixture##_ShapeMatches)    \
  {                                                          \
    const std::vector<int> shape = {2, 3};                   \
    EXPECT_EQ(tensor->GetShape(), shape);                    \
  }                                                          \
  TEST_F(test_fixture##Test, test_fixture##_SizeMatches)     \
  {                                                          \
    EXPECT_EQ(tensor->GetSize(), 6);                         \
  }                                                          \
  TEST_F(test_fixture##Test, test_fixture##_DataSizeMatches) \
  {                                                          \
    EXPECT_EQ(tensor->GetData().size(), 6);                  \
  }                                                          \
  TEST_F(test_fixture##Test, test_fixture##_DiffSizeMatches) \
  {                                                          \
    EXPECT_EQ(tensor->GetDiff().size(), 6);                  \
  }

#define DEFINE_DATA_AND_DIFF(data_name, diff_name) \
    data_name.resize(6); \
    diff_name.resize(6); \
    for (int i = 0; i < 6; i++) { \
      data_name[i] = i + 1; \
    } \
    diff_name[0] = 1; \
    diff_name[1] = 3; \
    diff_name[2] = 5; \
    diff_name[3] = 2; \
    diff_name[4] = 4; \
    diff_name[5] = 6;

#define DEFINE_TESNOR(tensor_name) \
  const std::vector<int> tensor_name##_shape {2, 3}; \
  try { \
    tensor_name = std::make_shared<my_tensor::Tensor<>>(tensor_name##_shape); \
  } catch (my_tensor::ShapeError& e) { \
    std::cerr << e.what() << std::endl; \
    FAIL() << "Failed to construct tensor."; \
  }

#define DATA_EQUAL_TEST                     \
  for (int i = 0; i < 6; i++)               \
  {                                         \
    EXPECT_EQ(tensor->GetData()[i], i + 1); \
  }

#define DIFF_EQUAL_TEST               \
  EXPECT_EQ(tensor->GetDiff()[0], 1); \
  EXPECT_EQ(tensor->GetDiff()[1], 3); \
  EXPECT_EQ(tensor->GetDiff()[2], 5); \
  EXPECT_EQ(tensor->GetDiff()[3], 2); \
  EXPECT_EQ(tensor->GetDiff()[4], 4); \
  EXPECT_EQ(tensor->GetDiff()[5], 6);

#define TEST_DATA(test_fixture)                        \
  TEST_F(test_fixture##Test, test_fixture##_DataEqual) \
  {                                                    \
    DATA_EQUAL_TEST                                    \
  }

#define TEST_DIFF(test_fixture)                        \
  TEST_F(test_fixture##Test, test_fixture##_DiffEqual) \
  {                                                    \
    DIFF_EQUAL_TEST                                    \
  }

#define TEST_DATA_AND_DIFF(test_fixture) \
  TEST_DATA(test_fixture)                \
  TEST_DIFF(test_fixture)

#endif // TEST_INCLUDE_TEST_UTILS_CUH_
