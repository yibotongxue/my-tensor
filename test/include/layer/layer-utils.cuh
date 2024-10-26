// Copyright 2024 yibotongxue

#ifndef TEST_INCLUDE_LAYER_LAYER_UTILS_CUH_
#define TEST_INCLUDE_LAYER_LAYER_UTILS_CUH_

#include <algorithm>
#include <functional>
#include <memory>
#include <ranges>
#include <tuple>
#include <vector>

#define BACKWARD_TEST(layer_class, layer_name, device)                         \
  TEST_F(layer_class##device##Test, BackwardDiff) {                            \
    layer_name->Forward##device(bottom, top);                                  \
    layer_name->Backward##device(top, bottom);                                 \
    std::vector<float> bottom_diff(bottom->Get##device##Diff().begin(),        \
                                   bottom->Get##device##Diff().end());         \
    my_tensor::TensorPtr<> new_bottom =                                        \
        std::make_shared<my_tensor::Tensor<>>(shape);                          \
    std::vector<float> new_bottom_data(30000);                                 \
    std::ranges::transform(data, new_bottom_data.begin(),                      \
                           [](float x) { return x + 0.001; });                 \
    new_bottom->Set##device##Data(new_bottom_data);                            \
    my_tensor::TensorPtr<> new_top =                                           \
        std::make_shared<my_tensor::Tensor<>>(shape);                          \
    layer_name->Forward##device(new_bottom, new_top);                          \
    std::vector<float> new_top_data(new_top->Get##device##Data().begin(),      \
                                    new_top->Get##device##Data().end());       \
    const std::vector<float> top_data(top->Get##device##Data().begin(),        \
                                      top->Get##device##Data().end());         \
    std::vector<float> results(new_top_data.size());                           \
                                                                               \
    std::transform(new_top_data.begin(), new_top_data.end(), top_data.begin(), \
                   results.begin(),                                            \
                   [](float x, float y) { return (x - y) / 0.001f; });         \
    std::ranges::transform(results, diff, results.begin(),                     \
                           std::multiplies<float>());                          \
    for (int i = 0; i < 30000; i++) {                                          \
      EXPECT_NEAR(results[i], bottom_diff[i], 0.01);                           \
    }                                                                          \
  }

#endif  // TEST_INCLUDE_LAYER_LAYER_UTILS_CUH_
