// Copyright 2024 yibotongxue

#ifndef TEST_INCLUDE_LAYER_LAYER_UTILS_HPP_
#define TEST_INCLUDE_LAYER_LAYER_UTILS_HPP_

#include <algorithm>
#include <functional>
#include <memory>
#include <ranges>  // NOLINT
#include <tuple>
#include <vector>

#define BACKWARD_TEST(layer_class, layer_name, device)                         \
  TEST_F(layer_class##device##Test, BackwardDiff) {                            \
    layer_name->Forward##device(bottom_vec, top_vec);                          \
    layer_name->Backward##device(top_vec, bottom_vec);                         \
    my_tensor::TensorPtr<> new_bottom =                                        \
        std::make_shared<my_tensor::Tensor<>>(shape);                          \
    std::vector<float> new_bottom_data(30000);                                 \
    std::ranges::transform(data, new_bottom_data.begin(),                      \
                           [](float x) { return x + 0.001; });                 \
    new_bottom->Set##device##Data(new_bottom_data.data(),                      \
                                  new_bottom_data.size());                     \
    my_tensor::TensorPtr<> new_top =                                           \
        std::make_shared<my_tensor::Tensor<>>(shape);                          \
    layer_name->Forward##device({new_bottom}, {new_top});                      \
    std::vector<float> results(new_top->GetSize());                            \
                                                                               \
    std::ranges::transform(SPAN_DATA(new_top, float), SPAN_DATA(top, float),   \
                           results.begin(),                                    \
                           [](float x, float y) { return (x - y) / 0.001f; }); \
    std::ranges::transform(results, diff, results.begin(),                     \
                           std::multiplies<float>());                          \
    for (int i = 0; i < 30000; i++) {                                          \
      EXPECT_NEAR(results[i], bottom->Get##device##Diff(i), 0.01);             \
    }                                                                          \
  }

#endif  // TEST_INCLUDE_LAYER_LAYER_UTILS_HPP_
