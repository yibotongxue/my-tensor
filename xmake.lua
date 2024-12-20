-- modified based on https://gitee.com/pku-vcl/vcx2024

set_project("my-tensor")
set_version("2.0.0")
set_xmakever("2.6.9")
set_languages("cxx20")

add_rules("mode.debug", "mode.release", "mode.profile")
add_cuflags("-G", "--extended-lambda")
add_cugencodes("native")

add_requires("gtest", "pybind11", "nlohmann_json")

local tensor_src = {
    "src/synced-vector.cu",
    "src/tensor.cu"
}

local data_src = {
    "src/dataset.cc",
    "src/data-loader.cu"
}

local layer_cpu_src = {
    "src/relu.cc",
    "src/sigmoid.cc",
    "src/linear.cc",
    "src/conv.cc",
    "src/pooling.cc",
    "src/softmax.cc",
    "src/loss-with-softmax.cc"
}

local layer_src = {
    "src/layer.cu",
    "src/json-loader.cc",
    "src/layer-parameter.cc",
    "src/filler.cu",
    "src/relu.cu",
    "src/sigmoid.cu",
    "src/flatten.cu",
    "src/linear.cu",
    "src/conv.cu",
    "src/pooling.cu",
    "src/softmax.cu",
    "src/loss-with-softmax.cu"
}

target("common_lib")
    set_kind("static")
    add_includedirs("include", {public = true})
    add_headerfiles("include/common.hpp")
    add_files("src/common.cu")
    add_links("cublas", "curand")

target("tensor_lib")
    set_kind("static")
    add_deps("common_lib")
    add_includedirs("include", {public = true})
    add_files(tensor_src)

target("data_lib")
    set_kind("static")
    add_deps("tensor_lib")
    add_includedirs("include", {public = true})
    add_files(data_src)

target("blas_lib")
    set_kind("static")
    add_deps("tensor_lib")
    add_includedirs("include", {public = true})
    add_files("src/blas.cu")

target("im2col_cpu")
    set_kind("static")
    add_defines("CPU_ONLY")
    add_includedirs("include", {public = true})
    add_files("src/im2col.cc")

target("im2col_lib")
    set_kind("static")
    add_deps("im2col_cpu")
    add_includedirs("include", {public = true})
    add_files("src/im2col.cu")

target("layer_cpu")
    set_kind("static")
    add_packages("nlohmann_json", {public = true})
    add_deps("im2col_cpu")
    add_includedirs("include", {public = true})
    add_files(layer_cpu_src)
    add_defines("CPU_ONLY")

target("layer_lib")
    set_kind("static")
    add_deps("tensor_lib")
    add_deps("blas_lib")
    add_deps("im2col_lib")
    add_deps("layer_cpu")
    add_includedirs("include", {public = true})
    add_files(layer_src)

target("tensor_test")
    set_kind("binary")
    add_packages("gtest")
    add_deps("tensor_lib")
    add_includedirs("include")
    add_includedirs("test/include")
    add_files("test/tensor-test.cu")

target("blas_test")
    set_kind("binary")
    add_packages("gtest")
    add_deps("blas_lib")
    add_files("test/blas-test.cu")

target("im2col_test")
    set_kind("binary")
    add_packages("gtest")
    add_deps("im2col_lib")
    add_deps("tensor_lib")
    add_files("test/im2col-test.cu")

target("relu_test")
    set_kind("binary")
    add_packages("gtest")
    add_deps("layer_lib")
    add_includedirs("test/include")
    add_files("test/relu-test.cu")

target("sigmoid_test")
    set_kind("binary")
    add_packages("gtest")
    add_deps("layer_lib")
    add_includedirs("test/include")
    add_files("test/sigmoid-test.cu")

target("linear_test")
    set_kind("binary")
    add_packages("gtest")
    add_deps("layer_lib")
    add_includedirs("test/include")
    add_files("test/linear-test.cu")

target("conv_test")
    set_kind("binary")
    add_packages("gtest")
    add_deps("layer_lib")
    add_includedirs("test/include")
    add_files("test/conv-test.cu")

target("pooling_test")
    set_kind("binary")
    add_packages("gtest")
    add_deps("layer_lib")
    add_includedirs("test/include")
    add_files("test/pooling-test.cu")

target("softmax_test")
    set_kind("binary")
    add_packages("gtest")
    add_deps("layer_lib")
    add_includedirs("test/include")
    add_files("test/softmax-test.cu")

target("loss_with_softmax_test")
    set_kind("binary")
    add_packages("gtest")
    add_deps("layer_lib")
    add_includedirs("test/include")
    add_files("test/loss-with-softmax-test.cu")
