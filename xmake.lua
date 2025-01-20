-- modified based on https://gitee.com/pku-vcl/vcx2024

set_project("my-tensor")
set_version("2.0.0")
set_xmakever("2.6.9")
set_languages("cxx20")

add_rules("mode.debug", "mode.release", "mode.profile")
add_cuflags("-G", "--extended-lambda")
add_cugencodes("native")

add_requires("gtest", "nlohmann_json", "openblas", "spdlog")

local tensor_src = {
    "src/memory-util.cu",
    "src/synced-vector.cc",
    "src/tensor.cc"
}

local data_src = {
    "src/dataset.cc",
    "src/data-loader.cc"
}

local layer_src = {
    "src/layer.cc",
    "src/json-loader.cc",
    "src/layer-parameter.cc",
    "src/filler.cu",
    "src/filler.cc",
    "src/relu.cu",
    "src/relu.cc",
    "src/sigmoid.cu",
    "src/sigmoid.cc",
    "src/flatten.cc",
    "src/linear.cu",
    "src/linear.cc",
    "src/conv.cu",
    "src/conv.cc",
    "src/pooling.cu",
    "src/pooling.cc",
    "src/softmax.cu",
    "src/softmax.cc",
    "src/loss-with-softmax.cu",
    "src/loss-with-softmax.cc",
    "src/accuracy.cu",
    "src/accuracy.cc",
    "src/batchnorm.cu",
    "src/batchnorm.cc"
}

local solver_src = {
    "src/model-saver.cc",
    "src/solver.cc",
    "src/sgd-solver.cc",
    "src/sgd-with-momentum-solver.cc",
    "src/adamw-solver.cc"
}

target("common_lib")
    set_kind("static")
    add_packages("spdlog", {public = true})
    add_includedirs("include", {public = true})
    add_headerfiles("include/common.hpp")
    add_files("src/common.cc")
    add_files("src/cuda-context.cu")
    add_links("cublas", "curand")
    set_policy("build.cuda.devlink", true)

target("tensor_lib")
    set_kind("static")
    add_deps("common_lib")
    add_includedirs("include", {public = true})
    add_files(tensor_src)
    set_policy("build.cuda.devlink", true)

target("data_lib")
    set_kind("static")
    add_deps("tensor_lib")
    add_includedirs("include", {public = true})
    add_files(data_src)
    set_policy("build.cuda.devlink", true)

target("blas_lib")
    set_kind("static")
    add_packages("openblas")
    add_deps("tensor_lib")
    add_includedirs("include", {public = true})
    add_files("src/blas.cu")
    add_files("src/blas.cc")
    set_policy("build.cuda.devlink", true)

target("im2col_lib")
    set_kind("static")
    add_deps("common_lib")
    add_includedirs("include", {public = true})
    add_files("src/im2col.cu")
    add_files("src/im2col.cc")
    set_policy("build.cuda.devlink", true)

target("layer_lib")
    set_kind("static")
    add_packages("nlohmann_json", {public = true})
    add_deps("tensor_lib")
    add_deps("blas_lib")
    add_deps("im2col_lib")
    add_includedirs("include", {public = true})
    add_files(layer_src)
    set_policy("build.cuda.devlink", true)

target("net_lib")
    set_kind("static")
    add_deps("layer_lib")
    add_deps("data_lib")
    add_includedirs("include", {public = true})
    add_files("src/net.cc")
    set_policy("build.cuda.devlink", true)

target("solver_lib")
    set_kind("static")
    add_deps("net_lib")
    add_includedirs("include", {public = true})
    add_files(solver_src)
    set_policy("build.cuda.devlink", true)

target("tensor_test")
    set_kind("binary")
    add_packages("gtest")
    add_deps("tensor_lib")
    add_includedirs("include")
    add_includedirs("test/include")
    add_files("test/tensor-test.cc")

target("blas_test")
    set_kind("binary")
    add_packages("gtest")
    add_deps("blas_lib")
    add_files("test/blas-test.cc")

target("im2col_test")
    set_kind("binary")
    add_packages("gtest")
    add_deps("im2col_lib")
    add_deps("tensor_lib")
    add_files("test/im2col-test.cc")

target("relu_test")
    set_kind("binary")
    add_packages("gtest")
    add_deps("layer_lib")
    add_includedirs("test/include")
    add_files("test/relu-test.cc")

target("sigmoid_test")
    set_kind("binary")
    add_packages("gtest")
    add_deps("layer_lib")
    add_includedirs("test/include")
    add_files("test/sigmoid-test.cc")

target("linear_test")
    set_kind("binary")
    add_packages("gtest")
    add_deps("layer_lib")
    add_includedirs("test/include")
    add_files("test/linear-test.cc")

target("conv_test")
    set_kind("binary")
    add_packages("gtest")
    add_deps("layer_lib")
    add_includedirs("test/include")
    add_files("test/conv-test.cc")

target("pooling_test")
    set_kind("binary")
    add_packages("gtest")
    add_deps("layer_lib")
    add_includedirs("test/include")
    add_files("test/pooling-test.cc")

target("softmax_test")
    set_kind("binary")
    add_packages("gtest")
    add_deps("layer_lib")
    add_includedirs("test/include")
    add_files("test/softmax-test.cc")

target("loss_with_softmax_test")
    set_kind("binary")
    add_packages("gtest")
    add_deps("layer_lib")
    add_includedirs("test/include")
    add_files("test/loss-with-softmax-test.cc")

target("accuracy_test")
    set_kind("binary")
    add_packages("gtest")
    add_deps("layer_lib")
    add_includedirs("test/include")
    add_files("test/accuracy-test.cc")

target("batchnorm_test")
    set_kind("binary")
    add_packages("gtest")
    add_deps("layer_lib")
    add_includedirs("test/include")
    add_files("test/batchnorm-test.cc")

target("main")
    set_kind("binary")
    add_deps("layer_lib")
    add_deps("data_lib")
    add_deps("solver_lib")
    add_files("src/json-loader.cc")
    add_files("src/main.cc")

-- from ChatGPT
-- 添加一个目标用于批量运行测试
target("run_tests")
    set_kind("phony") -- 表示这是一个伪目标
    on_run(function ()
        -- 定义需要运行的测试列表
        local tests = {
            "tensor_test",
            "blas_test",
            "im2col_test",
            "relu_test",
            "sigmoid_test",
            "linear_test",
            "conv_test",
            "pooling_test",
            "softmax_test",
            "loss_with_softmax_test",
            "accuracy_test",
            "batchnorm_test"
        }

        -- 遍历运行每个测试
        for _, test in ipairs(tests) do
            os.exec("pwd")
            os.exec("xmake run %s", test)
        end
    end)

