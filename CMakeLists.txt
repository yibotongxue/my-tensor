cmake_minimum_required(VERSION 3.24)
project(my_tensor VERSION 0.1.0 LANGUAGES C CXX CUDA)

# Set default build type to debug
if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# Define DEBUG when building in debug type
if (CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DDEBUG)
endif()

# Set the CXX standard to CXX20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA Toolkit package
find_package(CUDAToolkit REQUIRED)

# Include CUDA Toolkit directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Set NVCC compile flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G --extended-lambda")

# Define a function to find gtest/gmock libraries
function(find_gtest)
    find_library(gtest libgtest.a PATHS third_parts/googletest/lib)
    find_library(gtest_main libgtest_main.a PATHS third_parts/googletest/lib)
    find_library(gmock libgmock.a PATHS third_parts/googletest/lib)
    find_library(gmock_main libgmock_main.a PATHS third_parts/googletest/lib)

    set(gtest_libs ${gtest} ${gtest_main} ${gmock} ${gmock_main} PARENT_SCOPE)
endfunction()
find_gtest()

# Set include directories for libraries
include_directories(include)
include_directories(third_parts)
include_directories(test/include)
include_directories(third_parts/googletest/googletest/include)
include_directories(third_parts/googletest/googlemock/include)

# Set CUDA properties function
function(set_cuda_properties target_name)
  set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES native)
  set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endfunction()

# Link gtest function
function(link_googletest target_name)
    target_link_libraries(${target_name} PRIVATE ${gtest_libs} ${ARGN} pthread)
endfunction()

# Enable testing
enable_testing()

# Define source files
set(JSON_SRC src/json-loader.cc)
set(TENSOR_SRC src/synced-vector.cu src/tensor.cu)
set(LAYER_SRC src/layer.cu src/json-loader.cc)
set(RELU_SRC src/relu.cu)
set(SIGMOID_SRC src/sigmoid.cu)
set(LINEAR_SRC src/linear.cu)
set(IM2COL_SRC src/im2col.cu)
set(CONV_SRC src/conv.cu)
set(POOLING_SRC src/pooling.cu)

# Add libraries
add_library(tensor_lib ${TENSOR_SRC})
set_cuda_properties(tensor_lib)

add_library(layer_lib ${LAYER_SRC})
set_cuda_properties(layer_lib)
target_link_libraries(layer_lib PUBLIC tensor_lib)

add_library(relu_lib ${RELU_SRC})
set_cuda_properties(relu_lib)
target_link_libraries(relu_lib PUBLIC layer_lib)

add_library(sigmoid_lib ${SIGMOID_SRC})
set_cuda_properties(sigmoid_lib)
target_link_libraries(sigmoid_lib PUBLIC layer_lib)

add_library(blas_lib src/handle.cu src/blas.cu)
set_cuda_properties(blas_lib)
target_link_libraries(blas_lib PUBLIC cublas)

add_library(linear_lib ${LINEAR_SRC})
set_cuda_properties(linear_lib)
target_link_libraries(linear_lib PUBLIC layer_lib blas_lib)

add_library(im2col_lib ${IM2COL_SRC})
set_cuda_properties(im2col_lib)

add_library(conv_lib ${CONV_SRC})
set_cuda_properties(conv_lib)
target_link_libraries(conv_lib PUBLIC layer_lib blas_lib im2col_lib)

add_library(pooling_lib ${POOLING_SRC})
set_cuda_properties(pooling_lib)
target_link_libraries(pooling_lib PUBLIC layer_lib)

# Add executables and tests
add_executable(tensor_test test/tensor-test.cu)
set_cuda_properties(tensor_test)
link_googletest(tensor_test tensor_lib)
add_test(NAME tensor_test COMMAND tensor_test)

add_executable(relu_test test/relu-test.cu)
set_cuda_properties(relu_test)
link_googletest(relu_test relu_lib)
add_test(NAME relu_test COMMAND relu_test)

add_executable(sigmoid_test test/sigmoid-test.cu)
set_cuda_properties(sigmoid_test)
link_googletest(sigmoid_test sigmoid_lib)
add_test(NAME sigmoid_test COMMAND sigmoid_test)

add_executable(blas_test test/blas-test.cu)
set_cuda_properties(blas_test)
link_googletest(blas_test blas_lib tensor_lib)
add_test(NAME blas_test COMMAND blas_test)

add_executable(linear_test test/linear-test.cu)
set_cuda_properties(linear_test)
link_googletest(linear_test linear_lib)
add_test(NAMD linear_test COMMAND linear_test)

add_executable(im2col_test test/im2col-test.cu)
set_cuda_properties(im2col_test)
link_googletest(im2col_test im2col_lib tensor_lib)
add_test(NAMD im2col_test COMMAND im2col_test)

add_executable(conv_test test/conv-test.cu)
set_cuda_properties(conv_test)
link_googletest(conv_test conv_lib)
add_test(NAMD conv_test COMMAND conv_test)

add_executable(pooling_test test/pooling-test.cu)
set_cuda_properties(pooling_test)
link_googletest(pooling_test pooling_lib)
add_test(NAMD pooling_test COMMAND pooling_test)

add_executable(json_test test/json-loader-test.cc src/json-loader.cc)
link_googletest(json_test)
add_test(NAMD json_test COMMAND json_test)

# Add the main target
add_executable(my_tensor main.cpp)
