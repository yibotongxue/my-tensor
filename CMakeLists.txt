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
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")

# Include gtest directories
include_directories(third_parts/googletest)
include_directories(third_parts/googletest/googletest/include)
include_directories(third_parts/googletest/googlemock/include)

# Include directories
include_directories(include)
include_directories(test/include)

# Find gtest libraries
find_library(gtest libgtest.a PATHS third_parts/googletest/lib)
find_library(gtest_main libgtest_main.a PATHS third_parts/googletest/lib)
find_library(gmock libgmock.a PATHS third_parts/googletest/lib)
find_library(gmock_main libgmock_main.a PATHS third_parts/googletest/lib)

# Set CUDA properties function
function(set_cuda_properties target_name)
  set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES native)
  set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endfunction()

# Link gtest function
function(link_googletest target_name)
    target_link_libraries(${target_name} PRIVATE ${gtest} ${gtest_main} ${gmock} ${gmock_main} pthread)
endfunction()

# Enable testing
enable_testing()

# Add executables and tests
add_executable(tensor_test src/tensor.cu test/tensor-test.cu)
set_cuda_properties(tensor_test)
link_googletest(tensor_test)
add_test(NAME tensor_test COMMAND tensor_test)

add_executable(relu_test src/tensor.cu src/layer.cu src/relu.cu test/relu-test.cu)
set_cuda_properties(relu_test)
link_googletest(relu_test)
add_test(NAME relu_test COMMAND relu_test)

add_executable(sigmoid_test src/tensor.cu src/layer.cu src/sigmoid.cu test/sigmoid-test.cu)
set_cuda_properties(sigmoid_test)
link_googletest(sigmoid_test)
add_test(NAME sigmoid_test COMMAND sigmoid_test)

# Add the main target
add_executable(my_tensor main.cpp)
