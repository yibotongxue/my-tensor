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

set (CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find CUDA Toolkit package
find_package(CUDAToolkit REQUIRED)

# Include CUDA Toolkit directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Set NVCC compile flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G --extended-lambda -Xcompiler -fPIC")

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
set(TENSOR_SRC src/synced-vector.cu src/tensor.cu)
set(LAYER_SRC src/layer.cu src/json-loader.cc src/layer-parameter.cc src/filler.cu
src/relu.cu src/sigmoid.cu src/flatten.cu src/linear.cu src/conv.cu
src/pooling.cu src/softmax.cu src/loss-with-softmax.cu)
set(BLAS_SRC src/blas.cu)
set(IM2COL_SRC src/im2col.cu)
set(DATA_SRC src/dataset.cc src/data-loader.cu)

# Add libraries
add_library(tensor_lib ${TENSOR_SRC})
set_cuda_properties(tensor_lib)

add_library(data_lib ${DATA_SRC})
set_cuda_properties(data_lib)
target_link_libraries(tensor_lib)

add_library(blas_lib src/handle.cu src/blas.cu)
set_cuda_properties(blas_lib)
target_link_libraries(blas_lib PUBLIC cublas)

add_library(im2col_lib ${IM2COL_SRC})
set_cuda_properties(im2col_lib)

add_library(layer_lib ${LAYER_SRC})
set_cuda_properties(layer_lib)
target_link_libraries(layer_lib PUBLIC tensor_lib blas_lib im2col_lib)

# # Add executables and tests
add_executable(tensor_test test/tensor-test.cu)
set_cuda_properties(tensor_test)
link_googletest(tensor_test tensor_lib)
add_test(NAME tensor_test COMMAND tensor_test)

add_executable(relu_test test/relu-test.cu)
set_cuda_properties(relu_test)
link_googletest(relu_test layer_lib)
add_test(NAME relu_test COMMAND relu_test)

add_executable(sigmoid_test test/sigmoid-test.cu)
set_cuda_properties(sigmoid_test)
link_googletest(sigmoid_test layer_lib)
add_test(NAME sigmoid_test COMMAND sigmoid_test)

add_executable(blas_test test/blas-test.cu)
set_cuda_properties(blas_test)
link_googletest(blas_test blas_lib tensor_lib)
add_test(NAME blas_test COMMAND blas_test)

add_executable(linear_test test/linear-test.cu)
set_cuda_properties(linear_test)
link_googletest(linear_test layer_lib)
add_test(NAMD linear_test COMMAND linear_test)

add_executable(im2col_test test/im2col-test.cu)
set_cuda_properties(im2col_test)
link_googletest(im2col_test im2col_lib tensor_lib)
add_test(NAMD im2col_test COMMAND im2col_test)

add_executable(conv_test test/conv-test.cu)
set_cuda_properties(conv_test)
link_googletest(conv_test layer_lib)
add_test(NAMD conv_test COMMAND conv_test)

add_executable(pooling_test test/pooling-test.cu)
set_cuda_properties(pooling_test)
link_googletest(pooling_test layer_lib)
add_test(NAMD pooling_test COMMAND pooling_test)

add_executable(softmax_test test/softmax-test.cu)
set_cuda_properties(softmax_test)
link_googletest(softmax_test layer_lib)
add_test(NAMD softmax_test COMMAND softmax_test)

add_executable(loss_with_softmax_test test/loss-with-softmax-test.cu)
set_cuda_properties(loss_with_softmax_test)
link_googletest(loss_with_softmax_test layer_lib)
add_test(NAMD loss_with_softmax_test COMMAND loss_with_softmax_test)

add_executable(json_test test/json-loader-test.cc src/json-loader.cc src/layer-parameter.cc)
link_googletest(json_test)
add_test(NAMD json_test COMMAND json_test)

add_executable(mnist src/mnist.cu)
set_cuda_properties(mnist)
target_link_libraries(mnist PRIVATE layer_lib data_lib)

# Add the main target
add_executable(my_tensor main.cpp)


##### Begin fetch pybind11 #####
include(FetchContent)

FetchContent_Declare(
  pybind11_sources
  GIT_REPOSITORY https://gitee.com/mirrors/pybind11.git
  GIT_TAG        v2.13
)

if(NOT pybind11_sources_POPULATED)
  FetchContent_Populate(pybind11_sources)

  add_subdirectory(
    ${pybind11_sources_SOURCE_DIR}
    ${pybind11_sources_BINARY_DIR}
    )
endif()

##### End fetch pybind11 #####

set(PYBIND11_NEWPYTHON ON)
# find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

pybind11_add_module(mytensor python/tensor-facade.cu python/layer-facade.cu python/tensor-pybind.cu)
set_cuda_properties(mytensor)
target_link_libraries(mytensor PUBLIC tensor_lib layer_lib data_lib)

# For make install
install(TARGETS mytensor
    LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}"
)

enable_testing()

add_test(
  NAME TensorTest
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/python/tensor-test.py
)

add_test(
  NAME ReluTest
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/python/relu-test.py
)

add_test(
  NAME SigmoidTest
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/python/sigmoid-test.py
)

add_test(
  NAME LinearTest
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/python/linear-test.py
)

add_test(
  NAME ConvolutionTest
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/python/conv-test.py
)

add_test(
  NAME PoolingTest
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/python/pooling-test.py
)

add_test(
  NAME SoftmaxTest
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/python/softmax-test.py
)

add_test(
  NAME CrossEntropyLossTest
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/python/cross-entropy-loss-test.py
)

add_test(
  NAME DatasetTest
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/python/dataset-test.py
)

set_tests_properties(DatasetTest PROPERTIES
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test/python
)

# For Scikit-build-core
# install(TARGETS mytensor LIBRARY DESTINATION .)

