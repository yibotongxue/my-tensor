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

# Set include directories for libraries
include_directories(include)
include_directories(third_parts)

# Set CUDA properties function
function(set_cuda_properties target_name)
  set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES native)
  set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endfunction()

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

# add_library(data_lib ${DATA_SRC})
# set_cuda_properties(data_lib)
# target_link_libraries(tensor_lib)

# add_library(blas_lib src/handle.cu src/blas.cu)
# set_cuda_properties(blas_lib)
# target_link_libraries(blas_lib PUBLIC cublas)

# add_library(im2col_lib ${IM2COL_SRC})
# set_cuda_properties(im2col_lib)

# add_library(layer_lib ${LAYER_SRC})
# set_cuda_properties(layer_lib)
# target_link_libraries(layer_lib PUBLIC tensor_lib blas_lib im2col_lib)

include(FetchContent)

FetchContent_Declare(
  pybind11_sources
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.13
)

if(NOT pybind11_sources_POPULATED)
  FetchContent_Populate(pybind11_sources)

  add_subdirectory(
    ${pybind11_sources_SOURCE_DIR}
    ${pybind11_sources_BINARY_DIR}
    )
endif()

set(PYBIND11_NEWPYTHON ON)
# find_package(pybind11 CONFIG REQUIRED)

add_library(tensor_facade_lib python/tensor-facade.cu)
target_link_libraries(tensor_facade_lib PUBLIC tensor_lib)

pybind11_add_module(mytensor python/tensor-pybind.cu)
set_cuda_properties(mytensor)
target_link_libraries(mytensor PUBLIC  tensor_facade_lib)
install(TARGETS mytensor LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/my_tensor)
