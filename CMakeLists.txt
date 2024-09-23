cmake_minimum_required(VERSION 3.24) # set the minimun version of cmake
project(my_tensor VERSION 0.1.0 LANGUAGES CUDA CXX) # the project name, version and language

# set default build type to debug
if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# define DEBUG when build in debug type
if (CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DDEBUG)
endif()

# set the cxx standard to cxx20
set(CMAKE_CXX_STANDARD 20)
# require the cxx standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# check CUDA language
include(CheckLanguage)
check_language(CUDA)

# find CUDAToolkit package
find_package(CUDAToolkit REQUIRED)

# include the head files in CUDAToolkit include directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# set the nvcc compile flags
set(CUDA_NVCC_FLAGS "-g -G")

# include gtest include directories
include_directories(third_parts/googletest)  # 编译gtest
include_directories(third_parts/googletest/googletest/include)  # 包含gtest等头文件
include_directories(third_parts/googletest/googlemock/include)  # 包含gmock等头文件

# include directories
include_directories(include)

# find the library of gtest
find_library(gtest libgtest.a third_parts/googletest/lib)
find_library(gtest_main libgtest_main.a third_parts/googletest/lib)
find_library(gmock libgmock.a third_parts/googletest/lib)
find_library(gmock_main libgmock_main.a third_parts/googletest/lib)

# enable test
enable_testing()

# add the main target
add_executable(my_tensor main.cpp)
