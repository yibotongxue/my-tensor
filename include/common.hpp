// Copyright 2024 yibotongxue

#ifndef INCLUDE_COMMON_HPP_
#define INCLUDE_COMMON_HPP_

#ifndef CPU_ONLY
#include <cublas_v2.h>
#include <curand.h>
#endif  // CPU_ONLY

#include <memory>
#include <random>

namespace my_tensor {
class MyTensorContext {
 public:
  enum DeviceType : uint8_t { CPU, GPU };  // class DeviceType

  ~MyTensorContext();

  MyTensorContext(const MyTensorContext&) = delete;
  MyTensorContext& operator=(const MyTensorContext&) = delete;

  static MyTensorContext& Get();

  inline static DeviceType device_type() { return Get().device_type_; }
  inline static void set_device_type(DeviceType type) {
    Get().device_type_ = type;
  }
  inline static bool on_cpu() { return device_type() == CPU; }
  inline static bool on_gpu() { return device_type() == GPU; }
  inline static std::mt19937& random_eigine() { return Get().random_engine_; }

#ifndef CPU_ONLY
  inline static cublasHandle_t& cublas_handle() { return Get().cublas_handle_; }
  inline static curandGenerator_t& curand_generator() {
    return Get().curand_generator_;
  }
#endif  // CPU_ONLY

 protected:
  DeviceType device_type_;
  std::mt19937 random_engine_;

#ifndef CPU_ONLY
  cublasHandle_t cublas_handle_;
  curandGenerator_t curand_generator_;
#endif  // CPU_ONLY

 private:
  MyTensorContext();
};  // class MyTensorContext
}  // namespace my_tensor

#endif  // INCLUDE_COMMON_HPP_
