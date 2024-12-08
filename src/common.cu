// Copyright 2024 yibotongxue

#include <unistd.h>

#include <ctime>
#include <iostream>
#include <thread>  // NOLINT

#include "common.hpp"
#include "utils.hpp"

namespace my_tensor {

// random seeding
// from https://github.com/BVLC/caffe.git
int64_t cluster_seedgen(void) {
  int64_t s, seed, pid;
  FILE* f = fopen("/dev/urandom", "rb");
  if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
    fclose(f);
    return seed;
  }

  std::cerr << "System entropy source not available, "
               "using fallback algorithm to generate seed instead.";
  if (f) fclose(f);

  pid = getpid();
  s = time(NULL);
  seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}

MyTensorContext::~MyTensorContext() {
  if (cublas_handle_) CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  if (curand_generator_) {
    CURAND_CHECK(curandDestroyGenerator(curand_generator_));
  }
}

MyTensorContext& MyTensorContext::Get() {
  static thread_local MyTensorContext instance;
  return instance;
}

MyTensorContext::MyTensorContext()
    : device_type_(CPU),
      random_engine_(std::random_device{}()),  // NOLINT
      cublas_handle_(nullptr),
      curand_generator_(nullptr) {
  // from https://github.com/BVLC/caffe.git
  // Try to create a cublas handler, and report an error if failed (but we will
  // keep the program running as one might just want to run CPU code).
  if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Cannot create Cublas handle. Cublas won't be available.";
  }
  // Try to create a curand handler.
  if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT) !=
          CURAND_STATUS_SUCCESS ||
      curandSetPseudoRandomGeneratorSeed(
          curand_generator_, cluster_seedgen()) != CURAND_STATUS_SUCCESS) {
    std::cerr << "Cannot create Curand generator. Curand won't be available.";
  }
}

}  // namespace my_tensor
