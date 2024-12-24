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
}

MyTensorContext& MyTensorContext::Get() {
  static thread_local MyTensorContext instance;
  return instance;
}

MyTensorContext::MyTensorContext()
    : device_type_(CPU),
      random_engine_(std::random_device{}())  // NOLINT
      {
  // from https://github.com/BVLC/caffe.git
  // Try to create a cublas handler, and report an error if failed (but we will
  // keep the program running as one might just want to run CPU code).
}

}  // namespace my_tensor
