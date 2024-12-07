// Copyright 2024 yibotongxue

#ifndef INCLUDE_HANDLE_HPP_
#define INCLUDE_HANDLE_HPP_

#include <cublas_v2.h>

#include <memory>

namespace my_tensor {
class Handle;
using HandlePtr = std::shared_ptr<Handle>;

class Handle {
 public:
  ~Handle() { cublasDestroy(h_); }

  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;

  static HandlePtr GetInstance() {
    if (handle_.get() == nullptr) {
      handle_ = std::shared_ptr<Handle>(new Handle());
    }
    return handle_;
  }

  cublasHandle_t& GetHandle() { return h_; }

 private:
  cublasHandle_t h_;

  static HandlePtr handle_;

  Handle() { cublasCreate(&h_); }
};  // class Handle

extern HandlePtr handle;
}  // namespace my_tensor

#endif  // INCLUDE_HANDLE_HPP_
