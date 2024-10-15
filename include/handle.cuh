#ifndef INCLUDE_HANDLE_CUH_
#define INCLUDE_HANDLE_CUH_

#include <cublas_v2.h>

#include <memory>

namespace my_tensor {
class Handle {
public:
  ~Handle() {
    cublasDestroy(h_);
  }

  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;

  static Handle* GetInstance() {
    if (handle_ == nullptr) {
      handle_ = new Handle();
    }
    return handle_;
  }

  cublasHandle_t& GetHandle() {
    return h_;
  }

private:
  cublasHandle_t h_;

  static Handle* handle_;

  Handle() {
    cublasCreate(&h_);
  }
};  // class Handle

extern Handle* handle;
}  // namespace my_tensor

#endif  // INCLUDE_HANDLE_CUH_
