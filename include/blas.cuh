#ifndef INCLUDE_BLAS_CUH_
#define INCLUDE_BLAS_CUH_

#include <handle.cuh>
#include <tensor.cuh>
#include <error.h>

#include <cublas_v2.h>

namespace my_tensor {
template <typename T>
Tensor<T> operator+(const Tensor<T>& lhs, const Tensor<T>& rhs) {
  throw BlasError("Unimplemention error.");
}

template <>
Tensor<> operator+(const Tensor<>& lhs, const Tensor<>& rhs);
}

#endif  // INCLUDE_BLAS_CUH_
