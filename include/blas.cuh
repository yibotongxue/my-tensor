#ifndef INCLUDE_BLAS_CUH_
#define INCLUDE_BLAS_CUH_

#include <utils.cuh>
#include <error.h>

namespace my_tensor {

template <typename T>
void matmul(const T *A, const T *B, T *C, const int m, const int k, const int n) {
  BLAS_UNIMPLEMENTION
}

template <>
void matmul(const float *A, const float *B, float *C, const int m, const int k, const int n);

template <typename T>
void transpose_matmul(const T *A, const T *B, T *C, const int m, const int k, const int n) {
  BLAS_UNIMPLEMENTION
}

template <>
void transpose_matmul(const float *A, const float *B, float *C, const int m, const int k, const int n);

template <typename T>
void matmul_transpose(const T *A, const T *B, T *C, const int m, const int k, const int n) {
  BLAS_UNIMPLEMENTION
}

template <>
void matmul_transpose(const float *A, const float *B, float *C, const int m, const int k, const int n);

template <typename T>
void transpose_matmul_transpose(const T *A, const T *B, T *C, const int m, const int k, const int n) {
  BLAS_UNIMPLEMENTION
}

template <>
void transpose_matmul_transpose(const float *A, const float *B, float *C, const int m, const int k, const int n);

}  // namespace my_tensor

#endif  // INCLUDE_BLAS_CUH_
