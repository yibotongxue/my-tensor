#ifndef INCLUDE_BLAS_CUH_
#define INCLUDE_BLAS_CUH_

#include <utils.cuh>
#include <error.h>

namespace my_tensor {

// template <typename T>
// void transpose_matmul_transpose(const T *A, const T *B, T *C, const int m, const int k, const int n) {
//   BLAS_UNIMPLEMENTION
// }

// template <>
// void transpose_matmul_transpose(const float *A, const float *B, float *C, const int m, const int k, const int n);

template <typename T>
void matmul(const T *A, const T *B, T *C, const int m,
    const int k, const int n, const int batch_count = 1,
    const int broadcast = 0) {
  BLAS_UNIMPLEMENTION
}

template <>
void matmul(const float *A, const float *B, float *C,
    const int m, const int k, const int n,
    const int batch_count, const int broadcast);

template <typename T>
void transpose_matmul(const T *A, const T *B, T *C,
    const int m, const int k, const int n,
    const int batch_count = 1, const int broadcast = 0) {
  BLAS_UNIMPLEMENTION
}

template <>
void transpose_matmul(const float *A, const float *B,
    float *C, const int m, const int k, const int n,
    const int batch_count, const int broadcast);

template <typename T>
void matmul_transpose(const T *A, const T *B, T *C,
    const int m, const int k, const int n,
    const int batch_count = 1, const int broadcast = 0) {
  BLAS_UNIMPLEMENTION
}

template <>
void matmul_transpose(const float *A, const float *B,
    float *C, const int m, const int k, const int n,
    const int batch_count, const int broadcast);

template <typename T>
void transpose_matmul_transpose(const T *A, const T *B,
    T *C, const int m, const int k, const int n,
    const int batch_count = 1, const int broadcast = 0) {
  BLAS_UNIMPLEMENTION
}

template <>
void transpose_matmul_transpose(const float *A,
    const float *B, float *C, const int m, const int k,
    const int n, const int batch_count,
    const int broadcast);

template <typename T>
void add_row_vector(T *mat, const T *vec, const int m, const int n) {
  BLAS_UNIMPLEMENTION
}

template <>
void add_row_vector(float *mat, const float *vec, const int m, const int n);

template <typename T>
void add_col_vector(T *mat, const T *vec, const int m, const int n) {
  BLAS_UNIMPLEMENTION
}

template <>
void add_col_vector(float *mat, const float *vec, const int m, const int n);

template <typename T>
T tensor_sum(const T *tensor, const int cnt) {
  BLAS_UNIMPLEMENTION
}

template <>
float tensor_sum(const float *tensor, const int cnt);

template <typename T>
void row_sum(const T *mat, T *result, const int m, const int n) {
  BLAS_UNIMPLEMENTION
}

template <>
void row_sum(const float *mat, float *result, const int m, const int n);

template <typename T>
void col_sum(const T *mat, T *result, const int m, const int n) {
  BLAS_UNIMPLEMENTION
}

template <>
void col_sum(const float *mat, float *result, const int m, const int n);
}  // namespace my_tensor

#endif  // INCLUDE_BLAS_CUH_
