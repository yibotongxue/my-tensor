// Copyright 2024 yibotongxue

#ifndef INCLUDE_BLAS_HPP_
#define INCLUDE_BLAS_HPP_

#include "error.hpp"
#include "utils.hpp"

namespace my_tensor {

template <typename T>
void matmul_gpu(const T *A, const T *B, T *C, const int m, const int k,
                const int n, const int batch_count = 1,
                const int broadcast = 0) {
  BLAS_UNIMPLEMENTION
}

template <typename T>
void matmul_cpu(const T *A, const T *B, T *C, const int m, const int k,
                const int n, const int batch_count = 1,
                const int broadcast = 0) {
  BLAS_UNIMPLEMENTION
}

template <>
void matmul_gpu(const float *A, const float *B, float *C, const int m,
                const int k, const int n, const int batch_count,
                const int broadcast);

template <>
void matmul_cpu(const float *A, const float *B, float *C, const int m,
                const int k, const int n, const int batch_count,
                const int broadcast);

template <typename T>
void transpose_matmul_gpu(const T *A, const T *B, T *C, const int m,
                          const int k, const int n, const int batch_count = 1,
                          const int broadcast = 0) {
  BLAS_UNIMPLEMENTION
}

template <typename T>
void transpose_matmul_cpu(const T *A, const T *B, T *C, const int m,
                          const int k, const int n, const int batch_count = 1,
                          const int broadcast = 0) {
  BLAS_UNIMPLEMENTION
}

template <>
void transpose_matmul_gpu(const float *A, const float *B, float *C, const int m,
                          const int k, const int n, const int batch_count,
                          const int broadcast);

template <>
void transpose_matmul_cpu(const float *A, const float *B, float *C, const int m,
                          const int k, const int n, const int batch_count,
                          const int broadcast);

template <typename T>
void matmul_transpose_gpu(const T *A, const T *B, T *C, const int m,
                          const int k, const int n, const int batch_count = 1,
                          const int broadcast = 0) {
  BLAS_UNIMPLEMENTION
}

template <typename T>
void matmul_transpose_cpu(const T *A, const T *B, T *C, const int m,
                          const int k, const int n, const int batch_count = 1,
                          const int broadcast = 0) {
  BLAS_UNIMPLEMENTION
}

template <>
void matmul_transpose_gpu(const float *A, const float *B, float *C, const int m,
                          const int k, const int n, const int batch_count,
                          const int broadcast);

template <>
void matmul_transpose_cpu(const float *A, const float *B, float *C, const int m,
                          const int k, const int n, const int batch_count,
                          const int broadcast);

template <typename T>
void transpose_matmul_transpose_gpu(const T *A, const T *B, T *C, const int m,
                                    const int k, const int n,
                                    const int batch_count = 1,
                                    const int broadcast = 0) {
  BLAS_UNIMPLEMENTION
}

template <typename T>
void transpose_matmul_transpose_cpu(const T *A, const T *B, T *C, const int m,
                                    const int k, const int n,
                                    const int batch_count = 1,
                                    const int broadcast = 0) {
  BLAS_UNIMPLEMENTION
}

template <>
void transpose_matmul_transpose_gpu(const float *A, const float *B, float *C,
                                    const int m, const int k, const int n,
                                    const int batch_count, const int broadcast);

template <>
void transpose_matmul_transpose_cpu(const float *A, const float *B, float *C,
                                    const int m, const int k, const int n,
                                    const int batch_count, const int broadcast);

template <typename T>
void add_row_vector_gpu(T *mat, const T *vec, const int m, const int n,
                        const int batch_count = 1) {
  BLAS_UNIMPLEMENTION
}

template <typename T>
void add_row_vector_cpu(T *mat, const T *vec, const int m, const int n,
                        const int batch_count = 1) {
  BLAS_UNIMPLEMENTION
}

template <>
void add_row_vector_gpu(float *mat, const float *vec, const int m, const int n,
                        const int batch_count);

template <>
void add_row_vector_cpu(float *mat, const float *vec, const int m, const int n,
                        const int batch_count);

template <typename T>
void add_col_vector_gpu(T *mat, const T *vec, const int m, const int n,
                        const int batch_count = 1) {
  BLAS_UNIMPLEMENTION
}

template <typename T>
void add_col_vector_cpu(T *mat, const T *vec, const int m, const int n,
                        const int batch_count = 1) {
  BLAS_UNIMPLEMENTION
}

template <>
void add_col_vector_gpu(float *mat, const float *vec, const int m, const int n,
                        const int batch_count);

template <>
void add_col_vector_cpu(float *mat, const float *vec, const int m, const int n,
                        const int batch_count);

template <typename T>
T tensor_sum_gpu(const T *tensor, const int cnt) {
  BLAS_UNIMPLEMENTION
}

template <typename T>
T tensor_sum_cpu(const T *tensor, const int cnt) {
  BLAS_UNIMPLEMENTION
}

template <>
float tensor_sum_gpu(const float *tensor, const int cnt);

template <>
float tensor_sum_cpu(const float *tensor, const int cnt);

template <typename T>
void row_sum_gpu(const T *mat, T *result, const int m, const int n,
                 const int batch_count = 1) {
  BLAS_UNIMPLEMENTION
}

template <typename T>
void row_sum_cpu(const T *mat, T *result, const int m, const int n,
                 const int batch_count = 1) {
  BLAS_UNIMPLEMENTION
}

template <>
void row_sum_gpu(const float *mat, float *result, const int m, const int n,
                 const int batch_count);

template <>
void row_sum_cpu(const float *mat, float *result, const int m, const int n,
                 const int batch_count);

template <typename T>
void col_sum_gpu(const T *mat, T *result, const int m, const int n,
                 const int batch_count = 1) {
  BLAS_UNIMPLEMENTION
}

template <typename T>
void col_sum_cpu(const T *mat, T *result, const int m, const int n,
                 const int batch_count = 1) {
  BLAS_UNIMPLEMENTION
}

template <>
void col_sum_gpu(const float *mat, float *result, const int m, const int n,
                 const int batch_count);

template <>
void col_sum_cpu(const float *mat, float *result, const int m, const int n,
                 const int batch_count);
}  // namespace my_tensor

#endif  // INCLUDE_BLAS_HPP_
