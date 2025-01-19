// Copyright 2024 yibotongxue

#ifndef INCLUDE_BLAS_HPP_
#define INCLUDE_BLAS_HPP_

#include "common.hpp"
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

/**
 * @brief add a row vector to a matrix
 *
 * @param mat 矩阵 batch_count * m * n
 * @param vec 向量 m
 * @param m 矩阵行数
 * @param n 矩阵列数
 * @param batch_count batch数
 * @param scale 缩放因子
 */
template <typename T>
void add_row_vector_gpu(T *mat, const T *vec, const int m, const int n,
                        const int batch_count = 1, const T scale = 1) {
  BLAS_UNIMPLEMENTION
}

/**
 * @brief add a row vector to a matrix
 *
 * @param mat 矩阵 batch_count * m * n
 * @param vec 向量 m
 * @param m 矩阵行数
 * @param n 矩阵列数
 * @param batch_count batch数
 * @param scale 缩放因子
 */
template <typename T>
void add_row_vector_cpu(T *mat, const T *vec, const int m, const int n,
                        const int batch_count = 1, const T scale = 1) {
  BLAS_UNIMPLEMENTION
}

/**
 * @brief add a row vector to a matrix
 *
 * @param mat 矩阵 batch_count * m * n
 * @param vec 向量 m
 * @param m 矩阵行数
 * @param n 矩阵列数
 * @param batch_count batch数
 * @param scale 缩放因子
 */
template <>
void add_row_vector_gpu(float *mat, const float *vec, const int m, const int n,
                        const int batch_count, const float scale);

/**
 * @brief add a row vector to a matrix
 *
 * @param mat 矩阵 batch_count * m * n
 * @param vec 向量 m
 * @param m 矩阵行数
 * @param n 矩阵列数
 * @param batch_count batch数
 * @param scale 缩放因子
 */
template <>
void add_row_vector_cpu(float *mat, const float *vec, const int m, const int n,
                        const int batch_count, const float scale);

template <typename T>
void add_col_vector_gpu(T *mat, const T *vec, const int m, const int n,
                        const int batch_count = 1, const T scale = 1) {
  BLAS_UNIMPLEMENTION
}

template <typename T>
void add_col_vector_cpu(T *mat, const T *vec, const int m, const int n,
                        const int batch_count = 1, const T scale = 1) {
  BLAS_UNIMPLEMENTION
}

template <>
void add_col_vector_gpu(float *mat, const float *vec, const int m, const int n,
                        const int batch_count, const float scale);

template <>
void add_col_vector_cpu(float *mat, const float *vec, const int m, const int n,
                        const int batch_count, const float scale);

template <typename T>
void multiply_row_vector_cpu(T *mat, const T *vec, const int m, const int n,
                             const int batch_count = 1);

extern template void multiply_row_vector_cpu(float *mat, const float *vec,
                                             const int m, const int n,
                                             const int batch_count);

template <typename T>
void multiply_row_vector_gpu(T *mat, const T *vec, const int m, const int n,
                             const int batch_count = 1);

extern template void multiply_row_vector_gpu(float *mat, const float *vec,
                                             const int m, const int n,
                                             const int batch_count);

template <typename T>
void multiply_col_vector_cpu(T *mat, const T *vec, const int m, const int n,
                             const int batch_count = 1);

extern template void multiply_col_vector_cpu(float *mat, const float *vec,
                                             const int m, const int n,
                                             const int batch_count);

template <typename T>
void multiply_col_vector_gpu(T *mat, const T *vec, const int m, const int n,
                             const int batch_count = 1);

extern template void multiply_col_vector_gpu(float *mat, const float *vec,
                                             const int m, const int n,
                                             const int batch_count);

template <typename T>
void divide_row_vector_cpu(T *mat, const T *vec, const int m, const int n,
                           const int batch_count = 1, const T eps = 0);

extern template void divide_row_vector_cpu(float *mat, const float *vec,
                                           const int m, const int n,
                                           const int batch_count,
                                           const float eps);

template <typename T>
void divide_row_vector_gpu(T *mat, const T *vec, const int m, const int n,
                           const int batch_count = 1, const T eps = 0);

extern template void divide_row_vector_gpu(float *mat, const float *vec,
                                           const int m, const int n,
                                           const int batch_count,
                                           const float eps);

template <typename T>
void divide_col_vector_cpu(T *mat, const T *vec, const int m, const int n,
                           const int batch_count = 1, const T eps = 0);

extern template void divide_col_vector_cpu(float *mat, const float *vec,
                                           const int m, const int n,
                                           const int batch_count,
                                           const float eps);

template <typename T>
void divide_col_vector_gpu(T *mat, const T *vec, const int m, const int n,
                           const int batch_count = 1, const T eps = 0);

extern template void divide_col_vector_gpu(float *mat, const float *vec,
                                           const int m, const int n,
                                           const int batch_count,
                                           const float eps);

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
                 const int batch_count = 1, T *helper_vec = nullptr) {
  BLAS_UNIMPLEMENTION
}

template <typename T>
void row_sum_cpu(const T *mat, T *result, const int m, const int n,
                 const int batch_count = 1, T *helper_vec = nullptr) {
  BLAS_UNIMPLEMENTION
}

template <>
void row_sum_gpu(const float *mat, float *result, const int m, const int n,
                 const int batch_count, float *helper_vec);

template <>
void row_sum_cpu(const float *mat, float *result, const int m, const int n,
                 const int batch_count, float *helper_vec);

template <typename T>
void col_sum_gpu(const T *mat, T *result, const int m, const int n,
                 const int batch_count = 1, T *helper_vec = nullptr) {
  BLAS_UNIMPLEMENTION
}

template <typename T>
void col_sum_cpu(const T *mat, T *result, const int m, const int n,
                 const int batch_count = 1, T *helper_vec = nullptr) {
  BLAS_UNIMPLEMENTION
}

template <>
void col_sum_gpu(const float *mat, float *result, const int m, const int n,
                 const int batch_count, float *helper_vec);

template <>
void col_sum_cpu(const float *mat, float *result, const int m, const int n,
                 const int batch_count, float *helper_vec);

template <typename T>
void mytensor_gemv_cpu(const T *A, const T *x, T *y, const int m, const int n,
                       const T alpha, const T beta) {
  BLAS_UNIMPLEMENTION
}

// template <>
// void mytensor_gemv_cpu(const float *A, const float *x, float *y, const int m,
//                        const int n, const float alpha, const float beta);

template <typename T>
void mytensor_gemv_gpu(const T *A, const T *x, T *y, const int m, const int n,
                       const T alpha, const T beta) {
  BLAS_UNIMPLEMENTION
}

// template <>
// void mytensor_gemv_gpu(const float *A, const float *x, float *y, const int m,
//                        const int n, const float alpha, const float beta);

template <typename T>
void add_two_vec_gpu(T *lhs, const T *rhs, const T k, const int n) {
  BLAS_UNIMPLEMENTION
}

template <typename T>
void add_two_vec_cpu(T *lhs, const T *rhs, const T k, const int n) {
  BLAS_UNIMPLEMENTION
}

template <>
void add_two_vec_gpu(float *lhs, const float *rhs, const float k, const int n);

template <>
void add_two_vec_cpu(float *lhs, const float *rhs, const float k, const int n);

// lhs = lhs + rhs * k
template <typename T>
inline void add_two_vec(float *lhs, const float *rhs, const float k,
                        const int n) {
  if (MyTensorContext::on_cpu()) {
    add_two_vec_cpu(lhs, rhs, k, n);
  } else {
    add_two_vec_gpu(lhs, rhs, k, n);
  }
}

template <typename T>
void scale_cpu(T *x, const int n, const T k) {
  BLAS_UNIMPLEMENTION
}

template <typename T>
void scale_gpu(T *x, const int n, const T k) {
  BLAS_UNIMPLEMENTION
}

template <>
void scale_cpu(float *x, const int n, const float k);

template <>
void scale_gpu(float *x, const int n, const float k);

template <typename T>
inline void scale(T *x, const int n, const T k) {
  if (MyTensorContext::on_cpu()) {
    scale_cpu(x, n, k);
  } else {
    scale_gpu(x, n, k);
  }
}

template <typename T>
void square_cpu(const T *x, T *y, const int n);

extern template void square_cpu(const float *x, float *y, const int n);

template <typename T>
void square_gpu(const T *x, T *y, const int n);

extern template void square_gpu(const float *x, float *y, const int n);

template <typename T>
inline void square(const T *x, T *y, const int n) {
  if (MyTensorContext::on_cpu()) {
    square_cpu(x, y, n);
  } else {
    square_gpu(x, y, n);
  }
}

template <typename T>
void sqrt_cpu(const T *x, T *y, const int n);

extern template void sqrt_cpu(const float *x, float *y, const int n);

template <typename T>
void sqrt_gpu(const T *x, T *y, const int n);

extern template void sqrt_gpu(const float *x, float *y, const int n);

template <typename T>
inline void sqrt(const T *x, T *y, const int n) {
  if (MyTensorContext::on_cpu()) {
    sqrt_cpu(x, y, n);
  } else {
    sqrt_gpu(x, y, n);
  }
}

template <typename T>
void divide_two_vec_cpu(const T *lhs, const T *rhs, T *result, const int n);

extern template void divide_two_vec_cpu(const float *lhs, const float *rhs,
                                        float *result, const int n);

template <typename T>
void divide_two_vec_gpu(const T *lhs, const T *rhs, T *result, const int n);

extern template void divide_two_vec_gpu(const float *lhs, const float *rhs,
                                        float *result, const int n);

template <typename T>
inline void divide_two_vec(const T *lhs, const T *rhs, T *result, const int n) {
  if (MyTensorContext::on_cpu()) {
    divide_two_vec_cpu(lhs, rhs, result, n);
  } else {
    divide_two_vec_gpu(lhs, rhs, result, n);
  }
}

template <typename T>
void multiply_two_vec_cpu(const T *lhs, const T *rhs, T *result, const int n);

extern template void multiply_two_vec_cpu(const float *lhs, const float *rhs,
                                          float *result, const int n);

template <typename T>
void multiply_two_vec_gpu(const T *lhs, const T *rhs, T *result, const int n);

extern template void multiply_two_vec_gpu(const float *lhs, const float *rhs,
                                          float *result, const int n);

template <typename T>
inline void multiply_two_vec(const T *lhs, const T *rhs, T *result,
                             const int n) {
  if (MyTensorContext::on_cpu()) {
    multiply_two_vec_cpu(lhs, rhs, result, n);
  } else {
    multiply_two_vec_gpu(lhs, rhs, result, n);
  }
}

template <typename T>
void vec_add_num_cpu(const T *vec, T *result, const T num, const int n);

extern template void vec_add_num_cpu(const float *vec, float *result,
                                     const float num, const int n);

template <typename T>
void vec_add_num_gpu(const T *vec, T *result, const T num, const int n);

extern template void vec_add_num_gpu(const float *vec, float *result,
                                     const float num, const int n);

template <typename T>
inline void vec_add_num(const T *vec, T *result, const T num, const int n) {
  if (MyTensorContext::on_cpu()) {
    vec_add_num_cpu(vec, result, num, n);
  } else {
    vec_add_num_gpu(vec, result, num, n);
  }
}

template <typename T>
void vec_divide_num_cpu(const T *vec, T *result, const T num, const int n);

extern template void vec_divide_num_cpu(const float *vec, float *result,
                                        const float num, const int n);

template <typename T>
void vec_divide_num_gpu(const T *vec, T *result, const T num, const int n);

extern template void vec_divide_num_gpu(const float *vec, float *result,
                                        const float num, const int n);

template <typename T>
inline void vec_divide_num(const T *vec, T *result, const T num, const int n) {
  if (MyTensorContext::on_cpu()) {
    vec_divide_num_cpu(vec, result, num, n);
  } else {
    vec_divide_num_gpu(vec, result, num, n);
  }
}
}  // namespace my_tensor

#endif  // INCLUDE_BLAS_HPP_
