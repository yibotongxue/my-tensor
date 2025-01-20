// Copyright 2024 yibotongxue

/**
 * @file blas.hpp
 * @brief 本文件包含各种 BLAS（基本线性代数子程序）操作的声明，适用于 CPU 和 GPU
 * 实现。这些操作包括矩阵乘法、 向量加法、缩放、元素级操作和归约操作。
 *
 * 这些函数是模板化的，以支持不同的数据类型，并为浮点类型提供了特化版本。
 * 操作分为 CPU 和 GPU 版本，并根据执行上下文（CPU 或 GPU）选择适当的版本。
 *
 * 本文件中声明了以下操作：
 * - 矩阵乘法（matmul）
 * - 转置矩阵乘法（transpose_matmul）
 * - 带转置矩阵的矩阵乘法（matmul_transpose）
 * - 带转置矩阵的转置矩阵乘法（transpose_matmul_transpose）
 * - 向矩阵添加行/列向量
 * - 矩阵与行/列向量的乘法
 * - 矩阵与行/列向量的除法
 * - 张量元素的求和
 * - 矩阵的行/列求和
 * - 通用矩阵-向量乘法（gemv）
 * - 两个向量的加法
 * - 向量的缩放
 * - 向量的元素级平方和平方根
 * - 两个向量的元素级除法和乘法
 * - 向量与标量的加法和除法
 *
 * 本文件还包括内联函数，根据执行上下文选择适当的 CPU 或 GPU 实现。
 *
 * @note 本文件中未提供 GPU 函数的实际实现，预计在其他地方定义。
 *
 * @namespace my_tensor
 * 包含 my-tensor 库所有 BLAS 操作的命名空间。
 */

#ifndef INCLUDE_BLAS_HPP_
#define INCLUDE_BLAS_HPP_

#include "common.hpp"
#include "error.hpp"
#include "utils.hpp"

namespace my_tensor {

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
void matmul_gpu(const T *A, const T *B, T *C, const int m, const int k,
                const int n, const int batch_count = 1,
                const int broadcast = 0) {
  BLAS_UNIMPLEMENTION
}

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
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

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
void transpose_matmul_gpu(const T *A, const T *B, T *C, const int m,
                          const int k, const int n, const int batch_count = 1,
                          const int broadcast = 0) {
  BLAS_UNIMPLEMENTION
}

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
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

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
void matmul_transpose_gpu(const T *A, const T *B, T *C, const int m,
                          const int k, const int n, const int batch_count = 1,
                          const int broadcast = 0) {
  BLAS_UNIMPLEMENTION
}

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
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

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
void transpose_matmul_transpose_gpu(const T *A, const T *B, T *C, const int m,
                                    const int k, const int n,
                                    const int batch_count = 1,
                                    const int broadcast = 0) {
  BLAS_UNIMPLEMENTION
}

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
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
template <Arithmetic T>
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
template <Arithmetic T>
  requires std::is_arithmetic<T>::value
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

template <Arithmetic T>
void add_col_vector_gpu(T *mat, const T *vec, const int m, const int n,
                        const int batch_count = 1, const T scale = 1) {
  BLAS_UNIMPLEMENTION
}

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
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

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
void multiply_row_vector_cpu(T *mat, const T *vec, const int m, const int n,
                             const int batch_count = 1);

extern template void multiply_row_vector_cpu(float *mat, const float *vec,
                                             const int m, const int n,
                                             const int batch_count);

template <Arithmetic T>
void multiply_row_vector_gpu(T *mat, const T *vec, const int m, const int n,
                             const int batch_count = 1);

extern template void multiply_row_vector_gpu(float *mat, const float *vec,
                                             const int m, const int n,
                                             const int batch_count);

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
void multiply_col_vector_cpu(T *mat, const T *vec, const int m, const int n,
                             const int batch_count = 1);

extern template void multiply_col_vector_cpu(float *mat, const float *vec,
                                             const int m, const int n,
                                             const int batch_count);

template <Arithmetic T>
void multiply_col_vector_gpu(T *mat, const T *vec, const int m, const int n,
                             const int batch_count = 1);

extern template void multiply_col_vector_gpu(float *mat, const float *vec,
                                             const int m, const int n,
                                             const int batch_count);

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
void divide_row_vector_cpu(T *mat, const T *vec, const int m, const int n,
                           const int batch_count = 1, const T eps = 0);

extern template void divide_row_vector_cpu(float *mat, const float *vec,
                                           const int m, const int n,
                                           const int batch_count,
                                           const float eps);

template <Arithmetic T>
void divide_row_vector_gpu(T *mat, const T *vec, const int m, const int n,
                           const int batch_count = 1, const T eps = 0);

extern template void divide_row_vector_gpu(float *mat, const float *vec,
                                           const int m, const int n,
                                           const int batch_count,
                                           const float eps);

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
void divide_col_vector_cpu(T *mat, const T *vec, const int m, const int n,
                           const int batch_count = 1, const T eps = 0);

extern template void divide_col_vector_cpu(float *mat, const float *vec,
                                           const int m, const int n,
                                           const int batch_count,
                                           const float eps);

template <Arithmetic T>
void divide_col_vector_gpu(T *mat, const T *vec, const int m, const int n,
                           const int batch_count = 1, const T eps = 0);

extern template void divide_col_vector_gpu(float *mat, const float *vec,
                                           const int m, const int n,
                                           const int batch_count,
                                           const float eps);

template <Arithmetic T>
T tensor_sum_gpu(const T *tensor, const int cnt) {
  BLAS_UNIMPLEMENTION
}

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
T tensor_sum_cpu(const T *tensor, const int cnt) {
  BLAS_UNIMPLEMENTION
}

template <>
float tensor_sum_gpu(const float *tensor, const int cnt);

template <>
float tensor_sum_cpu(const float *tensor, const int cnt);

template <Arithmetic T>
void row_sum_gpu(const T *mat, T *result, const int m, const int n,
                 const int batch_count = 1, T *helper_vec = nullptr) {
  BLAS_UNIMPLEMENTION
}

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
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

template <Arithmetic T>
void col_sum_gpu(const T *mat, T *result, const int m, const int n,
                 const int batch_count = 1, T *helper_vec = nullptr) {
  BLAS_UNIMPLEMENTION
}

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
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

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
void mytensor_gemv_cpu(const T *A, const T *x, T *y, const int m, const int n,
                       const T alpha, const T beta) {
  BLAS_UNIMPLEMENTION
}

// template <>
// void mytensor_gemv_cpu(const float *A, const float *x, float *y, const int m,
//                        const int n, const float alpha, const float beta);

template <Arithmetic T>
void mytensor_gemv_gpu(const T *A, const T *x, T *y, const int m, const int n,
                       const T alpha, const T beta) {
  BLAS_UNIMPLEMENTION
}

// template <>
// void mytensor_gemv_gpu(const float *A, const float *x, float *y, const int m,
//                        const int n, const float alpha, const float beta);

template <Arithmetic T>
void add_two_vec_gpu(T *lhs, const T *rhs, const T k, const int n) {
  BLAS_UNIMPLEMENTION
}

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
void add_two_vec_cpu(T *lhs, const T *rhs, const T k, const int n) {
  BLAS_UNIMPLEMENTION
}

template <>
void add_two_vec_gpu(float *lhs, const float *rhs, const float k, const int n);

template <>
void add_two_vec_cpu(float *lhs, const float *rhs, const float k, const int n);

// lhs = lhs + rhs * k
template <Arithmetic T>
  requires std::is_arithmetic<T>::value
inline void add_two_vec(float *lhs, const float *rhs, const float k,
                        const int n) {
  if (MyTensorContext::on_cpu()) {
    add_two_vec_cpu(lhs, rhs, k, n);
  } else {
    add_two_vec_gpu(lhs, rhs, k, n);
  }
}

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
void scale_cpu(T *x, const int n, const T k) {
  BLAS_UNIMPLEMENTION
}

template <Arithmetic T>
void scale_gpu(T *x, const int n, const T k) {
  BLAS_UNIMPLEMENTION
}

template <>
void scale_cpu(float *x, const int n, const float k);

template <>
void scale_gpu(float *x, const int n, const float k);

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
inline void scale(T *x, const int n, const T k) {
  if (MyTensorContext::on_cpu()) {
    scale_cpu(x, n, k);
  } else {
    scale_gpu(x, n, k);
  }
}

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
void square_cpu(const T *x, T *y, const int n);

extern template void square_cpu(const float *x, float *y, const int n);

template <Arithmetic T>
void square_gpu(const T *x, T *y, const int n);

extern template void square_gpu(const float *x, float *y, const int n);

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
inline void square(const T *x, T *y, const int n) {
  if (MyTensorContext::on_cpu()) {
    square_cpu(x, y, n);
  } else {
    square_gpu(x, y, n);
  }
}

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
void sqrt_cpu(const T *x, T *y, const int n);

extern template void sqrt_cpu(const float *x, float *y, const int n);

template <Arithmetic T>
void sqrt_gpu(const T *x, T *y, const int n);

extern template void sqrt_gpu(const float *x, float *y, const int n);

template <Arithmetic T>
inline void sqrt(const T *x, T *y, const int n) {
  if (MyTensorContext::on_cpu()) {
    sqrt_cpu(x, y, n);
  } else {
    sqrt_gpu(x, y, n);
  }
}

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
void divide_two_vec_cpu(const T *lhs, const T *rhs, T *result, const int n);

extern template void divide_two_vec_cpu(const float *lhs, const float *rhs,
                                        float *result, const int n);

template <Arithmetic T>
void divide_two_vec_gpu(const T *lhs, const T *rhs, T *result, const int n);

extern template void divide_two_vec_gpu(const float *lhs, const float *rhs,
                                        float *result, const int n);

template <Arithmetic T>
inline void divide_two_vec(const T *lhs, const T *rhs, T *result, const int n) {
  if (MyTensorContext::on_cpu()) {
    divide_two_vec_cpu(lhs, rhs, result, n);
  } else {
    divide_two_vec_gpu(lhs, rhs, result, n);
  }
}

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
void multiply_two_vec_cpu(const T *lhs, const T *rhs, T *result, const int n);

extern template void multiply_two_vec_cpu(const float *lhs, const float *rhs,
                                          float *result, const int n);

template <Arithmetic T>
void multiply_two_vec_gpu(const T *lhs, const T *rhs, T *result, const int n);

extern template void multiply_two_vec_gpu(const float *lhs, const float *rhs,
                                          float *result, const int n);

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
inline void multiply_two_vec(const T *lhs, const T *rhs, T *result,
                             const int n) {
  if (MyTensorContext::on_cpu()) {
    multiply_two_vec_cpu(lhs, rhs, result, n);
  } else {
    multiply_two_vec_gpu(lhs, rhs, result, n);
  }
}

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
void vec_add_num_cpu(const T *vec, T *result, const T num, const int n);

extern template void vec_add_num_cpu(const float *vec, float *result,
                                     const float num, const int n);

template <Arithmetic T>
void vec_add_num_gpu(const T *vec, T *result, const T num, const int n);

extern template void vec_add_num_gpu(const float *vec, float *result,
                                     const float num, const int n);

template <Arithmetic T>
inline void vec_add_num(const T *vec, T *result, const T num, const int n) {
  if (MyTensorContext::on_cpu()) {
    vec_add_num_cpu(vec, result, num, n);
  } else {
    vec_add_num_gpu(vec, result, num, n);
  }
}

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
void vec_divide_num_cpu(const T *vec, T *result, const T num, const int n);

extern template void vec_divide_num_cpu(const float *vec, float *result,
                                        const float num, const int n);

template <Arithmetic T>
void vec_divide_num_gpu(const T *vec, T *result, const T num, const int n);

extern template void vec_divide_num_gpu(const float *vec, float *result,
                                        const float num, const int n);

template <Arithmetic T>
  requires std::is_arithmetic<T>::value
inline void vec_divide_num(const T *vec, T *result, const T num, const int n) {
  if (MyTensorContext::on_cpu()) {
    vec_divide_num_cpu(vec, result, num, n);
  } else {
    vec_divide_num_gpu(vec, result, num, n);
  }
}
}  // namespace my_tensor

#endif  // INCLUDE_BLAS_HPP_
