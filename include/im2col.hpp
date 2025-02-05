// Copyright 2024 yibotongxue

#ifndef INCLUDE_IM2COL_HPP_
#define INCLUDE_IM2COL_HPP_

#include "error.hpp"
#include "utils.hpp"

namespace my_tensor {

template <Arithmetic T>
void Im2col_CPU(const int n, const T *data_im, const int channels,
                const int height, const int width, const int kernel_h,
                const int kernel_w, T *data_col);

template <Arithmetic T>
void Col2im_CPU(const int n, const T *data_col, const int channels,
                const int height, const int width, const int kernel_h,
                const int kernel_w, T *data_im);

template <Arithmetic T>
void Im2col_GPU(const int n, const T *data_im, const int channels,
                const int height, const int width, const int kernel_h,
                const int kernel_w, T *data_col);

template <Arithmetic T>
void Col2im_GPU(const int n, const T *data_col, const int channels,
                const int height, const int width, const int kernel_h,
                const int kernel_w, T *data_im);

extern template void Im2col_CPU<float>(const int n, const float *data_im,
                                       const int channels, const int height,
                                       int width, const int kernel_h,
                                       const int kernel_w, float *data_col);
extern template void Col2im_CPU<float>(const int n, const float *data_col,
                                       const int channels, const int height,
                                       const int width, const int kernel_h,
                                       const int kernel_w, float *data_im);
extern template void Im2col_GPU<float>(const int n, const float *data_im,
                                       const int channels, const int height,
                                       int width, const int kernel_h,
                                       const int kernel_w, float *data_col);
extern template void Col2im_GPU<float>(const int n, const float *data_col,
                                       const int channels, const int height,
                                       const int width, const int kernel_h,
                                       const int kernel_w, float *data_im);

}  // namespace my_tensor

#endif  // INCLUDE_IM2COL_HPP_
