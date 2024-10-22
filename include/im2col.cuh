#ifndef INCLUDE_IM2COL_CUH_
#define INCLUDE_IM2COL_CUH_

#include <error.h>
#include <utils.cuh>

namespace my_tensor {

template <typename T>
void im2col_cpu(
    const T *data_im, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w,
    T *data_col) {
  IM2COL_UNIMPLEMENTION
}

template <>
void im2col_cpu(
    const float *data_im, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w,
    float *data_col);

template <typename T>
void col2im_cpu(
    const T *data_col, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w,
    T *data_im) {
  IM2COL_UNIMPLEMENTION
}

template <>
void col2im_cpu(
    const float *data_col, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w,
    float *data_im);

template <typename T>
void im2col_gpu(
    const T *data_im, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w,
    T *data_col) {
  IM2COL_UNIMPLEMENTION
}

template <>
void im2col_gpu(
    const float *data_im, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w,
    float *data_col);

template <typename T>
void col2im_gpu(
    const T *data_col, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w,
    T *data_im) {
  IM2COL_UNIMPLEMENTION
}

template<>
void col2im_gpu(
    const float *data_col, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w,
    float *data_im);

}  // namespace my_tensor

#endif  // INCLUDE_IM2COL_CUH_
