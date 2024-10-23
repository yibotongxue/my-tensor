#include <im2col.cuh>
#include <iostream>

namespace my_tensor {

static inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename T>
void Im2col_CPU(const int n, const T *data_im, const int channels,
    const int height, const int width, const int kernel_h,
    const int kernel_w, T *data_col) {
  CHECK_KERNEL_SHAPE
  const int im_size = height * width;
  for (int channel = n * channels; channel--; data_im += im_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = kernel_row - (kernel_h - 1) / 2;
        for (int output_row = height; output_row; output_row--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_col = width; output_col; output_col--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = kernel_col - (kernel_w - 1) / 2;
            for (int output_col = width; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = *(data_im + input_row * width + input_col);
              } else {
                *(data_col++) = 0;
              }
              input_col += 1;
            }
          }
          input_row += 1;
        }
      }
    }
  }
}

template <typename T>
void Col2im_CPU(const int n, const T *data_col, const int channels,
    const int height, const int width, const int kernel_h,
    const int kernel_w, T *data_im) {
  CHECK_KERNEL_SHAPE
}

// namespace {
// template <typename T>
// __global__ void Im2col_kernel(const T *data_im,
//     const int height, const int width, const int kernel_h,
//     const int kernel_w, T *data_col, const int height_col,
//     const int width_col, const int im_size, const int col_size) {
//   int c = blockIdx.x;
//   int gidx = blockDim.x * blockIdx.x + threadIdx.x;
//   int tidx = threadIdx.x;
// }
// }  // namespace

template <typename T>
void Im2col_GPU(const int n, const T *data_im, const int channels,
    const int height, const int width, const int kernel_h,
    const int kernel_w, T *data_col) {
  CHECK_KERNEL_SHAPE
}

template <typename T>
void Col2im_GPU(const int n, const T *data_col, const int channels,
    const int height, const int width, const int kernel_h,
    const int kernel_w, T *data_im) {
  CHECK_KERNEL_SHAPE
}

template void Im2col_CPU<float>(const int n, const float *data_im,
    const int channels, const int height, int width,
    const int kernel_h, const int kernel_w, float *data_col);
template void Col2im_CPU<float>(const int n, const float *data_col,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, float *data_im);
template void Im2col_GPU<float>(const int n, const float *data_im,
    const int channels, const int height, int width,
    const int kernel_h, const int kernel_w, float *data_col);
template void Col2im_GPU<float>(const int n, const float *data_col,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, float *data_im);

}  // namespace my_tensor