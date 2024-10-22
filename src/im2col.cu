#include <im2col.cuh>
#include <iostream>

namespace my_tensor {

template <typename T>
void Im2col_CPU(const T *data_im, const int channels,
    const int height, const int width, const int kernel_h,
    const int kernel_w, T *data_col) {
  CHECK_KERNEL_SHAPE
  const int channels_col = channels;
  const int height_col = height * width;
  const int width_col = kernel_h * kernel_w;
  const int im_size = height * width;
  const int col_size = height_col * width_col;
  for (int i = 0; i < channels_col; i++) {
    for (int j = 0; j < height_col; j++) {
      for (int k = 0; k < width_col; k++) {
        int output_x = j / width;
        int output_y = j % width;
        int offset_x = k / kernel_w - kernel_h / 2;
        int offset_y = k % kernel_w - kernel_w / 2;
        int input_x = output_x + offset_x;
        int input_y = output_y + offset_y;
        data_col[i * col_size + j * width_col + k] = 
          (input_x >= 0 && input_x < height && input_y >= 0 && input_y < width) ?
          data_im[i * im_size + input_x * width + input_y] : 0;
      }
    }
  }
}

template <typename T>
void Col2im_CPU(const T *data_col, const int channels,
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
void Im2col_GPU(const T *data_im, const int channels,
    const int height, const int width, const int kernel_h,
    const int kernel_w, T *data_col) {
  CHECK_KERNEL_SHAPE
}

template <typename T>
void Col2im_GPU(const T *data_col, const int channels,
    const int height, const int width, const int kernel_h,
    const int kernel_w, T *data_im) {
  CHECK_KERNEL_SHAPE
}

template void Im2col_CPU<float>(const float *data_im,
    const int channels, const int height, int width,
    const int kernel_h, const int kernel_w, float *data_col);
template void Col2im_CPU<float>(const float *data_col,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, float *data_im);
template void Im2col_GPU<float>(const float *data_im,
    const int channels, const int height, int width,
    const int kernel_h, const int kernel_w, float *data_col);
template void Col2im_GPU<float>(const float *data_col,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, float *data_im);

}  // namespace my_tensor