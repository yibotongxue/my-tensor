#include <im2col.cuh>
#include <iostream>

namespace my_tensor {

template <>
void im2col_cpu(
    const float *data_im, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w,
    float *data_col) {
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
        int offset_x = k / kernel_w - 1;
        int offset_y = k % kernel_w - 1;
        int input_x = std::abs(output_x + offset_x);
        int input_y = std::abs(output_y + offset_y);
        if (input_x >= height) {
          input_x -= 2 * (input_x - height + 1);
        }
        if (input_y >= width) {
          input_y -= 2 * (input_y - width + 1);
        }
        data_col[i * col_size + j * width_col + k] = data_im[i * im_size + input_x * width + input_y];
      }
    }
  }
}

template <>
void col2im_cpu(
    const float *data_col, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w,
    float *data_im) {
  CHECK_KERNEL_SHAPE
}

template <>
void im2col_gpu(
    const float *data_im, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w,
    float *data_col) {
  CHECK_KERNEL_SHAPE
}

template <>
void col2im_gpu(
    const float *data_col, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w,
    float *data_im) {
  CHECK_KERNEL_SHAPE
}

}  // namespace my_tensor