// Copyright 2024 yibotongxue

#include <im2col.cuh>

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
  int im_size = height * width;
  memset(data_im, 0, channels * im_size * sizeof(T));
  for (int channel = n * channels; channel--; data_im += im_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = kernel_row - (kernel_h - 1) / 2;
        for (int output_row = height; output_row; output_row--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col += width;
          } else {
            int input_col = kernel_col - (kernel_w - 1) / 2;
            for (int output_col = width; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += 1;
            }
          }
          input_row += 1;
        }
      }
    }
  }
}

namespace {
// kernel_nums = n * channels * width * height = 48 * 3
// height: input height and output height = 8
// width: input width and output width = 6
// kernel_h: kernel height = 3
// kernel_w: kernel width = 3
// im_size = height * width = 48
// col_size = kernel_w * kernel_h * im_size = 9 * 48
template <typename T>
__global__ void Im2col_kernel(const T *data_im, const int kernel_nums,
                              const int height, const int width,
                              const int kernel_h, const int kernel_w,
                              T *data_col, const int im_size,
                              const int col_size) {
  CUDA_KERNEL_LOOP(index, kernel_nums) {
    int channel_index = index / im_size;
    int w_index = index % im_size;
    int h_output = w_index / width;
    int w_output = w_index % width;  // w_output will be 1
    T *data_col_write = data_col + channel_index * col_size + w_index;
    const T *data_im_read = data_im + channel_index * im_size;
    int h_offset = h_output - (kernel_h - 1) / 2;
    int w_offset = w_output % width - (kernel_w - 1) / 2;
    for (int i = 0; i < kernel_h; i++) {
      for (int j = 0; j < kernel_w; j++) {
        int h_read = h_offset + i;
        int w_read = w_offset + j;
        *data_col_write =
            (h_read >= 0 && h_read < height && w_read >= 0 && w_read < width)
                ? data_im_read[h_read * width + w_read]
                : 0;
        data_col_write += im_size;
      }
    }
  }
}
}  // namespace

// kernel_nums = n * channels * width * height
// height: input height and output height
// width: input width and output width
// kernel_h: kernel height
// kernel_w: kernel width
// im_size = height * width
// col_size = im_size * kernel_w * kernel_h
template <typename T>
void Im2col_GPU(const int n, const T *data_im, const int channels,
                const int height, const int width, const int kernel_h,
                const int kernel_w, T *data_col) {
  CHECK_KERNEL_SHAPE
  int kernel_nums = n * channels * width * height;
  int im_size = height * width;
  int col_size = kernel_w * kernel_h * im_size;
  Im2col_kernel<<<CudaGetBlocks(kernel_nums), kCudaThreadNum>>>(
      data_im, kernel_nums, height, width, kernel_h, kernel_w, data_col,
      im_size, col_size);
}

namespace {
template <typename T>
__global__ void Col2im_kernel(const T *data_col, const int kernel_nums,
                              const int height, const int width,
                              const int kernel_h, const int kernel_w,
                              T *data_im, const int im_size,
                              const int col_size) {
  CUDA_KERNEL_LOOP(index, kernel_nums) {
    int channel_index = index / im_size;
    int w_index = index % im_size;
    int write_h = w_index / width;
    int write_w = w_index % width;
    int h_offset = write_h + (kernel_h - 1) / 2;
    int w_offset = write_w + (kernel_w - 1) / 2;
    T val = 0;
    const T *data_read = data_col + channel_index * col_size;
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int read_h = h_offset - kernel_row;
        int read_w = w_offset - kernel_col;
        if (read_h >= 0 && read_h < height && read_w >= 0 && read_w < width) {
          val += data_read[read_h * width + read_w];
        }
        data_read += im_size;
      }
    }
    data_im[index] = val;
  }
}
}  // namespace

template <typename T>
void Col2im_GPU(const int n, const T *data_col, const int channels,
                const int height, const int width, const int kernel_h,
                const int kernel_w, T *data_im) {
  CHECK_KERNEL_SHAPE
  int im_size = height * width;
  int kernel_nums = n * channels * im_size;
  int col_size = kernel_w * kernel_h * im_size;
  Col2im_kernel<<<CudaGetBlocks(kernel_nums), kCudaThreadNum>>>(
      data_col, kernel_nums, height, width, kernel_h, kernel_w, data_im,
      im_size, col_size);
}

template void Im2col_CPU<float>(const int n, const float *data_im,
                                const int channels, const int height, int width,
                                const int kernel_h, const int kernel_w,
                                float *data_col);
template void Col2im_CPU<float>(const int n, const float *data_col,
                                const int channels, const int height,
                                const int width, const int kernel_h,
                                const int kernel_w, float *data_im);
template void Im2col_GPU<float>(const int n, const float *data_im,
                                const int channels, const int height, int width,
                                const int kernel_h, const int kernel_w,
                                float *data_col);
template void Col2im_GPU<float>(const int n, const float *data_col,
                                const int channels, const int height,
                                const int width, const int kernel_h,
                                const int kernel_w, float *data_im);

}  // namespace my_tensor
