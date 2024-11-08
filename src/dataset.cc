// Copyright 2024 yibotongxue

#include "dataset.h"

#include <cstdint>
#include <fstream>

namespace my_tensor {
namespace {
struct MNISTHeader {
  uint32_t magic_number;
  uint32_t num_images;
  uint32_t num_rows;
  uint32_t num_cols;
};

uint32_t ReverseInt(uint32_t i) {
  uint8_t ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return (static_cast<uint32_t>(ch1) << 24) +
         (static_cast<uint32_t>(ch2) << 16) +
         (static_cast<uint32_t>(ch3) << 8) + static_cast<uint32_t>(ch4);
}
}  // namespace

void MnistDataset::ReadImageFile() {
  std::ifstream file(image_file_path_, std::ios::binary);
  MNISTHeader header;
  file.read(reinterpret_cast<char*>(&header), sizeof(MNISTHeader));
  header.magic_number = ReverseInt(header.magic_number);
  header.num_images = ReverseInt(header.num_images);
  header.num_rows = ReverseInt(header.num_rows);
  header.num_cols = ReverseInt(header.num_cols);
  image_.resize(header.num_images * header.num_rows * header.num_cols);
  file.read(reinterpret_cast<char*>(image_.data()), image_.size());
  height_ = header.num_rows;
  width_ = header.num_cols;
}

void MnistDataset::ReadLabelFile() {
  std::ifstream file(label_file_path_, std::ios::binary);
  uint32_t magic_number, num_labels;
  file.read(reinterpret_cast<char*>(&magic_number), sizeof(uint32_t));
  file.read(reinterpret_cast<char*>(&num_labels), sizeof(uint32_t));
  magic_number = ReverseInt(magic_number);
  num_labels = ReverseInt(num_labels);
  label_.resize(num_labels);
  file.read(reinterpret_cast<char*>(label_.data()), num_labels);
}
}  // namespace my_tensor
