// Copyright 2024 yibotongxue

#include "dataset.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <ranges>  // NOLINT
#include <vector>

namespace my_tensor {
namespace {
struct MNISTHeader {
  uint32_t magic_number;
  uint32_t num_images;
  uint32_t num_rows;
  uint32_t num_cols;
};

uint32_t ReverseInt(uint32_t i) {
  int ch1, ch2, ch3, ch4;
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
  std::vector<uint8_t> uimg_data(header.num_images * header.num_rows *
                                 header.num_cols);
  file.read(reinterpret_cast<char*>(uimg_data.data()),
            image_.size() * sizeof(char));
  std::ranges::transform(uimg_data, image_.begin(), [](uint8_t val) -> float {
    return static_cast<float>(val) / 255.0f - 0.5;
  });
  this->height_ = header.num_rows;
  this->width_ = header.num_cols;
  spdlog::info("Read image file done");
}

void MnistDataset::ReadLabelFile() {
  std::ifstream file(label_file_path_, std::ios::binary);
  uint32_t magic_number, num_labels;
  file.read(reinterpret_cast<char*>(&magic_number), sizeof(uint32_t));
  file.read(reinterpret_cast<char*>(&num_labels), sizeof(uint32_t));
  magic_number = ReverseInt(magic_number);
  num_labels = ReverseInt(num_labels);
  std::vector<uint8_t> label(num_labels);
  file.read(reinterpret_cast<char*>(label.data()), num_labels * sizeof(char));
  label_.resize(num_labels);
  std::ranges::transform(label, label_.begin(), [](uint8_t val) -> int {
    return static_cast<int>(val);
  });
  spdlog::info("Read label file done");
}

void Cifar10Dataset::LoadData() {
  for (auto&& data_batch : data_batches_) {
    std::ifstream file(data_batch, std::ios::binary);
    std::vector<uint8_t> data((32 * 32 * 3 + 1) * 10000);
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(char));
    file.close();
    auto data_view = data | std::views::transform([](uint8_t val) -> float {
                       return static_cast<float>(val) / 255.0f - 0.5f;
                     });
    for (int i : std::views::iota(0, 10000)) {
      // std::cout << i << std::endl;
      label_.push_back(data[i * (32 * 32 * 3 + 1)]);
      // std::cout << "label: " << label_.back() << std::endl;
      auto image_data = data_view |
                        std::views::drop(i * (32 * 32 * 3 + 1) + 1) |
                        std::views::take(32 * 32 * 3);
      std::ranges::copy(image_data, std::back_inserter(image_));
    }
  }
  this->height_ = 32;
  this->width_ = 32;
  spdlog::info("Load data done");
}
}  // namespace my_tensor
