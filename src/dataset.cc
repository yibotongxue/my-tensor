// Copyright 2024 yibotongxue

#include "dataset.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <ranges>  // NOLINT
#include <vector>

#include "opencv2/opencv.hpp"

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

std::span<const float> ImageNetDataset::GetImageSpanBetweenAnd(int start,
                                                               int end) const {
  if (start < 0 || end > data_set_size_ || start >= end) {
    throw std::runtime_error("Invalid start and end index");
  }

  loaded_images_.clear();

  for (int i = start; i < end; ++i) {
    const std::string& image_path = image_paths_[indices_[i]];
    std::vector<float> image = LoadImage(image_path);
    loaded_images_.insert(loaded_images_.end(), image.begin(), image.end());
  }

  return {loaded_images_.data(), static_cast<size_t>(loaded_images_.size())};
}

std::span<const float> ImageNetDataset::GetLabelSpanBetweenAnd(int start,
                                                               int end) const {
  if (start < 0 || end > data_set_size_ || start >= end) {
    throw std::runtime_error("Invalid start and end index");
  }

  loaded_labels_.clear();

  for (int i = start; i < end; ++i) {
    const std::string& image_path = image_paths_[indices_[i]];
    int label = GetImageFolderIndex(indices_[i]);
    loaded_labels_.push_back(static_cast<float>(label));
  }

  return {loaded_labels_.data(), static_cast<size_t>(loaded_labels_.size())};
}

void ImageNetDataset::LoadData() {
  image_paths_.clear();
  image_indices_prefix_sum_.clear();

  for (const auto& entry :
       std::filesystem::directory_iterator(real_root_path_)) {
    if (entry.is_directory()) {
      const std::string folder_name = entry.path().filename().string();

      std::vector<std::string> images_in_folder =
          GetImagesInFolder(entry.path());
      image_paths_.insert(image_paths_.end(), images_in_folder.begin(),
                          images_in_folder.end());
      image_indices_prefix_sum_.push_back(image_paths_.size());
    }
  }
  data_set_size_ = image_paths_.size();
  indices_.resize(data_set_size_);
  std::ranges::copy(std::views::iota(0, data_set_size_), indices_.begin());
  std::random_device rd;
  std::mt19937 g(rd());
  std::ranges::shuffle(indices_, g);
  spdlog::info("Load data done");
}

std::vector<std::string> ImageNetDataset::GetImagesInFolder(
    const std::filesystem::path& folder_path) const {
  std::vector<std::string> image_paths;
  for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
    if (entry.is_regular_file() && IsImageFile(entry.path())) {
      image_paths.push_back(entry.path().string());
    }
  }
  return image_paths;
}

bool ImageNetDataset::IsImageFile(
    const std::filesystem::path& file_path) const {
  std::string extension = file_path.extension().string();
  std::ranges::transform(extension, extension.begin(), ::tolower);
  return extension == ".jpg" || extension == ".jpeg" || extension == ".png";
}

std::vector<float> ImageNetDataset::LoadImage(
    const std::string& image_path) const {
  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("Failed to load image: " + image_path);
  }

  cv::Mat resized_image;
  cv::resize(image, resized_image, cv::Size(224, 224));

  resized_image.convertTo(resized_image, CV_32F, 1.0 / 255.0);

  std::vector<float> image_data(resized_image.begin<float>(),
                                resized_image.end<float>());

  return image_data;
}

}  // namespace my_tensor
