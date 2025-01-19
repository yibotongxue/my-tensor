// Copyright 2024 yibotongxue

#ifndef INCLUDE_DATASET_HPP_
#define INCLUDE_DATASET_HPP_

#include <cstdint>
#include <functional>
#include <memory>
#include <span>  // NOLINT
#include <string>
#include <vector>

#include "error.hpp"

namespace my_tensor {

class Dataset {
 public:
  explicit Dataset(const std::string& image_file_path,
                   const std::string& label_file_path)
      : image_file_path_(image_file_path), label_file_path_(label_file_path) {}

  virtual ~Dataset() = default;

  virtual void LoadData() = 0;

  virtual int GetHeight() const = 0;
  virtual int GetWidth() const = 0;
  virtual int GetSize() const = 0;
  virtual std::span<const float> GetImageSpanBetweenAnd(int start,
                                                        int end) const = 0;
  virtual std::span<const uint8_t> GetLabelSpanBetweenAnd(int start,
                                                          int end) const = 0;

 protected:
  std::string image_file_path_;
  std::string label_file_path_;
};  // class Dataset

class LoadInMemoryDataset : public Dataset {
 public:
  explicit LoadInMemoryDataset(const std::string& image_file_path,
                               const std::string& label_file_path)
      : Dataset(image_file_path, label_file_path) {}

  int GetHeight() const override { return height_; }
  int GetWidth() const override { return width_; }
  int GetSize() const override { return label_.size(); }
  std::span<const float> GetImageSpanBetweenAnd(int start,
                                                int end) const override {
    return {image_.data() + start * height_ * width_,
            static_cast<size_t>((end - start) * height_ * width_)};
  }
  std::span<const uint8_t> GetLabelSpanBetweenAnd(int start,
                                                  int end) const override {
    return {label_.data() + start, static_cast<size_t>(end - start)};
  }

 protected:
  std::vector<float> image_;
  std::vector<uint8_t> label_;
  int height_;
  int width_;
};  // class LoadInMemoryDataset

class MnistDataset final : public LoadInMemoryDataset {
 public:
  explicit MnistDataset(const std::string& image_file_path,
                        const std::string& label_file_path)
      : LoadInMemoryDataset(image_file_path, label_file_path) {}

  void LoadData() override {
    ReadImageFile();
    ReadLabelFile();
  }

 private:
  void ReadImageFile();
  void ReadLabelFile();
};  // class MnistDataset

using DatasetPtr = std::shared_ptr<Dataset>;

inline std::function<DatasetPtr(const std::string&, const std::string&)>
GetDatasetCreater(const std::string& type) {
  if (type == "mnist") {
    return [](const std::string& image_file_path,
              const std::string& label_file_path) -> DatasetPtr {
      return std::make_shared<MnistDataset>(image_file_path, label_file_path);
    };
  } else {
    throw DataError("Unsupported data set type.");
  }
}
}  // namespace my_tensor

#endif  // INCLUDE_DATASET_HPP_
