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
  explicit Dataset(const std::string& data_files_path, bool is_train)
      : data_files_root_(data_files_path), is_train_(is_train) {}

  virtual ~Dataset() = default;

  virtual void LoadData() = 0;

  virtual int GetHeight() const = 0;
  virtual int GetWidth() const = 0;
  virtual int GetChannel() const = 0;
  virtual int GetSize() const = 0;
  virtual std::span<const float> GetImageSpanBetweenAnd(int start,
                                                        int end) const = 0;
  virtual std::span<const float> GetLabelSpanBetweenAnd(int start,
                                                        int end) const = 0;

 protected:
  std::string data_files_root_;
  bool is_train_;
};  // class Dataset

class LoadInMemoryDataset : public Dataset {
 public:
  explicit LoadInMemoryDataset(const std::string& data_files_path,
                               bool is_train)
      : Dataset(data_files_path, is_train) {}

  int GetHeight() const override { return height_; }
  int GetWidth() const override { return width_; }
  int GetSize() const override { return label_.size(); }
  std::span<const float> GetImageSpanBetweenAnd(int start,
                                                int end) const override {
    return {
        image_.data() + start * GetChannel() * height_ * width_,
        static_cast<size_t>((end - start) * height_ * GetChannel() * width_)};
  }
  std::span<const float> GetLabelSpanBetweenAnd(int start,
                                                int end) const override {
    return {label_.data() + start, static_cast<size_t>(end - start)};
  }

 protected:
  std::vector<float> image_;
  std::vector<float> label_;
  int height_;
  int width_;
};  // class LoadInMemoryDataset

class MnistDataset final : public LoadInMemoryDataset {
 public:
  explicit MnistDataset(const std::string& data_files_root, bool is_train)
      : LoadInMemoryDataset(data_files_root, is_train) {
    if (is_train) {
      image_file_path_ = data_files_root + "/train-images-idx3-ubyte";
      label_file_path_ = data_files_root + "/train-labels-idx1-ubyte";
    } else {
      image_file_path_ = data_files_root + "/t10k-images-idx3-ubyte";
      label_file_path_ = data_files_root + "/t10k-labels-idx1-ubyte";
    }
  }

  int GetChannel() const override { return 1; }

  void LoadData() override {
    ReadImageFile();
    ReadLabelFile();
  }

 private:
  void ReadImageFile();
  void ReadLabelFile();

  std::string image_file_path_;
  std::string label_file_path_;
};  // class MnistDataset

class Cifar10Dataset final : public LoadInMemoryDataset {
 public:
  explicit Cifar10Dataset(const std::string& data_files_root, bool is_train)
      : LoadInMemoryDataset(data_files_root, is_train) {
    if (is_train) {
      data_batches_ = {
          data_files_root + "/data_batch_1.bin",
          data_files_root + "/data_batch_2.bin",
          data_files_root + "/data_batch_3.bin",
          data_files_root + "/data_batch_4.bin",
          data_files_root + "/data_batch_5.bin",
      };
    } else {
      data_batches_ = {data_files_root + "/test_batch.bin"};
    }
  }

  int GetChannel() const override { return 3; }

  void LoadData() override;

 private:
  std::vector<std::string> data_batches_;
};  // class Cifar10Dataset

using DatasetPtr = std::shared_ptr<Dataset>;

inline std::function<DatasetPtr(const std::string&, bool)> GetDatasetCreater(
    const std::string& type) {
  if (type == "mnist") {
    return [](const std::string& data_files_root, bool is_train) {
      return std::make_shared<MnistDataset>(data_files_root, is_train);
    };
  } else if (type == "cifar-10") {
    return [](const std::string& data_files_root, bool is_train) {
      return std::make_shared<Cifar10Dataset>(data_files_root, is_train);
    };
  } else {
    throw DataError("Unsupported data set type.");
  }
}
}  // namespace my_tensor

#endif  // INCLUDE_DATASET_HPP_
