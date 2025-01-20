// Copyright 2024 yibotongxue

#ifndef INCLUDE_DATASET_HPP_
#define INCLUDE_DATASET_HPP_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>  // NOLINT
#include <string>
#include <vector>

#include "error.hpp"

namespace my_tensor {

/**
 * @brief 数据集基类
 *
 * @details 数据集基类，定义了数据集的基本属性和方法，包括以下公共接口
 * - LoadData: 加载数据集
 * - GetHeight: 获取图像高度
 * - GetWidth: 获取图像宽度
 * - GetChannel: 获取图像通道数
 * - GetSize: 获取数据集大小
 * - GetImageSpanBetweenAnd: 获取指定范围内的图像数据
 * - GetLabelSpanBetweenAnd: 获取指定范围内的标签数据
 */
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

/**
 * @brief 内存加载数据集，用于可以全部加载到内存的数据集
 *
 * @details 内存加载数据集，继承自数据集基类，实现了数据集的加载方法
 */
class LoadInMemoryDataset : public Dataset {
 public:
  explicit LoadInMemoryDataset(const std::string& data_files_path,
                               bool is_train)
      : Dataset(data_files_path, is_train) {}

  [[nodiscard]] int GetHeight() const override { return height_; }
  [[nodiscard]] int GetWidth() const override { return width_; }
  [[nodiscard]] int GetSize() const override { return label_.size(); }
  [[nodiscard]] std::span<const float> GetImageSpanBetweenAnd(
      int start, int end) const override {
    return {
        image_.data() + start * GetChannel() * height_ * width_,
        static_cast<size_t>((end - start) * height_ * GetChannel() * width_)};
  }
  [[nodiscard]] std::span<const float> GetLabelSpanBetweenAnd(
      int start, int end) const override {
    return {label_.data() + start, static_cast<size_t>(end - start)};
  }

 protected:
  std::vector<float> image_;
  std::vector<float> label_;
  int height_;
  int width_;
};  // class LoadInMemoryDataset

/**
 * @brief MNIST数据集
 *
 * @details MNIST数据集，继承自内存加载数据集，实现了数据集的加载方法
 */
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

  [[nodiscard]] int GetChannel() const override { return 1; }

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

/**
 * @brief CIFAR-10数据集
 *
 * @details CIFAR-10数据集，继承自内存加载数据集，实现了数据集的加载方法
 */
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

  [[nodiscard]] constexpr int GetChannel() const override { return 3; }

  void LoadData() override;

 private:
  std::vector<std::string> data_batches_;
};  // class Cifar10Dataset

class ImageNetDataset final : public Dataset {
 public:
  explicit ImageNetDataset(const std::string& data_files_root, bool is_train)
      : Dataset(data_files_root, is_train) {
    if (is_train) {
      real_root_path_ = data_files_root + "/train";
    } else {
      real_root_path_ = data_files_root + "/val";
    }
  }

  [[nodiscard]] int GetHeight() const override { return 224; }
  [[nodiscard]] int GetWidth() const override { return 224; }
  [[nodiscard]] int GetChannel() const override { return 3; }
  [[nodiscard]] int GetSize() const override { return data_set_size_; }

  [[nodiscard]] std::span<const float> GetImageSpanBetweenAnd(
      int start, int end) const override;

  [[nodiscard]] std::span<const float> GetLabelSpanBetweenAnd(
      int start, int end) const override;

  void LoadData() override;

 private:
  std::vector<int> indices_;
  std::vector<std::string> image_paths_;
  std::vector<int> image_indices_prefix_sum_;
  int data_set_size_;
  std::string real_root_path_;

  mutable std::vector<float> loaded_images_;
  mutable std::vector<float> loaded_labels_;

  [[nodiscard]] constexpr int GetImageFolderIndex(int index) const {
    auto it = std::upper_bound(image_indices_prefix_sum_.begin(),
                               image_indices_prefix_sum_.end(), index);
    return static_cast<int>(it - image_indices_prefix_sum_.begin()) - 1;
  }

  [[nodiscard]] constexpr int GetImageIndexInFolder(int index) const {
    return index - image_indices_prefix_sum_[GetImageFolderIndex(index)];
  }

  [[nodiscard]] std::vector<std::string> GetImagesInFolder(
      const std::filesystem::path& folder_path) const;

  [[nodiscard]] bool IsImageFile(const std::filesystem::path& file_path) const;

  [[nodiscard]] std::vector<float> LoadImage(
      const std::string& image_path) const;
};  // class ImageNetDataset

using DatasetPtr = std::shared_ptr<Dataset>;

/**
 * @brief 获取数据集创建函数
 *
 * @details 获取数据集创建函数，根据数据集类型返回对应的数据集创建函数
 *
 * @param type 数据集类型
 * @return 数据集创建函数
 */
[[nodiscard]] inline std::function<DatasetPtr(const std::string&, bool)>
GetDatasetCreater(const std::string& type) {
  if (type == "mnist") {
    return [](const std::string& data_files_root, bool is_train) -> DatasetPtr {
      return std::make_shared<MnistDataset>(data_files_root, is_train);
    };
  } else if (type == "cifar-10") {
    return [](const std::string& data_files_root, bool is_train) -> DatasetPtr {
      return std::make_shared<Cifar10Dataset>(data_files_root, is_train);
    };
  } else if (type == "imagenet") {
    return [](const std::string& data_files_root, bool is_train) -> DatasetPtr {
      return std::make_shared<ImageNetDataset>(data_files_root, is_train);
    };
  } else {
    throw DataError("Unsupported data set type.");
  }
}
}  // namespace my_tensor

#endif  // INCLUDE_DATASET_HPP_
