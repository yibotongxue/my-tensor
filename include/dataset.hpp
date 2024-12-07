// Copyright 2024 yibotongxue

#ifndef INCLUDE_DATASET_HPP_
#define INCLUDE_DATASET_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace my_tensor {
class Dataset {
 public:
  explicit Dataset(const std::string& image_file_path,
                   const std::string& label_file_path)
      : image_file_path_(image_file_path), label_file_path_(label_file_path) {}

  virtual ~Dataset() = default;

  void LoadData() {
    ReadImageFile();
    ReadLabelFile();
  }

  int GetHeight() const { return height_; }
  int GetWidth() const { return width_; }
  const std::vector<float>& GetImage() const { return image_; }
  const std::vector<uint8_t>& GetLabel() const { return label_; }
  int GetSize() const { return label_.size(); }

 protected:
  std::vector<float> image_;
  std::vector<uint8_t> label_;
  int height_;
  int width_;
  std::string image_file_path_;
  std::string label_file_path_;

  virtual void ReadImageFile() = 0;
  virtual void ReadLabelFile() = 0;
};  // class Dataset

class MnistDataset : public Dataset {
 public:
  explicit MnistDataset(const std::string& image_file_path,
                        const std::string& label_file_path)
      : Dataset(image_file_path, label_file_path) {}

 private:
  void ReadImageFile() override;
  void ReadLabelFile() override;
};  // class MnistDataset

using DatasetPtr = std::shared_ptr<Dataset>;
}  // namespace my_tensor

#endif  // INCLUDE_DATASET_HPP_
