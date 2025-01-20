// Copyright 2024 yibotongxue

#ifndef INCLUDE_DATA_LOADER_HPP_
#define INCLUDE_DATA_LOADER_HPP_

#include <array>
#include <memory>
#include <vector>

#include "dataset.hpp"
#include "tensor.hpp"

namespace my_tensor {

/**
 * @brief 数据加载器
 *
 * @details 数据加载器，用于加载数据集，实现了数据集的加载方法
 */
class DataLoader {
 public:
  DataLoader(DatasetPtr dataset, int batch_size)
      : dataset_(dataset), batch_size_(batch_size), index_(0) {}

  /**
   * @brief 判断是否有下一个批次
   *
   * @return 是否有下一个批次
   */
  [[nodiscard]] bool HasNext() const {
    return index_ + batch_size_ <= dataset_->GetSize();
  }

  /**
   * @brief 获取下一个批次
   *
   * @return 下一个批次
   */
  [[nodiscard]] std::array<std::shared_ptr<Tensor<float>>, 2> GetNext();

  /**
   * @brief 重置数据加载器
   */
  void Reset() noexcept {
    Shuffle();
    index_ = 0;
  }

  /**
   * @brief 获取图像形状
   *
   * @return 图像形状
   */
  [[nodiscard]] std::vector<int> GetImageShape() const {
    return {batch_size_, dataset_->GetChannel(), dataset_->GetHeight(),
            dataset_->GetWidth()};
  }

  /**
   * @brief 获取标签形状
   *
   * @return 标签形状
   */
  [[nodiscard]] std::vector<int> GetLabelShape() const { return {batch_size_}; }

  /**
   * @brief 打乱数据集
   *
   * @todo 实现数据集打乱方法
   */
  void Shuffle() noexcept {}

 private:
  // 数据集指针
  DatasetPtr dataset_;
  // 批次大小
  int batch_size_;
  // 索引
  int index_;
};  // class DataLoader
}  // namespace my_tensor

#endif  // INCLUDE_DATA_LOADER_HPP_
