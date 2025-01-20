// Copyright 2024 yibotongxue

#ifndef INCLUDE_COMMON_HPP_
#define INCLUDE_COMMON_HPP_

#include <memory>
#include <random>

namespace my_tensor {
/**
 * @class MyTensorContext
 * @brief 管理张量操作的上下文，包括设备类型和随机引擎。
 */
class MyTensorContext {
 public:
  /**
   * @enum DeviceType
   * @brief 表示设备类型（CPU或GPU）的枚举。
   */
  enum DeviceType : uint8_t { CPU, GPU };

  /**
   * @brief MyTensorContext的析构函数。
   */
  ~MyTensorContext();

  /**
   * @brief 删除的拷贝构造函数以防止复制。
   */
  MyTensorContext(const MyTensorContext&) = delete;

  /**
   * @brief 删除的拷贝赋值运算符以防止复制。
   */
  MyTensorContext& operator=(const MyTensorContext&) = delete;

  /**
   * @brief 获取MyTensorContext的单例实例。
   * @return 单例实例的引用。
   */
  static MyTensorContext& Get();

  /**
   * @brief 获取当前设备类型。
   * @return 当前设备类型。
   */
  inline static DeviceType device_type() { return Get().device_type_; }

  /**
   * @brief 设置设备类型。
   * @param type 要设置的设备类型。
   */
  inline static void set_device_type(DeviceType type) {
    Get().device_type_ = type;
  }

  /**
   * @brief 检查当前设备类型是否为CPU。
   * @return 如果设备类型是CPU则返回true，否则返回false。
   */
  inline static bool on_cpu() { return device_type() == CPU; }

  /**
   * @brief 检查当前设备类型是否为GPU。
   * @return 如果设备类型是GPU则返回true，否则返回false。
   */
  inline static bool on_gpu() { return device_type() == GPU; }

  /**
   * @brief 获取随机引擎。
   * @return 随机引擎的引用。
   */
  inline static std::mt19937& random_eigine() { return Get().random_engine_; }

 protected:
  DeviceType device_type_;      ///< 当前设备类型。
  std::mt19937 random_engine_;  ///< 随机引擎。

 private:
  /**
   * @brief 单例模式的私有构造函数。
   */
  MyTensorContext();
};  // class MyTensorContext
}  // namespace my_tensor

#endif  // INCLUDE_COMMON_HPP_
