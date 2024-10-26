// Copyright 2024 yibotongxue

// Copyright [2024] [yibotongxue]
#ifndef INCLUDE_LAYER_PARAMETER_CUH_
#define INCLUDE_LAYER_PARAMETER_CUH_

#include <string>

namespace my_tensor {

// Copyright 2024 林毅波
//
// 本代码片段修改自 Moonshot AI 提供的示例代码
//
// 原始代码版权归 Moonshot AI 所有
//
// 许可证：MIT License
//
// 感谢 Moonshot AI 提供的示例代码
class LayerParameter {
 public:
  std::string name;
  std::string type;

  LayerParameter() : name(""), type("") {}

  virtual std::string Serialize() const {
    return "";
  }

  static LayerParameter Deserialize() {
    return LayerParameter();
  }
};

}  // namespace my_tensor

#endif  // INCLUDE_LAYER_PARAMETER_CUH_
