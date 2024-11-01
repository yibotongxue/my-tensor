// Copyright 2024 yibotongxue

#ifndef INCLUDE_FILLER_PARAMETER_HPP_
#define INCLUDE_FILLER_PARAMETER_HPP_

#include <memory>

namespace my_tensor {

enum class InitMode { kXavier, kConstant };  // enum class InitMode

class FillerParameter {
 public:
  explicit FillerParameter(InitMode mode) : init_mode_(mode) {}
  virtual ~FillerParameter() = default;
  InitMode init_mode_;
};  // class FillerParameter

class ConstantFillerParameter final : public FillerParameter {
 public:
  explicit ConstantFillerParameter(const int val)
      : FillerParameter(InitMode::kConstant), val_(val) {}
  int val_;
};  // class ConstantFillerParameter

class XavierFillerParameter final : public FillerParameter {
 public:
  XavierFillerParameter() : FillerParameter(InitMode::kXavier) {}
};  // class XavierFillerParameter

using FillerParameterPtr = std::shared_ptr<FillerParameter>;

}  // namespace my_tensor

#endif  // INCLUDE_FILLER_PARAMETER_HPP_
