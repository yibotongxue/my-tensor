// Copyright 2024 yibotongxue

#ifndef INCLUDE_FILLER_PARAMETER_HPP_
#define INCLUDE_FILLER_PARAMETER_HPP_

#include <memory>

namespace my_tensor {

enum class InitMode { kZero, kXavier, kConstant, kHe };  // enum class InitMode

class FillerParameter {
 public:
  explicit FillerParameter(InitMode mode) : init_mode_(mode) {}
  virtual ~FillerParameter() = default;
  InitMode init_mode_;
};  // class FillerParameter

class ZeroFillerParameter final : public FillerParameter {
 public:
  ZeroFillerParameter() : FillerParameter(InitMode::kZero) {}
};  // class ZeroParameter

class ConstantFillerParameter final : public FillerParameter {
 public:
  ConstantFillerParameter() : FillerParameter(InitMode::kConstant) {}
  int val_;
};  // class ConstantFillerParameter

class XavierFillerParameter final : public FillerParameter {
 public:
  XavierFillerParameter() : FillerParameter(InitMode::kXavier) {}
  int n_in_;
  int n_out_;
};  // class XavierFillerParameter

class HeFillerParameter final : public FillerParameter {
 public:
  HeFillerParameter() : FillerParameter(InitMode::kHe) {}
  int n_;
};  // class HeFillerParameter

using FillerParameterPtr = std::shared_ptr<FillerParameter>;

inline FillerParameterPtr CreateFillerParameter(InitMode init) {
  switch (init) {
    case InitMode::kZero:
      return std::make_shared<ZeroFillerParameter>();
    case InitMode::kConstant:
      return std::make_shared<ConstantFillerParameter>();
    case InitMode::kXavier:
      return std::make_shared<XavierFillerParameter>();
    case InitMode::kHe:
      return std::make_shared<HeFillerParameter>();
    default:
      throw FillerError("Unimplemention error.");
  }
}

}  // namespace my_tensor

#endif  // INCLUDE_FILLER_PARAMETER_HPP_
