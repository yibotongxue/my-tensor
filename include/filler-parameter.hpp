// Copyright 2024 yibotongxue

#ifndef INCLUDE_FILLER_PARAMETER_HPP_
#define INCLUDE_FILLER_PARAMETER_HPP_

#include <memory>
#include <string>

#include "error.h"

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

inline FillerParameterPtr CreateFillerParameter(const std::string& init_name) {
  if (init_name == "zero") {
    return std::make_shared<ZeroFillerParameter>();
  } else if (init_name == "constant") {
    return std::make_shared<ConstantFillerParameter>();
  } else if (init_name == "xavier") {
    return std::make_shared<XavierFillerParameter>();
  } else if (init_name == "he") {
    return std::make_shared<HeFillerParameter>();
  } else {
    throw FillerError("Unimplemention error.");
  }
}

}  // namespace my_tensor

#endif  // INCLUDE_FILLER_PARAMETER_HPP_
