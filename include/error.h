// Copyright 2024 yibotongxue

#ifndef INCLUDE_ERROR_H_
#define INCLUDE_ERROR_H_

#include <stdexcept>

namespace my_tensor {
class VectorError : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};  // class VectorError

class ShapeError : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};  // class ShapeError

class BlasError : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};  // class BlasError

class LinearError : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};  // class LinearError

class Im2colError : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};  // class Im2colError

class ConvError : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};  // class ConvError
}  // namespace my_tensor

#endif  // INCLUDE_ERROR_H_
