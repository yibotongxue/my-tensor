// Copyright 2024 yibotongxue

#ifndef INCLUDE_ERROR_H_
#define INCLUDE_ERROR_H_

#include <stdexcept>

namespace my_tensor {
class FileError : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};  // class FileError

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

class LayerError : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};  // class LayerError

class ReluError : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};  // class ReluError

class SigmoidError : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};  // class SigmoidError

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

class PoolingError : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};  // class PoolingError

class SoftmaxError : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};  // class SoftmaxError

class LossWithSoftmaxError : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};  // class LossWithSoftmaxError

class FillerError : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};  // class FillerError
}  // namespace my_tensor

#endif  // INCLUDE_ERROR_H_
