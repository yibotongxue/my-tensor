// Copyright 2024 yibotongxue

#ifndef INCLUDE_ERROR_HPP_
#define INCLUDE_ERROR_HPP_

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

class FlattenError : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};  // class FlattenError

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

class AccuracyError : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};  // class AccuracyError

class BatchNormError : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};  // class BatchNormError

class FillerError : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};  // class FillerError

class NetError : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};  // class NetError

class DataError : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};  // class DataError

class SolverError : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};  // class SolverError

class SchedulerError : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};  // class SchedulerError
}  // namespace my_tensor

#endif  // INCLUDE_ERROR_HPP_
