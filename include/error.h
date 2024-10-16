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
}  // namespace my_tensor

#endif  // INCLUDE_ERROR_H_
