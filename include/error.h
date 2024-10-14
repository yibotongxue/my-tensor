#ifndef INCLUDE_ERROR_H_
#define INCLUDE_ERROR_H_

#include <stdexcept>

namespace my_tensor {
class ShapeError : public std::runtime_error {
public:
  using runtime_error::runtime_error;
};  // class ShapeError
}  // namespace my_tensor

#endif  // INCLUDE_ERROR_H_
