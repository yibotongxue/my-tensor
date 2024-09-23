#ifndef MYTENSOR_INCLUDE_ERROR_H_
#define MYTENSOR_INCLUDE_ERROR_H_

#include <stdexcept>

namespace my_tensor {
class MemoryError : public std::runtime_error {
  using runtime_error::runtime_error;
};
}

#endif // MYTENSOR_INCLUDE_ERROR_H_
