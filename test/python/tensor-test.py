import mytensor as ts
import numpy as np

print(ts.Tensor)

tensor = ts.Tensor((1, 2))

print(tensor.shape())

tensor.reshape((2, 1))

print(tensor.shape())

tensor.set_data(np.array([[2], [3]]))

numpy_tensor = np.array(tensor)
print(numpy_tensor)
