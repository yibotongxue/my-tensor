import mytensor as ts
import numpy as np

dataset = ts.MnistDataset("../../data/train-images-idx3-ubyte", "../../data/train-labels-idx1-ubyte")

dataset.load_data()

print("Height:", dataset.get_height())
print("Width:", dataset.get_width())
print("Size:", dataset.get_size())

images = np.array(dataset.get_image()).reshape((-1, 28, 28))
labels = np.array(dataset.get_label())

images, labels = ts.Tensor.from_numpy(images), ts.Tensor.from_numpy(labels)

print(images.shape(), labels.shape())
