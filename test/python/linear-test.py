import mytensor as ts
import torch
import numpy as np
import unittest

class LinearTest(unittest.TestCase):
  def setUp(self):
    input1 = np.random.uniform(-10.0, 10.0, 1000)
    input1 = input1.reshape((100, 10))
    input2 = input1.copy()

    weight1 = np.random.uniform(-10.0, 10.0, 80)
    weight1 = weight1.reshape((10, 8))
    weight2 = weight1.copy()

    bias1 = np.zeros(8)
    bias2 = bias1.copy()

    self.tsinput = ts.Tensor.from_numpy(input1)
    self.torchinput = torch.from_numpy(input2)
    self.torchinput.requires_grad_(True)

    self.linear = ts.Linear(10, 8)
    self.linear.set_weight(ts.Tensor.from_numpy(weight1))
    self.linear.set_bias(ts.Tensor.from_numpy(bias1))

    self.torch_linear = torch.nn.Linear(10, 8)
    self.torch_linear.weight.data = torch.from_numpy(weight2).T
    self.torch_linear.bias.data = torch.from_numpy(bias2)

  def test_cpu_forward(self):
    tsoutput = self.linear.forward(self.tsinput)
    torchoutput = self.torch_linear.forward(self.torchinput)
    tsoutput_data = tsoutput.data()
    torchoutput_data = torchoutput.detach().numpy()
    self.assertTrue(np.allclose(tsoutput_data, torchoutput_data, atol=1e-3))

  def test_gpu_forward(self):
    self.tsinput.to_gpu()
    self.torchinput = self.torchinput.cuda()
    self.torch_linear = self.torch_linear.cuda()
    tsoutput = self.linear.forward(self.tsinput)
    torchoutput = self.torch_linear.forward(self.torchinput)
    tsoutput_data = tsoutput.data()
    torchoutput_data = torchoutput.cpu().detach().numpy()
    self.assertTrue(np.allclose(tsoutput_data, torchoutput_data, atol=1e-3))

  def tearDown(self):
    pass

if __name__ == '__main__':
  unittest.main()
