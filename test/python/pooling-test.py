import mytensor as ts
import torch
import numpy as np
import unittest

class PoolingTest(unittest.TestCase):
  def setUp(self):
    input1 = np.random.uniform(-10.0, 10.0, 614400)
    input1 = input1.reshape((100, 3, 32, 64))
    input2 = input1.copy()
    self.tsinput = ts.Tensor.from_numpy(input1)
    self.torchinput = torch.from_numpy(input2)
    self.torchinput.requires_grad_(True)
    self.pooling = ts.Pooling(3, 2, 2)
    self.torch_pooling = torch.nn.MaxPool2d(2, 2)

  def test_cpu_forward(self):
    tsoutput = self.pooling.forward(self.tsinput)
    torchoutput = self.torch_pooling(self.torchinput)
    tsoutput_data = tsoutput.data()
    torchoutput_data = torchoutput.detach().numpy()
    self.assertTrue(np.allclose(tsoutput_data, torchoutput_data, atol=1e-3))

  def test_gpu_forward(self):
    self.tsinput.to_gpu()
    self.torchinput = self.torchinput.cuda()
    self.torchinput.retain_grad()
    self.torch_pooling = self.torch_pooling.cuda()
    tsoutput = self.pooling.forward(self.tsinput)
    torchoutput = self.torch_pooling(self.torchinput)
    tsoutput_data = tsoutput.data()
    torchoutput_data = torchoutput.cpu().detach().numpy()
    self.assertTrue(np.allclose(tsoutput_data, torchoutput_data, atol=1e-3))

  def test_cpu_backward(self):
    tsoutput = self.pooling.forward(self.tsinput)
    torchoutput = self.torch_pooling(self.torchinput)
    output_grad1 = np.random.uniform(-10.0, 10.0, 153600)
    tsoutput.set_grad(output_grad1)
    output_grad1 = output_grad1.reshape((100, 3, 16, 32))
    output_grad2 = output_grad1.copy()
    torchoutput_grad = torch.from_numpy(output_grad2)
    self.tsinput = self.pooling.backward(tsoutput)
    torchoutput.backward(torchoutput_grad)
    tsinput_grad = self.tsinput.grad()
    torchinput_grad = self.torchinput.grad.numpy()
    self.assertTrue(np.allclose(tsinput_grad, torchinput_grad, atol=1e-3))

  def test_gpu_backward(self):
    self.tsinput.to_gpu()
    self.torchinput = self.torchinput.cuda()
    self.torchinput.retain_grad()
    self.torch_pooling = self.torch_pooling.cuda()
    tsoutput = self.pooling.forward(self.tsinput)
    torchoutput = self.torch_pooling(self.torchinput)
    output_grad1 = np.random.uniform(-10.0, 10.0, 153600)
    tsoutput.set_grad(output_grad1)
    output_grad1 = output_grad1.reshape((100, 3, 16, 32))
    output_grad2 = output_grad1.copy()
    torchoutput_grad = torch.from_numpy(output_grad2).cuda()
    self.tsinput = self.pooling.backward(tsoutput)
    torchoutput.backward(torchoutput_grad)
    tsinput_grad = self.tsinput.grad()
    torchinput_grad = self.torchinput.grad.cpu().numpy()
    self.assertTrue(np.allclose(tsinput_grad, torchinput_grad, atol=1e-3))

  def tearDown(self):
    pass

if __name__ == '__main__':
  unittest.main()
