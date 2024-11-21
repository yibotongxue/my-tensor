import mytensor as ts
import torch
import numpy as np
import unittest

class CrossEntropyLossTest(unittest.TestCase):
  def setUp(self):
    input1 = np.random.uniform(-10.0, 10.0, 1000)
    input1 = input1.reshape((100, 10))
    input2 = input1.copy()
    label1 = np.random.randint(0, 10, 100).astype(np.float32)
    label1 = label1.reshape((100,)).astype(np.longlong)
    label2 = label1.copy()
    self.tsinput = ts.Tensor.from_numpy(input1)
    self.torchinput = torch.from_numpy(input2)
    self.tslabel = ts.Tensor.from_numpy(label1)
    self.torchlabel = torch.from_numpy(label2)
    self.torchinput.requires_grad_(True)
    self.cross_entropy_loss = ts.CrossEntropyLoss(10)
    self.torch_cross_entropy_loss = torch.nn.CrossEntropyLoss()

  def test_cpu_forward(self):
    tsoutput = self.cross_entropy_loss.forward(self.tsinput, self.tslabel)
    torchoutput = self.torch_cross_entropy_loss.forward(self.torchinput, self.torchlabel)
    tsoutput_data = tsoutput.data()
    torchoutput_data = torchoutput.detach().numpy()
    self.assertTrue(np.allclose(tsoutput_data, torchoutput_data, atol=1e-3))

  def test_gpu_forward(self):
    self.tsinput.to_gpu()
    self.tslabel.to_gpu()
    self.torchinput = self.torchinput.cuda()
    self.torchinput.retain_grad()
    self.torchlabel = self.torchlabel.cuda()
    self.torch_cross_entropy_loss = self.torch_cross_entropy_loss.cuda()
    tsoutput = self.cross_entropy_loss.forward(self.tsinput, self.tslabel)
    torchoutput = self.torch_cross_entropy_loss.forward(self.torchinput, self.torchlabel)
    tsoutput_data = tsoutput.data()
    torchoutput_data = torchoutput.detach().cpu().numpy()
    self.assertTrue(np.allclose(tsoutput_data, torchoutput_data, atol=1e-3))


  def test_cpu_backward(self):
    tsoutput = self.cross_entropy_loss.forward(self.tsinput, self.tslabel)
    torchoutput = self.torch_cross_entropy_loss.forward(self.torchinput, self.torchlabel)
    tsoutput.set_grad([1])
    self.tsinput = self.cross_entropy_loss.backward(tsoutput)
    torchoutput.backward()
    tsinput_grad = self.tsinput.grad()
    torchinput_grad = self.torchinput.grad.numpy()
    self.assertTrue(np.allclose(tsinput_grad, torchinput_grad, atol=1e-3))

  def test_gpu_backward(self):
    self.tsinput.to_gpu()
    self.tslabel.to_gpu()
    self.torchinput = self.torchinput.cuda()
    self.torchinput.retain_grad()
    self.torchlabel = self.torchlabel.cuda()
    self.torch_cross_entropy_loss = self.torch_cross_entropy_loss.cuda()
    tsoutput = self.cross_entropy_loss.forward(self.tsinput, self.tslabel)
    torchoutput = self.torch_cross_entropy_loss.forward(self.torchinput, self.torchlabel)
    tsoutput.set_grad([1])
    self.tsinput = self.cross_entropy_loss.backward(tsoutput)
    torchoutput.backward()
    tsinput_grad = self.tsinput.grad()
    torchinput_grad = self.torchinput.grad.cpu().numpy()
    self.assertTrue(np.allclose(tsinput_grad, torchinput_grad, atol=1e-3))

  def tearDown(self):
    pass

if __name__ == '__main__':
  unittest.main()
