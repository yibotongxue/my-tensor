import mytensor as ts
import torch
import numpy as np
import unittest

class ConvolutionTest(unittest.TestCase):
  def setUp(self):
    input1 = np.random.uniform(-10.0, 10.0, 614400)
    input1 = input1.reshape((100, 3, 32, 64))
    input2 = input1.copy()

    kernel1 = np.random.uniform(-10.0, 10.0, 135)
    kernel1 = kernel1.reshape((5, 3, 3, 3))
    kernel2 = kernel1.copy()

    bias1 = np.random.uniform(-10.0, 10.0, 5)
    bias2 = bias1.copy()

    self.tsinput = ts.Tensor.from_numpy(input1)
    self.torchinput = torch.from_numpy(input2)
    self.torchinput.requires_grad_(True)

    self.conv = ts.Conv(3, 5, 3)
    self.conv.set_kernel(ts.Tensor.from_numpy(kernel1))
    self.conv.set_bias(ts.Tensor.from_numpy(bias1))

    self.torch_conv = torch.nn.Conv2d(3, 5, 3, padding=1)
    self.torch_conv.weight.data = torch.from_numpy(kernel2)
    self.torch_conv.bias.data = torch.from_numpy(bias2)

  def test_cpu_forward(self):
    tsoutput = self.conv.forward(self.tsinput)
    torchoutput = self.torch_conv.forward(self.torchinput)
    tsoutput_data = tsoutput.data()
    torchoutput_data = torchoutput.detach().numpy()
    self.assertTrue(np.allclose(tsoutput_data, torchoutput_data, atol=1e-3))

  def test_gpu_forward(self):
    self.tsinput.to_gpu()
    self.torchinput = self.torchinput.cuda()
    self.torch_conv = self.torch_conv.cuda()
    tsoutput = self.conv.forward(self.tsinput)
    torchoutput = self.torch_conv.forward(self.torchinput)
    tsoutput_data = tsoutput.data()
    torchoutput_data = torchoutput.cpu().detach().numpy()
    self.assertTrue(np.allclose(tsoutput_data, torchoutput_data, atol=1e-3))

  def test_cpu_backward_input(self):
    tsoutput = self.conv.forward(self.tsinput)
    torchoutput = self.torch_conv.forward(self.torchinput)
    grad_output = np.random.uniform(-10.0, 10.0, 1024000).astype(np.float32)
    tsoutput.set_grad(grad_output)
    grad_output = grad_output.reshape((100, 5, 32, 64))
    torch_grad_output = torch.from_numpy(grad_output)
    self.tsinput = self.conv.backward(tsoutput)
    tsinput_grad = self.tsinput.grad()
    torchoutput.backward(torch_grad_output)
    torchinput_grad = self.torchinput.grad.numpy()
    self.assertTrue(np.allclose(tsinput_grad, torchinput_grad, 1e-3, 1e-3))

  def test_gpu_backward_input(self):
    self.tsinput.to_gpu()
    self.torchinput = self.torchinput.cuda()
    self.torchinput.retain_grad()
    self.torch_conv = self.torch_conv.cuda()
    tsoutput = self.conv.forward(self.tsinput)
    torchoutput = self.torch_conv.forward(self.torchinput)
    grad_output = np.random.uniform(-10.0, 10.0, 1024000).astype(np.float32)
    tsoutput.set_grad(grad_output)
    grad_output = grad_output.reshape((100, 5, 32, 64))
    torch_grad_output = torch.from_numpy(grad_output).cuda()
    self.tsinput = self.conv.backward(tsoutput)
    tsinput_grad = self.tsinput.grad()
    torchoutput.backward(torch_grad_output)
    torchinput_grad = self.torchinput.grad.cpu().numpy()
    self.assertTrue(np.allclose(tsinput_grad, torchinput_grad, 1e-3, 1e-3))

  def test_cpu_backward_kernel(self):
    tsoutput = self.conv.forward(self.tsinput)
    torchoutput = self.torch_conv.forward(self.torchinput)
    grad_output = np.random.uniform(-10.0, 10.0, 1024000).astype(np.float32)
    tsoutput.set_grad(grad_output)
    grad_output = grad_output.reshape((100, 5, 32, 64))
    torch_grad_output = torch.from_numpy(grad_output)
    self.tsinput = self.conv.backward(tsoutput)
    tskernel_grad = self.conv.kernel().grad()
    torchoutput.backward(torch_grad_output)
    torchkernel_grad = self.torch_conv.weight.grad.numpy()
    self.assertTrue(np.allclose(tskernel_grad, torchkernel_grad, 1e-3))

  def test_gpu_backward_kernel(self):
    self.tsinput.to_gpu()
    self.torchinput = self.torchinput.cuda()
    self.torchinput.retain_grad()
    self.torch_conv = self.torch_conv.cuda()
    tsoutput = self.conv.forward(self.tsinput)
    torchoutput = self.torch_conv.forward(self.torchinput)
    grad_output = np.random.uniform(-10.0, 10.0, 1024000).astype(np.float32)
    tsoutput.set_grad(grad_output)
    grad_output = grad_output.reshape((100, 5, 32, 64))
    torch_grad_output = torch.from_numpy(grad_output).cuda()
    self.tsinput = self.conv.backward(tsoutput)
    tskernel_grad = self.conv.kernel().grad()
    torchoutput.backward(torch_grad_output)
    torchkernel_grad = self.torch_conv.weight.grad.cpu().numpy()
    self.assertTrue(np.allclose(tskernel_grad, torchkernel_grad, 1e-3))

  def test_cpu_backward_bias(self):
    tsoutput = self.conv.forward(self.tsinput)
    torchoutput = self.torch_conv.forward(self.torchinput)
    grad_output = np.random.uniform(-10.0, 10.0, 1024000).astype(np.float32)
    tsoutput.set_grad(grad_output)
    grad_output = grad_output.reshape((100, 5, 32, 64))
    torch_grad_output = torch.from_numpy(grad_output)
    self.tsinput = self.conv.backward(tsoutput)
    tsbias_grad = self.conv.bias().grad()
    torchoutput.backward(torch_grad_output)
    torchbias_grad = self.torch_conv.bias.grad.numpy()
    self.assertTrue(np.allclose(tsbias_grad, torchbias_grad, 1e-3))

  def test_gpu_backward_bias(self):
    self.tsinput.to_gpu()
    self.torchinput = self.torchinput.cuda()
    self.torchinput.retain_grad()
    self.torch_conv = self.torch_conv.cuda()
    tsoutput = self.conv.forward(self.tsinput)
    torchoutput = self.torch_conv.forward(self.torchinput)
    grad_output = np.random.uniform(-10.0, 10.0, 1024000).astype(np.float32)
    tsoutput.set_grad(grad_output)
    grad_output = grad_output.reshape((100, 5, 32, 64))
    torch_grad_output = torch.from_numpy(grad_output).cuda()
    self.tsinput = self.conv.backward(tsoutput)
    tsbias_grad = self.conv.bias().grad()
    torchoutput.backward(torch_grad_output)
    torchbias_grad = self.torch_conv.bias.grad.cpu().numpy()
    self.assertTrue(np.allclose(tsbias_grad, torchbias_grad, 1e-3))

  def tearDown(self):
    pass

if __name__ == '__main__':
  unittest.main()
