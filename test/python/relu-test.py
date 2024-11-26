import mytensor as ts
import torch
import numpy as np
import unittest

class ReluTest(unittest.TestCase):
    def setUp(self):
        input1 = np.random.uniform(-10.0, 10.0, 1000)
        input1.reshape((100, 10))
        input2 = input1.copy()
        self.tsinput = ts.Tensor.from_numpy(input1)
        self.torchinput = torch.from_numpy(input2)
        self.torchinput.requires_grad_(True)
        self.relu = ts.Relu()

    def test_cpu_forward(self):
        tsoutput = self.relu.forward(self.tsinput)
        torchoutput = torch.nn.functional.relu(self.torchinput)
        tsoutput_data = tsoutput.data()
        torchoutput_data = torchoutput.detach().numpy()
        self.assertTrue(np.allclose(tsoutput_data, torchoutput_data, atol=1e-3))

    def test_gpu_forward(self):
        self.tsinput.to_gpu()
        self.torchinput = self.torchinput.cuda()
        tsoutput = self.relu.forward(self.tsinput)
        torchoutput = torch.nn.functional.relu(self.torchinput)
        tsoutput_data = tsoutput.data()
        torchoutput_data = torchoutput.cpu().detach().numpy()
        self.assertTrue(np.allclose(tsoutput_data, torchoutput_data, atol=1e-3))

    def test_cpu_backward(self):
        tsoutput = self.relu.forward(self.tsinput)
        torchoutput = torch.nn.functional.relu(self.torchinput)
        output_grad1 = np.random.uniform(-10.0, 10.0, 1000)
        output_grad1.reshape((100, 10))
        output_grad2 = output_grad1.copy()
        tsoutput.set_grad(output_grad1)
        torchoutput_grad = torch.from_numpy(output_grad2)
        self.tsinput = self.relu.backward(tsoutput)
        torchinput_grad = torch.autograd.grad(torchoutput, self.torchinput, torchoutput_grad)[0]
        tsinput_grad = self.tsinput.grad()
        torchinput_grad = torchinput_grad.numpy()
        self.assertTrue(np.allclose(tsinput_grad, torchinput_grad, atol=1e-3))

    def test_gpu_backward(self):
        self.tsinput.to_gpu()
        self.torchinput = self.torchinput.cuda()
        tsoutput = self.relu.forward(self.tsinput)
        torchoutput = torch.nn.functional.relu(self.torchinput)
        output_grad1 = np.random.uniform(-10.0, 10.0, 1000)
        output_grad1.reshape((100, 10))
        output_grad2 = output_grad1.copy()
        tsoutput.set_grad(output_grad1)
        torchoutput_grad = torch.from_numpy(output_grad2).cuda()
        self.tsinput = self.relu.backward(tsoutput)
        torchinput_grad = torch.autograd.grad(torchoutput, self.torchinput, torchoutput_grad)[0]
        tsinput_grad = self.tsinput.grad()
        torchinput_grad = torchinput_grad.cpu().numpy()
        self.assertTrue(np.allclose(tsinput_grad, torchinput_grad, atol=1e-3))

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
