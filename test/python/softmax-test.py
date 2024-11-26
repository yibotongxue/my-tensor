import mytensor as ts
import torch
import numpy as np
import unittest

class SoftmaxTest(unittest.TestCase):
    def setUp(self):
        input1 = np.random.uniform(-10.0, 10.0, 1000)
        input1 = input1.reshape((100, 10))
        input2 = input1.copy()
        self.tsinput = ts.Tensor.from_numpy(input1)
        self.torchinput = torch.from_numpy(input2)
        self.torchinput.requires_grad_(True)
        self.softmax = ts.Softmax(10)

    def test_cpu_forward(self):
        tsoutput = self.softmax.forward(self.tsinput)
        torchoutput = torch.nn.functional.softmax(self.torchinput, -1)
        tsoutput_data = tsoutput.data()
        torchoutput_data = torchoutput.detach().numpy()
        self.assertTrue(np.allclose(tsoutput_data, torchoutput_data, 1e-3, 1e-3))

    def test_gpu_forward(self):
        self.tsinput.to_gpu()
        self.torchinput = self.torchinput.cuda()
        tsoutput = self.softmax.forward(self.tsinput)
        torchoutput = torch.nn.functional.softmax(self.torchinput, -1)
        tsoutput_data = tsoutput.data()
        torchoutput_data = torchoutput.cpu().detach().numpy()
        self.assertTrue(np.allclose(tsoutput_data, torchoutput_data, 1e-3, 1e-3))

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
