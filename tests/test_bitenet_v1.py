import unittest
import torch
import numpy as np
import random

from models.bitenet_v1.bitenet_v1 import BiteNetV1
from PIL import Image

class TestBiteNetV1(unittest.TestCase):

    def setUp(self):
        # Set seed
        seed = 1238
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)


    def test_output_size_conv_layers(self):
        model = BiteNetV1()
        x = torch.randn(1,3,384,384)
        x_1 = model.conv_1(x)
        x_2 = model.conv_2(x_1)
        x_3 = model.conv_3(x_2)

        self.assertEqual(x_1.shape, torch.Size([1,18,95,95]))
        self.assertEqual(x_2.shape, torch.Size([1,28,23,23]))
        self.assertEqual(x_3.shape, torch.Size([1,28,5,5]))

