import unittest
import torch
import numpy as np
import random

from src.bitenet_v1 import BiteNetV1
from src.utils import set_seed
from PIL import Image

class TestBiteNetV1(unittest.TestCase):

    def setUp(self):
        # Set seed
        seed = 1238
        set_seed(seed)

    def test_output_size_conv_layers(self):
        model = BiteNetV1()
        x = torch.randn(1,3,224,224)
        x_1 = model.conv_1(x)
        x_2 = model.conv_2(x_1)
        x_3 = model.conv_3(x_2)

        self.assertEqual(x_1.shape, torch.Size([1,18,55,55]))
        self.assertEqual(x_2.shape, torch.Size([1,28,13,13]))
        self.assertEqual(x_3.shape, torch.Size([1,28,3,3]))

