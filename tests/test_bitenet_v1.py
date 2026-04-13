import unittest
import torch
import numpy as np
import random

from src.bitenet_v1 import BiteNetV1
from src.utils import set_seed
from src.utils import build_toy_dataset
from src.utils import calculate_mean_and_std
from src.utils import make_class_to_idx_map

class TestBiteNetV1(unittest.TestCase):

    def setUp(self):
        # Set seed
        seed = 1238
        set_seed(seed)

        # Build toy dataset
        build_toy_dataset(samplings_n = 27, csv_path = 'tests')

    def test_output_size_conv_layers(self):
        model = BiteNetV1()
        x = torch.randn(1,3,224,224)
        x_1 = model.conv_1(x)
        x_2 = model.conv_2(x_1)
        x_3 = model.conv_3(x_2)

        self.assertEqual(x_1.shape, torch.Size([1,18,55,55]))
        self.assertEqual(x_2.shape, torch.Size([1,28,13,13]))
        self.assertEqual(x_3.shape, torch.Size([1,28,3,3]))

    def test_predict(self):
        img_dir_path = 'tests'
        img_dir_name = 'images'
        csv_path = 'tests'
        _ = make_class_to_idx_map(img_dir_path, img_dir_name, csv_path, for_test = True)

        img_path = 'tests/images/class_0/0.jpg'
        yaml_path = 'tests'
        csv_file = 'tests/toy_csv.csv'
        data_root = 'tests'
        _, _ = calculate_mean_and_std(csv_file, data_root, yaml_path)

        model = BiteNetV1()
        prediction = model.predict(img_path)
        self.assertIsInstance(prediction, torch.Tensor)

