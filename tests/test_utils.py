import unittest
import torch
import os
import numpy as np
import random

from PIL import Image
from utils import set_seed
from utils import AverageValueMeter
from utils import calculate_mean_and_std
from utils import get_list_of_img_paths
from utils import make_class_to_idx_map
from utils import get_class_index_from_path
from utils import build_toy_dataset


class TestUtils(unittest.TestCase):

    def setUp(self):
        # Set seed
        seed = 1238
        set_seed(seed)

        # Build toy dataset
        build_toy_dataset(samplings_n = 1)

    def test_average_value_meter(self):
        meter = AverageValueMeter()
        meter.add(1,2)
        meter.add(3,4)
        self.assertAlmostEqual(meter.value(), (1 * 2 + 3 * 4) / 6)

        meter.reset()
        self.assertIsNone(meter.value())

    def test_calculate_mean_and_std(self):
        # 1*2*2
        a = torch.Tensor([[[1,1], [1,1]]])
        # 1*2*2
        b = torch.Tensor([[[1,1], [1,1]]])
        x = [[a , 'a'] , [b , 'b']]
        mean, std = calculate_mean_and_std(x)
        self.assertIsInstance(mean , torch.Tensor)
        self.assertAlmostEqual(mean , torch.Tensor([1]))
        self.assertAlmostEqual(std , torch.Tensor([0]))

    def test_get_list_of_img_paths(self):
        img_dir_path = 'tests'
        img_dir_name = 'images'
        img_paths_from_img_dir = get_list_of_img_paths(img_dir_path, img_dir_name)
        self.assertIsInstance(img_paths_from_img_dir, list)
        self.assertIn('images/class_0/0.jpg', img_paths_from_img_dir)

    def test_make_class_to_idx_map(self):
        img_dir_path = 'tests'
        img_dir_name = 'images'
        class_to_idx_dict = make_class_to_idx_map(img_dir_path, img_dir_name)
        self.assertIn('class_0', class_to_idx_dict.keys())
        self.assertEqual(class_to_idx_dict['class_0'], 0)

    def test_get_class_index_from_path(self):
        img_dir_path = 'tests'
        img_dir_name = 'images'
        class_to_idx_dict = make_class_to_idx_map(img_dir_path, img_dir_name)

        img_path = img_dir_path + '/' + img_dir_name + '/class_0/0.jpg'
        indx = get_class_index_from_path(img_path, class_to_idx_dict)
        self.assertEqual(indx, 0)

