import unittest
import torch
import numpy as np
import random

from utils import set_seed
from utils import AverageValueMeter
from utils import calculate_mean_and_std
from utils import get_list_of_img_paths
from utils import make_class_to_idx_map
from utils import get_class_index_from_path


class TestUtils(unittest.TestCase):

    def setUp(self):
        # Set seed
        seed = 1238
        set_seed(seed)

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
        # TODO: change the path
        img_dir_path = '/Users/agatabenvegna/ML_LAB'
        img_dir_name = 'images'
        img_paths_from_img_dir = get_list_of_img_paths(img_dir_path, img_dir_name)
        self.assertIsInstance(img_paths_from_img_dir, list)
        self.assertIn('images/churros/77767.jpg', img_paths_from_img_dir)

    def test_make_class_to_idx_map(self):
        # TODO: change the path
        img_dir_path = '/Users/agatabenvegna/ML_LAB'
        img_dir_name = 'images'
        class_to_idx_dict = make_class_to_idx_map(img_dir_path, img_dir_name)
        self.assertIn('churros', class_to_idx_dict.keys())
        self.assertEqual(class_to_idx_dict['churros'], 23)

    def test_get_class_index_from_path(self):
        pass

