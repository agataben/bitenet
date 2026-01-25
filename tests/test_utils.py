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


class TestUtils(unittest.TestCase):

    def setUp(self):
        # Set seed
        seed = 1238
        set_seed(seed)

        # Make a toy image for tests
        # Image wants array with uint8 values from 0 to 255 for RGB image
        array = (np.random.rand(384, 384, 3) * 255).astype(np.uint8)
        img = Image.fromarray(array)
        os.makedirs('tests/images/toy', exist_ok=True)
        img.save('tests/images/toy/1234.jpg')

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
        self.assertIn('images/toy/1234.jpg', img_paths_from_img_dir)

    def test_make_class_to_idx_map(self):
        img_dir_path = 'tests'
        img_dir_name = 'images'
        class_to_idx_dict = make_class_to_idx_map(img_dir_path, img_dir_name)
        self.assertIn('toy', class_to_idx_dict.keys())
        self.assertEqual(class_to_idx_dict['toy'], 0)

    def test_get_class_index_from_path(self):
        img_dir_path = 'tests'
        img_dir_name = 'images'
        class_to_idx_dict = make_class_to_idx_map(img_dir_path, img_dir_name)

        img_path = img_dir_path + '/' + img_dir_name + '/toy/1234.jpg'
        indx = get_class_index_from_path(img_path, class_to_idx_dict)
        self.assertEqual(indx, 0)

