import unittest
import torch
import os
import numpy as np
import random
import yaml

from PIL import Image
from src.utils import set_seed
from src.utils import AverageValueMeter
from src.utils import calculate_mean_and_std
from src.utils import get_list_of_img_paths
from src.utils import make_class_to_idx_map
from src.utils import get_class_index_from_path
from src.utils import build_toy_dataset


class TestUtils(unittest.TestCase):

    def setUp(self):
        # Set seed
        seed = 1238
        set_seed(seed)

        # Build toy dataset
        build_toy_dataset(samplings_n = 1, csv_path = 'tests')

    def test_average_value_meter(self):
        meter = AverageValueMeter()
        meter.add(1,2)
        meter.add(3,4)
        self.assertAlmostEqual(meter.value(), (1 * 2 + 3 * 4) / 6)

        meter.reset()
        self.assertIsNone(meter.value())

    def test_calculate_mean_and_std(self):
        csv_path = 'tests/toy_csv.csv'
        data_root = 'tests'
        m, s = calculate_mean_and_std(csv_path, data_root)
        self.assertAlmostEqual(m[0].item(), 0.5, places = 2)
        self.assertAlmostEqual(m[1].item(), 0.5, places = 2)
        self.assertAlmostEqual(m[2].item(), 0.5, places = 2)
        self.assertAlmostEqual(s[0].item(), 0.21, places = 2)
        self.assertAlmostEqual(s[1].item(), 0.21, places = 2)
        self.assertAlmostEqual(s[2].item(), 0.21, places = 2)

    def test_mean_and_std_json_generation(self):
        csv_path = 'tests/toy_csv.csv'
        data_root = 'tests'
        yaml_path = 'tests'
        m, s = calculate_mean_and_std(csv_path, data_root, yaml_path)
        path = 'tests/*'
        existing_paths = glob(path)
        self.assertIn('tests/normalizzation.yaml', existing_paths)

    def test_mean_and_std_json_is_correct(self):
        csv_path = 'tests/toy_csv.csv'
        data_root = 'tests'
        yaml_path = 'tests'
        m, s = calculate_mean_and_std(csv_path, data_root, yaml_path)

        with open(os.path.join(yaml_path, 'normalizzation.yaml'), 'r') as f:
            config = yaml.safe_load(f)
        self.assertAlmostEqual(config['mean'][0], 0.5, places = 2)
        self.assertAlmostEqual(config['mean'][1], 0.5, places = 2)
        self.assertAlmostEqual(config['mean'][2], 0.5, places = 2)
        self.assertAlmostEqual(config['std'][0], 0.21, places = 2)
        self.assertAlmostEqual(config['std'][1], 0.21, places = 2)
        self.assertAlmostEqual(config['std'][2], 0.21, places = 2)

    def test_get_norm_parameters(self):
        csv_path = 'tests/toy_csv.csv'
        data_root = 'tests'
        yaml_path = 'tests'
        m, s = calculate_mean_and_std(csv_path, data_root, yaml_path)
        mean, std = get_norm_parameters(yaml_path)
        self.assertIsInstance(mean, torch.Tensor)
        self.assertIsInstance(std, torch.Tensor)
        self.assertAlmostEqual(mean[0].item(), 0.5, places = 2)
        self.assertAlmostEqual(mean[1].item(), 0.5, places = 2)
        self.assertAlmostEqual(mean[2].item(), 0.5, places = 2)
        self.assertAlmostEqual(std[0].item(), 0.21, places = 2)
        self.assertAlmostEqual(std[1].item(), 0.21, places = 2)
        self.assertAlmostEqual(std[2].item(), 0.21, places = 2)

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

