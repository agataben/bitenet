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
        pass

    def test_get_list_of_img_paths(self):
        pass

    def test_make_class_to_idx_map(self):
        pass

    def test_get_class_index_from_path(self):
        pass

