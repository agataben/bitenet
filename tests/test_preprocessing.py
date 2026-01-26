import unittest
import os
import numpy as np
import random

from PIL import Image
from utils import set_seed
from utils import build_toy_dataset
from dataset.preprocessing import download_food101
from dataset.preprocessing import build_image_path_label_df
from dataset.preprocessing import split_dataset

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Set seed
        seed = 1238
        set_seed(seed)

        # Build toy dataset
        build_toy_dataset(samplings_n = 1)

    def test_download_food101(self):
        pass

    def test_build_image_path_label_df(self):
        class_names, labels, data_df = build_image_path_label_df('tests', True)
        self.assertEqual(len(class_names), 1)
        self.assertEqual(class_names[0], 'class_0')
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0], 0)
        self.assertEqual(data_df.loc[0,'path'], 'images/class_0/0.jpg')
        self.assertEqual(data_df.loc[0,'label'], 0)

    def test_split_dataset(self):
        pass

