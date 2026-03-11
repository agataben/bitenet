import unittest
import os
import numpy as np
import random
import pandas as pd

from PIL import Image
from src.utils import set_seed
from src.utils import build_toy_dataset
from src.preprocessing import download_food101
from src.preprocessing import build_image_path_label_df
from src.preprocessing import split_dataset

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Set seed
        seed = 1238
        set_seed(seed)

        # Build toy dataset
        build_toy_dataset(samplings_n = 4)

    @unittest.skipIf(os.getenv("CI") == "true", "Skip in CI")
    def test_download_food101(self):
        data_root = download_food101()
        self.assertIn('/.cache/kagglehub/datasets/dansbecker/food-101/versions/1', data_root)

    def test_build_image_path_label_df(self):
        class_names, labels, dataset_df = build_image_path_label_df('tests', True)
        self.assertEqual(len(class_names), 4)
        self.assertIn('class_0', class_names)
        self.assertIn('class_1', class_names)
        self.assertIn('class_2', class_names)
        self.assertIn('class_3', class_names)
        self.assertEqual(len(labels), 4)
        self.assertIn(0, labels)
        self.assertIn(1, labels)
        self.assertIn(2, labels)
        self.assertIn(3, labels)
        self.assertEqual(len(dataset_df), 4)

    def test_split_dataset(self):
        class_names, labels, dataset_df = build_image_path_label_df('tests', True)
        train_df, test_df = split_dataset(dataset_df, train_size = 0.5, val_size = 0.0, test_size = 0.5)
        self.assertEqual(len(train_df), 2)
        self.assertEqual(len(test_df), 2)

