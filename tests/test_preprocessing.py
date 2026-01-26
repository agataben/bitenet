import unittest
import os
import numpy as np
import random

from PIL import Image
from utils import set_seed
from dataset.preprocessing import download_food101
from dataset.preprocessing import build_image_path_label_df

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Set seed
        seed = 1238
        set_seed(seed)

        # Make a toy image for tests
        # Image wants array with uint8 values from 0 to 255 for RGB image
        # TODO: make a function in utils
        array = (np.random.rand(384, 384, 3) * 255).astype(np.uint8)
        img = Image.fromarray(array)
        os.makedirs('tests/images/toy', exist_ok=True)
        img.save('tests/images/toy/1234.jpg')

    def test_download_food101(self):
        pass

    def test_build_image_path_label_df(self):
        class_names, labels, data_df = build_image_path_label_df('tests', True)
        self.assertEqual(len(class_names), 1)
        self.assertEqual(class_names[0], 'toy')
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0], 0)
        self.assertEqual(data_df.loc[0,'path'], 'images/toy/1234.jpg')
        self.assertEqual(data_df.loc[0,'label'], 0)

