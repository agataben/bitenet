import unittest
import torch
import numpy as np
import random
import os

from torch.utils.data import DataLoader
from training.training import train
from utils import set_seed
from torch import nn
from glob import glob

class TestTraining(unittest.TestCase):

    def setUp(self):
        # Set seed
        seed = 1238
        set_seed(seed)

        # Toy model
        in_features_n = 4
        out_features_n = 2
        self.toy_model = nn.Linear(in_features_n, out_features_n)

        # Toy loaders
        x = torch.tensor([1., 2., 3., 4.])
        y = torch.tensor(1)
        train_ds = [(x, y), (x, y)]
        test_ds = [(x, y), (x, y)]
        train = DataLoader(train_ds, batch_size = 1, num_workers = 0)
        test = DataLoader(test_ds, batch_size = 1, num_workers = 0)
        self.toy_loaders = {'train': train, 'test': test}

    def test_return_type(self):
        momentum = 0
        logdir = None
        ckpt_dir = None
        epochs = 1
        self.toy_model = train(self.toy_model, self.toy_loaders, momentum = momentum,
                               epochs = epochs, logdir = logdir, ckpt_dir = ckpt_dir)

        self.assertIsInstance(self.toy_model, nn.Linear)

    def test_weights_change(self):
        momentum = 0
        logdir = None
        ckpt_dir = None
        epochs = 1
        pre_train_prams = self.toy_model.parameters()
        self.toy_model = train(self.toy_model, self.toy_loaders, momentum = momentum,
                          epochs = epochs, logdir = logdir, ckpt_dir = ckpt_dir)
        post_train_params = self.toy_model.parameters()
        self.assertNotEqual(pre_train_prams, post_train_params)

    @unittest.skipIf(os.getenv("CI") == "true", "Skip in CI")
    def test_ckpt_dir_creation(self):
        momentum = 0
        logdir = None
        ckpt_dir = 'ckpt'
        epochs = 2
        self.toy_model = train(self.toy_model, self.toy_loaders, momentum = momentum,
                               epochs = epochs, logdir = logdir, ckpt_dir = ckpt_dir)
        try:
            os.makedirs(ckpt_dir)
        except Exception as e:
            self.assertIsInstance(e, FileExistsError)
            path = ckpt_dir + '/*'
            files = glob(path)

            self.assertEqual(len(files), 2)

