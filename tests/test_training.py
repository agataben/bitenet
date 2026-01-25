import unittest
import torch
import numpy as np
import random

from training.training import train
from utils import set_seed
from torch import nn

class TestTraining(unittest.TestCase):

    def setUp(self):
        # Set seed
        seed = 1238
        set_seed(seed)

        # Toy model
        in_features_n = 4
        out_features_n = 2
        toy_model = nn.Linear(in_features_n,out_features_n)

    def test_default_training_parameters(self):
        pass

