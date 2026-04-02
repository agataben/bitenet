import unittest

from src.bitenet_v3 import BiteNetV3


class TestBiteNetV3(unittest.TestCase):

    def test_output_size(self):
        model = BiteNetV3()
        output_size = model.structure.fc.out_features
        self.assertEqual(output_size, 27)

