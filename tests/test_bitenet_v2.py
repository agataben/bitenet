import unittest

from src.bitenet_v2 import BiteNetV2


class TestBiteNetV2(unittest.TestCase):

    def test_output_size(self):
        model = BiteNetV2()
        output_size = model.structure.classifier[6].out_features
        self.assertEqual(output_size, 27)

