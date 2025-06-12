import unittest

import numpy as np

from vif.utils.image_utils import box_similarity_removal


class TestImageUtils(unittest.TestCase):
    def test_box_similarity_removal(self):
        """Expecting further similar boxes to be removed"""
        boxes = [(119, 137, 154, 171), (120, 137, 155, 171), (187, 137, 222, 171)]
        expected = [(119, 137, 154, 171), (187, 137, 222, 171)]
        result = box_similarity_removal(boxes)
        self.assertEqual(result, expected)
