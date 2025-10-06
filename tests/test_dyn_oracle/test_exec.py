import unittest

from vif.falcon.oracle.guided_oracle.expressions import present
from vif.models.detection import SegmentationMask
from PIL import Image


class TestExec(unittest.TestCase):

    def test_simple_exec(self):

        original_features: list[SegmentationMask] = None
        custom_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "triangle")
        ]
        original_image: Image.Image = None
        custom_image: Image.Image = None

        globals = {"present": present}
        toev = """
def test_valid_customization():
    return present("triangle")
"""

        feat_dict = {
            hash(original_image.tobytes()): original_features,
            hash(custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        exec(toev, globals)
        expression = globals["test_valid_customization"]()
        result, feedback = expression.evaluate(
            original_image=original_image,
            custom_image=custom_image,
            segment_function=get_features,
        )

        self.assertTrue(result)

    def test_simple_exec_false(self):

        original_features: list[SegmentationMask] = None
        custom_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "triangl")
        ]
        original_image: Image.Image = None
        custom_image: Image.Image = None

        globals = {"present": present}
        toev = """
def test_valid_customization():
    return present("triangle")
"""
        exec(toev, globals)
        expression = globals["test_valid_customization"]()
        feat_dict = {
            hash(original_image.tobytes()): original_features,
            hash(custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        exec(toev, globals)
        expression = globals["test_valid_customization"]()
        result, feedback = expression.evaluate(
            original_image=original_image,
            custom_image=custom_image,
            segment_function=get_features,
        )

        self.assertFalse(result)
