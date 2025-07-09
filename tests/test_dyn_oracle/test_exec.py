import unittest

from vif.falcon.oracle.guided_oracle.expressions import added
from vif.models.detection import SegmentationMask
from PIL import Image

class TestExec(unittest.TestCase):

    def test_simple_exec(self):

        original_features: list[SegmentationMask] = None
        custom_features: list[SegmentationMask] = [SegmentationMask(0,0,0,0,None,"triangle")]
        original_image: Image.Image = None
        custom_image:Image.Image = None

        globals = {"added": added}
        toev = """
def test_valid_customization():
    return added("triangle")
"""
        exec(toev, globals)
        expression = globals["test_valid_customization"]()
        result,feedback = expression.evaluate(original_features,custom_features,original_image,custom_image)
        
        self.assertTrue(result)

        
    def test_simple_exec_false(self):

        original_features: list[SegmentationMask] = None
        custom_features: list[SegmentationMask] = [SegmentationMask(0,0,0,0,None,"triangl")]
        original_image: Image.Image = None
        custom_image:Image.Image = None

        globals = {"added": added}
        toev = """
def test_valid_customization():
    return added("triangle")
"""
        exec(toev, globals)
        expression = globals["test_valid_customization"]()
        result,feedback = expression.evaluate(original_features,custom_features,original_image,custom_image)
        
        self.assertFalse(result)