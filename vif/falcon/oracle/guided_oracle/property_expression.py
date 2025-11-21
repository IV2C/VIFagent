from collections.abc import Callable
from vif.falcon.oracle.guided_oracle.expressions import OracleExpression
from PIL import Image



class visual_property(OracleExpression):

    def __init__(self, property):
        self.property = property
        self.negated = False

    def __invert__(self):
        self.negated = True
        return self

    def evaluate(
        self,
        *,
        original_image: Image.Image,
        custom_image: Image.Image,
        check_property_function: Callable[
            [Image.Image, Image.Image, str,bool], tuple[bool, list[str]]
        ],
    ) -> tuple[bool, list[str]]:
        return check_property_function(original_image,custom_image,self.property,self.negated)
