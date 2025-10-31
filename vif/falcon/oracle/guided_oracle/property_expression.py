# Property condition => call to other llm to check property applied on the image

# TODO
from collections.abc import Callable
import re
from vif.baselines.models import RegexException
from vif.falcon.oracle.guided_oracle.expressions import OracleExpression
from loguru import logger
from openai import Client
from PIL import Image

from vif.prompts.property_check_prompt import PROPERTY_PROMPT
from vif.utils.image_utils import concat_images_horizontally, encode_image


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
