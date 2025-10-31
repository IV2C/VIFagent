from collections import defaultdict
from collections.abc import Callable
import re
from typing import Any
from loguru import logger
from openai import Client

from vif.baselines.models import RegexException
from vif.falcon.oracle.oracle import OracleModule, OracleResponse
from PIL import Image
import hashlib

from vif.models.detection import SegmentationMask
from vif.prompts.oracle_prompts import (
    ORACLE_CODE_PROMPT,
    ORACLE_CODE_BOOLEAN_SYSTEM_PROMPT,
    ORACLE_PROPERTY_USAGE_PROMPT,
)
from vif.prompts.property_check_prompt import PROPERTY_PROMPT
from vif.utils.detection_utils import get_segmentation_masks
from vif.utils.image_utils import concat_images_horizontally, encode_image

from vif.falcon.oracle.guided_oracle.property_expression import visual_property
from vif.falcon.oracle.guided_oracle.expressions import (
    OracleExpression,
    aligned,
    present,
    angle,
    placement,
    position,
    color,
    shape,
    size,
    within,
    mirrored,
)
from google import genai


class OracleGuidedCodeModule(OracleModule):
    def __init__(
        self,
        *,
        model,
        client: Client,
        temperature=0.3,
        visual_client: genai.Client,
        visual_model: str,
        property_model,
        property_client: Client,
        property_model_temperature=0.3,
    ):
        self.visual_client = visual_client
        self.visual_model = visual_model
        self.segmentation_cache: dict[int, list[SegmentationMask]] = defaultdict(list)
        self.segmentation_usage: dict[str, str] = defaultdict(list)
        self.property_usage: dict[str, str] = defaultdict(list)
        self.property_client = property_client
        self.property_model = property_model
        self.property_model_temperature = property_model_temperature

        super().__init__(
            client=client,
            temperature=temperature,
            model=model,
        )

    def get_oracle_code(
        self, instruction: str, base_image: Image.Image
    ) -> tuple[str, Any]:

        oracle_code_instructions = ORACLE_CODE_PROMPT.format(instruction=instruction)
        encoded_image = encode_image(base_image)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": ORACLE_CODE_BOOLEAN_SYSTEM_PROMPT
                    + "\n"
                    + ORACLE_PROPERTY_USAGE_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": oracle_code_instructions,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}"
                            },
                        },
                    ],
                },
            ],
        )
        cnt = response.choices[0].message.content
        pattern = r"```(?:\w+)?\n([\s\S]+?)```"
        id_match = re.search(pattern, cnt)

        if not id_match:
            return None

        oracle_method = id_match.group(1)
        return oracle_method, response.usage

    # If needs to be added back, then it should be implemented manually, with proper obervation
    # @retry(reraise=True, stop=stop_after_attempt(ORACLE_GENERATION_ATTEMPS))
    def get_oracle(
        self, instruction: str, base_image: Image.Image
    ) -> tuple[Callable[[Image.Image], OracleResponse], Any]:
        """Generates the code for the oracle, given the instruction and the base image

        Args:
            instruction (str): The instruction the llm will have to apply to the iamge
            base_image (Image.Image): the base image that the llm will have to edit

        Returns:
            tuple[Callable[[Image.Image], OracleResponse],Any]: A tuple containing the function and usage metrics(token counts)
        """

        self.segmentation_cache.clear()
        logger.info(f"Creating Oracle for instruction {instruction}")
        oracle_code, res_usage = self.get_oracle_code(instruction, base_image)

        available_functions = {
            "placement": placement,
            "position": position,
            "color": color,
            "angle": angle,
            "size": size,
            "within": within,
            "shape": shape,
            "present": present,
            "mirrored": mirrored,
            "aligned": aligned,
            "visual_property": visual_property,
        }

        oracle_code = self.normalize_oracle_function(oracle_code)
        exec(oracle_code, available_functions)

        logger.info(f"Oracle created: {oracle_code}")

        expression: OracleExpression = available_functions["test_valid_customization"]()

        @staticmethod
        def oracle(
            image: Image.Image,
        ) -> OracleResponse:
            self.segmentation_usage.clear()
            self.property_usage.clear()
            result, feedbacks = expression.evaluate(
                original_image=base_image,
                custom_image=image,
                segment_function=self.segments_from_features,
                check_property_function=self.eval_property,
            )
            return OracleResponse(
                result,
                feedbacks,
                evaluation_code=oracle_code,
                seg_token_usage=self.segmentation_usage,
                prop_token_usage=self.property_usage
            )

        return oracle, res_usage

    def segments_from_features(
        self, features: list[str], image: Image.Image
    ) -> tuple[list[SegmentationMask], Any]:

        already_computed_label = [
            label
            for label in self.segmentation_cache[
                hashlib.sha1(image.tobytes()).hexdigest()
            ]
        ]

        to_compute_features = [
            comp_feat
            for comp_feat in features
            if comp_feat not in already_computed_label
        ]
        cache_hit_features = [
            feat for feat in features if feat not in to_compute_features
        ]
        if len(cache_hit_features) > 0:
            logger.info(
                f"feature detection Cache hit for {','.join(cache_hit_features)}"
            )
        logger.info("Features to compute :[" + ",".join(to_compute_features) + "]")

        segments = self.segmentation_cache[hashlib.sha1(image.tobytes()).hexdigest()]
        if len(to_compute_features) > 0:
            segs, token_usage = get_segmentation_masks(
                image,
                self.visual_client,
                to_compute_features,
                self.visual_model,
            )
            segments += segs
            self.segmentation_usage[
                "".join(features) + str(hashlib.sha1(image.tobytes()).hexdigest())
            ] = token_usage

        return segments

    def normalize_oracle_function(self, function: str):
        return (
            function.replace(" and ", " & ").replace(" or ", " | ").replace("not ", "~")
        )

    def eval_property(
        self,
        original_image: Image.Image,
        custom_image: Image.Image,
        property_to_check: str,
        negated: bool,
    ) -> tuple[bool, list[str]]:

        concat_image = concat_images_horizontally([original_image, custom_image])
        encoded_image = encode_image(concat_image)
        response = self.property_client.chat.completions.create(
            model=self.property_model,
            temperature=self.property_model_temperature,
            messages=[
                {"role": "system", "content": PROPERTY_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": property_to_check,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}"
                            },
                        },
                    ],
                },
            ],
        )
        cnt = response.choices[0].message.content
        pattern = r"\\boxed{(True|False)}"
        id_match = re.search(pattern, cnt)

        if not id_match:
            raise RegexException(pattern, cnt)

        condition = id_match.group(1) == "True"
        self.property_usage[property_to_check] = response.usage
        if negated:
            condition = not condition
            feedback = (
                f'The property "{property_to_check}" is applied, but shouldn\'t be.'
            )
        else:
            feedback = (
                f'The property "{property_to_check}" is not applied, but should be.'
            )
        return (condition, [feedback] if not condition else [])
