from abc import abstractmethod
from collections.abc import Callable
import os
import re
from typing import Any
from loguru import logger
from openai import Client

from vif.falcon.oracle.oracle import OracleModule, OracleResponse
from PIL import Image

from vif.models.detection import SegmentationMask
from vif.prompts.oracle_prompts import (
    ORACLE_CODE_PROMPT,
    ORACLE_CODE_BOOLEAN_SYSTEM_PROMPT,
)
from vif.utils.image_utils import encode_image

from vif.falcon.oracle.guided_oracle.expressions import (
    OracleExpression,
    added,
    removed,
    angle,
    placement,
    position,
    color,
)


class OracleGuidedCodeModule(OracleModule):
    def __init__(
        self,
        *,
        model,
        client: Client,
        temperature=0.3,
    ):
        super().__init__(
            client=client,
            temperature=temperature,
            model=model,
        )

    def get_oracle_code(
        self, features: list[str], instruction: str, base_image: Image.Image
    ):

        feature_string = '["' + '","'.join([f for f in features]) + '"]'

        oracle_code_instructions = ORACLE_CODE_PROMPT.format(
            features=feature_string, instruction=instruction
        )
        encoded_image = encode_image(base_image)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": ORACLE_CODE_BOOLEAN_SYSTEM_PROMPT},
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
        response = response.choices[0].message.content
        pattern = r"```(?:\w+)?\n([\s\S]+?)```"
        id_match = re.search(pattern, response)

        if not id_match:

            return None

        oracle_method = id_match.group(1)
        return oracle_method

    def get_oracle(
        self,
        features_segments: list[SegmentationMask],
        instruction: str,
        base_image: Image.Image,
        detect_seg_masks_boxes: Callable[
            [list[str], Image.Image], list[SegmentationMask]
        ],
    ) -> Callable[[Image.Image], tuple[str, float, Any]]:
        logger.info("Creating Oracle")

        original_detected_segs = features_segments
        # first filtering the features depending on wether they have been detected in the original image
        features = list(set([seg.label for seg in original_detected_segs]))

        oracle_code = self.get_oracle_code(features, instruction, base_image)
        available_functions = {
            "added": added,
            "removed": removed,
            "color": color,
            "position": position,
            "placement": placement,
            "angle": angle,
        }

        if self.debug:
            with open(
                os.path.join(self.debug_folder, "oracle_code.py"), "w"
            ) as oracle_file:
                oracle_file.write(oracle_code)

        oracle_code = self.normalize_oracle_function(oracle_code)
        exec(oracle_code, available_functions)

        expression: OracleExpression = available_functions["test_valid_customization"]()

        @staticmethod
        def oracle(
            image: Image.Image,
        ) -> OracleResponse:
            custom_detected_segs = detect_seg_masks_boxes(features,image)
            result, feedbacks = expression.evaluate(
                original_detected_segs, custom_detected_segs, base_image, image
            )
            return OracleResponse(result, feedbacks, evaluation_code=oracle_code)

        return oracle

    def normalize_oracle_function(self, function: str):
        return (
            function.replace(" and ", " & ").replace(" or ", " | ").replace("not ", "~")
        )
