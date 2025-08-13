from abc import abstractmethod
from collections.abc import Callable
import json
import os
import re
from typing import Any
from loguru import logger
from openai import Client
from tenacity import retry, stop_after_attempt

from vif.env import ORACLE_GENERATION_ATTEMPS, SEGMENTATION_ATTEMPS
from vif.falcon.oracle.oracle import OracleModule, OracleResponse
from PIL import Image

from vif.models.detection import SegmentationMask, dataclassJSONEncoder
from vif.models.exceptions import InvalidMasksError
from vif.prompts.oracle_prompts import (
    ORACLE_CODE_PROMPT,
    ORACLE_CODE_BOOLEAN_SYSTEM_PROMPT,
)
from vif.utils.detection_utils import get_segmentation_masks, plot_segmentation_masks
from vif.utils.image_utils import encode_image

from vif.falcon.oracle.guided_oracle.expressions import (
    FeatureHolder,
    OracleExpression,
    added,
    removed,
    angle,
    placement,
    position,
    color,
)
from google import genai
from google.genai import types as genTypes


class OracleGuidedCodeModule(OracleModule):
    def __init__(
        self,
        *,
        model,
        client: Client,
        temperature=0.3,
        visual_client: genai.Client,
        visual_generation_content_config: genTypes.GenerateContentConfig,
        visual_model: str,
    ):
        self.visual_client = visual_client
        self.visual_generation_content_config = visual_generation_content_config
        self.visual_model = visual_model
        self.segmentation_cache: dict[str, SegmentationMask] = dict()
        super().__init__(
            client=client,
            temperature=temperature,
            model=model,
        )

    def get_oracle_code(self, instruction: str, base_image: Image.Image) -> tuple[str,Any]:

        oracle_code_instructions = ORACLE_CODE_PROMPT.format(instruction=instruction)
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
        cnt = response.choices[0].message.content
        pattern = r"```(?:\w+)?\n([\s\S]+?)```"
        id_match = re.search(pattern, cnt)

        if not id_match:
            return None

        oracle_method = id_match.group(1)
        return oracle_method, response.usage
    
    @retry(reraise=True, stop=stop_after_attempt(ORACLE_GENERATION_ATTEMPS))
    def get_oracle(
        self, instruction: str, base_image: Image.Image
    ) -> tuple[Callable[[Image.Image], tuple[str, float, Any]],Any]:
        """Generates the code for the oracle, givne the instruction and the base image

        Args:
            instruction (str): The instruction the llm will have to apply to the iamge
            base_image (Image.Image): the base image that the llm will have to edit

        Returns:
            tuple[Callable[[Image.Image], tuple[str, float, Any]],Any]: A tuple contriaing the function and usage metrics(token counts)
        """
        
        self.segmentation_cache.clear()
        logger.info(f"Creating Oracle for instruction {instruction}")
        oracle_code,res_usage = self.get_oracle_code(instruction, base_image)
        available_functions = {
            "added": added,
            "removed": removed,
            "color": color,
            "position": position,
            "placement": placement,
            "angle": angle,
            "FeatureHolder": FeatureHolder,
        }

        oracle_code = self.normalize_oracle_function(oracle_code)
        exec(oracle_code, available_functions)

        logger.info(f"Oracle created: {oracle_code}")

        expression: OracleExpression = available_functions["test_valid_customization"]()

        @staticmethod
        def oracle(
            image: Image.Image,
        ) -> OracleResponse:
            logger.info("segmenting original image")

            original_detected_segs = self.segments_from_features(
                FeatureHolder.feature_set, base_image, True
            )
            logger.info("segmenting customized image")
            custom_detected_segs = self.segments_from_features(
                FeatureHolder.feature_set, image
            )
            
            result, feedbacks = expression.evaluate(
                original_detected_segs, custom_detected_segs, base_image, image
            )
            return OracleResponse(result, feedbacks, evaluation_code=oracle_code)

        return oracle,res_usage

    @retry(reraise=True, stop=stop_after_attempt(SEGMENTATION_ATTEMPS))
    def segments_from_features(
        self, features: list[str], image: Image.Image, is_base_image: bool = False
    ) -> list[SegmentationMask]:
        if is_base_image:
            to_compute_features = [
                comp_feat
                for comp_feat in features
                if comp_feat not in self.segmentation_cache.keys()
            ]
            logger.debug("Features to compute " + ",".join(to_compute_features))
        else:
            to_compute_features = features

        segments = []
        if len(to_compute_features) > 0:
            segments = get_segmentation_masks(
                image,
                self.visual_client,
                to_compute_features,
                self.visual_model,
                self.visual_generation_content_config,
            )
        if is_base_image:
            segments = segments + [
                seg for feat, seg in self.segmentation_cache if feat in features
            ]

        return segments

    def normalize_oracle_function(self, function: str):
        return (
            function.replace(" and ", " & ").replace(" or ", " | ").replace("not ", "~")
        )
