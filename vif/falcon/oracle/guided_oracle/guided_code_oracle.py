from abc import abstractmethod
from collections.abc import Callable
import json
import os
import re
from typing import Any
from loguru import logger
from openai import Client

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

    def get_oracle_code(self, instruction: str, base_image: Image.Image):
        self.segmentation_call_nb = 0

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
        response = response.choices[0].message.content
        pattern = r"```(?:\w+)?\n([\s\S]+?)```"
        id_match = re.search(pattern, response)

        if not id_match:
            return None

        oracle_method = id_match.group(1)
        return oracle_method

    def get_oracle(
        self, instruction: str, base_image: Image.Image
    ) -> Callable[[Image.Image], tuple[str, float, Any]]:

        self.segmentation_cache.clear()
        logger.debug("Creating Oracle")
        oracle_code = self.get_oracle_code(instruction, base_image)
        available_functions = {
            "added": added,
            "removed": removed,
            "color": color,
            "position": position,
            "placement": placement,
            "angle": angle,
            "FeatureHolder":FeatureHolder
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
            logger.debug("segmenting original image")
            original_detected_segs = self.segments_from_features(
                FeatureHolder.feature_set, base_image, True
            )
            logger.debug("segmenting customized image")
            custom_detected_segs = self.segments_from_features(
                FeatureHolder.feature_set, image
            )
            result, feedbacks = expression.evaluate(
                original_detected_segs, custom_detected_segs, base_image, image
            )
            return OracleResponse(result, feedbacks, evaluation_code=oracle_code)

        return oracle

    def segments_from_features(
        self, features: list[str], image: Image.Image, is_base_image: bool = False
    ) -> list[SegmentationMask]:
        self.segmentation_call_nb = self.segmentation_call_nb + 1
        if is_base_image:
            to_compute_features = [
                comp_feat
                for comp_feat in features
                if comp_feat not in self.segmentation_cache.keys()
            ]
            logger.debug("Features to compute " + ",".join(to_compute_features))
        else:
            to_compute_features = features

        try:
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
        except InvalidMasksError as ime:
            if self.debug:
                with open(
                    os.path.join(
                        self.debug_folder,
                        f"segments_{self.segmentation_call_nb}_res.json",
                    ),
                    "w",
                ) as seg_file:
                    json.dump(ime, seg_file)
        if self.debug:
            with open(
                os.path.join(
                    self.debug_folder, f"segments_{self.segmentation_call_nb}.json"
                ),
                "w",
            ) as seg_file:
                json.dump(segments, seg_file, cls=dataclassJSONEncoder)
            image.save(
                os.path.join(
                    self.debug_folder,
                    f"image_{self.segmentation_call_nb}.png",
                )
            )
            debug_seg_image = plot_segmentation_masks(image, segments)
            debug_seg_image.save(
                os.path.join(
                    self.debug_folder,
                    f"image_{self.segmentation_call_nb}_segmented.png",
                )
            )
            self.segmentation_call_nb += 1
        return segments

    def normalize_oracle_function(self, function: str):
        return (
            function.replace(" and ", " & ").replace(" or ", " | ").replace("not ", "~")
        )
