from collections import defaultdict
from collections.abc import Callable
import re
from typing import Any
from loguru import logger
from openai import Client

from vif.baselines.models import RegexException
from vif.falcon.oracle.guided_oracle.feedback import FeedBack, FeedBacks
from vif.falcon.oracle.oracle import OracleModule, OracleResponse
from PIL import Image
import hashlib

from vif.models.detection import BoundingBox, SegmentationMask
from vif.models.exceptions import JsonFormatError, LLMDetectionException, ParsingError
from vif.prompts.oracle_prompts import (
    ORACLE_CODE_PROMPT,
    ORACLE_CODE_EXPR_SYSTEM_PROMPT,
    ORACLE_PROPERTY_USAGE_PROMPT,
)
from vif.prompts.property_check_prompt import PROPERTY_PROMPT
from vif.utils.detection_utils import get_bounding_boxes, get_segmentation_masks
from vif.utils.image_utils import concat_images_horizontally, encode_image

from vif.falcon.oracle.guided_oracle.property_expression import visual_property
from vif.falcon.oracle.guided_oracle.expressions import (
    OracleExpression,
    aligned,
    count,
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
        property_model,
        property_client: Client,
        property_model_temperature=0.3,
        segmentation_model: str = "gemini-2.5-flash",
        box_model: str = "gemini-2.0-flash",
    ):
        self.visual_client = visual_client
        self.segmentation_model = segmentation_model
        self.box_model = box_model
        self.segmentation_usage: dict[str, str] = defaultdict(list)
        self.box_usage: dict[str, str] = defaultdict(list)
        self.property_usage: dict[str, str] = defaultdict(list)
        self.property_client = property_client
        self.property_model = property_model
        self.property_model_temperature = property_model_temperature

        self.identified_boxes: list[BoundingBox] = []
        self.identified_segments: list[SegmentationMask] = []

        # Caching
        self.seg_cache: dict[int : list[SegmentationMask]] = defaultdict(list)
        self.box_cache: dict[int : list[BoundingBox]] = defaultdict(list)

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
        logger.debug(instruction)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": ORACLE_CODE_EXPR_SYSTEM_PROMPT
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
            "count": count,
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
            self.box_usage.clear()
            self.property_usage.clear()
            self.seg_cache.clear()
            self.box_cache.clear()

            self.identified_segments = []
            self.identified_boxes = []
            feedback = expression.evaluate(
                original_image=base_image,
                custom_image=image,
                segment_function=self.segments_from_feature,
                box_function=self.box_from_feature,
                check_property_function=self.eval_property,
            )
            return OracleResponse(
                feedback,
                evaluation_code=oracle_code,
                seg_token_usage=self.segmentation_usage,
                box_token_usage=self.box_usage,
                prop_token_usage=self.property_usage,
                boxes=self.identified_boxes,
                segments=self.identified_segments,
            )

        return oracle, res_usage, oracle_code

    def box_from_feature(self, feature: str, image: Image.Image) -> list[BoundingBox]:

        cache_key = hashlib.sha1(
            str((hashlib.sha1(image.tobytes()).hexdigest(), feature)).encode("utf8")
        ).hexdigest()

        if cache_key in self.box_cache:
            logger.info(f"Temp cache hit for feature {feature}")
            return self.box_cache[cache_key]

        try:
            boxes, token_usage = get_bounding_boxes(
                image, self.visual_client, feature, self.box_model, enable_logprob=True
            )
            # parsing errors and jsonFormatErrors are considered as failures to identify
            # other errors are considered full failures
        except (ParsingError, JsonFormatError) as pe:
            boxes = []
            token_usage = pe.token_data

        self.box_usage[feature + str(hashlib.sha1(image.tobytes()).hexdigest())] = (
            token_usage
        )

        self.identified_boxes.extend(boxes)

        # saving cache
        self.box_cache[cache_key] = boxes
        return boxes

    def segments_from_feature(
        self, feature: str, image: Image.Image
    ) -> list[SegmentationMask]:
        cache_key = hashlib.sha1(
            str((hashlib.sha1(image.tobytes()).hexdigest(), feature)).encode("utf8")
        ).hexdigest()
        if cache_key in self.seg_cache:
            logger.info(f"Temp cache hit for feature {feature}")
            return self.seg_cache[cache_key]
        try:
            segs, token_usage = get_segmentation_masks(
                image,
                self.visual_client,
                feature,
                self.segmentation_model,
                enable_logprob=False,
            )
            # parsing errors and jsonFormatErrors are considered as failures to identify
            # other errors are considered full failures
        except (ParsingError, JsonFormatError) as pe:
            segs = []
            token_usage = pe.token_data
        self.segmentation_usage[
            feature + str(hashlib.sha1(image.tobytes()).hexdigest())
        ] = token_usage

        self.identified_segments.extend(segs)

        # saving cache
        self.seg_cache[cache_key] = segs

        return segs

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
    ) -> FeedBacks:

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
        pattern = r"\\boxed{([0-1]\.?[0-9]?)}"
        id_match = re.search(pattern, cnt)

        if not id_match:
            raise RegexException(pattern=pattern, content=cnt)

        res_score = float(id_match.group(1))
        self.property_usage[property_to_check] = response.usage
        if negated:
            res_score = 1- res_score
            feedback = (
                f'The property "{property_to_check}" is applied, but shouldn\'t be.'
            )
        else:
            feedback = (
                f'The property "{property_to_check}" is not applied, but should be.'
            )
        return FeedBack(feedback,res_score)
