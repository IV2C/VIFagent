from abc import abstractmethod
import ast
from collections.abc import Callable
from typing import Any
from openai import Client
from vif.models.module import LLMmodule
from PIL import Image

from vif.prompts.identification_prompts import PINPOINT_PROMPT
from vif.utils.detection_utils import (
    SegmentationMask,
    get_boxes,
    get_segmentation_masks,
)
from vif.utils.image_utils import encode_image


class OracleModule(LLMmodule):
    def __init__(
        self,
        *,
        model,
        client: Client,
        temperature=0.3,
        debug: bool = False,
        debug_folder: str = ".tmp/debug",
    ):
        self.debug = debug
        self.debug_folder = debug_folder
        super().__init__(
            debug=debug,
            debug_folder=debug_folder,
            client=client,
            temperature=temperature,
            model=model,
        )

    def get_edit_list(
        self, features: list[str], instruction: str, base_image: Image.Image
    ):
        # getting the features to add/delete/modify
        feature_string = ",".join([f for f in features])
        pinpoint_instructions = PINPOINT_PROMPT.format(
            features=feature_string, instruction=instruction
        )
        encoded_image = encode_image(base_image)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": pinpoint_instructions,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            },
                        },
                    ],
                },
            ],
        )
        response = response.choices[0].message.content

        features_to_edit, features_to_delete, features_to_add = tuple(
            [
                ast.literal_eval(arr)
                for arr in response.split("ANSWER:")[1].strip().splitlines()
            ]
        )
        return (features_to_edit, features_to_delete, features_to_add)

    def detect_feat_boxes(self, features: list[str], base_image: Image.Image):
        return get_boxes(
            base_image,
            self.client,
            features,
            self.model,
            self.temperature,
        )

    def detect_seg_masks_boxes(
        self, features: list[str], base_image: Image.Image
    ) -> list[SegmentationMask]:
        return get_segmentation_masks(
            base_image,
            self.client,
            features,
            self.model,
            self.temperature,
        )

    @abstractmethod
    def get_oracle(
        self, features: list[str], instruction: str, base_image: Image.Image
    ) -> Callable[[Image.Image], tuple[bool, float, str, Any]]: ...
