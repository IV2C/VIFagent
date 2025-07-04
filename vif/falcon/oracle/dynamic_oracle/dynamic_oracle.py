from abc import abstractmethod
from collections.abc import Callable
from typing import Any
from loguru import logger
from openai import Client

from vif.falcon.oracle.oracle import OracleModule
from PIL import Image

from vif.prompts.oracle_prompts import ORACLE_CODE_PROMPT, ORACLE_CODE_BOOLEAN_SYSTEM_PROMPT
from vif.utils.image_utils import encode_image


class OracleDynamicBoxModule(OracleModule):
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

    def get_oracle_code(
        self, features: list[str], instruction: str, base_image: Image.Image
    ):
        # getting the features to add/delete/modify
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
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            },
                        },
                    ],
                },
            ],
        )
        response = response.choices[0].message.content
        #TODO
        raise NotImplementedError()

    def get_oracle(
        self, features: list[str], instruction: str, base_image: Image.Image
    ) -> Callable[[Image.Image], tuple[str, float, Any]]:
        logger.info("Creating Oracle")

        original_detected_boxes = self.detect_feat_boxes(features, base_image)
        # first filtering the features depending on wether they have been detected in the original image
        features = list(set([box["label"] for box in original_detected_boxes]))
        #TODO
        
        

    def normalize_oracle_function(function: str):
        return function.replace(" and ", " & ").replace(" or ", " | ").replace("not ", "~")

