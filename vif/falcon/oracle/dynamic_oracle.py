from collections.abc import Callable
from typing import Any
from loguru import logger
from openai import Client

from vif.falcon.oracle.oracle import OracleModule
from PIL import Image

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
        
    def get_oracle(
            self, features: list[str], instruction: str, base_image: Image.Image
        ) -> Callable[[Image.Image], tuple[str,float,Any]]:
            logger.info("Creating Oracle")

            original_detected_boxes = self.detect_feat_boxes(features, base_image)
            # first filtering the features depending on wether they have been detected in the original image
            features = list(set([box["label"] for box in original_detected_boxes]))
            # The two tasks are independent and can be executed in parallel
            

            features_to_edit, features_to_delete, features_to_add = self.get_edit_list(features, instruction, base_image)
