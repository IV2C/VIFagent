from collections.abc import Callable
from PIL import Image
from loguru import logger

from vif.falcon.edition import OracleEditionModule
from vif.falcon.oracle import OracleBoxModule
from vif.feature_search.feature_search import SearchModule
from vif.prompts.edition_prompts import SYSTEM_PROMPT_CLARIFY
from vif.utils.image_utils import encode_image
import os

class Falcon:
    def __init__(
        self,
        code_renderer: Callable[[str], Image.Image],
        edition_module: OracleEditionModule = None,
        search_module: SearchModule = None,
        identification_module: OracleBoxModule = None,
        debug=False,
        clarify_instruction=False,
        debug_folder=".tmp/debug",
    ):
        self.edition_module = edition_module
        self.code_renderer = code_renderer
        self.debug = debug
        self.debug_folder = debug_folder

        self.search_module = search_module

        self.identification_module = identification_module

        self.clarify_instruction = clarify_instruction
        
        pass
    
    def apply_instruction(self, code: str, instruction: str):
        """Applies the instruction to the code

        """
        # render image
        base_image = self.code_renderer(code)
        if self.clarify_instruction:
            logger.info("clarifying the instruction")
            instruction = self.apply_clarification(instruction, base_image)
            
        # unifying the code for easier parsing in steps ahead
        code = "\n".join(line.strip() for line in code.split("\n"))

        # VLM to get list of feature
        logger.info("Searching for features")

        features = self.search_module.get_features(base_image)
        if not features:
            logger.warning(
                f"Feature search failed, using un-commented code"
            )  # depending on the search module we cannot ensure it does not happen
            return code, base_image
        self.features = features
        
        
        logger.info("creating oracle")
        oracle = self.identification_module.get_oracle(
                    self.features, instruction, base_image
                )
        logger.info("Editing the code")
        response_code = self.edition_module.customize(instruction, code, oracle)
        return response_code
        
        

    def apply_clarification(self, instruction: str, base_image: Image.Image):
        encoded_image = encode_image(image=base_image)

        response = self.edition_module.client.chat.completions.create(
            model=self.edition_module.model,
            temperature=self.edition_module.temperature,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_CLARIFY,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": instruction,
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
        return response.choices[0].message.content