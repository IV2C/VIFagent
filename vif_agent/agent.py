import math
import os
import shutil
from typing import Iterable
from openai import OpenAI
from collections.abc import Callable
from PIL import Image
from sentence_transformers import SentenceTransformer
from vif_agent.feature import CodeImageMapping, MappedCode
from vif_agent.modules.edition.edition import LLMEditionModule
from vif_agent.modules.identification.identification import BoxIdentificationModule
from vif_agent.mutation.mutant import TexMutant
from vif_agent.utils import adjust_bbox, encode_image, norm_mse
from vif_agent.prompt import *
from vif_agent.mutation.tex_mutant_creator import (
    TexMappingMutantCreator,
    TexRegBrutalMutantCreator,
    TexRegMutantCreator,
    TexMutantCreator,
)
import json
import re
from functools import cache
from loguru import logger
import uuid
import sys

type Spans = tuple[list[tuple[int, int]], float]  # for lisibiilty

logger.remove()
logger.add(sys.stderr, level="INFO")


from enum import Enum


class IdModuleType(Enum):
    BOX = 0
    SEG = 1
    CLIP = 2


class VifAgent:
    def __init__(
        self,
        code_renderer: Callable[[str], Image.Image],
        edition_module: LLMEditionModule = None,
        search_client: OpenAI = None,
        search_model: str = None,
        search_model_temperature=None,
        identification_module: BoxIdentificationModule = None,
        debug=False,
        clarify_instruction=True,
        debug_folder=".tmp/debug",
        mutant_creator=TexMappingMutantCreator(),
    ):
        self.edition_module = edition_module
        self.code_renderer = code_renderer
        self.debug = debug
        self.debug_folder = debug_folder

        self.search_client = search_client
        self.search_model_temperature = search_model_temperature
        self.search_model = search_model

        self.identification_module = identification_module

        self.clarify_instruction = clarify_instruction

        self.mutant_creator = mutant_creator

    def apply_instruction(self, code: str, instruction: str):
        """Applies the instruction to the code, using the settings

        Args:
            code (str): _description_
            instruction (str): _description_

        Returns:
            _type_: _description_
        """
        
        """DEBUG"""
        if self.debug:
            self.debug_id = str(uuid.uuid4())
            os.mkdir(os.path.join(self.debug_folder, self.debug_id))
        """"""
        base_image = self.code_renderer(code)
        
        if self.clarify_instruction:
            logger.info("clarifying the instruction")
            instruction = self.apply_clarification(instruction, base_image)
        
        
        mapped_code = self.identify_features(code)




        logger.info("applying the instruction")
        response_code = self.edition_module.customize(mapped_code,instruction=instruction)
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

        new_instruction = response.choices[-1].message.content
        if self.debug:
            open(
                os.path.join(self.debug_folder, self.debug_id, "new_instruction.txt"),
                "w",
            ).write(new_instruction)

        return new_instruction

    def identify_features(self, code: str) -> MappedCode:
        """Identifies the features within the code by commenting it

        Args:
            code (str): The code where the features need to be identified
            comment_character (str): the character used to comment the code
        Returns:
            dict[str, list[Spans]]: a dictionnary associating the feature with a list of probable spans
        """
        """DEBUG"""
        if self.debug and not hasattr(self, "debug_id"):
            self.debug_id = str(uuid.uuid4())
            os.mkdir(os.path.join(self.debug_folder, self.debug_id))
        """"""

        # unifying the code for easier parsing in mutant creation
        code = "\n".join(line.strip() for line in code.split("\n"))
        # render image
        base_image = self.code_renderer(code)
        # VLM to get list of feature
        logger.info("Searching for features")
        encoded_image = encode_image(image=base_image)
        response = self.search_client.chat.completions.create(
            model=self.search_model,
            temperature=self.search_model_temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": FEATURE_IDENTIFIER_PROMPT,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            },
                        },
                    ],
                }
            ],
        )
        pattern = r"```(?:\w+)?\n([\s\S]+?)```"
        search_match = re.search(pattern, response.choices[0].message.content)
        if not search_match:
            logger.warning(
                f"Feature search failed, using un-commented code, unparseable response {response.choices[0].message.content}"
            )
            return code, base_image

        features_match = search_match.group(1)
        features = json.loads(features_match)
        self.features = features
        """DEBUG"""
        if self.debug:
            json.dump(
                features,
                open(
                    os.path.join(self.debug_folder, self.debug_id, "features.json"), "w"
                ),
            )
        """"""
        # identification using the identification module
        logger.info("Identifying features")
        return self.identification_module.identify(
            base_image=base_image, features=features, code=code
        )

    def __str__(self):
        return (
            f"VifAgent(model={self.model}, temperature={self.temperature}, "
            f"search_model={self.search_model}, search_model_temperature={self.search_model_temperature}, "
            f"identification_module={self.identification_module}"
            f"debug={self.debug}, debug_folder='{self.debug_folder}')"
        )
