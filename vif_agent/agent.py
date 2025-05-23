import math
import os
import shutil
from typing import Iterable
from openai import OpenAI
from collections.abc import Callable
from PIL import Image
from sentence_transformers import SentenceTransformer
from vif_agent.feature import CodeImageMapping, MappedCode
from vif_agent.modules.edition.edition import (
    EditionModule,
    LLMAgenticEditionModule,
    OracleEditionModule,
)
from vif_agent.modules.identification.identification import IdentificationModule
from vif_agent.modules.identification.mapping import (
    BoxIdentificationModule,
    IdentificationMappingModule,
)
from vif_agent.modules.identification.oracle import IdentificationOracleBoxModule
from vif_agent.modules.search.search import SearchModule
from vif_agent.mutation.mutant import TexMutant
from vif_agent.utils import adjust_bbox, encode_image, mse
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


class UnsatifiableConfig(Exception):
    pass


class VifAgent:
    def __init__(
        self,
        code_renderer: Callable[[str], Image.Image],
        edition_module: EditionModule = None,
        search_module: SearchModule = None,
        identification_module: IdentificationModule = None,
        debug=False,
        clarify_instruction=True,
        debug_folder=".tmp/debug",
        mutant_creator=TexMappingMutantCreator(),
    ):
        self.edition_module = edition_module
        self.code_renderer = code_renderer
        self.debug = debug
        self.debug_folder = debug_folder

        self.search_module = search_module

        self.identification_module = identification_module

        self.clarify_instruction = clarify_instruction

        self.mutant_creator = mutant_creator

        self.check_satifiable_config()

    def check_satifiable_config(self):
        if not self.search_module:
            raise UnsatifiableConfig("Search module is necessary")

        match self.edition_module:
            case LLMAgenticEditionModule():
                if not isinstance(
                    self.identification_module, IdentificationMappingModule
                ):
                    raise UnsatifiableConfig(
                        "LLMEdition module requires an identification mapping module"
                    )
            case OracleEditionModule():
                if not isinstance(
                    self.identification_module, IdentificationOracleBoxModule
                ):
                    raise UnsatifiableConfig(
                        "OracleEditionModule module requires an IdentificationOracleBoxModule module"
                    )

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

        # unifying the code for easier parsing in steps ahead
        code = "\n".join(line.strip() for line in code.split("\n"))
        # render image
        base_image = self.code_renderer(code)
        # VLM to get list of feature
        logger.info("Searching for features")

        features = self.search_module.get_features(base_image)
        if not features:
            logger.warning(
                f"Feature search failed, using un-commented code"
            )  # depending on the search module we cannot ensure it does not happen
            return code, base_image
        self.features = features

        logger.info("applying the instruction")

        ##Applying either by agent+tool or by oracle loop
        match self.edition_module:
            case LLMAgenticEditionModule():
                mapped_code = self.identification_module.identify(
                    base_image=base_image, features=features, code=code
                )
                response_code = self.edition_module.customize(
                    mapped_code=mapped_code, instruction=instruction
                )
            case OracleEditionModule():
                oracle = self.identification_module.get_oracle(
                    self.features, instruction, base_image
                )
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

        new_instruction = response.choices[-1].message.content
        if self.debug:
            open(
                os.path.join(self.debug_folder, self.debug_id, "new_instruction.txt"),
                "w",
            ).write(new_instruction)

        return new_instruction

    def __str__(self):
        return (
            f"VifAgent("
            f"search_module={self.search_module}"
            f"identification_module={self.identification_module}"
            f"edition_module={self.edition_module}"
            f"debug={self.debug}, debug_folder='{self.debug_folder}')"
        )
