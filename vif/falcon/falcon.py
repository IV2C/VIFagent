import datetime
import os
from collections.abc import Callable
from PIL import Image
from vif.CodeMapper.mapping import MappingModule
from vif.falcon.oracle.oracle import OracleModule
from vif.feature_identification.feature_identification import IdentificationModule
from vif.prompts.edition_prompts import SYSTEM_PROMPT_CLARIFY
from vif.falcon.edition import (
    EditionModule,
)


from loguru import logger
import sys

from vif.utils.image_utils import encode_image

type Spans = tuple[list[tuple[int, int]], float]  # for lisibiilty

logger.remove()
logger.add(sys.stderr, level="INFO")


class Falcon:
    def __init__(
        self,
        code_renderer: Callable[[str], Image.Image],
        oracle_module: OracleModule,
        edition_module: EditionModule,
        debug=False,
        clarify_instruction=False,
        debug_folder=".tmp/debug",
        mapping_module: MappingModule = None,
    ):
        self.code_renderer = code_renderer

        self.edition_module = edition_module
        self.oracle_module = oracle_module

        self.debug = debug
        self.debug_folder = debug_folder
        self.debug_instance_nb = 0
        if self.debug:
            self._uuid = datetime.datetime.now().strftime(r"%d%m-%H:%M:%S")
            logger.warning(
                f"Debug is activated, debug folder is {os.path.join(self.debug_folder, self._uuid)}"
            )
            os.mkdir(os.path.join(self.debug_folder, self._uuid))

    def set_instance_debug_folder(self):
        self.debug_instance_nb += 1
        if self.debug:
            inst_debug_folder = os.path.join(self.debug_folder, self._uuid, str(self.debug_instance_nb))
            os.mkdir(inst_debug_folder)
            self.oracle_module.debug_instance_creation(
                self.debug,inst_debug_folder
            )
            self.edition_module.debug_instance_creation(
                self.debug,inst_debug_folder
            )

    def apply_instruction(self, code: str, instruction: str):
        """Applies the instruction to the code, using the settings

        Args:
            code (str): _description_
            instruction (str): _description_

        Returns:
            _type_: _description_
        """


        self.set_instance_debug_folder()
        
        base_image = self.code_renderer(code)

        # code = "\n".join(line.strip() for line in code.split("\n"))
        # render image


        logger.info("Creating the oracle")
        oracle = self.oracle_module.get_oracle(
            instruction,
            base_image,
        )

        response_code = self.edition_module.customize(instruction, code, oracle)

        return response_code

    #TODO update code to be compaptible with client
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
                                "url": f"data:image/png;base64,{encoded_image}"
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
