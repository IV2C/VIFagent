import datetime
import os
from collections.abc import Callable
from PIL import Image
from vif.CodeMapper.mapping import MappingModule
from vif.falcon.oracle.oracle import OracleModule
from vif.prompts.edition_prompts import SYSTEM_PROMPT_CLARIFY
from vif.falcon.edition import (
    EditionModule,
)
import pickle

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
        observe=False,
        clarify_instruction=False,
        observe_folder=".tmp/debug",
        mapping_module: MappingModule = None,
    ):
        self.code_renderer = code_renderer

        self.edition_module = edition_module
        self.oracle_module = oracle_module

        self.observe = observe
        self.observe_folder = observe_folder
        if self.observe:
            self.ds_stored_name = (
                datetime.datetime.now().strftime(r"%d%m-%H:%M:%S") + ".pickle"
            )
            logger.warning(
                f"The observe parameter is activated, dict will be stored at {os.path.join(self.observe_folder, self.ds_stored_name)}"
            )
            if not os.path.exists(self.observe_folder):
                os.mkdir(self.observe_folder)

    def apply_instruction(self, code: str, instruction: str, optional_id: str = None):
        """Applies the instruction to the code, using the settings"""

        if self.observe and optional_id is None:
            raise ValueError("optionanl_id must be provided if observe is True")

        if optional_id is None:
            optional_id = "".join([s[0] for s in instruction.split(" ") if s])

        base_image = self.code_renderer(code)

        # code = "\n".join(line.strip() for line in code.split("\n"))

        logger.info("Creating the oracle")
        try:
            oracle = self.oracle_module.get_oracle(instruction, base_image)
        except AttributeError as ae:
            logger.error(
                f"Fatal error during oracle generation, oracle is none{str(ae)}"
            )

        response_code = self.edition_module.customize(
            instruction, code, oracle, optional_id
        )

        with open(
            os.path.join(self.observe_folder, self.ds_stored_name + ".pickle"), "wb"
        ) as obsfile:
            pickle.dump(self.edition_module.observe_list, obsfile)
        return response_code

    # TODO update code to be compaptible with client
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
        if self.observe:
            open(
                os.path.join(self.observe_folder, self.debug_id, "new_instruction.txt"),
                "w",
            ).write(new_instruction)

        return new_instruction

    def __str__(self):
        return (
            f"VifAgent("
            f"oracle_module={self.oracle_module}"
            f"edition_module={self.edition_module}"
            f"observe={self.observe}, observe_folder='{self.observe_folder}')"
        )
