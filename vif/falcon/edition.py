from collections.abc import Callable
from typing import Any
import PIL.Image
from vif.models.code import CodeEdit
from vif.models.misc import ToolCallError
from vif.models.module import LLMmodule
from vif.prompts.edition_prompts import IT_PROMPT
from vif.prompts.oracle_prompts import ORACLE_SYSTEM_PROMPT
from vif.utils.code_utils import apply_edits, get_annotated_code
from vif.utils.debug_utils import save_conversation


from vif.agent.tool_definitions import *
import json
from PIL import Image

import os
import datetime

from loguru import logger

from vif.utils.image_utils import encode_image
from vif.utils.renderer.tex_renderer import TexRenderer


class EditionModule:
    def __init__(self, debug=False, debug_folder=".tmp/debug", **kwargs):
        super().__init__(**kwargs)
        self.debug_folder=debug_folder
        self.debug = debug

    def gen_id(self):
        self._uuid = datetime.datetime.now().strftime(r"%d%m-%H%M%S")

    def get_id(self) -> str:
        return self._uuid

    def customize(instruction: str):
        pass

    def save_debug(self, instruction, code, image: Image.Image, step="", var_num="0"):
        if not self.debug:
            return
        id_d: str = self.get_id()
        os.mkdir(os.path.join(self.debug_folder, id_d))
        os.mkdir(os.path.join(self.debug_folder, id_d, str(step)))
        with open(
            os.path.join(self.debug_folder, id_d, str(step), var_num + ".tex"), "w"
        ) as debugd:
            debugd.write(code)
        with open(os.path.join(self.debug_folder, id_d, "instruction.txt"), "w") as ins:
            ins.write(instruction)
        image.save(os.path.join(self.debug_folder, id_d, str(step), var_num + ".png"))


class OracleEditionModule(EditionModule, LLMmodule):
    def __init__(
        self,
        *,
        client,
        model,
        temperature=0,
        debug=False,
        debug_folder=".tmp/debug",
        max_iterations=5,
        n=1,
        kept_mutants=1,
        code_renderer=TexRenderer().from_string_to_image,
    ):
        self.max_iterations = max_iterations
        self.n = n
        self.kept_mutants = kept_mutants
        self.code_renderer = code_renderer
        super().__init__(
            client=client,
            model=model,
            temperature=temperature,
            debug=debug,
            debug_folder=debug_folder,
        )

    def modify_code(self, edits: list, code: str) -> tuple[str, str]:
        logger.debug("applying modification")
        try:
            edits = [
                CodeEdit(edit["start"], edit["end"], edit["content"]) for edit in edits
            ]
            edited_code = apply_edits(code, edits)
            annotated_code = get_annotated_code(code)
        except ValueError as e:
            raise ToolCallError(str(e))
        return edited_code, annotated_code

    def initial_messages(
        self, instruction: str, initial_code: str, image: PIL.Image.Image
    ) -> list:
        encoded_image = encode_image(image)
        annotated_code = get_annotated_code(initial_code)
        messages = [
            {
                "role": "system",
                "content": ORACLE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": IT_PROMPT.format(
                            instruction=instruction, content=annotated_code
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ],
            },
        ]
        return messages

    def customize(
        self,
        instruction: str,
        code: str,
        oracle: Callable[
            [Image.Image], tuple[bool, tuple[float, bool, bool], str, Any]
        ],
    ):
        self.gen_id()  # update uuid, for debug purposes
        os.mkdir(os.path.join(self.debug_folder,self.get_id()))

        edited_code = code
        ## Send initial message
        base_image = self.code_renderer(code)
        messages = self.initial_messages(
            instruction=instruction, initial_code=code, image=base_image
        )
        logger.debug("Sending initial message")
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            tools=[modify_code_tool],
        )
        for step in range(self.max_iterations):
            self.save_conversation(messages)
            messages.append(response.choices[0].message)  # append the llm message
            if response.choices[0].message.tool_calls is not None:
                for tool_call in response.choices[0].message.tool_calls:
                    args = json.loads(tool_call.function.arguments)
                    try:
                        edited_code, annotated_code = self.modify_code(
                            args["edits"], edited_code
                        )

                    except ToolCallError as t:
                        annotated_code = str(
                            t
                        )  # still provide the tool call error to the llm

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": annotated_code,
                        }
                    )
            # run oracle
            edited_image = self.code_renderer(edited_code)
            (
                full_condition,
                _,
                report,
                debug_oracle_obj,
            ) = oracle(edited_image)

            self.save_debug(
                instruction=instruction,
                code=edited_code,
                image=edited_image,
                step=step,
                oracle_result=debug_oracle_obj,
                var_num="0",
            )

            # return if oracle satisfied
            if full_condition:
                return edited_code

            logger.debug("condition not yet satified, continuing edition")

            # iteration request, send report to LLM
            encoded_image = encode_image(edited_image)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": report,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            },
                        },
                    ],
                }
            )
            logger.debug("Send the report and image back to the LLM")
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
                tools=[modify_code_tool],
            )

        self.save_conversation(messages=messages)
        # stop on sufficient edit
        return edited_code

    def save_conversation(self, conv):
        save_conversation(conv, os.path.join(self.debug_folder,self.get_id()))

    def save_debug(self, instruction, code, image, oracle_result, step="", var_num="0"):
        super().save_debug(instruction, code, image, step, var_num)
        if not self.debug:
            return
        id_d: str = self.get_id()

        with open(
            os.path.join(self.debug_folder, id_d, str(step), var_num + ".data.json"),
            "w",
        ) as debugd:
            json.dump(oracle_result, debugd)
