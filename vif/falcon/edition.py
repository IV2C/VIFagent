from collections.abc import Callable
from typing import Any
import PIL.Image
from vif.falcon.oracle.oracle import OracleResponse
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
from vif.utils.renderer.tex_renderer import TexRenderer, TexRendererException


class EditionModule:
    def __init__(self, debug=False, debug_folder=".tmp/debug", **kwargs):
        super().__init__(**kwargs)
        self.debug_folder = debug_folder
        self.debug = debug

    def debug_instance_creation(self, debug: bool, debug_folder: str):
        self.debug = debug
        self.debug_folder = os.path.join(debug_folder, "edition")
        if debug:
            os.mkdir(self.debug_folder)

    def customize(instruction: str):
        pass

    def save_debug(self, code, image: Image.Image, step):
        if not self.debug:
            return
        with open(os.path.join(self.debug_folder, str(step) + ".tex"), "w") as debugd:
            debugd.write(code)
        image.save(os.path.join(self.debug_folder, str(step) + ".png"))


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
        code_renderer=TexRenderer().from_string_to_image,
    ):
        self.max_iterations = max_iterations
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
            annotated_code = get_annotated_code(edited_code)
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
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    },
                ],
            },
        ]
        return messages

    def customize(
        self,
        instruction: str,
        code: str,
        oracle: Callable[[Image.Image], OracleResponse],
    ):
        edited_code = code
        ## Send initial message
        base_image: Image.Image = self.code_renderer(code)

        """ DEBUG """
        if self.debug:
            with open(
                os.path.join(self.debug_folder, "initial.tex"),
                "w",
            ) as debugd:
                debugd.write(code)
            base_image.save(os.path.join(self.debug_folder, "initial.png"))

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
            try:
                edited_image = self.code_renderer(edited_code)

                oracle_response = oracle(edited_image)
                report = "The resulting image did not satify the instruction, here are some feedback:\n"+"\n".join(
                    oracle_response.feedbacks
                )
                self.save_debug(
                    instruction=instruction,
                    code=edited_code,
                    image=edited_image,
                    step=step,
                    oracle_response=oracle_response,
                )
            except TexRendererException as tre:
                oracle_response = OracleResponse(False, [])
                report = "Tex failed to compile, error: " + tre.extract_error()
            except ValueError as ve:
                oracle_response = OracleResponse(False, [])
                report = str(ve)
            # return if oracle satisfied
            if oracle_response.condition:
                self.save_conversation(messages)
                return edited_code

            logger.info("condition not yet satified, continuing edition")

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
                                "url": f"data:image/png;base64,{encoded_image}"
                            },
                        },
                    ],
                }
            )
            logger.info("Send the report and image back to the LLM")
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
                tools=[modify_code_tool],
            )

        self.save_conversation(messages)
        # stop on sufficient edit
        return edited_code

    def save_conversation(self, conv):
        save_conversation(conv, os.path.join(self.debug_folder))

    def save_debug(self, code, image, oracle_response:OracleResponse, step):
        super().save_debug(code, image, step)
        if not self.debug:
            return

        with open(
                os.path.join(self.debug_folder, f"response_oracle{str(step)}.txt"),
                "w",
            ) as debugd:
            debugd.write("\n".join(oracle_response.feedbacks))

        if oracle_response.score_object:
            with open(
                os.path.join(self.debug_folder, str(step) + ".data.json"),
                "w",
            ) as debugd:
                json.dump(oracle_response.score_object, debugd)
