from collections import defaultdict
from collections.abc import Callable
import re
from typing import Any
import PIL.Image
from vif.falcon.oracle.oracle import OracleResponse
from vif.models.code import CodeEdit
from vif.models.exceptions import InvalidMasksError, JsonFormatError, ParsingError
from vif.models.misc import ToolCallError
from vif.models.module import LLMmodule
from vif.prompts.edition_prompts import IT_PROMPT
from vif.prompts.oracle_prompts import ORACLE_SYSTEM_PROMPT
from vif.utils.code_utils import apply_edits, get_annotated_code
from vif.utils.debug_utils import save_conversation
from openai.types.completion_usage import CompletionUsage

from vif.agent.tool_definitions import *
import json
from PIL import Image

from loguru import logger

from vif.utils.image_utils import encode_image
from vif.utils.renderer.tex_renderer import TexRenderer, TexRendererException


class EditionModule:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def customize(instruction: str):
        pass


class OracleEditionModule(EditionModule, LLMmodule):
    def __init__(
        self,
        *,
        client,
        model,
        temperature=0,
        max_iterations=5,
        code_renderer=TexRenderer().from_string_to_image,
    ):
        self.max_iterations = max_iterations
        self.code_renderer = code_renderer
        super().__init__(
            client=client,
            model=model,
            temperature=temperature,
        )
        self.observe_list = []

    def set_existing_observe_list(self, observe_list):
        self.observe_list = observe_list

    def modify_code(self, edits: list, code: str) -> tuple[str, str]:
        try:
            edits = [
                CodeEdit(edit["start"], edit["end"], edit["content"]) for edit in edits
            ]
            logger.info(
                f"Applying modifications {','.join([str(edit) for edit in edits])}"
            )
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
        optional_id: str,
        oracle_metrics: CompletionUsage = None,
    ):

        ["instruction"].append(instruction)

        edited_code = code
        ## Send initial message
        base_image: Image.Image = self.code_renderer(code)

        messages = self.initial_messages(
            instruction=instruction, initial_code=code, image=base_image
        )
        logger.info("Sending initial message")
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            tools=[modify_code_tool],
        )

        for step in range(self.max_iterations):
            messages.append(response.choices[0].message)  # append the llm message
            logger.info(f"LLM response:{response}")
            pattern = r"```(?:\w+)?\n([\s\S]+?)```"

            edits_match: re.Match[str] = re.search(
                pattern, response.choices[0].message.content
            )
            edits = json.loads(edits_match.group(1))
            edited_code, annotated_code = self.modify_code(edits, edited_code)

            # run oracle
            ##default values for observability
            oracle_response, report, error_type, edited_image = (
                OracleResponse(False, []),
                None,
                None,
                None,
            )
            logger.info(f"code edited:\n{edited_code}")
            try:
                edited_image = self.code_renderer(edited_code)

                oracle_response = oracle(edited_image)
                report = (
                    "Here is the code after modification:"
                    + "\n"
                    + annotated_code
                    + "\n"
                    + "The resulting image did not satisfy the instruction, here are some feedback:\n"
                    + "\n".join(oracle_response.feedbacks)
                )
                error_type = "Oracle"
            except TexRendererException as tre:
                logger.info(f"Rendering failed for latex:\n{edited_code} ")
                oracle_response = OracleResponse(False, [])
                report = "Tex failed to compile, error: " + tre.extract_error()
                error_type = "TexRendererError"
            except ValueError as ve:
                logger.info(f"Value error {str(ve)} for latex:\n{edited_code} ")
                oracle_response = OracleResponse(False, [])
                report = str(ve)
                error_type = "ValueError"
            except (InvalidMasksError, ParsingError, JsonFormatError) as seg_error:
                logger.error(
                    f"fatal Error while segmenting the image:{str(seg_error)} initial code will be returned."
                )
                report = str(seg_error)
                error_type = "seg_error"

            if step != 0:
                oracle_metrics = {}

            # saving for observability
            self.append_observe_row(
                id=optional_id,
                instruction=instruction,
                original_code=code,
                original_image=base_image,
                oracle_code=oracle_response.evaluation_code,
                turn=step,
                custom_image=edited_image,
                custom_code=edited_code,
                oracle_condition=oracle_response.condition,
                oracle_report=report,
                error_type=error_type,
                edition_usage=response.usage.to_json(),
                oracle_usage=oracle_metrics.to_json(),
            )

            if error_type == "seg_error":
                return code

            # return if oracle satisfied
            if oracle_response.condition:
                return edited_code

            logger.info(f"report {report}")
            logger.info(f"condition not yet satified")

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

            # iteration
            logger.info("Send the report and image back to the LLM")
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
                tools=[modify_code_tool],
            )
        # stop on sufficient edit
        return edited_code

    def append_observe_row(
        self,
        id,
        instruction,
        original_code,
        original_image,
        oracle_code,
        turn,
        custom_image,
        custom_code,
        oracle_condition,
        oracle_report,
        error_type,
        edition_usage,
        oracle_usage,
    ):
        tmp_dict = {}
        tmp_dict["id"] = id
        tmp_dict["instruction"] = instruction
        tmp_dict["original_code"] = original_code
        tmp_dict["oracle_code"] = oracle_code
        tmp_dict["original_image"] = original_image
        tmp_dict["turn"] = turn
        tmp_dict["custom_image"] = custom_image
        tmp_dict["custom_code"] = custom_code
        tmp_dict["oracle_condition"] = oracle_condition
        tmp_dict["oracle_report"] = oracle_report
        tmp_dict["error_type"] = error_type
        tmp_dict["edition_usage"] = edition_usage
        tmp_dict["oracle_usage"] = oracle_usage

        self.observe_list.append(tmp_dict)
