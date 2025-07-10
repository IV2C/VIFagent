from collections.abc import Callable
import json
from loguru import logger
from openai import OpenAI
from vif.CodeMapper.feature import MappedCode
from vif.CodeMapper.mapping import ZoneIdentificationModule
from vif.feature_search.feature_search import SearchModule
from vif.models.code import CodeEdit
from vif.agent.tool_definitions import *
from PIL import Image

from vif.models.misc import ToolCallError
from vif.models.module import LLMmodule
from vif.prompts.edition_prompts import (
    EDITION_SYSTEM_PROMPT,
    IT_PROMPT,
    SYSTEM_PROMPT_CLARIFY,
)
from vif.utils.debug_utils import save_conversation
from vif.utils.image_utils import encode_image
from vif.utils.renderer.tex_renderer import TexRendererException


class FeatureAgent(LLMmodule):
    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        code_renderer: Callable[[str], Image.Image],
        temperature: float = 0.0,
        search_module: SearchModule = None,
        identification_module: ZoneIdentificationModule = None,
        debug=False,
        debug_folder=".tmp/debug",
        used_tools: list = [
            feature_find_tool,
            render_tool,
            modify_code_tool,
            finish_customization_tool,
        ],
        clarify_instruction=False,
    ):
        super().__init__(
            client=client,
            model=model,
            temperature=temperature,
            debug_folder=debug_folder,
            debug=debug,
        )
        self.code_renderer = code_renderer
        self.identification_module = identification_module
        self.search_module = search_module
        self.conv_data = {
            "get_feature_location_calls": 0,
            "render_code_calls": 0,
            "modify_code_calls": 0,
            "get_feature_location_errors": 0,
            "render_code_errors": 0,
            "modify_code_errors": 0,
            "finish_customization_calls": 0,
            "unknown_tool_calls": [],
        }
        self.used_tools = used_tools
        self.clarify_instruction = clarify_instruction

    def get_feature_location(self, feature_name: str) -> list[tuple[list[str], float]]:
        mappings = self.mapped_code.get_cimappings(feature_name)
        return [
            ([self.mapped_code.code[span[0] : span[1]] for span in mapping.spans], prob)
            for (mapping, prob) in mappings[:3]  # only the 3 most probable
        ]

    def render_code(self) -> Image.Image:
        try:
            rendered_image = self.code_renderer(self.mapped_code.code)
        except TexRendererException as e:
            raise ToolCallError(e.extract_error())
        return rendered_image

    def modify_code(self, edits: list) -> str:
        try:
            edits = [
                CodeEdit(edit["start"], edit["end"], edit["content"]) for edit in edits
            ]
            self.mapped_code.apply_edits(edits)
            edited_code = self.mapped_code.get_annotated()
        except ValueError as e:
            raise ToolCallError(str(e))
        return edited_code

    def apply_instruction(self, code: str, instruction: str):
        """Applies the instruction to the code"""

        if self.clarify_instruction:
            logger.info("clarifying the instruction")
            instruction = self.apply_clarification(instruction, base_image)
        # unifying the code for easier parsing in steps ahead
        code = "\n".join(line.strip() for line in code.split("\n"))
        # render image
        base_image = self.code_renderer(code)
        logger.info("Searching for features")

        features = self.search_module.get_features(base_image)

        mapped_code = self.identification_module.identify(
            base_image=base_image, features=features, code=code
        )

        self.mapped_code = mapped_code

        messages = [
            {"role": "system", "content": EDITION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": IT_PROMPT.format(
                    instruction=instruction, content=mapped_code.get_annotated()
                ),
            },
        ]

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            tools=self.used_tools,
        )
        # handling tool calls
        while True:
            save_conversation(messages, ".tmp/edition")
            messages.append(completion.choices[0].message)  # append the Model's message
            if completion.choices[0].message.tool_calls is not None:
                for tool_call in completion.choices[0].message.tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    self.conv_data[name + "_calls"] = (
                        self.conv_data[name + "_calls"] + 1
                    )
                    result = None
                    result_render = None
                    try:
                        if name == "get_feature_location":
                            result = str(self.get_feature_location(**args))
                        if name == "render_code":
                            result = self.render_code(**args)
                            encoded_image = encode_image(image=result)
                            result = "Rendering worked"
                            result_render = {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded_image}"
                                },
                            }
                        if name == "modify_code":
                            result = str(self.modify_code(**args))
                    except ToolCallError as t:
                        self.conv_data[name + "_errors"] = (
                            self.conv_data[name + "_errors"] + 1
                        )
                        result = str(t)
                    # edition ending condition
                    if name == "finish_customization":
                        self.messages = messages
                        return self.mapped_code.code

                    if result is None:
                        self.conv_data["unknown_tool_calls"].append(name)
                        result = "Unknown tool with name: " + name

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )
                    if result_render is not None:
                        messages.append({"role": "user", "content": [result_render]})

            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
                tools=self.used_tools,
            )

    def apply_clarification(self, instruction: str, base_image: Image.Image):
        encoded_image = encode_image(image=base_image)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
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
        return response.choices[0].message.content
