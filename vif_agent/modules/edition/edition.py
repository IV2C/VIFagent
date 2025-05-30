from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
import PIL.Image
from openai import OpenAI

from vif_agent.feature import CodeEdit, MappedCode
from vif_agent.modules.edition.prompt import EDITION_SYSTEM_PROMPT, IT_PROMPT
from vif_agent.modules.edition.tool_definitions import *
import json
from PIL import Image

from vif_agent.renderer.tex_renderer import TexRenderer
from vif_agent.utils import encode_image


class EditionModule:
    def customize(instruction: str):
        pass


class LLMEditionModule(EditionModule):
    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        temperature: float = 0.0,
        debug=False,
        debug_folder=".tmp/debug",
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.debug = debug
        self.debug_folder = debug_folder


class LLMAgenticEditionModule(LLMEditionModule):
    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        temperature: float = 0.0,
        debug=False,
        debug_folder=".tmp/debug",
        used_tools: list = [
            feature_find_tool,
            render_tool,
            modify_code_tool,
            finish_customization_tool,
        ],
        code_renderer: Callable[
            [str], Image.Image
        ] = TexRenderer().from_string_to_image,
    ):
        super().__init__(
            client=client,
            model=model,
            temperature=temperature,
            debug_folder=debug_folder,
            debug=debug,
        )
        self.code_renderer = code_renderer

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

    def get_feature_location(self, feature_name: str) -> list[tuple[list[str], float]]:
        mappings = self.mapped_code.get_cimappings(feature_name)
        return [
            ([self.mapped_code.code[span[0] : span[1]] for span in mapping.spans], prob)
            for (mapping, prob) in mappings[:3]  # only the 3 most probable
        ]

    def render_code(self) -> Image:
        try:
            rendered_image = self.code_renderer(self.mapped_code.code)
        except Exception as e:
            raise ToolCallError(str(e))
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

    def customize(self, instruction: str, mapped_code: MappedCode) -> str:
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
                                    "url": f"data:image/jpeg;base64,{encoded_image}"
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


class ToolCallError(Exception):
    pass


class OracleEditionModule(LLMEditionModule):
    def __init__(
        self, *, client, model, temperature=0, debug=False, debug_folder=".tmp/debug"
    ):
        super().__init__(
            client=client,
            model=model,
            temperature=temperature,
            debug=debug,
            debug_folder=debug_folder,
        )

    def send_initial_message(self, instruction: str, image: PIL.Image.Image):
        raise NotImplementedError()  # TODO
        encoded_image = encode_image(encoded_image)
        response = self.edition_module.client.chat.completions.create(
            model=self.edition_module.model,
            temperature=self.edition_module.temperature,
            messages=[
                {
                    "role": "system",
                    "content": None,
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

    def customize(
        instruction: str,
        code: str,
        oracle: Callable[
            [Image.Image], tuple[bool, tuple[float, bool, bool], str, Any]
        ],
    ):
        raise NotImplementedError()  # TODO


