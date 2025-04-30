from collections.abc import Callable
from dataclasses import dataclass
from openai import OpenAI

from vif_agent.feature import CodeEdit, MappedCode
from vif_agent.modules.edition.prompt import EDITION_SYSTEM_PROMPT
from vif_agent.modules.edition.tool_definitions import *
import json
from PIL import Image

from vif_agent.renderer.tex_renderer import TexRenderer
    
class LLMEditionModule:
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
        code_renderer: Callable[[str], Image.Image] = TexRenderer().from_string_to_image,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.debug = debug
        self.debug_folder = debug_folder
        self.used_tools = used_tools
        self.code_renderer = code_renderer

    def get_feature_location(self, feature_name: str) -> list[tuple[list[str], float]]:
        mappings = self.mapped_code.get_cimappings(feature_name)
        return [
            ([self.mapped_code.code[span[0] : span[1]] for span in mapping.spans], prob)
            for (mapping, prob) in mappings[:3]
        ]

    def render_code(self)->Image:
        return self.code_renderer(self.mapped_code.code)

    def modify_code(self, edits: list[CodeEdit]) -> str:
        
        pass

    def customize(self, mapped_code: MappedCode, instruction: str) -> str:
        self.mapped_code = mapped_code
        messages = [
            {"role": "system", "content": EDITION_SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            tools=self.used_tools,
        )
        # handling tool calls

        for tool_call in completion.choices[0].message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if name == "get_feature_location":
                result = self.get_feature_location(**args)
            if name == "render_code":
                result = self.render_code(**args)
            if name == "modify_code":
                result = self.modify_code(**args)
            if name == "finish_customization":
                return self.mapped_code.code

            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": str(result)}
            )

        pass
