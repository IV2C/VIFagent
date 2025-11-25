from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import re
import traceback
from openai import Client
from regex import Match
from vif.baselines.models import RegexException, RequestException
from vif.baselines.verifiers_baseline.ViperGPT_adapt.ViperGPT_prompts import (
    VIPER_CODEGEN_PROMPT,
)
from vif.baselines.verifiers_baseline.ViperGPT_adapt.image_patch import (
    ImagePatch,
    best_image_match,
    distance,
)
from vif.baselines.verifiers_baseline.ver_baseline import TexVerBaseline
from vif.models.detection import BoundingBox


def eval_code(code: str, initial_image, customized_image):
    ImagePatch.token_usage = defaultdict(list)
    ImagePatch.boxes = list()
    ImagePatch.box_cache = defaultdict(list)
    globals = {
        "ImagePatch": ImagePatch,
        "distance": distance,
        "best_image_match": best_image_match,
    }
    exec(code, globals)
    return (
        globals["execute_command"](initial_image, customized_image),
        ImagePatch.token_usage,
        ImagePatch.boxes,
    )


@dataclass
class ViperGPTMetadata:
    generated_function: str
    boxes: list[BoundingBox]


class ViperGPTVerifier(TexVerBaseline):
    def __init__(self, *args, model, client: Client, temperature, **kwargs):

        self.model = model
        self.client = client
        self.temperature = temperature

        super().__init__(*args, **kwargs)

    def get_config_metadata(self):
        return {
            "name": "ViperGPTVerifier",
            "model": self.model,
            "temperature": self.temperature,
        }

    def get_code(self, ver_eval_input):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": VIPER_CODEGEN_PROMPT.format(
                            instruction=ver_eval_input.initial_instruction
                        ),
                    }
                ],
            },
        ]

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
            )
        except Exception as e:
            raise RequestException(messages=messages, wrapped_exception=e)

        cnt = response.choices[0].message.content
        pattern = r"```(?:\w+)?\n([\s\S]+?)```"
        id_match: Match[str] = re.search(pattern, cnt)

        if not id_match:
            raise RegexException(pattern=pattern, content=cnt)

        return id_match.group(1), response.usage

    def assess_customization(self, ver_eval_input):
        try:
            generated_function, usage = self.get_code(ver_eval_input)
            ver_eval_input.additional_metadata["generated_function"] = generated_function
        except Exception as e:
            ver_eval_input.errors["code_generation"] = [traceback.format_exc()]
            return ver_eval_input

        try:
            condition, usages, boxes = eval_code(
            generated_function,
            ver_eval_input.initial_image,
            ver_eval_input.initial_solution_image,
        )
        except:
            ver_eval_input.errors["code_execution"] = [traceback.format_exc()]
            return ver_eval_input
        
        metadata = ViperGPTMetadata(generated_function, boxes)
        ver_eval_input.additional_metadata = asdict(metadata)
        ver_eval_input.usage_metadata = {"Base": [usage]}
        ver_eval_input.usage_metadata["box"] = usages["box"]
        ver_eval_input.usage_metadata["simple_query"] = usages["simple_query"]
        ver_eval_input.usage_metadata["llm_query"] = usages["llm_query"]

        ver_eval_input.classified_score = 1.0 if condition else 0.0
        return ver_eval_input
