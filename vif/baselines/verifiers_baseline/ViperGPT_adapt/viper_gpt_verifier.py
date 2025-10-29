from dataclasses import asdict, dataclass
import json
import re
from openai import Client
from regex import Match
from vif.baselines.models import RegexException, RequestException
from vif.baselines.verifiers_baseline.ViperGPT_adapt.ViperGPT_prompts import (
    VIPER_CODEGEN_PROMPT,
)
from vif.baselines.verifiers_baseline.ViperGPT_adapt.image_patch import ImagePatch
from vif.baselines.verifiers_baseline.ver_baseline import TexVerBaseline


def eval_code(code: str, initial_image, customized_image):
    globals = {"ImagePatch": ImagePatch}
    exec(code, globals)
    return globals["execute_command"](initial_image, customized_image)

@dataclass
class ViperGPTMetadata:
    generated_function:str

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
    def assess_customization(self, ver_eval_input):
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

        generated_function = id_match.group(1)
        condition = eval_code(
            generated_function,
            ver_eval_input.initial_image,
            ver_eval_input.initial_solution_image,
        )
        
        metadata = ViperGPTMetadata(generated_function)
        ver_eval_input.additional_metadata = asdict(metadata)
        
        ver_eval_input.classified = condition
        return ver_eval_input
