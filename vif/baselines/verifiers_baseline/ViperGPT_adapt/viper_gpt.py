import json
import re
from openai import Client
from vif.baselines.models import RegexException, RequestException
from vif.baselines.verifiers_baseline.ViperGPT_adapt.ViperGPT_prompts import VIPER_CODEGEN_PROMPT
from vif.baselines.verifiers_baseline.ver_baseline import TexVerBaseline
from vif.utils.image_utils import concat_images_horizontally, encode_image



tools = [
    {
        "type": "function",
        "name": "eval_code",
        "description": "Interprets python code",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code that will be executed",
                },
            },
            "required": ["sign"],
        },
    },
]


def eval_code(code: str, initial_image, customized_image):
    globals = {
        
    }
    exec(code, globals)
    return globals["code_result"](initial_image,customized_image)


class ViperGPTVerifier(TexVerBaseline):
    def __init__(self, *args, model, client: Client, temperature, **kwargs):

        self.model = model
        self.client = client
        self.temperature = temperature

        super().__init__(*args, **kwargs)

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
                tools=tools,
            )
        except Exception as e:
            raise RequestException(messages=messages, wrapped_exception=e)

        tool_call_res = response.choices[0].message

        for tool_call in tool_call_res.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            if name == "eval_code":
                result = str(
                    eval_code(
                        code=args["code"],
                        initial_image=ver_eval_input.initial_image,
                        customized_image=ver_eval_input.initial_solution_image,
                    )
                )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

        final_response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            tools=tools,
        )

        cnt = final_response.choices[0].message.content
        pattern = r"\\boxed{(True|False)}"
        id_match = re.search(pattern, cnt)

        if not id_match:
            raise RegexException(pattern=pattern, content=cnt)

        condition = id_match.group(1) == "True"
        ver_eval_input.classified = condition
        return ver_eval_input
