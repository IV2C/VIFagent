import json
import re
import traceback
from openai import Client
from vif.baselines.models import CodeExecException, RegexException, RequestException
from vif.baselines.verifiers_baseline.ver_baseline import TexVerBaseline
from vif.utils.image_utils import concat_images_horizontally, encode_image


IMAGE_CODE_VERIFY_SYSTEM_PROMPT: str = """
You are a verification agent. Your task is to determine whether a given customization instruction has been correctly applied to an image.
You will be given the initial images and the instruction.
You have access to a Python code execution tool called eval_code(code).
eval_code takes a string of Python code as input and executes it.

- Within this code, you must define a python function called verify_customization that takes the variables initial_image and customized_image, which are both PIL images, of varying size.
- This function must return any object, that you will have access as the result of the call.
- You can use the libraries cv2, PIL, and numpy, which you must import explicitly before using them.

The function has the following signature:
```
def verify_customization(initial_image, customized_image):
    ...
```

You must create the full function, including imports and signature.
Do not call the function yourself, only implement it, its execution will be handled automatically and its output will be returned.
Ensure the code is executable, and there are no errors in it.

Your goal is to verify, by using Python code, whether the instruction has been correctly applied.
You must always call the tool eval code, unless you are satisfied with the tool call output and want to answer.
Then, your final response must always contain the final answer in the format:
\\boxed{score}

With score being a float between 0 and 1.
0.0 => not applied at all.
1.0 => Perfectly applied.
"""

IMAGE_CODE_VERIFY_PROMPT: str = """
INSTRUCTION:
{instruction}
"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "eval_code",
            "description": "Interprets python code",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code that will be executed, No bacticks allowed, just python code",
                    },
                },
                "additionalProperties": False,
                "required": ["code"],
            },
        },
    }
]


def eval_code(code: str, initial_image, customized_image):
    globals = {}
    exec(code, globals)
    return globals["verify_customization"](initial_image, customized_image)


class VisualCodeVerifier(TexVerBaseline):
    def __init__(self, *args, model, client: Client, temperature, **kwargs):

        self.model = model
        self.client = client
        self.temperature = temperature

        super().__init__(*args, **kwargs)

    def get_config_metadata(self):
        return {
            "name": "VisualCodeVerifier",
            "model": self.model,
            "temperature": self.temperature,
        }

    def assess_customization(self, ver_eval_input):
        ver_eval_input.usage_metadata = {"Base": []}

        ver_eval_input.errors["Base"] = []
        ver_eval_input.errors["final_request_regex"] = []

        encoded_image = encode_image(ver_eval_input.initial_image)

        messages = [
            {
                "role": "system",
                "content": IMAGE_CODE_VERIFY_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": IMAGE_CODE_VERIFY_PROMPT.format(
                            instruction=ver_eval_input.initial_instruction
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    },
                ],
            },
        ]

        max_iterations = 3
        iteration_count = 0
        while iteration_count < max_iterations:
            iteration_count += 1
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature,
                    tools=tools,
                    extra_body={"reasoning": {"effort": "low"}},
                )
                messages.append(response.choices[0].message.model_dump())
                ver_eval_input.usage_metadata["Base"].append(response.usage)
            except Exception as e:
                ver_eval_input.errors["Base"].append(
                    RequestException(messages=messages, wrapped_exception=e).json_dump()
                )
                return ver_eval_input
            if response.choices[0].message.tool_calls is not None:

                tool_call = response.choices[0].message.tool_calls[0]
                args = json.loads(tool_call.function.arguments)

                try:
                    result = str(
                        eval_code(
                            code=args["code"],
                            initial_image=ver_eval_input.initial_image,
                            customized_image=ver_eval_input.initial_solution_image,
                        )
                    )
                except Exception as e:
                    result = traceback.format_exc()
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
            else:
                break

        cnt = response.choices[0].message.content
        pattern = r"\\boxed{([0-1]\.?[0-9]?)}"
        id_match = re.search(pattern, cnt)

        if not id_match:
            ver_eval_input.errors["final_request_regex"].append(
                RegexException(pattern=pattern, content=cnt).json_dump()
            )
            return ver_eval_input

        ver_eval_input.classified_score = float(id_match.group(1))

        ver_eval_input.additional_metadata["conversation"] = messages

        return ver_eval_input
