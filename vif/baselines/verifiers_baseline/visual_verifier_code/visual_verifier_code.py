import json
import re
from openai import Client
from vif.baselines.models import RegexException, RequestException
from vif.baselines.verifiers_baseline.ver_baseline import TexVerBaseline
from vif.utils.image_utils import concat_images_horizontally, encode_image


IMAGE_CODE_VERIFY_SYSTEM_PROMPT: str = """
You are a verification agent. Your task is to determine whether a given customization instruction has been correctly applied to an image.
You will be given the initial and customized images, and the instruction.
You have access to a Python code execution tool called eval_code(code).
eval_code takes a string of Python code as input and executes it.

- Within this code, you must define a python function called verify_customization that takes the variables initial_image and customized_image, which are both OpenCV images, of varying size.
- This function must return any object, that you will have access as the result of the call.
- You can use the libraries cv2, PIL, and numpy, which you must import explicitly before using them.

The function has the following signature:
```
def verify_customization(initial_image, customized_image):
    ...
```

You must create the full function, including imports and signature.

Your goal is to verify, by using Python code, whether the instruction has been correctly applied.
Your final response must always contain the final answer in the format:
\\boxed{True} or \\boxed{False}

Return True if the instruction has been applied; False otherwise.
"""

IMAGE_CODE_VERIFY_PROMPT: str = """
INSTRUCTION:
{instruction}
"""

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
    globals = {}
    exec(code, globals)
    return globals["verify_customization"](initial_image,customized_image)


class VisualCodeVerifier(TexVerBaseline):
    def __init__(self, *args, model, client: Client, temperature, **kwargs):

        self.model = model
        self.client = client
        self.temperature = temperature

        super().__init__(*args, **kwargs)

    def assess_customization(self, ver_eval_input):

        concat_image = concat_images_horizontally(
            [ver_eval_input.initial_image, ver_eval_input.initial_solution_image]
        )
        encoded_image = encode_image(concat_image)

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
