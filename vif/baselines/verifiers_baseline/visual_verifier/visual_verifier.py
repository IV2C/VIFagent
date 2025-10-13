import re
from openai import Client
from vif.baselines.models import RegexException, RequestException
from vif.baselines.verifiers_baseline.ver_baseline import TexVerBaseline
from vif.utils.image_utils import concat_images_horizontally, encode_image

IMAGE_VERIFY_SYSTEM_PROMPT: str = """
You are a verification agent, your task is to assess whether a given customization instruction has been applied on a image or not.
You will be given the initial and customized images and the instruction.

Your response must always contain the final answer in the format:
\\boxed{True} or \\boxed{False}

Answer with True when the instruction is applied, False when it is not.
"""

TEXT_IMAGE_VERIFY_PROMPT: str = """
INSTRUCTION:
{instruction}
"""


class VisualVerifier(TexVerBaseline):
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

        messages = (
            [
                {
                    "role": "system",
                    "content": IMAGE_VERIFY_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": TEXT_IMAGE_VERIFY_PROMPT.format(
                                instruction=ver_eval_input.initial_instruction
                            ),
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

        try:
            response = self.client.chat.completions.create(
                messages=messages, model=self.model, temperature=self.temperature
            )
        except Exception as e:
            raise RequestException(messages=messages, wrapped_exception=e)

        cnt = response.choices[0].message.content
        pattern = r"\\boxed{(True|False)}"
        id_match = re.search(pattern, cnt)

        if not id_match:
            raise RegexException(pattern=pattern, content=cnt)

        condition = id_match.group(1) == "True"
        ver_eval_input.classified = condition
        return ver_eval_input
