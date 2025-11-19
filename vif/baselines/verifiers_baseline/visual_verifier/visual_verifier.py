import re
from openai import Client
from vif.baselines.models import RegexException, RequestException
from vif.baselines.verifiers_baseline.ver_baseline import TexVerBaseline
from vif.utils.image_utils import concat_images_horizontally, encode_image

IMAGE_VERIFY_SYSTEM_PROMPT: str = """
You are a verification agent, your task is to assess whether a given customization instruction has been applied on a image or not.
You will be given the initial and customized images and the instruction.

Your response must always contain the final answer in the format:
\\boxed{score}

With score being a score between 0 and 1.
0.0 => not applied at all.
1.0 => Perfectly applied.
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

    def get_config_metadata(self):
        return {
            "name": "VisualVerifier",
            "model": self.model,
            "temperature": self.temperature,
        }

    def assess_customization(self, ver_eval_input):
        ver_eval_input.errors["base"] =[]

        concat_image = concat_images_horizontally(
            [ver_eval_input.initial_image, ver_eval_input.initial_solution_image]
        )
        encoded_image = encode_image(concat_image)

        messages = [
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
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    },
                ],
            },
        ]

        try:
            response = self.client.chat.completions.create(
                messages=messages, model=self.model, temperature=self.temperature
            )
        except Exception as e:
            ver_eval_input.errors["base"].append(
                RequestException(messages=messages, wrapped_exception=e).json_dump()
            )
            return ver_eval_input

        cnt = response.choices[0].message.content
        ver_eval_input.additional_metadata["response_content"] = cnt
        pattern = r"\\boxed{([0-1]\.?[0-9]?)}"
        id_match = re.search(pattern, cnt)

        if not id_match:
            ver_eval_input.errors["base"].append(RegexException(pattern=pattern, content=cnt).json_dump())
            return ver_eval_input

        ver_eval_input.classified_score = float(id_match.group(1))

        ver_eval_input.usage_metadata = {"Base": [response.usage]}

        return ver_eval_input
