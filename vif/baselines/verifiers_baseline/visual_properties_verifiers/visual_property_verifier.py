import re
from openai import Client
from pydantic import BaseModel
from vif.baselines.models import CompletionUsage, RegexException, RequestException
from vif.baselines.verifiers_baseline.ver_baseline import TexVerBaseline
from vif.utils.image_utils import concat_images_horizontally, encode_image

GET_PROPERTIES_SYSTEM_PROMPT: str = """
You are a verification agent. You will be given an instruction and an original image.
Your task is to list the properties that must be applied to the image to ensure that the requested customization is correctly made.

Your answer must always end with the following format:
Properties:
- p1
- p2
...

Each property must be an open-ended string of reasonable length on a single line, that describes a required visual or structural change.
"""

GET_PROPERTIES_PROMPT: str = """
INSTRUCTION:
{instruction}
"""

PROPERTY_EVALUATION_SYSTEM_PROMPT: str = """
You are an image evaluation agent.
You will be given two concatenated images (first = original, second = modified), and a property.
Your task is to determine whether the specified property has been applied to the second image.

Your response must always contain the final answer in the format:
\\boxed{True} or \\boxed{False}
"""
PROPERTY_EVALUATION_PROMPT: str = """
Property to verify:
{property}
"""


class VisualPropertiesVerifierMetadata(BaseModel):
    properties_eval: dict[str, bool]


class VisualPropertiesVerifier(TexVerBaseline):
    def __init__(self, *args, model, client: Client, temperature, **kwargs):

        self.model = model
        self.client = client
        self.temperature = temperature

        super().__init__(*args, **kwargs)

    def get_config_metadata(self):
        return {
            "name": "VisualPropertyVerifier",
            "model": self.model,
            "temperature": self.temperature,
        }

    def get_properties(
        self, initial_image, instruction
    ) -> tuple[list[str], CompletionUsage]:
        encoded_image = encode_image(initial_image)

        messages = (
            [
                {
                    "role": "system",
                    "content": GET_PROPERTIES_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": GET_PROPERTIES_PROMPT.format(
                                instruction=instruction
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
        pattern = r"Properties:\n((?:- .+\n?)+)"
        id_match = re.search(pattern, cnt)

        if not id_match:
            raise RegexException(pattern=pattern, content=cnt)

        properties = id_match.group(1).split("-")
        return properties, response.usage

    def check_property_applied(self, initial_image, customized_image, property):
        concat_image = concat_images_horizontally([initial_image, customized_image])
        encoded_image = encode_image(concat_image)

        messages = (
            [
                {
                    "role": "system",
                    "content": PROPERTY_EVALUATION_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": PROPERTY_EVALUATION_PROMPT.format(
                                property=property
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
        return condition, response.usage

    def assess_customization(self, ver_eval_input):
        properties, property_usage = self.get_properties(
            ver_eval_input.initial_image, ver_eval_input.initial_instruction
        )

        conditions_res = [
            self.check_property_applied(
                ver_eval_input.initial_image,
                ver_eval_input.initial_solution_image,
                property,
            )
            for property in properties
        ]
        conditions = [cond for cond, usage in conditions_res]
        property_check_usages = [usage for cond, usage in conditions_res]
        ver_eval_input.classified = all(conditions)

        prop_metadata = VisualPropertiesVerifierMetadata(
            {property: condition for property, condition in zip(properties, conditions)}
        )
        ver_eval_input.usage_metadata = {
            "property_gen": [property_usage],
            "property_check": property_check_usages,
        }

        ver_eval_input.additional_metadata = prop_metadata
        return ver_eval_input
