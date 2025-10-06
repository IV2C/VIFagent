# Property condition => call to other llm to check property applied on the image

# TODO
import re
from vif.falcon.oracle.guided_oracle.expressions import OracleExpression
from loguru import logger
from openai import Client
from PIL import Image

from vif.prompts.property_check_prompt import PROPERTY_PROMPT
from vif.utils.image_utils import concat_images_horizontally, encode_image


class property(OracleExpression):

    def __init__(self, property):
        self.property = property

    def __invert__(self):
        self.negated = True
        return self
    
    def evaluate(
        self,
        *,
        original_image: Image.Image,
        custom_image: Image.Image,
        client: Client,
        model: str,
        temperature: float,
    ) -> tuple[bool, list[str]]:

        concat_image = concat_images_horizontally([original_image, custom_image])
        encoded_image = encode_image(concat_image)
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": PROPERTY_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.property,
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
        cnt = response.choices[0].message.content
        pattern = r"\\boxed{(True|False)}"
        id_match = re.search(pattern, cnt)

        if not id_match:
            return None
        
        bool_response = bool(id_match.group(1))
        #TODO self negated and feedback toussatoussa
        
        return bool_response
