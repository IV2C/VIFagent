import json
from pyexpat import features
import re

from loguru import logger
from build.lib.vif_agent.prompt import DETECTION_PROMPT
from vif_agent.utils import adjust_bbox, encode_image
from PIL import Image
from openai import OpenAI


def get_boxes(
    image: Image.Image,
    client:OpenAI,
    model,
    temperature,
):
    encoded_image = encode_image(image=image)

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": DETECTION_PROMPT.format(
                            labels=", ".join(features["features"])
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ],
            }
        ],
    )
    pattern = r"```(?:\w+)?\n([\s\S]+?)```"
    id_match = re.search(pattern, response.choices[0].message.content)

    if not id_match:
        
        return None

    json_boxes = id_match.group(1)
    detected_boxes = json.loads(json_boxes)
    detected_boxes = [adjust_bbox(box, image) for box in detected_boxes]
    return detected_boxes
