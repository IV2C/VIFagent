import json
import os
import re
from openai import OpenAI
from build.lib.vif_agent.prompt import FEATURE_IDENTIFIER_PROMPT
from vif_agent.modules.identification.utils import get_boxes
from vif_agent.renderer.tex_renderer import TexRenderer
from vif_agent.utils import encode_image

renderer = TexRenderer()
monkey_code = open("tests/resources/mapped_code/monkey_sad.tex", "r").read()
image = renderer.from_string_to_image(monkey_code)

image.save("tests/resources/mapped_code/monkey_sad.png")

client = OpenAI(
    api_key=os.environ.get("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
model = "gemini-2.0-flash"
temperature = 0.3

features = [
    "left brown monkey ear",
    "right brown monkey ear",
    "left pink monkey ear",
    "right pink monkey ear",
]

boxes = get_boxes(
    client=client, image=image, model=model, temperature=temperature, features=features
)


with open("tests/resources/mapped_code/monkey_sad_boxes.json", "w") as mkbx:
    json.dump(boxes, mkbx)
