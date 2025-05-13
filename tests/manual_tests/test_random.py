import json
import os
import re
from openai import OpenAI
from build.lib.vif_agent.prompt import FEATURE_IDENTIFIER_PROMPT
from vif_agent.modules.identification.utils import get_boxes
from vif_agent.renderer.tex_renderer import TexRenderer
from vif_agent.utils import encode_image

renderer = TexRenderer()
monkey_code = open("tests/resources/monkey.tex", "r").read()
image = renderer.from_string_to_image(monkey_code)

client = OpenAI(
    api_key=os.environ.get("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
model = "gemini-2.0-flash"
temperature = 0.3


# getting feature
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
                    "text": FEATURE_IDENTIFIER_PROMPT,
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
search_match = re.search(pattern, response.choices[0].message.content)


features_match = search_match.group(1)
features = json.loads(features_match)
print("Features")
print(features)

boxes = get_boxes(
    client=client, image=image, model=model, temperature=temperature, features=features
)


with open("tests/resources/mapped_code/monkey_boxes", "w") as mkbx:
    json.dump(boxes, mkbx)

for box in boxes:
    print("----------------------")
    print(box["label"])
    print(box["box_2d"])
