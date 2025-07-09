import base64
from io import BytesIO
import math
import os
from PIL import Image
import numpy as np

from vif.utils.image_utils import write_base64_to_image


def save_conversation(messages: list, debug_path: str):
    with open(os.path.join(debug_path, "conversation.txt"), "w") as conv:
        id = 0
        for message in messages:
            if not isinstance(message, dict):
                message = message.__dict__

            match message["role"]:
                case "assistant":
                    conv.write("Assistant:\n")
                    if message["content"] is not None:
                        conv.write("Message: " + message["content"] + "\n")
                    if message["tool_calls"] is not None:
                        for tool in message["tool_calls"]:
                            conv.write(
                                tool.id
                                + " : "
                                + tool.function.name
                                + f"({tool.function.arguments})\n"
                            )
                case "user":
                    conv.write("User:\n")
                    if isinstance(message["content"], list):
                        for content in message["content"]:
                            match content["type"]:
                                case "text":
                                    conv.write(content["text"] + "\n")
                                case "image_url":
                                    image_b64 = content["image_url"]["url"]
                                    write_base64_to_image(
                                        image_b64,
                                        os.path.join(debug_path, str(id) + ".png"),
                                    )
                                    conv.write(str(id) + "\n")
                                    id += 1
                    else:
                        conv.write(message["content"] + "\n")

                case "tool":
                    conv.write("Tool:\n")
                    conv.write(
                        message["tool_call_id"] + " : " + message["content"] + "\n"
                    )
            conv.write("_________________________________________________\n")
