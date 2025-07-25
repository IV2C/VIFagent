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


from google import genai
from google.genai import types as genTypes


def save_conversation_google(contents: list[genTypes.Content], debug_path: str):
    with open(os.path.join(debug_path, "conversation.txt"), "w") as conv:
        id = 0
        for content in contents:
            match content.role:
                case "model":
                    conv.write("Assistant:\n")
                    for part in content.parts:
                        if part.thought is not None:
                            conv.write("Thoughts: " + part.text + "\n")
                        if part.text is not None:
                            conv.write("Message: " + part.text + "\n")
                        if part.function_call is not None:
                            conv.write(
                                part.function_call.id
                                + " : "
                                + part.function_call.name
                                + f"({part.function_call.args})\n"
                            )
                case "user":
                    conv.write("User:\n")
                    for part in content.parts:
                        if part.text is not None:
                            conv.write("Message: " + part.text + "\n")
                        if part.file_data is not None:
                            image_b64 = part.file_data.file_uri
                            write_base64_to_image(
                                image_b64,
                                os.path.join(debug_path, str(id) + ".png"),
                            )
                            conv.write(str(id) + "\n")
                            id += 1
                case "tool":
                    for part in content.parts:
                        conv.write("Tool:\n")
                        conv.write(
                            part.function_response.id
                            + " : "
                            + part.function_response.response
                            + "\n"
                        )
            conv.write("_________________________________________________\n")
