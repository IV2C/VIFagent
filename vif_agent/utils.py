import base64
from io import BytesIO
from PIL import Image
import numpy as np


def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_image(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def norm_mse(image1: Image.Image, image2: Image.Image):
    """Compute Mean Squared Error between two PIL images."""
    arr1 = np.array(image1, dtype=np.float32)
    arr2 = np.array(image2, dtype=np.float32)

    if arr1.shape != arr2.shape:
        raise ValueError("Images must have the same dimensions")

    max_val = max(arr1.max(), arr2.max())
    if max_val == 0:
        return 0.0  # Avoid divide-by-zero

    # Normalize to [0, 1]
    arr1 /= max_val
    arr2 /= max_val

    return np.mean((arr1 - arr2) ** 2)


def adjust_bbox(box, image: Image.Image):
    adjust = lambda box_k, cursize: (box_k / 1000) * cursize
    box_2d = box["box_2d"]
    new_box = (
        int(adjust(box_2d[1], image.width)),
        int(adjust(box_2d[0], image.height)),
        int(adjust(box_2d[3], image.width)),
        int(adjust(box_2d[2], image.height)),
    )
    box["box_2d"] = new_box
    return box


import uuid


def show_conversation(messages: list):
    with open(".tmp/edition/conversation.txt", "w") as conv:
        conv.write("")#cleaning the file
    with open(".tmp/edition/conversation.txt", "a") as conv:
        id = 0
        for message in messages:
            if not isinstance(message,dict):
                message = message.__dict__
            
            match message["role"]:
                case "assistant":
                    conv.write("Assistant:\n")
                    if message["content"] is not None:
                        conv.write("Message: " + message["content"] + "\n")
                    if message["tool_calls"] is not None:
                        for tool in message["tool_calls"]:
                            conv.write(
                                tool.id + " : " + tool.function.name + f"({tool.function.arguments})\n"
                            )
                case "user":
                    conv.write("User:\n")
                    if isinstance(message["content"], list):
                        image_b64 = message["content"][0]["image_url"]["url"]
                        write_base64_to_image(
                            image_b64, ".tmp/edition/" + str(id) + ".png"
                        )
                        conv.write(str(id) + "\n")
                        id += 1
                    else:
                        conv.write(message['content'] + "\n")

                case "tool":
                    conv.write("Tool:\n")
                    conv.write(message["tool_call_id"] + " : " + message["content"]+"\n")
            conv.write("_________________________________________________\n")


def write_base64_to_image(base64_str: str, output_path: str) -> None:
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]
    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(base64_str))
