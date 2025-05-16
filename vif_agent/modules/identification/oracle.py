import math
from vif_agent.modules.identification.identification import IdentificationModule
from PIL import Image
from collections.abc import Callable
from openai import Client
from vif_agent.modules.identification.prompt import PINPOINT_PROMPT
from vif_agent.modules.identification.utils import dsim_box, get_boxes
import ast

from vif_agent.utils import encode_image, get_used_pixels, mse, se


class IdentificationOracleBoxModule(IdentificationModule):
    def __init__(
        self,
        *,
        model,
        client: Client,
        temperature=0.3,
        debug: bool = False,
        debug_folder: str = ".tmp/debug",
        pinpoint_client: Client = None,
        pinpoint_model=None,
        pinpoint_model_temperature=None,
    ):
        self.pinpoint_client = pinpoint_client or client
        self.pinpoint_model = pinpoint_model or model
        self.pinpoint_model_temperature = pinpoint_model_temperature or temperature

        self.identification_client = client
        self.identification_model = model
        self.identification_model_temperature = temperature
        super().__init__(debug=debug, debug_folder=debug_folder)

    def get_oracle(
        self, features: list[str], instruction: str, base_image: Image.Image
    ) -> Callable[[Image.Image], tuple[list[tuple[str, float]], bool]]:
        feature_string = ",".join([f for f in features])
        pinpoint_instructions = PINPOINT_PROMPT.format(
            features=feature_string, instruction=instruction
        )
        encoded_image = encode_image(base_image)
        response = self.pinpoint_client.chat.completions.create(
            model=self.pinpoint_model,
            temperature=self.pinpoint_model_temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": pinpoint_instructions,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            },
                        },
                    ],
                },
            ],
        )
        response = response.choices[0].message.content
        features_to_edit, features_to_delete, features_to_add = tuple(
            [
                ast.literal_eval(arr)
                for arr in response.split("ANSWER:")[1].strip().splitlines()
            ]
        )

        original_detected_boxes = get_boxes(
            base_image,
            self.identification_client,
            features,
            self.identification_model,
            self.identification_model_temperature,
        )

        @staticmethod
        def oracle(image: Image.Image) -> tuple[bool,tuple[float,bool,bool]]:
            """Assert that an image solution is a right solution

            Args:
                image (Image.Image): image customized by the LLM

            Returns:
                tuple[bool,tuple[float,bool,bool]]: tuple containing a boolean for wether the proposed solution is right, and another tuple with the edit_score(float), added_condition(bool) and delete_condition(bool)
            """
            customized_detected_boxes = get_boxes(
                image,
                self.identification_client,
                features + features_to_add,
                self.identification_model,
                self.identification_model_temperature,
            )
            customized_detected_boxes_dict = {
                box["label"]: box["box_2d"] for box in customized_detected_boxes
            }

            def get_score(
                boxes: list[tuple[int, int, int, int]],
                image_1: Image.Image,
                image_2: Image.Image,
            ):
                
                
                # first "union" score
                cropped_im1 = Image.new("RGB", image_1.size, (0, 0, 0))
                cropped_im2 = Image.new("RGB", image_2.size, (0, 0, 0))
                for box in boxes:
                    region = image_1.crop(box)
                    cropped_im1.paste(region, box)
                    region2 = image_2.crop(box)
                    cropped_im2.paste(region2, box)
                #computing the number of pixel not black and dividing the squarred error by it
                cropped_im1.save("debug_cropped1.png")
                cropped_im2.save("debug_cropped2.png")
                used_pixels = get_used_pixels(cropped_im1)
                boxes_union_mse = se(cropped_im1, cropped_im2)/used_pixels

                # Second "anti-union" score
                hid_im1 = image_1.copy()
                hid_im2 = image_2.copy()
                for box in boxes:
                    region = image_2.crop(box)
                    crop_size = (box[2] - box[0], box[3] - box[1])
                    hid_im1.paste(
                        Image.new("RGB", crop_size, (0, 0, 0)),
                        box,
                    )
                    hid_im2.paste(
                        Image.new("RGB", crop_size, (0, 0, 0)),
                        box,
                    )  # put black image on the boxes 
                #this time dividing by the prod of the size minus the number of "anti-used" pixels
                hid_im1.save("debug_hidden1.png")
                hid_im2.save("debug_hidden2.png")
                box_antiunion_mse = se(hid_im1, hid_im2)/(math.prod(image_1.size)-used_pixels)
                return float(boxes_union_mse / (1+box_antiunion_mse))

            # computing scores
            
            
            features_to_edit_boxes = [
                box["box_2d"]
                for box in original_detected_boxes + customized_detected_boxes
                if box["label"] in features_to_edit
            ]
            edit_score = get_score(features_to_edit_boxes,base_image,image)
            edit_condition = (
                edit_score > 0.3
            )  # TODO adjust parameter
            
            added_condition = all(
                [
                    feature_to_add in customized_detected_boxes_dict
                    for feature_to_add in features_to_add
                ]
            )
            deleted_condition = all(
                [
                    feature_to_delete not in customized_detected_boxes_dict
                    for feature_to_delete in features_to_delete
                ]
            )

            full_condition = edit_condition and added_condition and deleted_condition

            return full_condition, ( edit_score, added_condition,deleted_condition)

        return oracle

    def __str__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  identification_model={self.identification_model},\n"
            f"  identification_model_temperature={self.identification_model_temperature},\n"
            f"  pinpoint_model={self.pinpoint_model},\n"
            f"  pinpoint_model_temperature={self.pinpoint_model_temperature},\n"
            f"  debug={self.debug},\n"
            f"  debug_folder='{self.debug_folder}'\n"
            f")"
        )
