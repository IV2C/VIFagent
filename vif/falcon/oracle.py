import math
from typing import Any

from loguru import logger
import torch
from PIL import Image
from collections.abc import Callable
from openai import Client
from vif.models.module import LLMmodule
from vif.prompts.identification_prompts import PINPOINT_PROMPT
from vif.utils.detection_utils import get_boxes
import ast

from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

from vif.utils.image_utils import encode_image, get_used_pixels, se


class OracleBoxModule(LLMmodule):
    def __init__(
        self,
        *,
        model,
        client: Client,
        temperature=0.3,
        debug: bool = False,
        debug_folder: str = ".tmp/debug",
    ):
        self.debug = debug
        self.debug_folder = debug_folder
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        super().__init__(
            debug=debug,
            debug_folder=debug_folder,
            client=client,
            temperature=temperature,
            model=model,
        )

    def compute_selfembedding(self, features: list[str]):
        # embedding the features
        self.emb_features = self.embedding_model.encode(features)
        self_similarities = self.embedding_model.similarity(
            self.emb_features, self.emb_features
        )
        max_selfsim = torch.quantile(self_similarities[self_similarities < 1], 0.75)

        self.max_authorized_selfsim = max_selfsim + ((1 - max_selfsim) / 2)

    def get_featureset(
        self, features: list[str], instruction: str, base_image: Image.Image
    ):
        # getting the features to add/delete/modify
        feature_string = ",".join([f for f in features])
        pinpoint_instructions = PINPOINT_PROMPT.format(
            features=feature_string, instruction=instruction
        )
        encoded_image = encode_image(base_image)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
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
        return (features_to_edit, features_to_delete, features_to_add)

    def detect_feat_boxes(self, features: list[str], base_image: Image.Image):
        return get_boxes(
            base_image,
            self.client,
            features,
            self.model,
            self.temperature,
        )

    def get_oracle(
        self, features: list[str], instruction: str, base_image: Image.Image
    ) -> Callable[[Image.Image], tuple[bool, tuple[float, bool, bool], str, Any]]:
        logger.info("Creating Oracle")

        # The three tasks are independent and can be executed in parallel
        with ThreadPoolExecutor() as executor:
            f1 = executor.submit(self.get_featureset, features, instruction, base_image)
            f2 = executor.submit(self.compute_selfembedding, features)
            f3 = executor.submit(self.detect_feat_boxes, features, base_image)

            features_to_edit, features_to_delete, features_to_add = f1.result()
            _ = f2.result()
            original_detected_boxes = f3.result()

        # Removing the added features if they are too close to an already existing feature, using an ambedding model
        if len(features_to_add) > 0 and not (
            len(features_to_delete) == 0 and len(features_to_edit) == 0
        ):
            emb_add_features = self.embedding_model.encode(features_to_add)
            existing_features_sim = self.embedding_model.similarity(
                emb_add_features, self.emb_features
            )
            features_to_add = [
                added_feat
                for sim, added_feat in zip(existing_features_sim, features_to_add)
                if sim.max() < self.max_authorized_selfsim
            ]

        @staticmethod
        def oracle(
            image: Image.Image,
        ) -> tuple[bool, tuple[float, bool, bool], str, Any]:
            """Assert that an image solution is a right solution

            Args:
                image (Image.Image): image customized by the LLM

            Returns:
                tuple[bool,tuple[float,bool,bool],str,Any]: tuple containing a boolean for whether the proposed solution is right, and another tuple with the edit_score(float), added_condition(bool) and delete_condition(bool), and finally the report and a debug object
            """

            customized_detected_boxes = get_boxes(
                image,
                self.client,
                features + features_to_add,
                self.model,
                self.temperature,
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
                # computing the number of pixel not black and dividing the squarred error by it
                cropped_im1.save("debug_cropped1.png")
                cropped_im2.save("debug_cropped2.png")
                used_pixels = get_used_pixels(cropped_im1)
                boxes_union_mse = se(cropped_im1, cropped_im2) / used_pixels

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
                # this time dividing by the prod of the size minus the number of "anti-used" pixels
                hid_im1.save("debug_hidden1.png")
                hid_im2.save("debug_hidden2.png")
                box_antiunion_mse = se(hid_im1, hid_im2) / (
                    math.prod(image_1.size) - used_pixels
                )
                return float(boxes_union_mse / (1 + box_antiunion_mse))

            # computing scores and conditions

            if len(features_to_edit) == 0:
                edit_condition = True
                edit_score = -1
            else:
                features_to_edit_boxes = [
                    box["box_2d"]
                    for box in original_detected_boxes + customized_detected_boxes
                    if box["label"] in features_to_edit
                ]
                edit_score = get_score(features_to_edit_boxes, base_image, image)
                edit_condition = edit_score > 0.1  # TODO adjust parameter

            not_added_features = [
                feature_to_add
                for feature_to_add in features_to_add
                if feature_to_add not in customized_detected_boxes_dict
            ]

            added_condition = len(not_added_features) == 0

            not_removed_features = [
                feature_to_delete
                for feature_to_delete in features_to_delete
                if feature_to_delete in customized_detected_boxes_dict
            ]

            deleted_condition = len(not_removed_features) == 0

            full_condition = edit_condition and added_condition and deleted_condition

            # for observability
            debug_object = {
                "to_add": features_to_add,
                "to_delete": features_to_delete,
                "to_edit": features_to_edit,
                "original_boxes": customized_detected_boxes_dict,
                "customized_boxes": customized_detected_boxes_dict,
            }

            # creating report
            add_report: str = (
                f"The features [{', '.join(not_added_features)}] were not added to the resulting image.\n"
            )
            edit_report: str = (
                f"The features [{', '.join(features_to_edit)}] were not edited in the resulting image, here is the current score for your edit: {str(round(edit_score,3))}\n"
            )
            deleted_report: str = (
                f"The features [{', '.join(not_removed_features)}] were not removed to the resulting image.\n"
            )
            full_report = (
                add_report
                if not added_condition
                else (
                    edit_report
                    if not edit_condition
                    else "" + deleted_report if not deleted_condition else ""
                )
            )

            return (
                full_condition,
                (edit_score, added_condition, deleted_condition),
                full_report,
                debug_object,
            )

        return oracle