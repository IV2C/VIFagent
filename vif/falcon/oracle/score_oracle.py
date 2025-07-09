from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

from loguru import logger
import torch
from PIL import Image
from collections.abc import Callable
from openai import Client
from vif.falcon.oracle.oracle import OracleModule, OracleResponse
from vif.models.module import LLMmodule
from vif.prompts.identification_prompts import PINPOINT_PROMPT
from vif.utils.detection_utils import get_boxes
import ast

from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

from vif.utils.image_utils import adjust_bbox, encode_image, get_used_pixels, se

import imagehash


class OracleScoreBoxModule(OracleModule):
    """Initial implementation of the oracle module, based on score computation from feature similarity/disimilarity
    NOTE: Not tested"""

    def __init__(
        self,
        *,
        model,
        client: Client,
        temperature=0.3,
        debug: bool = False,
        debug_folder: str = ".tmp/debug",
    ):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        super().__init__(
            debug=debug,
            debug_folder=debug_folder,
            client=client,
            temperature=temperature,
            model=model,
        )

    def get_edit_list(
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

    def compute_selfembedding(self, features: list[str]):
        # embedding the features
        self.emb_features = self.embedding_model.encode(features)
        self_similarities = self.embedding_model.similarity(
            self.emb_features, self.emb_features
        )
        max_selfsim = torch.quantile(self_similarities[self_similarities < 1], 0.75)

        self.max_authorized_selfsim = max_selfsim + ((1 - max_selfsim) / 2)

    def get_oracle(
        self, features: list[str], instruction: str, base_image: Image.Image
    ) -> Callable[[Image.Image], OracleResponse]:
        logger.info("Creating Oracle")

        original_detected_boxes = self.detect_feat_boxes(features, base_image)
        # first filtering the features depending on wether they have been detected in the original image
        features = list(set([box["label"] for box in original_detected_boxes]))
        # The two tasks are independent and can be executed in parallel
        with ThreadPoolExecutor() as executor:
            f1 = executor.submit(self.get_edit_list, features, instruction, base_image)
            f2 = executor.submit(self.compute_selfembedding, features)

            features_to_edit, features_to_delete, features_to_add = f1.result()
            _ = f2.result()

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
        ) -> tuple[bool, str, Any]:
            """Assert that an image solution has edited the right features

            Args:
                image (Image.Image): image customized by the LLM

            """

            customized_detected_boxes = get_boxes(
                image,
                self.client,
                features + features_to_add,
                self.model,
                self.temperature,
            )
            customized_detected_boxes_labels = [
                box["label"] for box in customized_detected_boxes
            ]

            # computing scores and conditions
            ##  edit score
            if len(features_to_edit) == 0:
                edit_condition = True
                edit_score = -1
            else:

                edit_score = OracleScoreBoxModule.get_score(
                    features_to_edit,
                    original_detected_boxes,
                    customized_detected_boxes,
                    base_image,
                    image,
                )
                edit_condition = edit_score > 0.1  # TODO adjust parameter
            ## add score
            not_added_features = [
                feature_to_add
                for feature_to_add in features_to_add
                if feature_to_add not in customized_detected_boxes_labels
            ]

            added_condition = len(not_added_features) == 0
            ## removed score
            not_removed_features = [
                feature_to_delete
                for feature_to_delete in features_to_delete
                if feature_to_delete in customized_detected_boxes_labels
            ]

            deleted_condition = len(not_removed_features) == 0

            full_condition = edit_condition and added_condition and deleted_condition

            # for observability
            score_object = {
                "edit_score": edit_score,
                "to_add": features_to_add,
                "to_delete": features_to_delete,
                "to_edit": features_to_edit,
                "original_boxes": original_detected_boxes,
                "customized_boxes": customized_detected_boxes,
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
                full_report,
                score_object,
            )

        return oracle

    @staticmethod
    def ensure_match_ori_custom(
        all_feature_lists: tuple[list[str], list[str], list[str]],
        original_detected_labels: set[str],
        customized_detected_labels: set[str],
    ) -> tuple[float, str]:
        """Ensures that the detected boxes in the custom image satify the following condition
        detected_labels-features_to_add = original_labels-feature_to_remove

        """
        features_to_edit, features_to_add, features_to_remove = all_feature_lists

        # ensure all features to edit are there in the modified image
        features_to_edit_not_detected = [
            '"' + feat_to_edit + '"'
            for feat_to_edit in features_to_edit
            if feat_to_edit not in customized_detected_labels
        ]

        if len(features_to_edit_not_detected) > 0:
            feedback = f"The feature(s) {",".join(features_to_edit_not_detected)} was supposed to be edited, but was not detected in the customized image."
            return tuple(0.0, feedback)

        # ensure that the same features have been detected in the original and customized image
        customized_labels_sub = customized_detected_labels.difference(
            set(features_to_add)
        )
        original_label_sub = original_detected_labels.difference(
            set(features_to_remove)
        )

        if not_in_customized := original_label_sub - customized_labels_sub != set():
            nc_f = ['"' + nc + '"' for nc in not_in_customized]
            feedback = (
                f"The feature(s) {",".join(nc_f)} were not detected in the customized image, they must have been hidden or altered."
                if len(nc_f) > 1
                else f"The feature {nc_f[0]} was not detected in the customized image, It must have been hidden or altered."
            )
            return tuple(0.0, feedback)
        return None

    @staticmethod
    def get_score(
        all_feature_lists: tuple[list[str], list[str], list[str]],
        original_detected_boxes: list[tuple],
        customized_detected_boxes: list[tuple],
        image_1: Image.Image,
        image_2: Image.Image,
    ) -> tuple[float, str]:
        """computes an edit score based on the boxes detected in both images, and the list of features to edit.

        Note: It is a precondition of this function that

        Returns:
            tuple[float,EditReason,str]: The score, and open string feedback
        """

        features_to_edit = all_feature_lists[0]

        customized_detected_labels = set(
            [box["label"] for box in customized_detected_boxes]
        )
        original_detected_labels = set(
            [box["label"] for box in original_detected_boxes]
        )

        if (
            return_value := OracleScoreBoxModule.ensure_match_ori_custom(
                all_feature_lists, original_detected_labels, customized_detected_labels
            )
            != NotImplemented
        ):
            return return_value

        # asjudting the boxes to the images
        original_features_cropped = {
            box["label"]: adjust_bbox(box, image_1)[
                "box_2d"
            ]  # TODO make a copy before modifying
            for box in original_detected_boxes
        }

        customized_features_cropped = {
            box["label"]: adjust_bbox(box, image_2)[
                "box_2d"
            ]  # TODO make a copy before modifying
            for box in customized_detected_boxes
        }

        # Getting the boxes to edit
        get_box_to_edit = lambda boxes: [
            box for label, box in boxes.items() if label in features_to_edit
        ]

        original_features_to_edit = get_box_to_edit(original_features_cropped)
        customized_features_to_edit = get_box_to_edit(customized_features_cropped)

        # encompass feature location difference
        # first "union" score
        cropped_im1 = Image.new("RGB", image_1.size, (0, 0, 0))
        for box in original_features_to_edit:
            region = image_1.crop(box)
            cropped_im1.paste(region, box)
        cropped_im2 = Image.new("RGB", image_2.size, (0, 0, 0))
        for box in customized_features_to_edit:
            region2 = image_2.crop(box)
            cropped_im2.paste(region2, box)
        # computing the number of pixel not black and dividing the squarred error by it
        # used_pixels = get_used_pixels(cropped_im1)#TODO check wether this value might be useful
        hash_ori_box = imagehash.phash(cropped_im1)
        hash_cust_box = imagehash.phash(cropped_im2)
        phash_diff_modified = hash_ori_box - hash_cust_box

        # Second "anti-union" score
        hid_im1 = image_1.copy()
        hid_im2 = image_2.copy()
        for box in original_features_to_edit:
            region = image_1.crop(box)
            crop_size = (box[2] - box[0], box[3] - box[1])
            hid_im1.paste(
                Image.new("RGB", crop_size, (0, 0, 0)),
                box,
            )
        for box in customized_features_to_edit:
            region = image_2.crop(box)
            crop_size = (box[2] - box[0], box[3] - box[1])
            hid_im2.paste(
                Image.new("RGB", crop_size, (0, 0, 0)),
                box,
            )  # put black image on the boxes

        hash_ori_box_ne = imagehash.phash(cropped_im1)
        hash_cust_box_ne = imagehash.phash(cropped_im2)
        phash_diff_non_modified = hash_ori_box_ne - hash_cust_box_ne

        phash_score = float(
            phash_diff_modified / (phash_diff_modified + phash_diff_non_modified)
        )

        return phash_score

    @staticmethod
    def get_score_size_dependent(
        boxes: list[tuple[int, int, int, int]],
        image_1: Image.Image,
        image_2: Image.Image,
    ):
        """First implementation of image dependence

        Args:
            boxes (list[tuple[int, int, int, int]]): _description_
            image_1 (Image.Image): _description_
            image_2 (Image.Image): _description_

        Returns:
            _type_: _description_
        """

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
