from collections.abc import Callable
import json
import os
from PIL import Image
import unittest
from unittest.mock import MagicMock, PropertyMock

from openai import OpenAI

import vif_agent.modules.identification.oracle
from vif_agent.renderer.tex_renderer import TexRenderer


class TestMappedCode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.monkey_features = [
                "light blue background circle",
                "brown monkey head outline",
                "pink monkey face",
                "left brown monkey ear",
                "right brown monkey ear",
                "left pink monkey ear",
                "right pink monkey ear",
                "left black monkey eye",
                "right black monkey eye",
                "left pink monkey nostril",
                "right pink monkey nostril",
                "pink curved monkey mouth",
            ]
        

        cls.monkey_code = open("tests/resources/monkey.tex", "r").read()
        cls.monkey_image = Image.open("tests/resources/mapped_code/monkey.png")

        def original_image_get_boxes(image, client, features, model, temperature):
            return json.load(open("tests/resources/mapped_code/monkey_boxes.json", "r"))

        vif_agent.modules.identification.oracle.get_boxes = original_image_get_boxes

        client_mock = OpenAI(
            api_key="",
            base_url="",
        )
        mock_choice = MagicMock()
        mock_choice.message.content ="""
ANSWER:
["left brown monkey ear","right brown monkey ear","left pink monkey ear","right pink monkey ear"]
[]
[]"""
        mock_res = MagicMock()
        mock_res.choices = [mock_choice]
        
        
        client_mock.chat.completions.create = MagicMock(
            return_value=mock_res
        )

        cls.instruction = "Increase the size of the monkey's ears by a lot"
        cls.box_oracle_module = (
            vif_agent.modules.identification.oracle.IdentificationOracleBoxModule(
                model=None, client=client_mock
            )
        )
        cls.oracle: Callable[[Image.Image], tuple[list[tuple[str, float]], bool]] = (
            cls.box_oracle_module.get_oracle(cls.monkey_features,cls.instruction,cls.monkey_image)
        )

    def test_oracle_right_edit(self):

        def customized_image_get_boxes(image, client, features, model, temperature):
            return json.load(
                open("tests/resources/mapped_code/monkey_bigger_ears_boxes.json", "r")
            )

        modified_ears_monkey = open(
            "tests/resources/mapped_code/monkey_bigger_ears.tex", "r"
        ).read()
        renderer = TexRenderer()
        modified_ears_monkey_image = renderer.from_string_to_image(modified_ears_monkey)
        vif_agent.modules.identification.oracle.get_boxes = customized_image_get_boxes

        full_condition, (edit_score, added_condition, deleted_condition) = self.oracle(
            modified_ears_monkey_image
        )

        self.assertEqual(
            (full_condition, (edit_score, added_condition, deleted_condition)),
            (True, (0.3285546898841858, True, True)),
        )

        #TODO add other tests for failing upon wrong addition deletion etc
        
    def test_oracle_no_edit(self):

        def customized_image_get_boxes(image, client, features, model, temperature):
            return json.load(
                open("tests/resources/mapped_code/monkey_boxes.json", "r")
            )

        modified_ears_monkey = open(
            "tests/resources/mapped_code/monkey.tex", "r"
        ).read()
        renderer = TexRenderer()
        modified_ears_monkey_image = renderer.from_string_to_image(modified_ears_monkey)
        vif_agent.modules.identification.oracle.get_boxes = customized_image_get_boxes

        full_condition, (edit_score, added_condition, deleted_condition) = self.oracle(
            modified_ears_monkey_image
        )

        self.assertEqual(
            (full_condition, (edit_score, added_condition, deleted_condition)),
            (False, (0, True, True)),
        )

    def test_oracle_wrong_edit(self):

        def customized_image_get_boxes(image, client, features, model, temperature):
            return json.load(
                open("tests/resources/mapped_code/monkey_sad_boxes.json", "r")
            )

        modified_ears_monkey = open(
            "tests/resources/mapped_code/monkey_sad.tex", "r"
        ).read()
        renderer = TexRenderer()
        modified_ears_monkey_image = renderer.from_string_to_image(modified_ears_monkey)
        vif_agent.modules.identification.oracle.get_boxes = customized_image_get_boxes

        full_condition, (edit_score, added_condition, deleted_condition) = self.oracle(
            modified_ears_monkey_image
        )

        self.assertEqual(
            (full_condition, (edit_score, added_condition, deleted_condition)),
            (False, (0, True, True)),
        )