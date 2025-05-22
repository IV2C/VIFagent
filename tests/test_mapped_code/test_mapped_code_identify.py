import unittest
import json
from PIL import Image
import numpy as np
from vif_agent.feature import CodeImageMapping
import vif_agent.modules.identification.mapping


class TestMappedCodeIdentify(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.monkey_features =[
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

        def new_get_boxes(image, client, features, model, temperature):
            return json.load(open("tests/resources/mapped_code/monkey_boxes.json", "r"))

        vif_agent.modules.identification.mapping.get_boxes = new_get_boxes

    def test_identify_simple(self):

        box_id_module = (
            vif_agent.modules.identification.mapping.BoxIdentificationModule(
                client=None,
                model=None,
            )
        )

        mapped_code = box_id_module.identify(
            base_image=self.monkey_image,
            code=self.monkey_code,
            features=self.monkey_features,
        )

        self.assertEqual(
            mapped_code.feature_map["left black monkey eye"][0][0].spans, [(1383, 1430)]
        )
