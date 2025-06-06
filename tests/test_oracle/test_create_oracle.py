from collections.abc import Callable
from PIL import Image
import unittest
from unittest.mock import MagicMock

from openai import OpenAI

import vif.falcon.oracle


class TestCreateOracle(unittest.TestCase):

    def test_get_oracle_similar_embedding(self):
        """
        tests that similar embedding in added features are removed
        """
        client_mock = OpenAI(
            api_key="",
            base_url="",
        )
        mock_choice = MagicMock()
        mock_choice.message.content = """
ANSWER:
[]
[]
["left grey eye of monkey"]"""
        monkey_features = [
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
        mock_res = MagicMock()
        mock_res.choices = [mock_choice]

        client_mock.chat.completions.create = MagicMock(return_value=mock_res)
        box_oracle_module = (
            vif.falcon.oracle.OracleBoxModule(
                model=None, client=client_mock
            )
        )
        box_oracle_module.detect_feat_boxes = MagicMock(return_value="")
        
        oracle: Callable[[Image.Image], tuple[list[tuple[str, float]], bool]] = (
            box_oracle_module.get_oracle(
                monkey_features, "",  Image.open("tests/resources/mapped_code/monkey.png")
            )
        )
        oracle=oracle.__func__
        closure_vars = {var: cell.cell_contents for var, cell in zip(oracle.__code__.co_freevars, oracle.__closure__)}

        self.assertEqual([],closure_vars["features_to_add"])
