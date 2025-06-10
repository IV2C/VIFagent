import unittest

import torch

from vif.CodeMapper.feature import CodeImageMapping, MappedCode
from vif.agent.agent import FeatureAgent
from vif.models.misc import ToolCallError
from vif.utils.renderer.tex_renderer import TexRenderer


class TestMappedCode(unittest.TestCase):

    def test_edition_modify_raises(self):
        # input
        code = """        dog
        cat
        
        foo
        bar
        foo
        """

        edits: list = [
            {"start":2,"end":5,"content":"        cat"},
            {"start":4,"end":5,"content":"        dog"}
        ]

        mapped_code = MappedCode(code=code, feature_map=None, image=None)
        editionmodule: FeatureAgent = FeatureAgent(client=None, model="",code_renderer=TexRenderer().from_string_to_image)
        editionmodule.mapped_code = mapped_code

        with self.assertRaises(ToolCallError):
            editionmodule.modify_code(edits)

    def test_edition_feature_location(self):
        # input
        code = """        dog
        cat
        
        foo
        bar
        foo
        """

        dummy_mapping: dict[str, list[tuple[CodeImageMapping, float]]] = {
            "foo": [
                (
                    CodeImageMapping(
                        spans=[(41, 44)], box_zone=None, segment_zone=None
                    ),
                    torch.tensor(0.5),
                ),
                (
                    CodeImageMapping(
                        spans=[(65, 68)], box_zone=None, segment_zone=None
                    ),
                    torch.tensor(0.4),
                ),
            ],
            "cat": [
                (
                    CodeImageMapping(
                        spans=[(20, 23)], box_zone=None, segment_zone=None
                    ),
                    torch.tensor(0.7),
                ),
                (
                    CodeImageMapping(spans=[(5, 10)], box_zone=None, segment_zone=None),
                    torch.tensor(0.1),
                ),
            ],
        }
        mapped_code = MappedCode(code=code, feature_map=dummy_mapping, image=None)
        editionmodule: FeatureAgent = FeatureAgent(client=None, model="",code_renderer=TexRenderer().from_string_to_image)
        editionmodule.mapped_code = mapped_code

        # even if cat !=cats, similarities are normalizes between 0 and 1, so we get the same probs as the ones in dummy_mapping
        expected = [
            (["cat"], torch.tensor([0.7])),
            (["   do"], torch.tensor([0.1])),
            (["foo"], torch.tensor([0.0])),
        ]

        result = editionmodule.get_feature_location("cats")
        self.assertEqual(expected, result)
