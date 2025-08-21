import unittest

from vif.falcon.oracle.guided_oracle.expressions import (
    OracleExpression,
    added,
    angle,
    color,
    placement,
    position,
    removed,
)
from vif.models.detection import SegmentationMask
from PIL import Image
from parameterized import parameterized, parameterized_class


class TestExpression(unittest.TestCase):

    def __init__(self, methodName="runTest"):
        self.original_image: Image.Image = Image.new("1", (2, 2))
        self.original_image.putdata([0, 0, 0, 1])
        self.custom_image: Image.Image = Image.new("1", (2, 2))
        self.custom_image.putdata([0, 0, 1, 0])
        super().__init__(methodName)

    @parameterized.expand(
        [
            ("right", (40, 40, 60, 60), (40, 30, 60, 50)),
            ("left", (40, 30, 60, 50), (40, 40, 60, 60)),
            ("up", (30, 40, 50, 60), (40, 40, 60, 60)),
            ("down", (40, 40, 60, 60), (30, 40, 50, 60)),
        ]
    )
    def test_placement_valid(self, direction, boxA, boxB):
        original_features: list[SegmentationMask] = None
        custom_features: list[SegmentationMask] = [
            SegmentationMask(*boxA, None, "triangle"),
            SegmentationMask(*boxB, None, "rectangle"),
        ]
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return placement("triangle", "rectangle", direction)

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
        )
        self.assertTrue(result)
        self.assertEqual([], feedback)

    @parameterized.expand(
        [
            (
                "left",
                (40, 40, 60, 60),
                (40, 30, 60, 50),
                "The feature triangle is not on the left of the feature rectangle",
            ),
            (
                "right",
                (40, 30, 60, 50),
                (40, 40, 60, 60),
                "The feature triangle is not on the right of the feature rectangle",
            ),
            (
                "down",
                (30, 40, 50, 60),
                (40, 40, 60, 60),
                "The feature triangle is not under the feature rectangle",
            ),
            (
                "up",
                (40, 40, 60, 60),
                (30, 40, 50, 60),
                "The feature triangle is not above the feature rectangle",
            ),
        ]
    )
    def test_placement_invalid(self, direction, boxA, boxB, feedback_expected):
        original_features: list[SegmentationMask] = None
        custom_features: list[SegmentationMask] = [
            SegmentationMask(*boxA, None, "triangle"),
            SegmentationMask(*boxB, None, "rectangle"),
        ]
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return placement("triangle", "rectangle", direction)

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
        )
        self.assertFalse(result)
        self.assertEqual([feedback_expected], feedback)

    @parameterized.expand(
        [
            (
                "left",
                (40, 40, 60, 60),
                (40, 30, 60, 50),
                "The feature triangle is not on the left of the feature rectangle",
            ),
            (
                "right",
                (40, 30, 60, 50),
                (40, 40, 60, 60),
                "The feature triangle is not on the right of the feature rectangle",
            ),
            (
                "down",
                (30, 40, 50, 60),
                (40, 40, 60, 60),
                "The feature triangle is not under the feature rectangle",
            ),
            (
                "up",
                (40, 40, 60, 60),
                (30, 40, 50, 60),
                "The feature triangle is not above the feature rectangle",
            ),
        ]
    )
    def test_placement_valid_negated(self, direction, boxA, boxB, feedback_expected):
        original_features: list[SegmentationMask] = None
        custom_features: list[SegmentationMask] = [
            SegmentationMask(*boxA, None, "triangle"),
            SegmentationMask(*boxB, None, "rectangle"),
        ]
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return ~placement("triangle", "rectangle", direction)

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
        )
        self.assertTrue(result)
        self.assertEqual([], feedback)

    @parameterized.expand(
        [
            (
                "vertical",
                (30, 30, 50, 50),
                (40, 40, 60, 60),
                (30, 30, 50, 50),
                (50, 40, 70, 60),
            ),
            (
                "horizontal",
                (30, 30, 50, 50),
                (40, 40, 60, 60),
                (30, 30, 50, 50),
                (40, 50, 60, 70),
            ),
        ]
    )
    def test_position_valid(
        self, axis, originalBoxA, originalBoxB, customBoxA, customBoxB
    ):
        original_features: list[SegmentationMask] = [
            SegmentationMask(*originalBoxA, None, "triangle"),
            SegmentationMask(*originalBoxB, None, "rectangle"),
        ]
        custom_features: list[SegmentationMask] = [
            SegmentationMask(*customBoxA, None, "triangle"),
            SegmentationMask(*customBoxB, None, "rectangle"),
        ]
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return position("triangle", "rectangle", 2, axis)

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
        )
        self.assertTrue(result, feedback)
        self.assertEqual([], feedback)

    @parameterized.expand(
        [
            (
                "horizontal",
                (30, 30, 50, 50),
                (40, 40, 60, 60),
                (30, 30, 50, 50),
                (50, 40, 70, 60),
            ),
            (
                "vertical",
                (30, 30, 50, 50),
                (40, 40, 60, 60),
                (30, 30, 50, 50),
                (40, 50, 60, 70),
            ),
        ]
    )
    def test_position_valid_negated(
        self,
        axis,
        originalBoxA,
        originalBoxB,
        customBoxA,
        customBoxB,
    ):
        original_features: list[SegmentationMask] = [
            SegmentationMask(*originalBoxA, None, "triangle"),
            SegmentationMask(*originalBoxB, None, "rectangle"),
        ]
        custom_features: list[SegmentationMask] = [
            SegmentationMask(*customBoxA, None, "triangle"),
            SegmentationMask(*customBoxB, None, "rectangle"),
        ]
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return ~position("triangle", "rectangle", 2, axis)

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
        )
        self.assertTrue(result, feedback)
        self.assertEqual([], feedback)

    @parameterized.expand(
        [
            (
                "vertical",
                (30, 30, 50, 50),
                (40, 40, 60, 60),
                (30, 30, 50, 50),
                (50, 40, 70, 60),
                "The vertical distance between triangle and rectangle is 20.0 which is too close to 20.0.",
            ),
            (
                "horizontal",
                (30, 30, 50, 50),
                (40, 40, 60, 60),
                (30, 30, 50, 50),
                (40, 50, 60, 70),
                "The horizontal distance between triangle and rectangle is 20.0 which is too close to 20.0.",
            ),
        ]
    )
    def test_position_invalid_negated(
        self,
        axis,
        originalBoxA,
        originalBoxB,
        customBoxA,
        customBoxB,
        feedback_expected,
    ):
        original_features: list[SegmentationMask] = [
            SegmentationMask(*originalBoxA, None, "triangle"),
            SegmentationMask(*originalBoxB, None, "rectangle"),
        ]
        custom_features: list[SegmentationMask] = [
            SegmentationMask(*customBoxA, None, "triangle"),
            SegmentationMask(*customBoxB, None, "rectangle"),
        ]
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return ~position("triangle", "rectangle", 2, axis)

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
        )
        self.assertFalse(result)
        self.assertEqual([feedback_expected], feedback)

    @parameterized.expand(
        [
            (40),
            (45),
            (50),
            (-40),
            (-45),
            (-50),
            (130),
            (135),
            (140),
            (-130),
            (-135),
            (-140),
        ]
    )
    def test_rotated_valid(self, degree):
        import pickle

        original_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/rgb_stc.pickle", "rb").read()
        )
        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/stc_rgb_rotc.pickle", "rb").read()
        )
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return angle("blue square", degree)

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
        )
        self.assertTrue(result)
        self.assertEqual([], feedback)

    @parameterized.expand(
        [
            (85),
            (90),
            (95),
            (-85),
            (-90),
            (-95),
            (175),
            (180),
            (185),
            (-175),
            (-180),
            (-185),
        ]
    )
    def test_rotated_invalid(self, degree):
        import pickle

        original_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/rgb_stc.pickle", "rb").read()
        )
        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/stc_rgb_rotc.pickle", "rb").read()
        )
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return angle("blue square", degree)

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
        )
        expected_feedback = f"The feature blue square should be rotated by {degree} degrees, but is rotated by -135,45,-45,135 degrees."
        self.assertFalse(result)
        self.assertEqual([expected_feedback], feedback)

    @parameterized.expand(
        [
            (85),
            (90),
            (95),
            (-85),
            (-90),
            (-95),
            (175),
            (180),
            (185),
            (-175),
            (-180),
            (-185),
        ]
    )
    def test_rotated_negated_valid(self, degree):
        import pickle

        original_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/rgb_stc.pickle", "rb").read()
        )
        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/stc_rgb_rotc.pickle", "rb").read()
        )
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return ~angle("blue square", degree)

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
        )
        self.assertTrue(result)
        self.assertEqual([], feedback)

    @parameterized.expand(
        [
            (40),
            (45),
            (50),
            (-40),
            (-45),
            (-50),
            (130),
            (135),
            (140),
            (-130),
            (-135),
            (-140),
        ]
    )
    def test_rotated_negated_invalid(self, degree):
        import pickle

        original_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/rgb_stc.pickle", "rb").read()
        )
        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/stc_rgb_rotc.pickle", "rb").read()
        )
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return ~angle("blue square", degree)

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
        )
        expected_feedback = f"The feature blue square should not be rotated by {degree} degrees, and is rotated by -135,45,-45,135 degrees, which is too close/equal."
        self.assertFalse(result)
        self.assertEqual([expected_feedback], feedback)

    @parameterized.expand(["pale blue", "light blue", "very light blue"])
    def test_color_valid(self, color_expected):
        import pickle

        original_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/rgb_stc.pickle", "rb").read()
        )
        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/stc_rgb_rotc.pickle", "rb").read()
        )
        custom_image: Image.Image = Image.open("tests/resources/seg/stc_rgb_rotc.png")

        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):
            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return color("blue square", color_expected)

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, custom_image, get_features
        )
        self.assertTrue(result, feedback)
        self.assertEqual([], feedback)

    @parameterized.expand(["white", "green", "yellowish gray"])
    def test_color_invalid(self, color_expected):
        import pickle

        original_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/rgb_stc.pickle", "rb").read()
        )
        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/stc_rgb_rotc.pickle", "rb").read()
        )
        custom_image: Image.Image = Image.open("tests/resources/seg/stc_rgb_rotc.png")
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):
            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return color("blue square", color_expected)

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, custom_image, get_features
        )
        epected_feedback = f"The color of the feature blue square should have been {color_expected}, but is closer to light purple, very light purple, purple."

        self.assertFalse(result, feedback)
        self.assertEqual([epected_feedback], feedback)

    @parameterized.expand(["white", "green", "yellowish gray"])
    def test_color_negated_valid(self, color_expected):
        import pickle

        original_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/rgb_stc.pickle", "rb").read()
        )
        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/stc_rgb_rotc.pickle", "rb").read()
        )
        custom_image: Image.Image = Image.open("tests/resources/seg/stc_rgb_rotc.png")
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):
            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return ~color("blue square", color_expected)

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, custom_image, get_features
        )
        self.assertTrue(result, feedback)
        self.assertEqual([], feedback)

    @parameterized.expand(["pale blue", "light blue", "very light blue"])
    def test_color_negated_invalid(self, color_expected):
        import pickle

        original_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/rgb_stc.pickle", "rb").read()
        )
        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/stc_rgb_rotc.pickle", "rb").read()
        )
        custom_image: Image.Image = Image.open("tests/resources/seg/stc_rgb_rotc.png")

        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):
            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return ~color("blue square", color_expected)

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, custom_image, get_features
        )
        epected_feedback = f"The color of the feature blue square should not have been {color_expected}, but is still too close to {color_expected}."

        self.assertFalse(result, feedback)
        self.assertEqual([epected_feedback], feedback)
