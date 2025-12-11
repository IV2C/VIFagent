import pickle
import unittest

from vif.falcon.oracle.guided_oracle.expressions import (
    OracleExpression,
    angle,
    aligned,
    color,
    count,
    mirrored,
    placement,
    position,
    size,
    shape,
    within,
)
from vif.models.detection import BoundingBox, SegmentationMask
from PIL import Image
from parameterized import parameterized


class TestExpression(unittest.TestCase):
    """Note: Does not test expressions which are property(), hence no need for llm instantiation"""

    def __init__(self, methodName="runTest"):
        self.original_image: Image.Image = Image.new("1", (2, 2))
        self.original_image.putdata([0, 0, 0, 1])
        self.custom_image: Image.Image = Image.new("1", (2, 2))
        self.custom_image.putdata([0, 0, 1, 0])
        self.maxDiff = None
        super().__init__(methodName)

    @parameterized.expand(
        [
            ("right", (40, 40, 60, 60), (40, 30, 60, 50)),
            ("left", (40, 30, 60, 50), (40, 40, 60, 60)),
            ("over", (30, 40, 50, 60), (40, 40, 60, 60)),
            ("under", (40, 40, 60, 60), (30, 40, 50, 60)),
        ]
    )
    def test_placement_valid(self, direction, boxA, boxB):

        feat_dict = {
            hash((self.custom_image.tobytes(), "triangle")): [
                BoundingBox(*boxA, "triangle")
            ],
            hash((self.custom_image.tobytes(), "rectangle")): [
                BoundingBox(*boxB, "rectangle")
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return placement("triangle", "rectangle", direction)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            segment_function=get_features,
            box_function=get_features,
        )
        self.assertAlmostEqual(1.0, feedback.probability)
        expected = None

        self.assertEqual(expected, feedback.tojson(threshold=0.5))

    @parameterized.expand(
        [
            (
                "left",
                (40, 40, 60, 60),
                (40, 30, 60, 50),
                "The triangle is not on the left of the rectangle",
            ),
            (
                "right",
                (40, 30, 60, 50),
                (40, 40, 60, 60),
                "The triangle is not on the right of the rectangle",
            ),
            (
                "under",
                (30, 40, 50, 60),
                (40, 40, 60, 60),
                "The triangle is not under the rectangle",
            ),
            (
                "over",
                (40, 40, 60, 60),
                (30, 40, 50, 60),
                "The triangle is not above the rectangle",
            ),
        ]
    )
    def test_placement_invalid(self, direction, boxA, boxB, feedback_expected):
        feat_dict = {
            hash((self.custom_image.tobytes(), "triangle")): [
                BoundingBox(*boxA, "triangle")
            ],
            hash((self.custom_image.tobytes(), "rectangle")): [
                BoundingBox(*boxB, "rectangle")
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return placement("triangle", "rectangle", direction)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            box_function=get_features,
            segment_function=get_features,
        )
        self.assertAlmostEqual(feedback.probability, 0.0)

        expected = {
            "type": "FeedBackAndList",
            "probability": 0,
            "items": [
                {
                    "type": "FeedBack",
                    "name": "placement",
                    "score": 0,
                    "feedback": feedback_expected,
                    "probability": 0,
                    "negated": False,
                }
            ],
        }
        self.assertEqual(expected, feedback.tojson(0.5))

    @parameterized.expand(
        [
            (
                "left",
                (40, 40, 60, 60),
                (40, 30, 60, 50),
            ),
            (
                "right",
                (40, 30, 60, 50),
                (40, 40, 60, 60),
            ),
            (
                "under",
                (30, 40, 50, 60),
                (40, 40, 60, 60),
            ),
            (
                "over",
                (40, 40, 60, 60),
                (30, 40, 50, 60),
            ),
        ]
    )
    def test_placement_valid_negated(self, direction, boxA, boxB):
        feat_dict = {
            hash((self.custom_image.tobytes(), "triangle")): [
                BoundingBox(*boxA, "triangle")
            ],
            hash((self.custom_image.tobytes(), "rectangle")): [
                BoundingBox(*boxB, "rectangle")
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return ~placement("triangle", "rectangle", direction)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            box_function=get_features,
            segment_function=get_features,
        )
        self.assertAlmostEqual(1.0, feedback.probability, delta=0.01)
        expected = None

        self.assertEqual(expected, feedback.tojson(threshold=0.5))

    @parameterized.expand(
        [
            (
                "right",
                (40, 40, 60, 60),
                (40, 35, 60, 60),
                (40, 30, 60, 50),
                (40, 30, 60, 50),
            ),
            (
                "left",
                (40, 30, 60, 50),
                (40, 35, 60, 50),
                (40, 40, 60, 60),
                (40, 40, 60, 60),
            ),
            (
                "over",
                (30, 40, 50, 60),
                (35, 40, 50, 60),
                (40, 40, 60, 60),
                (40, 40, 60, 60),
            ),
            (
                "under",
                (40, 40, 60, 60),
                (35, 40, 60, 60),
                (30, 40, 50, 60),
                (30, 40, 50, 60),
            ),
        ]
    )
    def test_placement_valid_multiple(self, direction, boxTA, boxTB, boxRA, boxRB):

        feat_dict = {
            hash((self.custom_image.tobytes(), "triangles")): [
                BoundingBox(*boxTA, "triangleA"),
                BoundingBox(*boxTB, "triangleB"),
            ],
            hash((self.custom_image.tobytes(), "rectangles")): [
                BoundingBox(*boxRA, "rectangleA"),
                BoundingBox(*boxRB, "rectangleB"),
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return placement("triangles", "rectangles", direction)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            segment_function=get_features,
            box_function=get_features,
        )
        self.assertAlmostEqual(feedback.probability, 1.0)
        expected = None
        self.assertEqual(expected, feedback.tojson(0.5))

    @parameterized.expand(
        [
            (
                "right",
                (40, 40, 60, 50),
                (40, 65, 60, 80),
                (40, 45, 60, 55),
                (40, 60, 60, 65),
            )
        ]
    )
    def test_placement_invalid_multiple(self, direction, boxTA, boxTB, boxRA, boxRB):

        feat_dict = {
            hash((self.custom_image.tobytes(), "triangles")): [
                BoundingBox(*boxTA, "triangleA"),
                BoundingBox(*boxTB, "triangleB"),
            ],
            hash((self.custom_image.tobytes(), "rectangles")): [
                BoundingBox(*boxRA, "rectangleA"),
                BoundingBox(*boxRB, "rectangleB"),
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return placement("triangles", "rectangles", direction)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            segment_function=get_features,
            box_function=get_features,
        )
        self.assertAlmostEqual(feedback.probability, 0.0)

        expected = {
            "type": "FeedBackAndList",
            "probability": 0,
            "items": [
                {
                    "type": "FeedBack",
                    "feedback": "The triangleA is not on the right of the rectangleA",
                    "probability": 0,
                    "score": 0,
                    "name": "placement",
                    "negated": False,
                },
                {
                    "type": "FeedBack",
                    "feedback": "The triangleA is not on the right of the rectangleB",
                    "score": 0,
                    "probability": 0,
                    "name": "placement",
                    "negated": False,
                },
            ],
        }
        self.assertEqual(expected, feedback.tojson(0.5))

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

        feat_dict = {
            hash((self.original_image.tobytes(), "triangle")): [
                BoundingBox(*originalBoxA, "triangle")
            ],
            hash((self.original_image.tobytes(), "rectangle")): [
                BoundingBox(*originalBoxB, "rectangle")
            ],
            hash((self.custom_image.tobytes(), "triangle")): [
                BoundingBox(*customBoxA, "triangle")
            ],
            hash((self.custom_image.tobytes(), "rectangle")): [
                BoundingBox(*customBoxB, "rectangle")
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return position("triangle", "rectangle", 2, axis)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            segment_function=get_features,
            box_function=get_features,
        )

        self.assertAlmostEqual(feedback.probability, 1.0, delta=0.05)

        expected = None
        self.assertEqual(expected, feedback.tojson(0.5))

    @parameterized.expand(
        [
            (
                (30, 30, 50, 50),
                (40, 40, 60, 60),
                (30, 30, 50, 50),
                (50, 40, 70, 60),
            )
        ]
    )
    def test_position_valid_multiple(
        self, originalBoxA, originalBoxB, customBoxA, customBoxB
    ):

        feat_dict = {
            hash((self.original_image.tobytes(), "triangle")): [
                BoundingBox(*originalBoxA, "triangle1"),
                BoundingBox(*originalBoxA, "triangle2"),
            ],
            hash((self.original_image.tobytes(), "rectangle")): [
                BoundingBox(*originalBoxB, "rectangle")
            ],
            hash((self.custom_image.tobytes(), "triangle")): [
                BoundingBox(*customBoxA, "triangle1"),
                BoundingBox(*customBoxA, "triangle2"),
            ],
            hash((self.custom_image.tobytes(), "rectangle")): [
                BoundingBox(*customBoxB, "rectangle")
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return position("triangle", "rectangle", 2, "vertical")

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            segment_function=get_features,
            box_function=get_features,
        )
        self.assertAlmostEqual(feedback.probability, 1.0, delta=0.1)

        expected = None
        self.assertEqual(expected, feedback.tojson(0.5))

    @parameterized.expand(
        [
            (
                (30, 30, 50, 50),
                (40, 40, 60, 60),
                (31, 30, 50, 50),
                (40, 30, 46, 50),
                (50, 40, 70, 60),
            )
        ]
    )
    def test_position_invalid_multiple(
        self, originalBoxA, originalBoxB, customBoxA, customBoxA2, customBoxB
    ):

        feat_dict = {
            hash((self.original_image.tobytes(), "triangle")): [
                BoundingBox(*originalBoxA, "triangle1"),
                BoundingBox(*originalBoxA, "triangle2"),
            ],
            hash((self.original_image.tobytes(), "rectangle")): [
                BoundingBox(*originalBoxB, "rectangle")
            ],
            hash((self.custom_image.tobytes(), "triangle")): [
                BoundingBox(*customBoxA, "triangle1"),
                BoundingBox(*customBoxA2, "triangle2"),
            ],
            hash((self.custom_image.tobytes(), "rectangle")): [
                BoundingBox(*customBoxB, "rectangle")
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return position("triangle", "rectangle", 2, "vertical")

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            segment_function=get_features,
            box_function=get_features,
        )
        self.assertAlmostEqual(feedback.probability, 0.165, delta=0.05)
        expected = {
            "type": "FeedBackAndList",
            "probability": 0.17,
            "items": [
                {
                    "type": "FeedBack",
                    "feedback": "The vertical distance between triangle1 and rectangle was supposed to be around 20.0, but was 19.5.",
                    "probability": 0.90,
                    "score": 0.025,
                    "name": "position",
                    "negated": False,
                },
                {
                    "type": "FeedBack",
                    "feedback": "The vertical distance between triangle2 and rectangle was supposed to be around 20.0, but was 17.0.",
                    "probability": 0.18,
                    "score": 0.15,
                    "name": "position",
                    "negated": False,
                },
            ],
        }
        self.assertEqual(expected, feedback.tojson(0.9))

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
        feat_dict = {
            hash((self.original_image.tobytes(), "triangle")): [
                BoundingBox(*originalBoxA, "triangle")
            ],
            hash((self.original_image.tobytes(), "rectangle")): [
                BoundingBox(*originalBoxB, "rectangle")
            ],
            hash((self.custom_image.tobytes(), "triangle")): [
                BoundingBox(*customBoxA, "triangle")
            ],
            hash((self.custom_image.tobytes(), "rectangle")): [
                BoundingBox(*customBoxB, "rectangle")
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return ~position("triangle", "rectangle", 2, axis)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            box_function=get_features,
            segment_function=get_features,
        )
        self.assertAlmostEqual(feedback.probability, 1.0, delta=0.5)
        expected = None
        self.assertEqual(expected, feedback.tojson(0.9))

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
        feat_dict = {
            hash((self.original_image.tobytes(), "triangle")): [
                BoundingBox(*originalBoxA, "triangle")
            ],
            hash((self.original_image.tobytes(), "rectangle")): [
                BoundingBox(*originalBoxB, "rectangle")
            ],
            hash((self.custom_image.tobytes(), "triangle")): [
                BoundingBox(*customBoxA, "triangle")
            ],
            hash((self.custom_image.tobytes(), "rectangle")): [
                BoundingBox(*customBoxB, "rectangle")
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return ~position("triangle", "rectangle", 2, axis)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            box_function=get_features,
            segment_function=get_features,
        )
        self.assertAlmostEqual(feedback.probability, 0.0, delta=0.05)
        expected = {
            "type": "FeedBackOrList",
            "probability": 0.05,
            "items": [
                {
                    "type": "FeedBack",
                    "feedback": feedback_expected,
                    "score": 0,
                    "probability": 0.05,
                    "name": "position",
                    "negated": True,
                }
            ],
        }
        self.assertEqual(expected, feedback.tojson(0.9))

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
        )  # works because the pickle file only contains the seg for blue square
        feat_dict = {
            hash((self.original_image.tobytes(), "blue square")): original_features,
            hash((self.custom_image.tobytes(), "blue square")): custom_features,
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return angle("blue square", degree)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            box_function=get_features,
            segment_function=get_features,
        )

        self.assertAlmostEqual(feedback.probability, 1.0)
        expected = None
        self.assertEqual(expected, feedback.tojson(0.9))

    @parameterized.expand(
        [
            (85, 0.0, 35),
            (90, 0.0, 40),
            (95, 0.0, 35),
            (-85, 0.0, 35),
            (-90, 0.0, 40),
            (-95, 0.0, 35),
            (175, 0.0, 35),
            (180, 0.0, 40),
            (185, 0.0, 45),
            (-175, 0.0, 35),
            (-180, 0.0, 40),
            (-185, 0.0, 45),
            (55, 0.5, 5),
        ]
    )
    def test_rotated_invalid(self, degree, expected_score, score):
        import pickle

        original_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/rgb_stc.pickle", "rb").read()
        )
        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/stc_rgb_rotc.pickle", "rb").read()
        )
        feat_dict = {
            hash((self.original_image.tobytes(), "blue square")): original_features,
            hash((self.custom_image.tobytes(), "blue square")): custom_features,
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return angle("blue square", degree)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        expected_feedback = f"The blue square should be rotated by {degree} degrees, but is rotated by -135,45,-45,135 degrees."

        self.assertAlmostEqual(feedback.probability, expected_score, delta=0.05)
        expected = {
            "type": "FeedBackAndList",
            "probability": expected_score,
            "items": [
                {
                    "type": "FeedBack",
                    "feedback": expected_feedback,
                    "probability": expected_score,
                    "score": score,
                    "name": "angle",
                    "negated": False,
                }
            ],
        }
        self.assertEqual(expected, feedback.tojson(0.9))

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
            hash((self.original_image.tobytes(), "blue square")): original_features,
            hash((self.custom_image.tobytes(), "blue square")): custom_features,
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return ~angle("blue square", degree)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            box_function=get_features,
            original_image=self.original_image,
            custom_image=self.custom_image,
            segment_function=get_features,
        )

        self.assertAlmostEqual(feedback.probability, 1.0)
        expected = None
        self.assertEqual(expected, feedback.tojson(0.9))

    @parameterized.expand(
        [
            (40, 0.0, 0.0),
            (45, 0.0, 0.0),
            (50, 0.0, 0.0),
            (-40, 0.0, 0.0),
            (-45, 0.0, 0.0),
            (-50, 0.0, 0.0),
            (130, 0.0, 0.0),
            (135, 0.0, 0.0),
            (140, 0.0, 0.0),
            (-130, 0.0, 0.0),
            (-135, 0.0, 0.0),
            (-140, 0.0, 0.0),
            (55, 0.5, 5),
        ]
    )
    def test_rotated_negated_invalid(self, degree, expected_score, score):
        import pickle

        original_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/rgb_stc.pickle", "rb").read()
        )
        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/stc_rgb_rotc.pickle", "rb").read()
        )
        feat_dict = {
            hash((self.original_image.tobytes(), "blue square")): original_features,
            hash((self.custom_image.tobytes(), "blue square")): custom_features,
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return ~angle("blue square", degree)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            box_function=get_features,
            original_image=self.original_image,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        expected_feedback = f"The blue square should not be rotated by {degree} degrees, and is rotated by -135,45,-45,135 degrees, which is too close/equal."
        self.assertAlmostEqual(feedback.probability, expected_score)
        expected = {
            "type": "FeedBackOrList",
            "probability": expected_score,
            "items": [
                {
                    "type": "FeedBack",
                    "feedback": expected_feedback,
                    "probability": expected_score,
                    "score": score,
                    "name": "angle",
                    "negated": True,
                }
            ],
        }
        self.assertEqual(expected, feedback.tojson(0.9))

    @parameterized.expand(["pale blue", "light blue", "blue"])
    def test_color_valid(self, color_expected):
        import pickle

        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/stc_rgb_rotc.pickle", "rb").read()
        )
        custom_image: Image.Image = Image.open("tests/resources/seg/stc_rgb_rotc.png")
        feat_dict = {
            hash((custom_image.tobytes(), "blue square")): custom_features,
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return color("blue square", color_expected)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=custom_image,
            segment_function=get_features,
        )
        self.assertAlmostEqual(feedback.probability, 0.95, delta=0.06)
        expected = None
        self.assertEqual(expected, feedback.tojson(0.9))

    @parameterized.expand(
        [("white", 0.0, 10), ("green", 0.0, 30), ("yellowish gray", 0.0, 16)]
    )
    def test_color_invalid(self, color_expected, expected_score, score):
        import pickle

        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/stc_rgb_rotc.pickle", "rb").read()
        )
        custom_image: Image.Image = Image.open("tests/resources/seg/stc_rgb_rotc.png")
        feat_dict = {
            hash((custom_image.tobytes(), "blue square")): custom_features,
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return color("blue square", color_expected)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            box_function=get_features,
            original_image=self.original_image,
            custom_image=custom_image,
            segment_function=get_features,
        )
        expected_feedback = f"The color of the blue square should have been {color_expected}, but is closer to very light purple, light purple, very light blue."
        self.assertAlmostEqual(feedback.probability, expected_score)
        expected = {
            "type": "FeedBackAndList",
            "probability": expected_score,
            "items": [
                {
                    "type": "FeedBack",
                    "feedback": expected_feedback,
                    "probability": expected_score,
                    "score": score,
                    "name": "color",
                    "negated": False,
                }
            ],
        }
        self.assertEqual(expected, feedback.tojson(0.9))

    @parameterized.expand(["white", "green", "yellowish gray"])
    def test_color_negated_valid(self, color_expected):
        import pickle

        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/stc_rgb_rotc.pickle", "rb").read()
        )
        custom_image: Image.Image = Image.open("tests/resources/seg/stc_rgb_rotc.png")
        feat_dict = {
            hash((custom_image.tobytes(), "blue square")): custom_features,
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return ~color("blue square", color_expected)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=custom_image,
            segment_function=get_features,
        )

        self.assertAlmostEqual(feedback.probability, 0.95, delta=0.06)
        expected = None
        self.assertEqual(expected, feedback.tojson(0.9))

    @parameterized.expand([("pale blue", 2), ("light blue", 3), ("very light blue", 2)])
    def test_color_negated_invalid(self, color_expected, score):
        import pickle

        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/stc_rgb_rotc.pickle", "rb").read()
        )
        custom_image: Image.Image = Image.open("tests/resources/seg/stc_rgb_rotc.png")

        feat_dict = {
            hash((custom_image.tobytes(), "blue square")): custom_features,
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return ~color("blue square", color_expected)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=custom_image,
            segment_function=get_features,
        )
        expected_feedback = f"The color of the blue square should not have been {color_expected}, but is still too close to {color_expected}."

        self.assertAlmostEqual(feedback.probability, 0.0, delta=0.05)
        expected = {
            "type": "FeedBackOrList",
            "probability": 0.0,
            "items": [
                {
                    "type": "FeedBack",
                    "feedback": expected_feedback,
                    "probability": 0.0,
                    "score": score,
                    "name": "color",
                    "negated": True,
                }
            ],
        }
        self.assertEqual(expected, feedback.tojson(0.9))

    @parameterized.expand(
        [
            ((1, 1.5), (50, 50, 60, 60), (50, 50, 65, 60)),
            ((2, 2), (50, 50, 60, 60), (45, 50, 65, 70)),
        ]
    )
    def test_resize_valid(self, ratio, box_ori, box_cust):
        original_features: list[BoundingBox] = [
            BoundingBox(*box_ori, "triangle"),
        ]
        custom_features: list[BoundingBox] = [
            BoundingBox(*box_cust, "triangle"),
        ]
        feat_dict = {
            hash((self.original_image.tobytes(), "triangle")): original_features,
            hash((self.custom_image.tobytes(), "triangle")): custom_features,
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return size("triangle", ratio)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=self.custom_image,
            segment_function=get_features,
        )

        self.assertAlmostEqual(feedback.probability, 1.0)
        expected = None
        self.assertEqual(expected, feedback.tojson(0.1))

    @parameterized.expand(
        [
            (
                (1, 1.5),
                (50, 50, 60, 60),
                (40, 40, 60, 60),
                (50, 50, 65, 60),
                (40, 40, 70, 60),
            ),
            (
                (2, 2),
                (50, 50, 60, 60),
                (10, 50, 60, 70),
                (45, 50, 65, 70),
                (20, 60, 120, 100),
            ),
        ]
    )
    def test_resize_valid_multiple(
        self, ratio, box_oriA, box_oriB, box_custA, box_custB
    ):
        original_features: list[BoundingBox] = [
            BoundingBox(*box_oriA, "triangleA"),
            BoundingBox(*box_oriB, "triangleB"),
        ]
        custom_features: list[BoundingBox] = [
            BoundingBox(*box_custA, "triangleA"),
            BoundingBox(*box_custB, "triangleB"),
        ]
        feat_dict = {
            hash((self.original_image.tobytes(), "triangle")): original_features,
            hash((self.custom_image.tobytes(), "triangle")): custom_features,
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return size("triangle", ratio)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        self.assertAlmostEqual(feedback.probability, 1.0)
        expected = None
        self.assertEqual(expected, feedback.tojson(0.1))

    @parameterized.expand(
        [
            (
                (1, 1.5),
                (50, 50, 60, 60),
                (40, 40, 60, 60),
                (50, 50, 65, 61),
                (40, 40, 72, 60),
            )
        ]
    )
    def test_resize_invalid_multiple(
        self, ratio, box_oriA, box_oriB, box_custA, box_custB
    ):
        original_features: list[BoundingBox] = [
            BoundingBox(*box_oriA, "triangleA"),
            BoundingBox(*box_oriB, "triangleB"),
        ]
        custom_features: list[BoundingBox] = [
            BoundingBox(*box_custA, "triangleA"),
            BoundingBox(*box_custB, "triangleB"),
        ]
        feat_dict = {
            hash((self.original_image.tobytes(), "triangle")): original_features,
            hash((self.custom_image.tobytes(), "triangle")): custom_features,
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return size("triangle", ratio)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=self.custom_image,
            segment_function=get_features,
        )

        self.assertAlmostEqual(feedback.probability, 1 / 3)
        expected = {
            "type": "FeedBackAndList",
            "probability": 0.33,
            "items": [
                {
                    "type": "FeedBackAnd",
                    "probability": 0.67,
                    "feedbackA": None,
                    "feedbackB": {
                        "type": "FeedBack",
                        "feedback": "The triangleB was resized on y by a ratio of 1.6, but should have been by a ratio of 1.5",
                        "probability": 0.67,
                    },
                },
                {
                    "type": "FeedBackAnd",
                    "probability": 0.5,
                    "feedbackA": {
                        "type": "FeedBack",
                        "feedback": "The triangleA was resized on x by a ratio of 1.1, but should have been by a ratio of 1",
                        "probability": 0.5,
                    },
                    "feedbackB": None,
                },
            ],
        }
        self.assertEqual(expected, feedback.tojson(0.9))

    @parameterized.expand(
        [
            (
                (1, 2),
                (50, 50, 60, 60),
                (50, 50, 65, 60),
                [
                    "The triangle was resized on y by a ratio of 1.5, but should have been by a ratio of 2"
                ],
                0.25,
                0.0,
                0.12,
                0.95,
                0.11
            ),
            (
                (3, 3),
                (50, 50, 60, 60),
                (45, 50, 65, 70),
                [
                    "The triangle was resized on y by a ratio of 2.0, but should have been by a ratio of 3",
                    "The triangle was resized on x by a ratio of 2.0, but should have been by a ratio of 3",
                ],
                1/3,
                1/3, 0.02,
                0.02,
                0.0
            ),
        ]
    )
    def test_resize_invalid(self, ratio, box_ori, box_cust, expected_feedback, score0,score1,prob0,prob1,prob):
        original_features: list[BoundingBox] = [
            BoundingBox(*box_ori, "triangle"),
        ]
        custom_features: list[BoundingBox] = [
            BoundingBox(*box_cust, "triangle"),
        ]
        feat_dict = {
            hash((self.original_image.tobytes(), "triangle")): original_features,
            hash((self.custom_image.tobytes(), "triangle")): custom_features,
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return size("triangle", ratio)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=self.custom_image,
            segment_function=get_features,
        )

        self.assertAlmostEqual(feedback.probability, prob,delta=0.05)
        expected = {
            "type": "FeedBackAndList",
            "probability": prob,
            "items": [
                {
                    "type": "FeedBackAnd",
                    "probability": prob,
                    "feedbackA": (
                        (
                            {
                                "type": "FeedBack",
                                "feedback": expected_feedback[1],
                                "probability": prob1,
                                "score": score1,
                                "name": "size",
                                "negated": False,
                            }
                        )
                        if len(expected_feedback) > 1
                        else None
                    ),
                    "feedbackB": {
                        "type": "FeedBack",
                        "feedback": expected_feedback[0],
                        "probability": prob0,
                        "score": score0,
                        "name": "size",
                        "negated": False,
                    },
                }
            ],
        }
        self.assertEqual(expected, feedback.tojson(0.9))

    @parameterized.expand(
        [
            (
                (1, 3),
                (50, 50, 60, 60),
                (50, 50, 65, 60),
            ),
            (
                (4, 2),
                (50, 50, 60, 60),
                (45, 50, 65, 70),
            ),
        ]
    )
    def test_resize_negated_valid(self, ratio, box_ori, box_cust):
        original_features: list[BoundingBox] = [
            BoundingBox(*box_ori, "triangle"),
        ]
        custom_features: list[BoundingBox] = [
            BoundingBox(*box_cust, "triangle"),
        ]
        feat_dict = {
            hash((self.original_image.tobytes(), "triangle")): original_features,
            hash((self.custom_image.tobytes(), "triangle")): custom_features,
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return ~size("triangle", ratio)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            box_function=get_features,
            original_image=self.original_image,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        expected = None
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 1.0)

    @parameterized.expand(
        [
            (
                (0.9, 1.6),
                (50, 50, 60, 60),
                (50, 50, 65, 60),
                [
                    "The triangle was resized on x by a ratio of 1.0, which is too close to 0.9",
                    "The triangle was resized on y by a ratio of 1.5, which is too close to 1.6",
                ],
                (0.56, 0.31, 0.69),
            ),
            (
                (2.1, 2.1),
                (50, 50, 60, 60),
                (45, 50, 65, 70),
                [
                    "The triangle was resized on x by a ratio of 2.0, which is too close to 2.1",
                    "The triangle was resized on y by a ratio of 2.0, which is too close to 2.1",
                ],
                (0.24, 0.24, 0.42),
            ),
        ]
    )
    def test_resize_negated_invalid(
        self, ratio, box_ori, box_cust, expected_feedback, scores
    ):
        original_features: list[BoundingBox] = [
            BoundingBox(*box_ori, "triangle"),
        ]
        custom_features: list[BoundingBox] = [
            BoundingBox(*box_cust, "triangle"),
        ]
        feat_dict = {
            hash((self.original_image.tobytes(), "triangle")): original_features,
            hash((self.custom_image.tobytes(), "triangle")): custom_features,
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return ~size("triangle", ratio)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            box_function=get_features,
            segment_function=get_features,
        )
        expected = {
            "type": "FeedBackOrList",
            "probability": scores[2],
            "items": [
                {
                    "type": "FeedBackOr",
                    "probability": scores[2],
                    "feedbackA": {
                        "type": "FeedBack",
                        "feedback": expected_feedback[0],
                        "probability": scores[0],
                    },
                    "feedbackB": {
                        "type": "FeedBack",
                        "feedback": expected_feedback[1],
                        "probability": scores[1],
                    },
                }
            ],
        }
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, scores[2], delta=0.05)

    @parameterized.expand(
        [
            ("triangle", "triangle"),
            ("square", "square"),
            ("circle", "circle"),
        ]
    )
    def test_shape_valid(self, feature, shape_expected):
        load_mask = lambda shape_w: pickle.loads(
            open(f"tests/resources/seg/simple_masks/{shape_w}.pickle", "rb").read()
        )
        custom_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, load_mask(feature), feature),
        ]
        feat_dict = {
            hash((self.custom_image.tobytes(), feature)): custom_features,
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return shape(feature, shape_expected)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        expected = None
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 1.0)

    @parameterized.expand(
        [
            ("triangle", "triangle"),
        ]
    )
    def test_shape_valid_multiple(self, feature, shape_expected):
        load_mask = lambda shape_w: pickle.loads(
            open(f"tests/resources/seg/simple_masks/{shape_w}.pickle", "rb").read()
        )

        feat_dict = {
            hash((self.custom_image.tobytes(), "triangle")): [
                SegmentationMask(0, 0, 0, 0, load_mask("triangle"), "triangleA"),
                SegmentationMask(0, 0, 0, 0, load_mask("triangle"), "triangleB"),
            ]
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return shape(feature, shape_expected)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        expected = None
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 1.0)

    @parameterized.expand(
        [
            ("triangle", "triangle"),
        ]
    )
    def test_shape_invalid_multiple(self, feature, shape_expected):
        load_mask = lambda shape_w: pickle.loads(
            open(f"tests/resources/seg/simple_masks/{shape_w}.pickle", "rb").read()
        )

        feat_dict = {
            hash((self.custom_image.tobytes(), "triangle")): [
                SegmentationMask(0, 0, 0, 0, load_mask("triangle"), "triangleA"),
                SegmentationMask(0, 0, 0, 0, load_mask("square"), "triangleB"),
            ]
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return shape(feature, shape_expected)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=self.custom_image,
            segment_function=get_features,
        )

        expected = {
            "type": "FeedBackAndList",
            "probability": 0.0,
            "items": [
                {
                    "type": "FeedBack",
                    "feedback": "The triangleB should be in the shape of a triangle, but looks more like a rectangle.",
                    "probability": 0.0,
                }
            ],
        }
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 0.0)

    @parameterized.expand(
        [
            ("triangle", "square", "triangle"),
            ("square", "circle", "rectangle"),
            ("circle", "triangle", "circle"),
        ]
    )
    def test_shape_invalid(self, feature, shape_expected, shapes_found):
        load_mask = lambda shape_w: pickle.loads(
            open(f"tests/resources/seg/simple_masks/{shape_w}.pickle", "rb").read()
        )

        feat_dict = {
            hash((self.custom_image.tobytes(), "triangle")): [
                SegmentationMask(0, 0, 0, 0, load_mask("triangle"), "triangle"),
            ],
            hash((self.custom_image.tobytes(), "square")): [
                SegmentationMask(0, 0, 0, 0, load_mask("square"), "square")
            ],
            hash((self.custom_image.tobytes(), "circle")): [
                SegmentationMask(0, 0, 0, 0, load_mask("circle"), "circle")
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return shape(feature, shape_expected)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        expected = {
            "type": "FeedBackAndList",
            "probability": 0.0,
            "items": [
                {
                    "type": "FeedBack",
                    "feedback": f"The {feature} should be in the shape of a {shape_expected}, but looks more like a {shapes_found}.",
                    "probability": 0.0,
                }
            ],
        }
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 0.0)

    @parameterized.expand(
        [
            ("triangle", "square"),
            ("square", "circle"),
            ("circle", "triangle"),
        ]
    )
    def test_shape_negated_valid(self, feature, shape_expected):
        load_mask = lambda shape_w: pickle.loads(
            open(f"tests/resources/seg/simple_masks/{shape_w}.pickle", "rb").read()
        )
        feat_dict = {
            hash((self.custom_image.tobytes(), "triangle")): [
                SegmentationMask(0, 0, 0, 0, load_mask("triangle"), "triangle"),
            ],
            hash((self.custom_image.tobytes(), "square")): [
                SegmentationMask(0, 0, 0, 0, load_mask("square"), "square")
            ],
            hash((self.custom_image.tobytes(), "circle")): [
                SegmentationMask(0, 0, 0, 0, load_mask("circle"), "circle")
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return ~shape(feature, shape_expected)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            segment_function=get_features,
            box_function=get_features,
        )
        expected = None
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 1.0)

    @parameterized.expand(
        [
            ("triangle", "triangle"),
            ("square", "square"),
            ("circle", "circle"),
        ]
    )
    def test_shape_negated_invalid(self, feature, shape_expected):
        load_mask = lambda shape_w: pickle.loads(
            open(f"tests/resources/seg/simple_masks/{shape_w}.pickle", "rb").read()
        )
        feat_dict = {
            hash((self.custom_image.tobytes(), "triangle")): [
                SegmentationMask(0, 0, 0, 0, load_mask("triangle"), "triangle"),
            ],
            hash((self.custom_image.tobytes(), "square")): [
                SegmentationMask(0, 0, 0, 0, load_mask("square"), "square")
            ],
            hash((self.custom_image.tobytes(), "circle")): [
                SegmentationMask(0, 0, 0, 0, load_mask("circle"), "circle")
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return ~shape(feature, shape_expected)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            box_function=get_features,
            segment_function=get_features,
        )

        expected = {
            "type": "FeedBackOrList",
            "probability": 0.0,
            "items": [
                {
                    "type": "FeedBack",
                    "feedback": f"The {feature} should not be in the shape of a {shape_expected}, but still looks like a {shape_expected}.",
                    "probability": 0.0,
                }
            ],
        }
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 0.0)

    # WITHIN

    def test_within_valid(self):
        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/dog_eyes_face.pickle", "rb").read()
        )
        feat_dict = {
            hash((self.custom_image.tobytes(), "left eye")): [
                [f for f in custom_features if f.label == "left eye"][0]
            ],
            hash((self.custom_image.tobytes(), "dog's face")): [
                [f for f in custom_features if f.label == "dog's face"][0]
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return within("left eye", "dog's face")

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        expected = None
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 1.0)

    def test_within_valid_multiple(self):
        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/dog_eyes_face.pickle", "rb").read()
        )
        feat_dict = {
            hash((self.custom_image.tobytes(), "eyes")): [
                [f for f in custom_features if f.label == "left eye"][0],
                [f for f in custom_features if f.label == "right eye"][0],
            ],
            hash((self.custom_image.tobytes(), "dog's face")): [
                [f for f in custom_features if f.label == "dog's face"][0]
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return within("eyes", "dog's face")

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        expected = None
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 1.0)

    def test_within_invalid_multiple(self):
        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/dog_eyes_face.pickle", "rb").read()
        )
        feat_dict = {
            hash((self.custom_image.tobytes(), "eyes")): [
                [f for f in custom_features if f.label == "left eye"][0],
                [f for f in custom_features if f.label == "right eye"][0],
            ],
            hash((self.custom_image.tobytes(), "right eye")): [
                [f for f in custom_features if f.label == "right eye"][0]
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return within("eyes", "right eye")

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=self.custom_image,
            segment_function=get_features,
        )

        expected = {
            "type": "FeedBackAndList",
            "probability": 0.0,
            "items": [
                {
                    "type": "FeedBack",
                    "feedback": "The left eye should be contained in the feature right eye, but isn't.",
                    "probability": 0.0,
                }
            ],
        }
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 0.0)

    def test_within_invalid(self):
        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/dog_eyes_face.pickle", "rb").read()
        )
        feat_dict = {
            hash((self.custom_image.tobytes(), "left eye")): [
                [f for f in custom_features if f.label == "left eye"][0]
            ],
            hash((self.custom_image.tobytes(), "right eye")): [
                [f for f in custom_features if f.label == "right eye"][0]
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return within("left eye", "right eye")

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            box_function=get_features,
            segment_function=get_features,
        )
        expected = {
            "type": "FeedBackAndList",
            "probability": 0.0,
            "items": [
                {
                    "type": "FeedBack",
                    "feedback": "The left eye should be contained in the feature right eye, but isn't.",
                    "probability": 0.0,
                }
            ],
        }
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 0.0)

    def test_within_negated_valid(self):
        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/dog_eyes_face.pickle", "rb").read()
        )
        feat_dict = {
            hash((self.custom_image.tobytes(), "right eye")): [
                [f for f in custom_features if f.label == "left eye"][0]
            ],
            hash((self.custom_image.tobytes(), "dog's face")): [
                [f for f in custom_features if f.label == "dog's face"][0]
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return ~within("dog's face", "right eye")

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        expected = None
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 1.0, delta=0.05)

    def test_within_negated_invalid(self):
        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/dog_eyes_face.pickle", "rb").read()
        )
        feat_dict = {
            hash((self.custom_image.tobytes(), "left eye")): [
                [f for f in custom_features if f.label == "left eye"][0]
            ],
            hash((self.custom_image.tobytes(), "dog's face")): [
                [f for f in custom_features if f.label == "dog's face"][0]
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return ~within("left eye", "dog's face")

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        expected = {
            "type": "FeedBackOrList",
            "probability": 0.0,
            "items": [
                {
                    "type": "FeedBack",
                    "feedback": f"The left eye should not be contained in the feature dog's face, but is actually within it.",
                    "probability": 0.0,
                }
            ],
        }
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 0.0, delta=0.05)

    # Mirrored

    def test_mirrored_vertical_valid(self):
        original_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/dog_mod_normal.pickle", "rb").read()
        )
        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/dog_mod_vertical.pickle", "rb").read()
        )
        original_image = Image.open("tests/resources/seg/dog_mod_normal.png")
        custom_image = Image.open("tests/resources/seg/dog_mod_vertical.png")
        feat_dict = {
            hash((original_image.tobytes(), "dog's face")): [
                [f for f in original_features if f.label == "dog's face"][0]
            ],
            hash((custom_image.tobytes(), "dog's face")): [
                [f for f in custom_features if f.label == "dog's face"][0]
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return mirrored("dog's face", "vertical")

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=original_image,
            box_function=get_features,
            custom_image=custom_image,
            segment_function=get_features,
        )
        expected = None
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 1.0, delta=0.05)

    def test_mirrored_vertical_invalid(self):
        original_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/dog_mod_normal.pickle", "rb").read()
        )
        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/dog_mod_horizontal.pickle", "rb").read()
        )
        original_image = Image.open("tests/resources/seg/dog_mod_normal.png")
        custom_image = Image.open("tests/resources/seg/dog_mod_horizontal.png")
        feat_dict = {
            hash((original_image.tobytes(), "dog's face")): [
                [f for f in original_features if f.label == "dog's face"][0]
            ],
            hash((custom_image.tobytes(), "dog's face")): [
                [f for f in custom_features if f.label == "dog's face"][0]
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return mirrored("dog's face", "vertical")

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=original_image,
            box_function=get_features,
            custom_image=custom_image,
            segment_function=get_features,
        )
        expected = {
            "type": "FeedBackAndList",
            "probability": 0.67,
            "items": [
                {
                    "type": "FeedBack",
                    "feedback": f"The dog's face should be mirrored along the vertical axis.",
                    "probability": 0.67,
                }
            ],
        }
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 0.67, delta=0.05)

    def test_mirrored_horizontal_valid(self):
        original_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/dog_mod_normal.pickle", "rb").read()
        )
        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/dog_mod_horizontal.pickle", "rb").read()
        )
        original_image = Image.open("tests/resources/seg/dog_mod_normal.png")
        custom_image = Image.open("tests/resources/seg/dog_mod_horizontal.png")
        feat_dict = {
            hash((original_image.tobytes(), "dog's face")): [
                [f for f in original_features if f.label == "dog's face"][0]
            ],
            hash((custom_image.tobytes(), "dog's face")): [
                [f for f in custom_features if f.label == "dog's face"][0]
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return mirrored("dog's face", "horizontal")

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=original_image,
            box_function=get_features,
            custom_image=custom_image,
            segment_function=get_features,
        )
        expected = None
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 1.0, delta=0.05)

    def test_mirrored_horizontal_invalid(self):
        original_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/dog_mod_normal.pickle", "rb").read()
        )
        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/dog_mod_vertical.pickle", "rb").read()
        )
        original_image = Image.open("tests/resources/seg/dog_mod_normal.png")
        custom_image = Image.open("tests/resources/seg/dog_mod_vertical.png")
        feat_dict = {
            hash((original_image.tobytes(), "dog's face")): [
                [f for f in original_features if f.label == "dog's face"][0]
            ],
            hash((custom_image.tobytes(), "dog's face")): [
                [f for f in custom_features if f.label == "dog's face"][0]
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return mirrored("dog's face", "horizontal")

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=original_image,
            box_function=get_features,
            custom_image=custom_image,
            segment_function=get_features,
        )

        expected = {
            "type": "FeedBackAndList",
            "probability": 0.67,
            "items": [
                {
                    "type": "FeedBack",
                    "feedback": "The dog's face should be mirrored along the horizontal axis.",
                    "probability": 0.67,
                }
            ],
        }
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 0.67, delta=0.05)

    def test_mirrored_vertical_negative_invalid(self):
        original_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/dog_mod_normal.pickle", "rb").read()
        )
        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/dog_mod_vertical.pickle", "rb").read()
        )
        original_image = Image.open("tests/resources/seg/dog_mod_normal.png")
        custom_image = Image.open("tests/resources/seg/dog_mod_vertical.png")
        feat_dict = {
            hash((original_image.tobytes(), "dog's face")): [
                [f for f in original_features if f.label == "dog's face"][0]
            ],
            hash((custom_image.tobytes(), "dog's face")): [
                [f for f in custom_features if f.label == "dog's face"][0]
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return ~mirrored("dog's face", "vertical")

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=original_image,
            box_function=get_features,
            custom_image=custom_image,
            segment_function=get_features,
        )
        expected = {
            "type": "FeedBackOrList",
            "probability": 0.0,
            "items": [
                {
                    "type": "FeedBack",
                    "feedback": "The dog's face should not be mirrored along the vertical axis.",
                    "probability": 0.0,
                }
            ],
        }
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 0.0, delta=0.05)

    def test_mirrored_horizontal_negative_invalid(self):
        original_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/dog_mod_normal.pickle", "rb").read()
        )
        custom_features: list[SegmentationMask] = pickle.loads(
            open("tests/resources/seg/dog_mod_horizontal.pickle", "rb").read()
        )
        original_image = Image.open("tests/resources/seg/dog_mod_normal.png")
        custom_image = Image.open("tests/resources/seg/dog_mod_horizontal.png")
        feat_dict = {
            hash((original_image.tobytes(), "dog's face")): [
                [f for f in original_features if f.label == "dog's face"][0]
            ],
            hash((custom_image.tobytes(), "dog's face")): [
                [f for f in custom_features if f.label == "dog's face"][0]
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return ~mirrored("dog's face", "horizontal")

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            box_function=get_features,
            original_image=original_image,
            custom_image=custom_image,
            segment_function=get_features,
        )
        expected = {
            "type": "FeedBackOrList",
            "probability": 0.0,
            "items": [
                {
                    "type": "FeedBack",
                    "feedback": "The dog's face should not be mirrored along the horizontal axis.",
                    "probability": 0.0,
                }
            ],
        }
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 0.0, delta=0.05)

    @parameterized.expand(
        [
            (
                (40, 40, 60, 60),
                (70, 45, 80, 55),
                "vertical",
            ),
            (
                (40, 40, 60, 60),
                (44, 70, 54, 80),
                "horizontal",
            ),
        ]
    )
    def test_aligned_valid(self, feature_a_box, feature_b_box, axis):

        feat_dict = {
            hash((self.custom_image.tobytes(), "circle")): [
                BoundingBox(*feature_a_box, "circle"),
            ],
            hash((self.custom_image.tobytes(), "square")): [
                BoundingBox(*feature_a_box, "square"),
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return aligned("square", "circle", axis)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        expected = None
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 1.0)

    @parameterized.expand(
        [
            (
                (70, 45, 80, 55),
                (40, 40, 60, 60),
                "horizontal",
            ),
            (
                (44, 70, 54, 80),
                (40, 40, 60, 60),
                "vertical",
            ),
        ]
    )
    def test_aligned_invalid(self, feature_a_box, feature_b_box, axis):
        feat_dict = {
            hash((self.custom_image.tobytes(), "circle")): [
                BoundingBox(*feature_a_box, "circle"),
            ],
            hash((self.custom_image.tobytes(), "square")): [
                BoundingBox(*feature_b_box, "square"),
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return aligned("circle", "square", axis)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        expected = {
            "type": "FeedBackAndList",
            "probability": 0.0,
            "items": [
                {
                    "type": "FeedBack",
                    "feedback": f"The circle should be aligned {axis}ly w.r.t. the features circle and square.",
                    "probability": 0.0,
                }
            ],
        }
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 0.0, delta=0.05)

    @parameterized.expand(
        [
            (
                (40, 40, 60, 60),
                (70, 45, 80, 55),
                "vertical",
            ),
            (
                (40, 40, 60, 60),
                (44, 70, 54, 80),
                "horizontal",
            ),
        ]
    )
    def test_aligned_valid_multiple(self, feature_a_box, feature_b_box, axis):

        feat_dict = {
            hash((self.custom_image.tobytes(), "circle")): [
                BoundingBox(*feature_a_box, "circleA"),
                BoundingBox(*feature_a_box, "circleB"),
            ],
            hash((self.custom_image.tobytes(), "square")): [
                BoundingBox(*feature_a_box, "square"),
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return aligned("square", "circle", axis)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=self.custom_image,
            segment_function=get_features,
        )

        expected = None
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 1.0, delta=0.05)

    @parameterized.expand(
        [
            (
                (40, 40, 60, 60),
                (170, 45, 180, 55),
                "horizontal",
            ),
            (
                (40, 40, 60, 60),
                (44, 170, 54, 180),
                "vertical",
            ),
        ]
    )
    def test_aligned_invalid_multiple(self, feature_a_box, feature_b_box, axis):

        feat_dict = {
            hash((self.custom_image.tobytes(), "circle")): [
                BoundingBox(*feature_a_box, "circleA"),
                BoundingBox(*feature_b_box, "circleB"),
            ],
            hash((self.custom_image.tobytes(), "square")): [
                BoundingBox(*feature_a_box, "square"),
            ],
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return aligned("square", "circle", axis)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=self.custom_image,
            segment_function=get_features,
        )

        expected = {
            "type": "FeedBackAndList",
            "probability": 0.0,
            "items": [
                {
                    "type": "FeedBack",
                    "feedback": f"The circleB should be aligned {axis}ly w.r.t. the features square and circle.",
                    "probability": 0.0,
                }
            ],
        }
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 0.0, delta=0.05)

    # Count

    def test_count_valid(self):

        feat_dict = {
            hash((self.custom_image.tobytes(), "circles")): [
                BoundingBox(1, 1, 1, 1, "circleA"),
                BoundingBox(1, 1, 1, 1, "circleB"),
            ]
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return count("circles", 2)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        expected = None
        self.assertEqual(expected, feedback.tojson(0.7))
        self.assertAlmostEqual(feedback.probability, 0.8, delta=0.05)

    def test_count_invalid(self):

        feat_dict = {
            hash((self.custom_image.tobytes(), "circles")): [
                BoundingBox(1, 1, 1, 1, "circleA"),
                BoundingBox(1, 1, 1, 1, "circleB"),
            ]
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return count("circles", 5)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        expected = {
            "type": "FeedBack",
            "feedback": "The number of circles is 2, but should be 5.",
            "probability": 0.2,
        }
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 0.2, delta=0.05)

    def test_count_negated_valid(self):

        feat_dict = {
            hash((self.custom_image.tobytes(), "circles")): [
                BoundingBox(1, 1, 1, 1, "circleA"),
                BoundingBox(1, 1, 1, 1, "circleB"),
            ]
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return ~count("circles", 3)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=self.custom_image,
            segment_function=get_features,
        )

        expected = None
        self.assertEqual(expected, feedback.tojson(0.7))
        self.assertAlmostEqual(feedback.probability, 0.8, delta=0.05)

    def test_count_negated_invalid(self):

        feat_dict = {
            hash((self.custom_image.tobytes(), "circles")): [
                BoundingBox(1, 1, 1, 1, "circleA"),
                BoundingBox(1, 1, 1, 1, "circleB"),
            ]
        }

        def get_features(feature: str, image: Image.Image):
            return feat_dict[hash((image.tobytes(), feature))]

        def test_valid_customization() -> bool:
            return ~count("circles", 2)

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            box_function=get_features,
            custom_image=self.custom_image,
            segment_function=get_features,
        )

        expected = {
            "type": "FeedBack",
            "feedback": "The number of circles should not be 2.",
            "probability": 0.2,
        }
        self.assertEqual(expected, feedback.tojson(0.9))
        self.assertAlmostEqual(feedback.probability, 0.2, delta=0.05)
