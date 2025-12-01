import unittest


from vif.falcon.oracle.guided_oracle.expressions import OracleExpression, present
from vif.models.detection import BoundingBox, SegmentationMask
from PIL import Image


class TestExpression(unittest.TestCase):
    """Note: Does not test expressions which are property(), hence no need for llm instantiation"""

    def __init__(self, methodName="runTest"):
        self.original_image: Image.Image = Image.new("1", (2, 2))
        self.original_image.putdata([0, 0, 0, 1])
        self.custom_image: Image.Image = Image.new("1", (2, 2))
        self.custom_image.putdata([0, 0, 1, 0])
        super().__init__(methodName)

    def test_addition(self):
        custom_features: list[BoundingBox] = [
            BoundingBox(0, 0, 0, 0, "triangle", 0.65)
        ]

        def get_features(features: list[str], image: Image.Image):
            return custom_features

        def test_valid_customization() -> bool:
            return present("triangle")

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            segment_function=get_features,
            box_function=get_features,
        )
        self.assertAlmostEqual(feedback.probability, 1.0)
        expected = None
        self.assertEqual(expected, feedback.tojson(0.9))

    def test_removal(self):
        custom_features: list[BoundingBox] = [
        ]

        def get_features(features: list[str], image: Image.Image):
            return custom_features

        def test_valid_customization() -> bool:
            return ~present("rectangle")

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            segment_function=get_features,
            box_function =get_features
        )
        self.assertAlmostEqual(feedback.probability, 1.0)
        expected = None
        self.assertEqual(expected, feedback.tojson(0.9))

    def test_removal_prob(self):
        custom_features: list[BoundingBox] = [
            BoundingBox(0, 0, 0, 0, "rectangle",0.05)
        ]

        def get_features(features: list[str], image: Image.Image):
            return custom_features

        def test_valid_customization() -> bool:
            return ~present("rectangle")

        expression: OracleExpression = test_valid_customization()
        feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            segment_function=get_features,
            box_function =get_features
        )
        self.assertAlmostEqual(feedback.probability, 1/3,delta=0.05)
        expected = None
        self.assertEqual(expected, feedback.tojson(0.9))

    def test_and_valid(self):

        custom_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "triangle", 0.6, 0.5),
            SegmentationMask(0, 0, 0, 0, None, "rectangle", 0.6, 0.5),
            SegmentationMask(0, 0, 0, 0, None, "circle"),
        ]
        feat_dict = {
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return present("rectangle") & present("triangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        self.assertTrue(result)
        self.assertEqual([], feedback)

    def test_and_invalid(self):
        original_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "circle")
        ]
        custom_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "rectangle", 0.6, 0.5),
            SegmentationMask(0, 0, 0, 0, None, "circle"),
        ]
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return present("rectangle") & present("triangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        self.assertFalse(result)
        self.assertEqual(
            ["The feature triangle is not in the customized image."], feedback
        )

    def test_not_present_invalid(self):
        original_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "circle")
        ]
        custom_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "circle"),
            SegmentationMask(0, 0, 0, 0, None, "rectangle", 0.6, 0.5),
        ]

        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return ~present("rectangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        self.assertFalse(result)
        self.assertEqual(
            ["The feature rectangle is still present in the customized image."],
            feedback,
        )

    def test_not_present_plus_present(self):
        original_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "circle")
        ]
        custom_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "circle"),
            SegmentationMask(0, 0, 0, 0, None, "triangle", 0.5, 0.6),
            SegmentationMask(0, 0, 0, 0, None, "rectangle", 0.7, 0.6),
        ]
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return ~present("rectangle") & present("triangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        self.assertFalse(result)
        self.assertEqual(
            [
                "The feature rectangle is still present in the customized image.",
                "The feature triangle is not in the customized image.",
            ],
            feedback,
        )

    def test_or_valid(self):
        original_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "circle")
        ]
        custom_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "rectangle", 0.7, 0.3),
            SegmentationMask(0, 0, 0, 0, None, "circle"),
        ]
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return present("rectangle") | present("triangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        self.assertTrue(result)
        self.assertEqual([], feedback)

    def test_or_invalid(self):
        original_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "circle")
        ]
        custom_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "circle"),
            SegmentationMask(0, 0, 0, 0, None, "rectangle", 0.5, 0.2),
            SegmentationMask(0, 0, 0, 0, None, "triangle", 0.5, 0.2),
        ]
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return present("rectangle") | present("triangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        self.assertFalse(result)
        self.assertEqual(
            [
                "One of these conditions should have been valid, but none were\n - The feature rectangle is not in the customized image.\n - The feature triangle is not in the customized image."
            ],
            feedback,
        )

    def test_or_inverted_valid(self):
        original_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "circle")
        ]
        custom_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "circle"),
        ]
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return ~(present("rectangle") | present("triangle"))

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        self.assertTrue(result)
        self.assertEqual([], feedback)

    def test_or_inverted_invalid(self):
        original_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "circle")
        ]
        custom_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "circle"),
            SegmentationMask(0, 0, 0, 0, None, "rectangle", 0.8, 0.5),
            SegmentationMask(0, 0, 0, 0, None, "triangle", 0.8, 0.5),
        ]
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return ~(present("rectangle") | present("triangle"))

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        self.assertFalse(result)
        self.assertEqual(
            [
                "The feature rectangle is still present in the customized image.",
                "The feature triangle is still present in the customized image.",
            ],
            feedback,
        )

    def test_or_inverted_invalid_one(self):
        original_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "circle")
        ]
        custom_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "circle"),
            SegmentationMask(0, 0, 0, 0, None, "rectangle", 0.8, 0.5),
        ]
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return ~(present("rectangle") | present("triangle"))

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_image=self.original_image,
            custom_image=self.custom_image,
            segment_function=get_features,
        )
        self.assertFalse(result)
        self.assertEqual(
            ["The feature rectangle is still present in the customized image."],
            feedback,
        )
