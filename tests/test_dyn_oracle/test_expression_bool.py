import unittest


from vif.falcon.oracle.guided_oracle.expressions import OracleExpression, added, removed
from vif.models.detection import SegmentationMask
from PIL import Image


class TestExpression(unittest.TestCase):

    def __init__(self, methodName="runTest"):
        self.original_image: Image.Image = Image.new("1", (2, 2))
        self.original_image.putdata([0, 0, 0, 1])
        self.custom_image: Image.Image = Image.new("1", (2, 2))
        self.custom_image.putdata([0, 0, 1, 0])
        super().__init__(methodName)

    def test_addition(self):
        custom_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "triangle")
        ]

        def get_features(features: list[str], image: Image.Image):
            return custom_features

        def test_valid_customization() -> bool:
            return added("triangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
        )
        self.assertTrue(result)
        self.assertEqual([], feedback)

    def test_removal(self):
        custom_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "triangle"),
            SegmentationMask(0, 0, 0, 0, None, "circle"),
        ]

        def get_features(features: list[str], image: Image.Image):
            return custom_features

        def test_valid_customization() -> bool:
            return removed("rectangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
        )
        self.assertTrue(result)
        self.assertEqual([], feedback)

    def test_and_valid(self):

        original_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "circle")
        ]
        custom_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "triangle"),
            SegmentationMask(0, 0, 0, 0, None, "rectangle"),
            SegmentationMask(0, 0, 0, 0, None, "circle"),
        ]
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return added("rectangle") & added("triangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
        )
        self.assertTrue(result)
        self.assertEqual([], feedback)

    def test_and_invalid(self):
        original_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "circle")
        ]
        custom_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "rectangle"),
            SegmentationMask(0, 0, 0, 0, None, "circle"),
        ]
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return added("rectangle") & added("triangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
        )
        self.assertFalse(result)
        self.assertEqual(
            ["The feature triangle is not in the customized image."], feedback
        )

    def test_not_added(self):
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
            return ~added("rectangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
        )
        self.assertTrue(result)
        self.assertEqual([], feedback)
        self.assertEqual(type(expression), removed)

    def test_not_added_invalid(self):
        original_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "circle")
        ]
        custom_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "circle"),
            SegmentationMask(0, 0, 0, 0, None, "rectangle"),
        ]

        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return ~added("rectangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
        )
        self.assertFalse(result)
        self.assertEqual(
            ["The feature rectangle is still present in the customized image."],
            feedback,
        )

    def test_not_removed(self):
        original_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "circle")
        ]
        custom_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "rectangle"),
            SegmentationMask(0, 0, 0, 0, None, "circle"),
        ]

        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return ~removed("rectangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
        )
        self.assertTrue(result)
        self.assertEqual([], feedback)
        self.assertEqual(type(expression), added)

    def test_not_removed_invalid(self):
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
            return ~removed("rectangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
        )
        self.assertFalse(result)
        self.assertEqual(
            ["The feature rectangle is not in the customized image."], feedback
        )

    def test_not_removed_plus_added(self):
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
            return ~removed("rectangle") & added("triangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
        )
        self.assertFalse(result)
        self.assertEqual(
            [
                "The feature rectangle is not in the customized image.",
                "The feature triangle is not in the customized image.",
            ],
            feedback,
        )

    def test_or_valid(self):
        original_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "circle")
        ]
        custom_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "rectangle"),
            SegmentationMask(0, 0, 0, 0, None, "circle"),
        ]
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return added("rectangle") | added("triangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
        )
        self.assertTrue(result)
        self.assertEqual([], feedback)

    def test_or_invalid(self):
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
            return added("rectangle") | added("triangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
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
            return ~(added("rectangle") | added("triangle"))

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
        )
        self.assertTrue(result)
        self.assertEqual([], feedback)

    def test_or_inverted_invalid(self):
        original_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "circle")
        ]
        custom_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "circle"),
            SegmentationMask(0, 0, 0, 0, None, "rectangle"),
            SegmentationMask(0, 0, 0, 0, None, "triangle"),
        ]
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return ~(added("rectangle") | added("triangle"))

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
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
            SegmentationMask(0, 0, 0, 0, None, "rectangle"),
        ]
        feat_dict = {
            hash(self.original_image.tobytes()): original_features,
            hash(self.custom_image.tobytes()): custom_features,
        }

        def get_features(features: list[str], image: Image.Image):

            return feat_dict[hash(image.tobytes())]

        def test_valid_customization() -> bool:
            return ~(added("rectangle") | added("triangle"))

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            self.original_image, self.custom_image, get_features
        )
        self.assertFalse(result)
        self.assertEqual(
            ["The feature rectangle is still present in the customized image."],
            feedback,
        )
