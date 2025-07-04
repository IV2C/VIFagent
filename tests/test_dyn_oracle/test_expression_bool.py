import unittest

from vif.falcon.oracle.dynamic_oracle.expressions import (
    OracleExpression,
    added,
    removed,
)
from vif.models.detection import SegmentationMask
from PIL import Image


class TestExpression(unittest.TestCase):

    def test_addition(self):
        original_features: list[SegmentationMask] = None
        custom_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "triangle")
        ]
        original_image: Image.Image = None
        custom_image: Image.Image = None

        def test_valid_customization() -> bool:
            return added("triangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_features, custom_features, original_image, custom_image
        )
        self.assertTrue(result)
        self.assertEqual([], feedback)

    def test_removal(self):
        original_features: list[SegmentationMask] = None
        custom_features: list[SegmentationMask] = [
            SegmentationMask(0, 0, 0, 0, None, "triangle"),
            SegmentationMask(0, 0, 0, 0, None, "circle"),
        ]
        original_image: Image.Image = None
        custom_image: Image.Image = None

        def test_valid_customization() -> bool:
            return removed("rectangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_features, custom_features, original_image, custom_image
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
        original_image: Image.Image = None
        custom_image: Image.Image = None

        def test_valid_customization() -> bool:
            return added("rectangle") & added("triangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_features, custom_features, original_image, custom_image
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
        original_image: Image.Image = None
        custom_image: Image.Image = None

        def test_valid_customization() -> bool:
            return added("rectangle") & added("triangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_features, custom_features, original_image, custom_image
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
        original_image: Image.Image = None
        custom_image: Image.Image = None

        def test_valid_customization() -> bool:
            return ~added("rectangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_features, custom_features, original_image, custom_image
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
        original_image: Image.Image = None
        custom_image: Image.Image = None

        def test_valid_customization() -> bool:
            return ~added("rectangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_features, custom_features, original_image, custom_image
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
        original_image: Image.Image = None
        custom_image: Image.Image = None

        def test_valid_customization() -> bool:
            return ~removed("rectangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_features, custom_features, original_image, custom_image
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
        original_image: Image.Image = None
        custom_image: Image.Image = None

        def test_valid_customization() -> bool:
            return ~removed("rectangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_features, custom_features, original_image, custom_image
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
        original_image: Image.Image = None
        custom_image: Image.Image = None

        def test_valid_customization() -> bool:
            return ~removed("rectangle") & added("triangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_features, custom_features, original_image, custom_image
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
        original_image: Image.Image = None
        custom_image: Image.Image = None

        def test_valid_customization() -> bool:
            return added("rectangle") | added("triangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_features, custom_features, original_image, custom_image
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
        original_image: Image.Image = None
        custom_image: Image.Image = None

        def test_valid_customization() -> bool:
            return added("rectangle") | added("triangle")

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_features, custom_features, original_image, custom_image
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
        original_image: Image.Image = None
        custom_image: Image.Image = None

        def test_valid_customization() -> bool:
            return ~(added("rectangle") | added("triangle"))

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_features, custom_features, original_image, custom_image
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
        original_image: Image.Image = None
        custom_image: Image.Image = None

        def test_valid_customization() -> bool:
            return ~(added("rectangle") | added("triangle"))

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_features, custom_features, original_image, custom_image
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
        original_image: Image.Image = None
        custom_image: Image.Image = None

        def test_valid_customization() -> bool:
            return ~(added("rectangle") | added("triangle"))

        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_features, custom_features, original_image, custom_image
        )
        self.assertFalse(result)
        self.assertEqual(
            ["The feature rectangle is still present in the customized image."],
            feedback,
        )
