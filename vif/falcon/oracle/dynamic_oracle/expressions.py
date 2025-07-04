#################### Oracle condition "function" which actually are classes, for easier feedback creation ###############

from abc import abstractmethod
from typing import Any
from PIL import Image

from vif.models.detection import SegmentationMask


class OracleExpression:
    def __init__(self):
        pass

    def __and__(self, other):
        return OracleAndExpr(self, other)

    def __or__(self, other):
        return OracleOrExpr(self, other)

    @abstractmethod
    def __invert__(self):
        pass

    @abstractmethod
    def evaluate(
        self,
        original_features: list[SegmentationMask],
        custom_features: list[SegmentationMask],
        original_image: Image.Image,
        custom_image: Image.Image,
    ) -> tuple[bool, list[str]]:
        """evaluates the condition

        Returns:
            tuple[bool,str]: a boolean from the condition and feedback
        """
        pass


class OracleCondition(OracleExpression):
    def __init__(self, feature: str):
        self.feature = feature
        pass


class OracleBynaryExpr(OracleExpression):
    def __init__(self, exprA: OracleExpression, exprB: OracleExpression):
        self.exprA = exprA
        self.exprB = exprB
        pass

    def evaluate(
        self,
        original_features,
        custom_features,
        original_image,
        custom_image,
    ):
        pass


class OracleOrExpr(OracleBynaryExpr):

    def __invert__(self):
        return ~self.exprA & ~self.exprB

    def evaluate(
        self,
        original_features,
        custom_features,
        original_image,
        custom_image,
    ):
        res1, feedback1 = self.exprA.evaluate(
            original_features, custom_features, original_image, custom_image
        )

        res2, feedback2 = self.exprB.evaluate(
            original_features, custom_features, original_image, custom_image
        )
        condition = res1 or res2
        if not condition:
            feedbacks = feedback1 + feedback2
            feedbacks = [
                f"One of these conditions should have been valid, but none were\n - {'\n - '.join(feedbacks)}"
            ]
        else:
            feedbacks = []

        return (condition, feedbacks)


class OracleAndExpr(OracleBynaryExpr):

    def __invert__(self):
        return ~self.exprA | ~self.exprB

    def evaluate(
        self,
        original_features,
        custom_features,
        original_image,
        custom_image,
    ):
        res1, feedback1 = self.exprA.evaluate(
            original_features, custom_features, original_image, custom_image
        )

        res2, feedback2 = self.exprB.evaluate(
            original_features, custom_features, original_image, custom_image
        )
        feedbacks = feedback1 + feedback2

        return (res1 and res2, feedbacks)


class added(OracleCondition):
    def __init__(self, feature):
        super().__init__(feature)

    def __invert__(self):
        return removed(self.feature)

    def evaluate(
        self,
        original_features,
        custom_features,
        original_image,
        custom_image,
    ):
        condition = any(m.label == self.feature for m in custom_features)
        return (
            condition,
            (
                [f"The feature {self.feature} is not in the customized image."]
                if not condition
                else []
            ),
        )


class removed(OracleCondition):
    def __init__(self, feature):
        super().__init__(feature)

    def __invert__(self):
        return added(self.feature)

    def evaluate(
        self,
        original_features,
        custom_features,
        original_image,
        custom_image,
    ):
        condition = all(m.label != self.feature for m in custom_features)
        return (
            condition,
            (
                [
                    f"The feature {self.feature} is still present in the customized image."
                ]
                if not condition
                else []
            ),
        )


############################# IMAGE ORACLE ##################################
from enum import Enum

############### UTILS ##############


class Direction(str, Enum):
    right = "right"
    up = "up"
    down = "down"
    left = "left"


class Axis(str, Enum):
    horizontal = "horizontal"
    vertical = "vertical"


def get_box_for_feature(feature: str, features: list[SegmentationMask]):
    return next((seg for seg in features if seg.label == feature), None)


def distance(coordA: tuple[int, int], coordB: tuple[int, int]):
    return (abs(coordB[0] - coordA[0]), abs(coordB[1] - coordA[1]))


def get_box_center(box: SegmentationMask):
    center = lambda b, a: a + (b - a) / 2
    return (
        center(box.x1, box.x0),
        center(box.y1, box.y0),
    )


############## ORACLES #############


class placement(OracleCondition):
    def __init__(self, feature: str, other_feature: str, direction: Direction):
        self.other_feature = other_feature
        self.direction = direction
        super().__init__(feature)

    def __invert__(self):
        opposite = {
            Direction.left: Direction.right,
            Direction.right: Direction.left,
            Direction.up: Direction.down,
            Direction.down: Direction.up,
        }

        self.direction = opposite.get(self.direction, self.direction)

        return self

    def evaluate(
        self, original_features, custom_features, original_image, custom_image
    ):
        box_featA = get_box_for_feature(self.feature, custom_features)
        box_featB = get_box_for_feature(self.other_feature, custom_features)

        center_boxA = get_box_center(box_featA)
        center_boxB = get_box_center(box_featB)

        direction_eval_map = {
            Direction.left: (
                lambda m1, m2: m1[0] < m2[0],
                f"The feature {self.feature} is not on the left of the feature {self.other_feature}",
            ),
            Direction.right: (
                lambda m1, m2: m1[0] > m2[0],
                f"The feature {self.feature} is not on the right of the feature {self.other_feature}",
            ),
            Direction.up: (
                lambda m1, m2: m1[1] < m2[1],
                f"The feature {self.feature} is not above the feature {self.other_feature}",
            ),
            Direction.down: (
                lambda m1, m2: m1[1] > m2[1],
                f"The feature {self.feature} is not under the feature {self.other_feature}",
            ),
        }

        condition_func, feedback = direction_eval_map[self.direction]
        condition = condition_func(center_boxA, center_boxB)

        return (condition, [feedback] if not condition else [])


class position(OracleCondition):
    def __init__(
        self, feature: str, other_feature: str, ratio: float, axis: Axis
    ):
        self.other_feature = other_feature
        self.axis = axis
        self.ratio = ratio
        self.negated = False
        super().__init__(feature)

    def __invert__(self):
        self.negated = True
        return self  # TODO

    def evaluate(
        self, original_features, custom_features, original_image, custom_image
    ):
        original_box_featA = get_box_for_feature(self.feature, original_features)
        original_box_featB = get_box_for_feature(self.other_feature, original_features)
        custom_box_featA = get_box_for_feature(self.feature, custom_features)
        custom_box_featB = get_box_for_feature(self.other_feature, custom_features)

        center_ori_boxA = get_box_center(original_box_featA)
        center_ori_boxB = get_box_center(original_box_featB)
        center_custom_boxA = get_box_center(custom_box_featA)
        center_custom_boxB = get_box_center(custom_box_featB)

        original_distance = distance(center_ori_boxA, center_ori_boxB)
        custom_distance = distance(center_custom_boxA, center_custom_boxB)

        axis_eval_map = {
            Axis.horizontal: self.horizontal_oracle,
            Axis.vertical: self.vertical_oracle,
        }
        condition, feedback = axis_eval_map[self.axis](
            original_distance, custom_distance
        )

        return (condition, [feedback] if not condition else [])

    def horizontal_oracle(
        self, d1: tuple[int, int], d2: tuple[int, int]
    ) -> tuple[bool, str]:
        expected_distance = self.ratio * d1[0]
        condition = (
            expected_distance - 0.1 * expected_distance
            <= d2[0]
            <= expected_distance + 0.1 * expected_distance
        )
        if self.negated:
            feedback = f"The horizontal distance between {self.feature} and {self.other_feature} is {d2[0]} which is too close to {expected_distance}."
            return (not condition,feedback)
        
        feedback = f"The horizontal distance between {self.feature} and {self.other_feature} was supposed to be around {expected_distance}, but was {d2[0]}."
        return (condition, feedback)

    def vertical_oracle(
        self, d1: tuple[int, int], d2: tuple[int, int]
    ) -> tuple[bool, str]:
        expected_distance = self.ratio * d1[1]
        condition = (
            expected_distance - 0.1 * expected_distance
            <= d2[1]
            <= expected_distance + 0.1 * expected_distance
        )
        if self.negated:
            feedback = f"The vertical distance between {self.feature} and {self.other_feature} is {d2[1]} which is too close to {expected_distance}."
            return (not condition,feedback)
        feedback = f"The vertical distance between {self.feature} and {self.other_feature} was supposed to be around {expected_distance}, but was {d2[1]}."
        return (condition, feedback)
