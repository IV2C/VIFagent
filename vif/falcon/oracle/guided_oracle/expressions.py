#################### Oracle condition "function" which actually are classes, for easier feedback creation ###############

from abc import abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import math
from typing import Any, Self
from PIL import Image
import numpy as np

from vif.models.detection import SegmentationMask
from vif.utils.image_utils import compute_overlap, crop_with_box, rotate_mask

from sentence_transformers import SentenceTransformer

import matplotlib.colors as mcolors

color_model = SentenceTransformer("CharlyR/clip_distilled_rgb_emb")
xkcd_colors = [key.removeprefix("xkcd:") for key in mcolors.XKCD_COLORS.keys()]
accepted_color_ratio = math.floor((1 / 10) * len(xkcd_colors))


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
        original_image: Image.Image,
        custom_image: Image.Image,
        segment_function: Callable[[list[str], Image.Image], list[SegmentationMask]],
    ) -> tuple[bool, list[str]]:
        """evaluates the condition

        Returns:
            tuple[bool,str]: a boolean from the condition and feedback
        """
        pass


class OracleCondition(OracleExpression):

    def __init__(self, feature: str):
        self.feature = feature


class OracleBynaryExpr(OracleExpression):
    def __init__(self, exprA: OracleExpression, exprB: OracleExpression):
        self.exprA = exprA
        self.exprB = exprB

    def evaluate(self, original_image, custom_image, segment_function):
        pass


class OracleOrExpr(OracleBynaryExpr):

    def __invert__(self):
        return ~self.exprA & ~self.exprB

    def evaluate(self, original_image, custom_image, segment_function):
        res1, feedback1 = self.exprA.evaluate(
            original_image, custom_image, segment_function
        )

        res2, feedback2 = self.exprB.evaluate(
            original_image, custom_image, segment_function
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

    def evaluate(self, original_image, custom_image, segment_function):
        res1, feedback1 = self.exprA.evaluate(
            original_image, custom_image, segment_function
        )

        res2, feedback2 = self.exprB.evaluate(
            original_image, custom_image, segment_function
        )
        feedbacks = feedback1 + feedback2

        return (res1 and res2, feedbacks)


class added(OracleCondition):
    def __init__(self, feature):
        super().__init__(feature)

    def __invert__(self):
        return removed(self.feature)

    def evaluate(self, original_image, custom_image, segment_function):
        condition = any(
            m.label == self.feature
            for m in segment_function([self.feature], custom_image)
        )
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

    def evaluate(self, original_image, custom_image, segment_function):
        condition = all(
            m.label != self.feature
            for m in segment_function([self.feature], custom_image)
        )
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


def get_seg_for_feature(feature: str, features: list[SegmentationMask]):
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

    def evaluate(self, original_image, custom_image, segment_function):
        box_featA = get_seg_for_feature(
            self.feature, segment_function([self.feature], custom_image)
        )
        box_featB = get_seg_for_feature(
            self.other_feature, segment_function([self.feature], custom_image)
        )

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
    def __init__(self, feature: str, other_feature: str, ratio: float, axis: Axis):
        self.other_feature = other_feature
        self.axis = axis
        self.ratio = ratio
        self.negated = False
        super().__init__(feature)

    def __invert__(self):
        self.negated = True
        return self

    def evaluate(self, original_image, custom_image, segment_function):
        original_box_featA = get_seg_for_feature(
            self.feature, segment_function([self.feature], original_image)
        )
        original_box_featB = get_seg_for_feature(
            self.other_feature, segment_function([self.feature], original_image)
        )
        custom_box_featA = get_seg_for_feature(
            self.feature, segment_function([self.feature], custom_image)
        )
        custom_box_featB = get_seg_for_feature(
            self.other_feature, segment_function([self.feature], custom_image)
        )

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
            return (not condition, feedback)

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
            return (not condition, feedback)
        feedback = f"The vertical distance between {self.feature} and {self.other_feature} was supposed to be around {expected_distance}, but was {d2[1]}."
        return (condition, feedback)


class angle(OracleCondition):
    def __init__(self, feature: str, degree: int):
        self.degree = degree
        self.negated = False
        super().__init__(feature)

    def __invert__(self):
        self.negated = True
        return self

    def evaluate(self, original_image, custom_image, segment_function):
        ori_seg = get_seg_for_feature(
            self.feature, segment_function([self.feature], original_image)
        )
        custom_seg = get_seg_for_feature(
            self.feature, segment_function([self.feature], custom_image)
        )
        ori_box = (ori_seg.x0, ori_seg.x1, ori_seg.y0, ori_seg.y1)
        custom_box = (custom_seg.x0, custom_seg.x1, custom_seg.y0, custom_seg.y1)
        cropped1 = crop_with_box(ori_seg.mask, ori_box)
        cropped2 = crop_with_box(custom_seg.mask, custom_box)

        degrees = list(range(-180, 180, 5))
        args_list = [(deg, cropped1, cropped2) for deg in degrees]

        ious = defaultdict(list)
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.test_angle, args) for args in args_list]
            for future in as_completed(futures):
                iou, deg = future.result()
                ious[iou].append(
                    deg
                )  # using similarity as key make this resistant to rotation invariant

        sorted_IoUs = sorted(ious.items(), reverse=True)
        sorted_IoUs_degrees = [iou for ious in sorted_IoUs[:3] for iou in ious[1]]
        condition = any(deg - 3 < self.degree < deg + 3 for deg in sorted_IoUs_degrees)

        if self.negated:
            condition = not condition
            feedback = f"The feature {self.feature} should not be rotated by {self.degree} degrees, and is rotated by {",".join([str(io) for io in sorted_IoUs[0][1]+sorted_IoUs[1][1]])} degrees, which is too close/equal."
        else:
            feedback = f"The feature {self.feature} should be rotated by {self.degree} degrees, but is rotated by {",".join([str(io) for io in sorted_IoUs[0][1]+sorted_IoUs[1][1]])} degrees."

        return (condition, [feedback] if not condition else [])

    def test_angle(self, args):
        degree_test, cropped1, cropped2 = args
        rotated = rotate_mask(cropped1, angle=degree_test)
        iou = compute_overlap(rotated, cropped2)
        return round(iou, 2), degree_test


class color(OracleCondition):
    def __init__(self, feature: str, color_expected: str):
        self.color_expected = color_expected
        self.negated = False

        super().__init__(feature)

    def __invert__(self):
        self.negated = True
        return self

    def get_feature_color(self, feature_seg: SegmentationMask, image: Image.Image):
        img_np = np.array(image)

        masked_pixels = img_np[feature_seg.mask.astype(bool)]

        most_common = Counter(map(tuple, masked_pixels)).most_common(1)[0][0]
        return most_common

    def evaluate(self, original_image, custom_image, segment_function):
        req_features = get_seg_for_feature(
            self.feature, segment_function([self.feature], custom_image)
        )

        color_custom = self.get_feature_color(req_features, custom_image)

        color_custom = "rgb(" + ",".join([str(co) for co in color_custom]) + ")"

        embeddings = color_model.encode([color_custom])
        all_colors = [self.color_expected] + xkcd_colors
        embeddings_full_colors = color_model.encode(all_colors)

        similarities = color_model.similarity(embeddings, embeddings_full_colors)[0]

        top_n_indices = np.argsort(-similarities)[:accepted_color_ratio]
        col_keys_max_sim = np.array(all_colors)[top_n_indices]

        condition = (
            similarities[0] > 0.9
            or self.color_expected in col_keys_max_sim
            or any(self.color_expected in cur_col for cur_col in col_keys_max_sim)
        )
        feedback = f"The color of the feature {self.feature} should have been {self.color_expected}, but is closer to {", ".join(col_keys_max_sim[:3])}."

        if self.negated:
            condition = not condition
            feedback = f"The color of the feature {self.feature} should not have been {self.color_expected}, but is still too close to {self.color_expected}."

        return (condition, [feedback] if not condition else [])
