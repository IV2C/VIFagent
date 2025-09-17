#################### Oracle condition "function" which actually are classes, for easier feedback creation ###############
import torch
from abc import abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from PIL import Image
import numpy as np
import open_clip

from vif.models.detection import SegmentationMask
from vif.utils.image_utils import (
    compute_mask_IoU,
    crop_image_with_box,
    crop_mask_with_box,
    image_from_color_count,
    nmse_np,
    pad_center,
    rotate_mask,
    apply_mask,
)


### Color oracle settings

basic_colors = [
    "red",
    "green",
    "yellow",
    "blue",
    "brown",
    "purple",
    "pink",
    "orange",
    "grey",
]
bw = ["black", "white"]
attributes = ["very light ", "light ", "", "dark ", "very dark "]


def build_basic_colors():
    colors_out = bw

    for basic_color in basic_colors:
        for att in attributes:
            colors_out.append(att + basic_color)
    return [color_out + " color" for color_out in colors_out]


basic_colors = build_basic_colors()
accepted_color_ratio = math.floor((3 / 20) * len(basic_colors))


### Shape oracle settings
shapes = [
    "point",
    "line",
    "ray",
    "segment",
    "circle",
    "ellipse",
    "oval",
    "arc",
    "sector",
    "segment of circle",
    "triangle",
    "equilateral triangle",
    "isosceles triangle",
    "scalene triangle",
    "right triangle",
    "quadrilateral",
    "square",
    "rectangle",
    "parallelogram",
    "rhombus",
    "trapezoid",
    "kite",
    "pentagon",
    "hexagon",
    "heptagon",
    "octagon",
    "nonagon",
    "decagon",
    "dodecagon",
    "star",
    "crescent",
    "cross",
    "polygon",
    "regular polygon",
    "irregular polygon",
]


# Boolean expression


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


class present(OracleCondition):
    def __init__(self, feature):
        super().__init__(feature)

    def __invert__(self):
        return removed(self.feature)

    def evaluate(self, original_image, custom_image, segment_function):
        segments = segment_function([self.feature], custom_image)
        targeted_seg = get_seg_for_feature(self.feature, segments)

        condition = targeted_seg != None and targeted_seg.box_prob >= 0.6
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
        return present(self.feature)

    def evaluate(self, original_image, custom_image, segment_function):
        segments = segment_function([self.feature], custom_image)
        targeted_seg = get_seg_for_feature(self.feature, segments)

        condition = targeted_seg == None or targeted_seg.box_prob < 0.6

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
    over = "over"
    under = "under"
    left = "left"


class Axis(str, Enum):
    horizontal = "horizontal"
    vertical = "vertical"


def get_seg_for_feature(
    feature: str, features: list[SegmentationMask]
) -> SegmentationMask:
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
            Direction.over: Direction.under,
            Direction.under: Direction.over,
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
            Direction.over: (
                lambda m1, m2: m1[1] < m2[1],
                f"The feature {self.feature} is not above the feature {self.other_feature}",
            ),
            Direction.under: (
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
        cropped1 = crop_mask_with_box(ori_seg.mask, ori_box)
        cropped2 = crop_mask_with_box(custom_seg.mask, custom_box)

        degrees = list(range(-175, 180, 5))
        args_list = [(deg, cropped1, cropped2) for deg in degrees]

        ious = defaultdict(list)
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.test_angle, args) for args in args_list]
            for future in as_completed(futures):
                iou, deg = future.result()
                ious[iou].append(
                    deg
                )  # using similarity as key make this somewhat resistant to rotation invariant

        sorted_IoUs = sorted(ious.items(), reverse=True)
        sorted_IoUs_degrees = [iou for ious in sorted_IoUs[:5] for iou in ious[1]]
        condition = any(
            deg - 5 <= self.degree <= deg + 5 for deg in sorted_IoUs_degrees
        )

        if self.negated:
            condition = not condition
            feedback = f"The feature {self.feature} should not be rotated by {self.degree} degrees, and is rotated by {",".join([str(io) for io in sorted_IoUs[0][1]+sorted_IoUs[1][1]])} degrees, which is too close/equal."
        else:
            feedback = f"The feature {self.feature} should be rotated by {self.degree} degrees, but is rotated by {",".join([str(io) for io in sorted_IoUs[0][1]+sorted_IoUs[1][1]])} degrees."

        return (condition, [feedback] if not condition else [])

    def test_angle(self, args):
        degree_test, cropped1, cropped2 = args
        rotated = rotate_mask(cropped1, angle=degree_test)
        iou = compute_mask_IoU(rotated, cropped2)
        return round(iou, 2), degree_test


class color(OracleCondition):
    def __init__(self, feature: str, color_expected: str):
        self.color_expected = color_expected + " color"
        self.negated = False

        super().__init__(feature)

    def __invert__(self):
        self.negated = True
        return self

    def get_feature_colors(self, feature_seg: SegmentationMask, image: Image.Image):
        img_np = np.array(image)

        masked_pixels = img_np[feature_seg.mask.astype(bool)]

        most_common = Counter(map(tuple, masked_pixels)).most_common()
        ratio_common = int(0.9 * len(most_common))
        return most_common[:ratio_common]

    def get_image_features(self, image):
        image_embedded = preprocess(image).to(device).unsqueeze(0)
        with torch.no_grad():
            feat = clip_model.encode_image(image_embedded)  # add batch dim
            feat /= feat.norm(dim=-1, keepdim=True)
        return feat

    def evaluate(self, original_image, custom_image, segment_function):
        cust_features = get_seg_for_feature(
            self.feature, segment_function([self.feature], custom_image)
        )
        custom_colors_counts = self.get_feature_colors(cust_features, custom_image)
        color_image = image_from_color_count(custom_colors_counts)

        eval_colors = [self.color_expected] + basic_colors

        clip_encoded_color_names = clip_model.encode_text(clip_tokenizer(eval_colors))

        image_features = self.get_image_features(color_image)
        with torch.no_grad(), torch.autocast("cuda"):
            text_probs = (100.0 * image_features @ clip_encoded_color_names.T).softmax(
                dim=-1
            )[0]
        top_indice = np.argsort(-text_probs)[:accepted_color_ratio]

        most_similar_colors = np.array(eval_colors)[top_indice]

        condition = (
            text_probs[0] > 0.5
            or self.color_expected in most_similar_colors
            or any(self.color_expected in cur_col for cur_col in most_similar_colors)
        )

        feedback = f"The color of the feature {self.feature} should have been {self.color_expected.removesuffix(" color")}, but is closer to {", ".join([c.removesuffix(" color") for c in most_similar_colors[:3]])}."

        if self.negated:
            condition = not condition
            feedback = f"The color of the feature {self.feature} should not have been {self.color_expected.removesuffix(" color")}, but is still too close to {self.color_expected.removesuffix(" color")}."

        return (condition, [feedback] if not condition else [])


class size(OracleCondition):
    def __init__(self, feature: str, ratio: tuple[float, float]):
        self.ratio = ratio
        self.negated = False
        self.delta = 0.15
        super().__init__(feature)

    def __invert__(self):
        self.negated = True
        return self

    def evaluate(self, original_image, custom_image, segment_function):
        cust_features = get_seg_for_feature(
            self.feature, segment_function([self.feature], custom_image)
        )
        ori_features = get_seg_for_feature(
            self.feature, segment_function([self.feature], original_image)
        )

        x_ratio = (cust_features.x1 - cust_features.x0) / (
            ori_features.x1 - ori_features.x0
        )
        y_ratio = (cust_features.y1 - cust_features.y0) / (
            ori_features.y1 - ori_features.y0
        )

        feedback = []

        x_condition = (
            (self.ratio[0] - self.delta * self.ratio[0])
            < x_ratio
            < (self.ratio[0] + self.delta * self.ratio[0])
        )
        y_condition = (
            (self.ratio[1] - self.delta * self.ratio[1])
            < y_ratio
            < (self.ratio[1] + self.delta * self.ratio[1])
        )

        condition = x_condition and y_condition

        if self.negated:
            x_condition and feedback.append(
                f"The {self.feature} was resized on x by a ratio of {x_ratio}, which is too close to {self.ratio[0]}"
            )
            y_condition and feedback.append(
                f"The {self.feature} was resized on y by a ratio of {y_ratio}, which is too close to {self.ratio[1]}"
            )
            return (not condition, feedback if condition else [])
        else:
            not x_condition and feedback.append(
                f"The {self.feature} was resized on x by a ratio of {x_ratio}, but was should have been by a ratio of {self.ratio[0]}"
            )
            not y_condition and feedback.append(
                f"The {self.feature} was resized on y by a ratio of {y_ratio}, but was should have been by a ratio of {self.ratio[1]}"
            )

            return (condition, feedback if not condition else [])


##model settings for shape detection
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
device = torch.device("cpu")
clip_model = clip_model.to(device)
clip_model.eval()
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")


class shape(OracleCondition):
    def __init__(self, feature: str, shape: str):
        self.negated = False
        self.req_shape = shape
        super().__init__(feature)

    def __invert__(self):
        self.negated = True
        return self

    def get_image_features(self, image):
        image_embedded = preprocess(image).to(device).unsqueeze(0)
        with torch.no_grad():
            feat = clip_model.encode_image(image_embedded)  # add batch dim
            feat /= feat.norm(dim=-1, keepdim=True)
        return feat

    def evaluate(self, original_image, custom_image, segment_function):
        cust_features = get_seg_for_feature(
            self.feature, segment_function([self.feature], custom_image)
        )
        mask_image = Image.fromarray(cust_features.mask)

        all_shapes = shapes + [self.req_shape]

        clip_encoded_shape_names = clip_model.encode_text(clip_tokenizer(all_shapes))

        image_features = self.get_image_features(mask_image)
        with torch.no_grad(), torch.autocast("cuda"):
            text_probs = (100.0 * image_features @ clip_encoded_shape_names.T).softmax(
                dim=-1
            )
        top_indice = np.argsort(-text_probs[0])[:3]

        most_similar_shapes = np.array(all_shapes)[top_indice]

        condition = self.req_shape in most_similar_shapes
        feedback = f"The feature {self.feature} should be in the shape of a {self.req_shape}, but looks more like a {','.join(most_similar_shapes)}."

        if self.negated:
            condition = not condition
            feedback = f"The feature {self.feature} should not be in the shape of a {self.req_shape}, but still looks like a {self.req_shape}."

        return (condition, [feedback] if not condition else [])


class within(OracleCondition):
    def __init__(self, feature: str, other_feature: str):
        self.other_feature = other_feature
        self.negated = False
        super().__init__(feature)

    def __invert__(self):
        self.negated = True
        return self

    def evaluate(self, original_image, custom_image, segment_function):
        custom_box_featA = get_seg_for_feature(
            self.feature, segment_function([self.feature], custom_image)
        )
        custom_box_featB = get_seg_for_feature(
            self.other_feature, segment_function([self.feature], custom_image)
        )

        score = (
            np.logical_and(custom_box_featA.mask, custom_box_featB.mask).sum()
            / custom_box_featA.mask.sum()
        )
        condition = round(score, 2) > 0.9

        feedback = f"The feature {self.feature} should be contained in the feature {self.other_feature}, but isn't."

        if self.negated:
            condition = not condition
            feedback = f"The feature {self.feature} should not be contained in the feature {self.other_feature}, but is actually within it."

        return (condition, [feedback] if not condition else [])


class mirrored(OracleCondition):
    def __init__(self, feature: str, axis: Axis):
        self.negated = False
        self.axis = axis
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
        cropped1 = apply_mask(original_image,ori_seg.mask)
        cropped2 = apply_mask(custom_image,custom_seg.mask)
                
        cropped1 = crop_image_with_box(cropped1, ori_box)
        cropped2 = crop_image_with_box(cropped2, custom_box)

        match self.axis:
            case Axis.vertical:
                mirrored_cropped1 = np.flip(cropped1, 0)
            case Axis.horizontal:
                mirrored_cropped1 = np.flip(cropped1, 1)
                
        h1, w1 = mirrored_cropped1.shape[:2]
        h2, w2 = cropped2.shape[:2]
        H = max(h1, h2)
        W = max(w1, w2)

        mirrored_cropped1 = pad_center(mirrored_cropped1, H, W)
        cropped2 = pad_center(cropped2, H, W)
        norm_mse = nmse_np(mirrored_cropped1, cropped2)
        condition = round(norm_mse, 2) < 0.1

        if self.negated:
            condition = not condition
            feedback = f"The feature {self.feature} should not be mirrored along the {self.axis} axis."
        else:
            feedback = f"The feature {self.feature} should be mirrored along the {self.axis} axis."
            
        return (condition, [feedback] if not condition else [])
