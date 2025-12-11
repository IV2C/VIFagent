#################### Oracle condition "function" which actually are classes, for easier feedback creation ###############
from dataclasses import dataclass
from typing import Self
from openai import Client
import torch
from abc import abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from PIL import Image
import numpy as np
import open_clip

from vif.falcon.oracle.guided_oracle.feedback import (
    FeedBack,
    FeedBackAnd,
    FeedBackAndList,
    FeedBackOr,
    FeedBackOrList,
    FeedBacks,
)
from vif.models.detection import BoundingBox, SegmentationMask
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
accepted_color_ratio = math.floor((1 / 10) * len(basic_colors))


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
        self, *, original_image: Image.Image, custom_image: Image.Image
    ) -> FeedBacks:
        """evaluates the condition

        Returns:
            FeedBacks: A feedback object containing probability and information about the edit
        """
        pass


class OracleCondition(OracleExpression):

    def __init__(self, feature: str, a: float, b: float):
        self.feature = feature

    @abstractmethod
    def evaluate(
        self,
        *,
        original_image: Image.Image,
        custom_image: Image.Image,
        segment_function: Callable[[list[str], Image.Image], list[SegmentationMask]],
        box_function: Callable[[list[str], Image.Image], list[BoundingBox]],
    ) -> FeedBacks:
        """evaluates the condition

        Returns: A feedback object containing probability and information about the edit
        """
        pass


class OracleBynaryExpr(OracleExpression):
    def __init__(self, exprA: OracleExpression, exprB: OracleExpression):
        self.exprA = exprA
        self.exprB = exprB

    def evaluate(self, *args, **kwargs):
        pass


class OracleOrExpr(OracleBynaryExpr):

    def __invert__(self):
        return ~self.exprA & ~self.exprB

    def evaluate(self, *args, **kwargs):
        feedback1 = self.exprA.evaluate(**kwargs)
        feedback2 = self.exprB.evaluate(**kwargs)

        feedbacks = FeedBackOr(feedback1, feedback2)

        return feedbacks


class OracleAndExpr(OracleBynaryExpr):

    def __invert__(self):
        return ~self.exprA | ~self.exprB

    def evaluate(self, *args, **kwargs):
        feedback1 = self.exprA.evaluate(**kwargs)
        feedback2 = self.exprB.evaluate(**kwargs)

        feedbacks = FeedBackAnd(feedback1, feedback2)

        return feedbacks


class present(OracleCondition):
    def __init__(self, feature):
        self.a = 0.3
        self.b = 20
        self.negated = False
        super().__init__(feature, a=self.a, b=self.b)

    def __invert__(self):
        self.negated = True
        return self

    def evaluate(
        self, *, original_image, custom_image, segment_function, box_function, **kwargs
    ):
        boxes = box_function(self.feature, custom_image)
        boxes = [box for box in boxes if box != None]
        if len(boxes) == 0:
            score = 0
        else:
            score = max([box.box_prob for box in boxes])

        if self.negated:
            score = 1 - score
            feeback_str = f"The feature {self.feature} is in the customized image, but should not be."
        else:
            feeback_str = f"The feature {self.feature} is not in the customized image."

        feeback = FeedBack(
            feeback_str,
            score=score,
            name=self.__class__.__name__,
            a=self.a,
            b=self.b,
        )

        return feeback


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


def distance(coordA: tuple[int, int], coordB: tuple[int, int]):
    return (abs(coordB[0] - coordA[0]), abs(coordB[1] - coordA[1]))


def get_box_center(box: BoundingBox):
    center = lambda b, a: a + (b - a) / 2
    return (
        center(box.x1, box.x0),
        center(box.y1, box.y0),
    )


def get_corresponding_features_detections(
    featuresA: list[BoundingBox | SegmentationMask],
    featuresB: list[BoundingBox | SegmentationMask],
) -> dict[str, str]:
    """Uses a clip model to compute the similarities between each labels, and make them correspond between each feature list"""
    a_labels = [f.label for f in featuresA]
    b_labels = [f.label for f in featuresB]

    if len(featuresA) > len(featuresB):
        tmp = b_labels
        b_labels = a_labels
        a_labels = tmp

    clip_encoded_a_labels = clip_model.encode_text(clip_tokenizer(a_labels))
    clip_encoded_b_labels = clip_model.encode_text(clip_tokenizer(b_labels))
    with torch.no_grad(), torch.autocast("cuda"):
        text_sims = clip_encoded_a_labels @ clip_encoded_b_labels.T

    text_sims = sorted(zip(text_sims, a_labels), key=lambda d: max(d[0]), reverse=True)
    label_correspondance = {}
    used_simlabel_indexes = []
    b_labels = np.array(b_labels)

    for sims, alabel in text_sims:
        m_sims = sims
        if len(used_simlabel_indexes) != 0:
            mask = torch.ones(sims.size(0), dtype=torch.bool)
            mask[used_simlabel_indexes] = False
            m_sims = sims[mask]
            if len(m_sims) == 0:
                break
            b_labels = np.delete(b_labels, used_simlabel_indexes)

        index_max = torch.argmax(m_sims)
        label_correspondance[alabel] = b_labels[index_max]
        used_simlabel_indexes.append(index_max.item())

    if len(featuresA) > len(featuresB):
        return {v: k for k, v in label_correspondance.items()}
    return label_correspondance


############## ORACLES #############


class placement(OracleCondition):
    def __init__(self, feature: str, other_feature: str, direction: Direction):
        self.other_feature = other_feature
        self.direction = direction

        self.direction_eval_map = {
            Direction.left: (
                lambda m1, m2: m1[0] < m2[0],
                lambda labelA, labelB: f"The {labelA} is not on the left of the {labelB}",
            ),
            Direction.right: (
                lambda m1, m2: m1[0] > m2[0],
                lambda labelA, labelB: f"The {labelA} is not on the right of the {labelB}",
            ),
            Direction.over: (
                lambda m1, m2: m1[1] < m2[1],
                lambda labelA, labelB: f"The {labelA} is not above the {labelB}",
            ),
            Direction.under: (
                lambda m1, m2: m1[1] > m2[1],
                lambda labelA, labelB: f"The {labelA} is not under the {labelB}",
            ),
        }
        self.negated = False
        self.a = 0.5
        self.b = 20
        super().__init__(feature, a=self.a, b=self.b)

    def __invert__(self):
        self.negated = True
        return self

    def evaluate(
        self, *, original_image, custom_image, segment_function, box_function, **kwargs
    ):
        boxes_featA = box_function(self.feature, custom_image)
        boxes_featB = box_function(self.other_feature, custom_image)

        if (
            feed := FeedBack.from_detection(boxes_featA, self.feature, "customized")
        ) or (
            feed := FeedBack.from_detection(
                boxes_featB, self.other_feature, "customized"
            )
        ):
            return feed

        label_centers_boxA = [
            (box_featA.label, get_box_center(box_featA)) for box_featA in boxes_featA
        ]
        label_centers_boxB = [
            (box_featB.label, get_box_center(box_featB)) for box_featB in boxes_featB
        ]

        condition_func, feedback_func = self.direction_eval_map[self.direction]
        conditions_feedbacks = [
            (condition_func(center_boxA, center_boxB), feedback_func(labelA, labelB))
            for labelA, center_boxA in label_centers_boxA
            for labelB, center_boxB in label_centers_boxB
        ]
        feedbacks = [
            FeedBack(feed, int(cond), self.__class__.__name__, a=self.a, b=self.b,negated=self.negated)
            for cond, feed in conditions_feedbacks
        ]

        return (
            FeedBackAndList(feedbacks)
            if not self.negated
            else FeedBackOrList(feedbacks)
        )


class position(OracleCondition):
    def __init__(self, feature: str, other_feature: str, ratio: float, axis: Axis):
        self.other_feature = other_feature
        self.axis = axis
        self.ratio = ratio
        self.negated = False
        self.a = 0.5
        self.b=10
        super().__init__(feature)

    def __invert__(self):
        self.negated = True
        return self

    def evaluate(
        self, *, original_image, custom_image, segment_function, box_function, **kwargs
    ):
        original_boxes_featA = box_function(self.feature, original_image)

        original_boxes_featB = box_function(self.other_feature, original_image)

        custom_boxes_featA = box_function(self.feature, custom_image)
        custom_boxes_featB = box_function(self.other_feature, custom_image)

        if (
            (
                feed := FeedBack.from_detection(
                    original_boxes_featA, self.feature, "original"
                )
            )
            or (
                feed := FeedBack.from_detection(
                    original_boxes_featB, self.other_feature, "original"
                )
            )
            or (
                feed := FeedBack.from_detection(
                    custom_boxes_featA, self.feature, "customized"
                )
            )
            or (
                feed := FeedBack.from_detection(
                    custom_boxes_featB, self.other_feature, "customized"
                )
            )
        ):
            return feed
        original_box_featB = original_boxes_featB[0]
        custom_box_featB = custom_boxes_featB[0]
        # keeping only the A features with the most similar names between original and modified
        feature_correspondance = get_corresponding_features_detections(
            original_boxes_featA, custom_boxes_featA
        )

        box_mapping = [
            (
                [boxA for boxA in original_boxes_featA if boxA.label == key_label][0],
                [boxA for boxA in custom_boxes_featA if boxA.label == value_label][0],
            )
            for key_label, value_label in feature_correspondance.items()
        ]

        centers_mapping = [
            (get_box_center(original_box), get_box_center(custom_box))
            for original_box, custom_box in box_mapping
        ]
        center_ori_boxB = get_box_center(original_box_featB)
        center_custom_boxB = get_box_center(custom_box_featB)

        original_distances = []
        custom_distances = []

        for center_original_box, center_custom_box in centers_mapping:
            original_distances.append(distance(center_original_box, center_ori_boxB))
            custom_distances.append(distance(center_custom_box, center_custom_boxB))

        axis_eval_map = {
            Axis.horizontal: self.horizontal_oracle,
            Axis.vertical: self.vertical_oracle,
        }
        feedbacks = [
            axis_eval_map[self.axis](original_distance, custom_distance, feat_name)
            for original_distance, custom_distance, (_, feat_name) in zip(
                original_distances, custom_distances, feature_correspondance.items()
            )
        ]

        return (
            FeedBackAndList(feedbacks)
            if not self.negated
            else FeedBackOrList(feedbacks)
        )

    def horizontal_oracle(
        self, d1: tuple[int, int], d2: tuple[int, int], feat_name
    ) -> FeedBack:
        expected_distance = self.ratio * d1[0]

        score = 1 - min(1, (abs(d2[0] - expected_distance) / (0.2 * expected_distance)))

        if self.negated:
            feedback = f"The horizontal distance between {feat_name} and {self.other_feature} is {d2[0]} which is too close to {expected_distance}."
            return FeedBack(feedback, 1 - score)

        feedback = f"The horizontal distance between {feat_name} and {self.other_feature} was supposed to be around {expected_distance}, but was {d2[0]}."
        return FeedBack(feedback, score)

    def vertical_oracle(
        self, d1: tuple[int, int], d2: tuple[int, int], feat_name
    ) -> tuple[bool, str]:
        expected_distance = self.ratio * d1[1]
        score = 1 - min(1, (abs(d2[1] - expected_distance) / (0.2 * expected_distance)))

        if self.negated:
            feedback = f"The vertical distance between {feat_name} and {self.other_feature} is {d2[1]} which is too close to {expected_distance}."
            return FeedBack(feedback, 1 - score)
        feedback = f"The vertical distance between {feat_name} and {self.other_feature} was supposed to be around {expected_distance}, but was {d2[1]}."
        return FeedBack(feedback, score)


class angle(OracleCondition):
    def __init__(self, feature: str, degree: int):
        self.degree = degree
        self.negated = False
        super().__init__(feature)

    def __invert__(self):
        self.negated = True
        return self

    def check_rotation(
        self, ori_seg: SegmentationMask, custom_seg: SegmentationMask
    ) -> FeedBack:
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
        sorted_IoUs_degrees = [iou for ious in sorted_IoUs[:3] for iou in ious[1]]

        deg_scores = [
            min(abs(self.degree - deg) / 10, 1) for deg in sorted_IoUs_degrees
        ]

        score = 1 - min(deg_scores)

        if self.negated:
            score = 1 - score
            feedback = f"The {ori_seg.label} should not be rotated by {self.degree} degrees, and is rotated by {",".join([str(io) for io in sorted_IoUs[0][1]+sorted_IoUs[1][1]])} degrees, which is too close/equal."
        else:
            feedback = f"The {ori_seg.label} should be rotated by {self.degree} degrees, but is rotated by {",".join([str(io) for io in sorted_IoUs[0][1]+sorted_IoUs[1][1]])} degrees."

        return FeedBack(feedback, score)

    def evaluate(
        self, *, original_image, custom_image, segment_function, box_function, **kwargs
    ):
        ori_segs = segment_function(self.feature, original_image)

        custom_segs = segment_function(self.feature, custom_image)
        if (feed := FeedBack.from_detection(ori_segs, self.feature, "original")) or (
            feed := FeedBack.from_detection(custom_segs, self.feature, "customized")
        ):
            return feed

        feature_correspondance = get_corresponding_features_detections(
            ori_segs, custom_segs
        )
        seg_mapping = [
            (
                [seg for seg in ori_segs if seg.label == key_label][0],
                [seg for seg in custom_segs if seg.label == value_label][0],
            )
            for key_label, value_label in feature_correspondance.items()
        ]
        cond_feed = [
            self.check_rotation(ori_seg, custom_seg)
            for ori_seg, custom_seg in seg_mapping
        ]

        return (
            FeedBackAndList(cond_feed)
            if not self.negated
            else FeedBackOrList(cond_feed)
        )

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

    def get_feature_colors(self, features_seg: SegmentationMask, image: Image.Image):
        img_np = np.array(image)

        masked_pixels = img_np[features_seg.mask.astype(bool)]

        most_common = Counter(map(tuple, masked_pixels)).most_common()
        ratio_common = int(0.9 * len(most_common))
        return most_common[:ratio_common]

    def get_image_features(self, image):
        image_embedded = preprocess(image).to(device).unsqueeze(0)
        with torch.no_grad():
            feat = clip_model.encode_image(image_embedded)  # add batch dim
            feat /= feat.norm(dim=-1, keepdim=True)
        return feat

    def check_color(self, image_features, clip_encoded_color_names, eval_colors, label):
        with torch.no_grad(), torch.autocast("cuda"):
            text_probs = (100.0 * image_features @ clip_encoded_color_names.T).softmax(
                dim=-1
            )[0]
        top_indice = np.argsort(-text_probs)  #

        ranked_colors = np.array(eval_colors)[top_indice]

        index = min(np.where(ranked_colors == self.color_expected)[0])

        if index < 5:
            score = 1.0
        else:
            score = 1 - min((index - 5) / accepted_color_ratio, 1)

        # condition = (
        #    text_probs[0] > 0.5
        #    or self.color_expected in most_similar_colors
        #    or any(self.color_expected in cur_col for cur_col in most_similar_colors)
        # )

        feedback = f"The color of the {label} should have been {self.color_expected.removesuffix(" color")}, but is closer to {", ".join([c.removesuffix(" color") for c in ranked_colors[:3]])}."

        if self.negated:
            score = 1 - score
            feedback = f"The color of the {label} should not have been {self.color_expected.removesuffix(" color")}, but is still too close to {self.color_expected.removesuffix(" color")}."

        return FeedBack(feedback, score)

    def evaluate(
        self, *, original_image, custom_image, segment_function, box_function, **kwargs
    ):
        cust_features = segment_function(self.feature, custom_image)
        if feed := FeedBack.from_detection(cust_features, self.feature, "customized"):
            return feed

        custom_colors_counts = [
            self.get_feature_colors(seg, custom_image) for seg in cust_features
        ]
        color_images = [
            image_from_color_count(custom_color_count)
            for custom_color_count in custom_colors_counts
        ]

        eval_colors = [self.color_expected] + basic_colors

        clip_encoded_color_names = clip_model.encode_text(clip_tokenizer(eval_colors))

        image_features = [
            self.get_image_features(color_image) for color_image in color_images
        ]
        cond_feed = [
            self.check_color(
                image_feature, clip_encoded_color_names, eval_colors, seg.label
            )
            for image_feature, seg in zip(image_features, cust_features)
        ]

        return (
            FeedBackAndList(cond_feed)
            if not self.negated
            else FeedBackOrList(cond_feed)
        )


class size(OracleCondition):
    def __init__(self, feature: str, ratio: tuple[float, float]):
        self.ratio = ratio
        self.negated = False
        self.delta = 0.2
        super().__init__(feature)

    def __invert__(self):
        self.negated = True
        return self

    def check_size(self, ori_feature: BoundingBox, cust_feature: BoundingBox):
        x_ratio = (cust_feature.x1 - cust_feature.x0) / (
            ori_feature.x1 - ori_feature.x0
        )
        y_ratio = (cust_feature.y1 - cust_feature.y0) / (
            ori_feature.y1 - ori_feature.y0
        )

        x_score = 1 - min(
            abs(x_ratio - self.ratio[0]) / (self.delta * self.ratio[0]), 1
        )
        y_score = 1 - min(
            abs(y_ratio - self.ratio[1]) / (self.delta * self.ratio[1]), 1
        )

        if self.negated:
            return FeedBackOr(
                FeedBack(
                    f"The {ori_feature.label} was resized on x by a ratio of {x_ratio}, which is too close to {self.ratio[0]}",
                    1 - x_score,
                ),
                FeedBack(
                    f"The {ori_feature.label} was resized on y by a ratio of {y_ratio}, which is too close to {self.ratio[1]}",
                    1 - y_score,
                ),
            )
        else:
            return FeedBackAnd(
                FeedBack(
                    f"The {ori_feature.label} was resized on x by a ratio of {x_ratio}, but should have been by a ratio of {self.ratio[0]}",
                    x_score,
                ),
                FeedBack(
                    f"The {ori_feature.label} was resized on y by a ratio of {y_ratio}, but should have been by a ratio of {self.ratio[1]}",
                    y_score,
                ),
            )

    def evaluate(
        self, *, original_image, custom_image, segment_function, box_function, **kwargs
    ):

        ori_features = box_function(self.feature, original_image)
        cust_features = box_function(self.feature, custom_image)
        if (
            feed := FeedBack.from_detection(ori_features, self.feature, "original")
        ) or (
            feed := FeedBack.from_detection(cust_features, self.feature, "customized")
        ):
            return feed

        feature_correspondance = get_corresponding_features_detections(
            ori_features, cust_features
        )
        seg_mapping = [
            (
                [seg for seg in ori_features if seg.label == key_label][0],
                [seg for seg in cust_features if seg.label == value_label][0],
            )
            for key_label, value_label in feature_correspondance.items()
        ]
        cond_feed = [
            self.check_size(ori_seg, custom_seg) for ori_seg, custom_seg in seg_mapping
        ]

        return (
            FeedBackAndList(cond_feed)
            if not self.negated
            else FeedBackOrList(cond_feed)
        )


##model settings for shape and color detection
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
device = torch.device("cpu")
clip_model = clip_model.to(device)
clip_model.eval()
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
accepted_shape_ratio = math.floor((1 / 10) * len(shapes))


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

    def check_shape(self, mask_image, all_shapes, clip_encoded_shape_names, label):
        image_features = self.get_image_features(mask_image)
        with torch.no_grad(), torch.autocast("cuda"):
            text_probs = (100.0 * image_features @ clip_encoded_shape_names.T).softmax(
                dim=-1
            )
        top_indices = np.argsort(-text_probs[0])

        ranked_shapes = np.array(all_shapes)[top_indices]

        index = min(np.where(ranked_shapes == self.req_shape)[0])

        if index < 2:
            score = 1.0
        else:
            score = 1 - min((index - 2) / accepted_shape_ratio, 1)

        feedback = f"The {label} should be in the shape of a {self.req_shape}, but looks more like a {ranked_shapes[0]}."

        if self.negated:
            score = 1 - score
            feedback = f"The {label} should not be in the shape of a {self.req_shape}, but still looks like a {self.req_shape}."

        return FeedBack(feedback, score)

    def evaluate(
        self, *, original_image, custom_image, segment_function, box_function, **kwargs
    ):
        cust_features = segment_function(self.feature, custom_image)
        if feed := FeedBack.from_detection(cust_features, self.feature, "customized"):
            return feed

        mask_images = [
            (Image.fromarray(cust_feature.mask), cust_feature.label)
            for cust_feature in cust_features
        ]

        all_shapes = shapes + [self.req_shape]

        clip_encoded_shape_names = clip_model.encode_text(clip_tokenizer(all_shapes))

        cond_feed = [
            self.check_shape(mask_image, all_shapes, clip_encoded_shape_names, label)
            for mask_image, label in mask_images
        ]

        return (
            FeedBackAndList(cond_feed)
            if not self.negated
            else FeedBackOrList(cond_feed)
        )


class within(OracleCondition):
    def __init__(self, feature: str, other_feature: str):
        self.other_feature = other_feature
        self.negated = False
        super().__init__(feature)

    def __invert__(self):
        self.negated = True
        return self

    def check_within(self, segA: SegmentationMask, segB: SegmentationMask):
        score = np.logical_and(segA.mask, segB.mask).sum() / segA.mask.sum()

        feedback = f"The {segA.label} should be contained in the feature {segB.label}, but isn't."

        if self.negated:
            score = 1 - score
            feedback = f"The {segA.label} should not be contained in the feature {segB.label}, but is actually within it."
        return FeedBack(feedback, score)

    def evaluate(
        self, *, original_image, custom_image, segment_function, box_function, **kwargs
    ):
        custom_segs_featA = segment_function(self.feature, custom_image)

        custom_segs_featB = segment_function(self.other_feature, custom_image)

        if (
            feed := FeedBack.from_detection(
                custom_segs_featA, self.feature, "customized"
            )
        ) or (
            feed := FeedBack.from_detection(
                custom_segs_featB, self.other_feature, "customized"
            )
        ):
            return feed
        custom_seg_featB = custom_segs_featB[0]

        cond_feed = [
            self.check_within(segA, custom_seg_featB) for segA in custom_segs_featA
        ]

        return (
            FeedBackAndList(cond_feed)
            if not self.negated
            else FeedBackOrList(cond_feed)
        )


class mirrored(OracleCondition):
    def __init__(self, feature: str, axis: Axis):
        self.negated = False
        self.axis = axis
        super().__init__(feature)

    def __invert__(self):
        self.negated = True
        return self

    def check_mirrored(
        self,
        ori_seg: SegmentationMask,
        custom_seg: SegmentationMask,
        original_image,
        custom_image,
    ):
        ori_box = (ori_seg.x0, ori_seg.x1, ori_seg.y0, ori_seg.y1)
        custom_box = (custom_seg.x0, custom_seg.x1, custom_seg.y0, custom_seg.y1)
        cropped1 = apply_mask(original_image, ori_seg.mask)
        cropped2 = apply_mask(custom_image, custom_seg.mask)

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
        inv_norm_mse = 1 - nmse_np(mirrored_cropped1, cropped2)
        if inv_norm_mse > 0.95:
            score = 1
        else:
            score = max((inv_norm_mse - 0.5) / 0.45, 0)

        if self.negated:
            score = 1 - score
            feedback = f"The {ori_seg.label} should not be mirrored along the {self.axis} axis."
        else:
            feedback = (
                f"The {ori_seg.label} should be mirrored along the {self.axis} axis."
            )

        return FeedBack(feedback, score)

    def evaluate(
        self, *, original_image, custom_image, segment_function, box_function, **kwargs
    ):
        ori_segs = segment_function(self.feature, original_image)
        custom_segs = segment_function(self.feature, custom_image)

        if (feed := FeedBack.from_detection(ori_segs, self.feature, "original")) or (
            feed := FeedBack.from_detection(custom_segs, self.feature, "customized")
        ):
            return feed

        feature_correspondance = get_corresponding_features_detections(
            ori_segs, custom_segs
        )
        seg_mapping = [
            (
                [seg for seg in ori_segs if seg.label == key_label][0],
                [seg for seg in custom_segs if seg.label == value_label][0],
            )
            for key_label, value_label in feature_correspondance.items()
        ]
        cond_feed = [
            self.check_mirrored(ori_seg, custom_seg, original_image, custom_image)
            for ori_seg, custom_seg in seg_mapping
        ]

        return (
            FeedBackAndList(cond_feed)
            if not self.negated
            else FeedBackOrList(cond_feed)
        )


import statistics


class aligned(OracleCondition):
    def __init__(self, feature: str, other_feature: str, axis: Axis):
        self.other_feature = other_feature
        self.axis = axis
        self.negated = False
        super().__init__(feature)

    def __invert__(self):
        self.negated = True
        return self

    def evaluate(
        self, *, original_image, custom_image, segment_function, box_function, **kwargs
    ):
        custom_boxes_featA = box_function(self.feature, custom_image)
        custom_boxes_featB = box_function(self.other_feature, custom_image)
        if (
            feed := FeedBack.from_detection(
                custom_boxes_featA, self.feature, "customized"
            )
        ) or (
            feed := FeedBack.from_detection(
                custom_boxes_featB, self.other_feature, "customized"
            )
        ):
            return feed

        box_center_boxes = [
            (box, get_box_center(box))
            for box in custom_boxes_featA + custom_boxes_featB
        ]

        accepted_x_delta = 0.1 * max((box.x1 - box.x0) for box, _ in box_center_boxes)
        accepted_y_delta = 0.1 * max((box.y1 - box.y0) for box, _ in box_center_boxes)

        x_med = statistics.median_low([center[0] for _, center in box_center_boxes])
        y_med = statistics.median_low([center[1] for _, center in box_center_boxes])

        label_score_x = [
            (
                box.label,
                (1 - min(abs(center[0] - x_med) / accepted_x_delta, 1)),
            )
            for box, center in box_center_boxes
        ]
        label_score_y = [
            (
                box.label,
                (1 - min(abs(center[1] - y_med) / accepted_y_delta, 1)),
            )
            for box, center in box_center_boxes
        ]

        match self.axis:
            case Axis.horizontal:
                label_conditions = label_score_y

            case Axis.vertical:
                label_conditions = label_score_x
        condition = all([condition for _, condition in label_conditions])
        feedbacks = [
            FeedBack(
                f"The {label} should be aligned {self.axis}ly w.r.t. the features {self.feature} and {self.other_feature}.",
                score,
            )
            for label, score in label_conditions
        ]
        if self.negated:
            condition = not condition
            feedbacks = [
                FeedBack(
                    f"The {label} should not be aligned {self.axis}ly w.r.t. the features {self.feature} and {self.other_feature}.",
                    1 - score,
                )
                for label, score in label_conditions
            ]
        return (
            FeedBackAndList(feedbacks)
            if not self.negated
            else FeedBackOrList(feedbacks)
        )


class count(OracleCondition):
    def __init__(self, feature: str, amount: int):
        self.amount = amount
        self.negated = False
        super().__init__(feature)

    def __invert__(self):
        self.negated = True
        return self

    def evaluate(
        self, *, original_image, custom_image, segment_function, box_function, **kwargs
    ):
        custom_boxes_feat = box_function(self.feature, custom_image)
        condition = len(custom_boxes_feat) == self.amount

        feedback = f"The number of {self.feature} is {len(custom_boxes_feat)}, but should be {self.amount}."

        if self.negated:
            condition = not condition
            feedback = f"The number of {self.feature} should not be {self.amount}."
        return FeedBack(feedback, 0.8 if condition else 0.2)
