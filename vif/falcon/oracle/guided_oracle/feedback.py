import json
from typing import Self, Union

type FeedBacks = FeedBack | FeedBackCond | FeedBackListBase
import math


def p_sigmoid(x, a, b):
    return 1 / (1 + math.exp(-b * (x - a)))


class FeedBack:
    def __init__(
        self,
        feedback: str,
        score: float,
        name: str,
        a: float = 50.0,
        b: float = 1,
        negated=False,
    ):
        self.feedback = feedback
        self.score = score
        self.a = a
        self.b = b
        self.negated = negated
        self.probability = p_sigmoid(score, self.a, self.b)
        if self.negated:
            self.probability = 1-self.probability
        self.name = name

    def tojson(self, threshold: float = 1) -> dict | None:
        if self.probability > threshold:
            return None
        return {
            "type": "FeedBack",
            "feedback": self.feedback,
            "probability": round(self.probability, 2),
            "name": self.name,
            "score": self.score,
            "negated": self.negated,
        }

    @staticmethod
    def from_detection(
        detection_items: list, feature_name: str, im_name: str = "customized"
    ):
        if len(detection_items) == 0:
            return FeedBack(
                f"No feature {feature_name} was detected in the {im_name} image", 0
            )
        return False


class FeedBackCond:
    def __init__(
        self, feedbackA: Union[Self, FeedBack], feedbackB: Union[Self, FeedBack]
    ):
        self.feedbackA = feedbackA
        self.feedbackB = feedbackB
        self.probability = 0

    def tojson(self, threshold: float = 1) -> dict | None:
        if self.probability > threshold:
            return None

        a = self.feedbackA.tojson(threshold)
        b = self.feedbackB.tojson(threshold)

        # If both children filtered out, skip the whole node
        if a is None and b is None:
            return None

        return {
            "type": self.__class__.__name__,
            "probability": round(self.probability, 2),
            "feedbackA": a,
            "feedbackB": b,
        }


class FeedBackOr(FeedBackCond):
    def __init__(self, feedbackA, feedbackB):
        super().__init__(feedbackA, feedbackB)
        self.probability = 1 - (
            (1 - self.feedbackA.probability) * (1 - self.feedbackB.probability)
        )


class FeedBackAnd(FeedBackCond):
    def __init__(self, feedbackA, feedbackB):
        super().__init__(feedbackA, feedbackB)
        self.probability = self.feedbackA.probability * self.feedbackB.probability


class FeedBackListBase:
    """Base class that enforces items are ONLY FeedBack."""

    def __init__(self, items: list[FeedBack]):
        if not items:
            raise ValueError("List cannot be empty")
        self.items = items
        self.probability = 0

    def tojson(self, threshold: float = 1):
        if self.probability > threshold:
            return None

        children = [x.tojson(threshold) for x in self.items]
        children = [c for c in children if c is not None]
        if not children:
            return None

        return {
            "type": self.__class__.__name__,
            "probability": round(self.probability, 2),
            "items": children,
        }


class FeedBackOrList(FeedBackListBase):
    def __init__(self, items: list[FeedBack]):
        super().__init__(items)
        p = 1
        for x in self.items:
            p *= 1 - x.probability
        self.probability = 1 - p


class FeedBackAndList(FeedBackListBase):
    def __init__(self, items: list[FeedBack]):
        super().__init__(items)
        p = 1
        for x in self.items:
            p *= x.probability
        self.probability = p
