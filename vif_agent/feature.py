from dataclasses import dataclass
from PIL import Image

from sentence_transformers import SentenceTransformer
import torch


type Box2D = tuple[float, float, float, float]
type Span = tuple[int, int]
import copy


@dataclass
class CodeEdit:
    start: int
    end: int
    content: str = None


@dataclass
class CodeImageMapping:
    """code to image mapping"""

    spans: list[Span]
    box_zone: Box2D
    segment_zone: Image.Image | None = None


class MappedCode:
    def __init__(
        self,
        image: Image.Image,
        code: str,
        feature_map: dict[str, list[tuple[CodeImageMapping, float]]] = None,
    ):
        self.image = image
        self.code = code
        self.feature_map = feature_map
        self.original_code = code

        if feature_map is not None:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.key_embeddings = self.embedding_model.encode(
                list(self.feature_map.keys())
            )

    def get_commented(self, comment_character: str = "%") -> str:
        if self.feature_map is None:
            return self.original_code

        char_id_feature: dict = (
            {}
        )  # mapping between the character index and the detected feature

        for feature_name, prob_mappings in self.feature_map.items():
            for mapping, prob in prob_mappings:
                features_for_char: list = char_id_feature.get(
                    mapping.spans[0][0], []
                )  # getting the first start char nb of the first tuple
                features_for_char.append((feature_name, prob))
                char_id_feature[mapping.spans[0][0]] = sorted(
                    features_for_char, key=lambda x: x[1], reverse=True
                )  # order the labels by the mse

        annotated_code = self.original_code
        # Annotate the code
        for characted_index, labels in sorted(char_id_feature.items(), reverse=True):
            labels = [label[0] for label in labels]  # removing the mse
            if len(labels) == len(
                self.feature_map
            ):  # all features have been detected for the modifications of this mutant, skipping
                continue
            selected_features = labels[0]  #
            annotated_code = (
                annotated_code[:characted_index]
                + comment_character
                + selected_features
                + "\n"
                + annotated_code[characted_index:]
            )
        return annotated_code

    def apply_edits(self, edits: list[CodeEdit]):
        """applies edits to the code

        span adaptations not done for now, because the llm only asks for it once

        Args:
            edits (list[CodeEdit]): The list of edits to apply to the code
        """

        edits = sorted(edits, key=lambda a: a.start, reverse=True)

        if any([edits[i].start < edits[i + 1].end for i in range(len(edits) - 1)]):
            raise ValueError("Ranges are overlaping, cancelling edits.")

        # Modifying code
        splitted_code = self.code.split("\n")
        for edit in edits:
            # -1 because the codeedit specify lines which start at 1, but not +1 for end even splice is [a,b[ because last index replaced

            splitted_code[edit.start - 1 : edit.end - 1] = []
            if edit.content is not None:
                splitted_code.insert(edit.start - 1, edit.content)

        self.code = "\n".join(splitted_code)
        return self.code

    def get_annotated(self):
        """Returns the code annotated with line numbers"""
        return "\n".join([str(i + 1) + "|" + line for (i, line) in enumerate(self.code.split("\n"))])

    def get_cimappings(self, feature: str) -> list[tuple[CodeImageMapping, float]]:
        """Gets a CodeImageMapping(parts of the code and the associated part of the image)
        from a string, i.e. given a string, computes which feature_names are the most similar,
        and return a list of the most probable CodeImageMapping

        Args:
            feature (str): Any string

        Returns:
            list[tuple[CodeImageMapping, float]]: Most probable part of the code/Image that the feature is in
        """
        if self.feature_map is None:
            return []

        asked_feature_embedding = self.embedding_model.encode(feature)
        similarities: torch.Tensor = self.embedding_model.similarity(
            self.key_embeddings, asked_feature_embedding
        )
        # normalize similarities
        min_sim = similarities.min()
        max_sim = similarities.max()
        similarities = (similarities - min_sim) / (max_sim - min_sim + 1e-8)

        adjusted_map: dict[str, list[tuple[CodeImageMapping, float]]] = {}

        for (feature_name, prob_mappings), similarity in zip(
            self.feature_map.items(), similarities
        ):
            adjusted_mappings = []
            for mapping, prob in prob_mappings:
                adjusted_mappings.append([mapping, prob * (similarity**10)])

            adjusted_map[feature_name] = adjusted_mappings

        all_mappings: list[tuple[CodeImageMapping, float]] = []

        for prob_map in adjusted_map.values():
            all_mappings = all_mappings + prob_map

        return sorted(all_mappings, key=lambda x: x[1], reverse=True)
